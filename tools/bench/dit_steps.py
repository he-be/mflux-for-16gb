import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from memstat import MemStat

import mflux.models.krea2.model.krea2_scheduler  # noqa: F401 — registers the euler/er_sde schedulers
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.krea2.latent_creator.krea2_latent_creator import Krea2LatentCreator
from mflux.models.krea2.model.krea2_sampler import Krea2Sampler
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.weights.krea2_lora_mapping import Krea2LoRAMapping
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition

# Plan M3, the one that decides the project: the q8 DiT and the 4-step LoRA resident in a
# process that holds neither the text encoder nor the VAE. Reads the conditioning M2 wrote,
# runs the denoise loop and writes the latent for M4 to decode. Run it under
# tools/swapwatch.py; a run that swapped is a failure, not a slow success.
#
# The escalation ladder from the plan is exposed as flags, to be tried in this order and
# only as far as needed: (1) nothing, (2) --cache-limit-gb / --clear-cache-each-step,
# (3) a raised iogpu.wired_limit_mb plus --wired-limit-gb, (4) --width/--height 768.


class Krea2DitBench:
    def __init__(self, model_path: Path, embeds_path: Path, out_path: Path, args: argparse.Namespace):
        self.model_path = model_path
        self.embeds_path = embeds_path
        self.out_path = out_path
        self.args = args
        self.timings: dict[str, float] = {}
        self.step_times: list[float] = []
        self.step_peaks: list[float] = []

    def run(self) -> dict:
        self._apply_limits()
        before = MemStat.capture(label="dit-before")
        before.report()

        transformer = self._build()
        bits = self._load(transformer)
        loaded_peak = mx.get_peak_memory()
        print(f"   DiT resident      : active {mx.get_active_memory() / 1e9:.2f} GB, peak {loaded_peak / 1e9:.2f} GB")

        embeds, prompt = self._read_embeds()
        latents, config = self._denoise(transformer, embeds)
        self._save(latents, prompt, config, bits)

        after = MemStat.capture(label="dit-after")
        result = {
            "stage": "M3-dit",
            "model_path": str(self.model_path),
            "prompt": prompt,
            "stored_bits": bits,
            "resolution": [config.width, config.height],
            "steps": config.num_inference_steps,
            "scheduler": "euler",
            "seed": self.args.seed,
            "lora": None if self.args.no_lora else self.args.lora_path,
            "lora_scale": None if self.args.no_lora else self.args.lora_scale,
            "compiled": bool(self.args.compile),
            "limits": {
                "cache_limit_gb": self.args.cache_limit_gb,
                "wired_limit_gb": self.args.wired_limit_gb,
                "clear_cache_each_step": self.args.clear_cache_each_step,
            },
            "timings_s": {k: round(v, 3) for k, v in self.timings.items()},
            "step_times_s": [round(t, 3) for t in self.step_times],
            "step_peak_memory_gb": [round(p / 1e9, 3) for p in self.step_peaks],
            "dit_resident_gb": round(loaded_peak / 1e9, 3),
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
            "active_memory_gb": round(mx.get_active_memory() / 1e9, 3),
            "cache_memory_gb": round(mx.get_cache_memory() / 1e9, 3),
            "latent_shape": list(latents.shape),
            "out_path": str(self.out_path),
            "memstat_before": {"claimable_gb": round(before.claimable_gb, 3), "when": before.when},
            "memstat_after": {"claimable_gb": round(after.claimable_gb, 3), "when": after.when},
        }
        self._report(result)
        return result

    def _apply_limits(self) -> None:
        if self.args.cache_limit_gb is not None:
            mx.set_cache_limit(int(self.args.cache_limit_gb * 1e9))
        if self.args.wired_limit_gb is not None:
            # Only meaningful once iogpu.wired_limit_mb has been raised to match: MLX
            # refuses a wired limit above what the system allows.
            mx.set_wired_limit(int(self.args.wired_limit_gb * 1e9))

    def _build(self) -> Krea2Transformer:
        start = time.perf_counter()
        transformer = Krea2Transformer()
        self.timings["build"] = time.perf_counter() - start
        return transformer

    def _load(self, transformer: Krea2Transformer) -> int | None:
        component = next(c for c in Krea2WeightDefinition.get_components() if c.name == "transformer")
        start = time.perf_counter()
        weights = WeightLoader.load_single_local(component=component, root_path=self.model_path)
        self.timings["read"] = time.perf_counter() - start

        start = time.perf_counter()
        bits = WeightApplier.apply_and_quantize_single(
            weights=weights,
            model=transformer,
            component=component,
            quantize_arg=None,
            quantization_predicate=Krea2WeightDefinition.quantization_predicate,
        )
        del weights
        # mx.load is lazy; without this the read above is only a mmap and the weights
        # would materialize inside the first step's timing.
        mx.eval(transformer)
        mx.clear_cache()
        self.timings["materialize"] = time.perf_counter() - start

        if not self.args.no_lora:
            start = time.perf_counter()
            LoRALoader.load_and_apply_lora(
                lora_mapping=Krea2LoRAMapping.get_mapping(),
                transformer=transformer,
                lora_paths=[self.args.lora_path],
                lora_scales=[self.args.lora_scale],
                bake_lora=True,
            )
            mx.eval(transformer)
            mx.clear_cache()
            self.timings["lora"] = time.perf_counter() - start
        return bits

    def _read_embeds(self) -> tuple[mx.array, str]:
        arrays, metadata = mx.load(str(self.embeds_path), return_metadata=True)
        embeds = arrays["embeds"]
        mx.eval(embeds)
        return embeds, metadata.get("prompt", "")

    def _denoise(self, transformer: Krea2Transformer, embeds: mx.array) -> tuple[mx.array, Config]:
        config = Config(
            model_config=ModelConfig.krea2(),
            num_inference_steps=self.args.steps,
            height=self.args.height,
            width=self.args.width,
            guidance=1.0,
            scheduler="euler",
        )
        sigmas = config.scheduler.sigmas
        latents = Krea2LatentCreator.create_noise(self.args.seed, config.height, config.width)
        mx.eval(latents)

        predict = self._make_predict(transformer, embeds)
        stepper = Krea2Sampler.make_stepper("euler", sigmas, self.args.seed)

        loop_start = time.perf_counter()
        for t in range(config.num_inference_steps):
            start = time.perf_counter()
            ts = sigmas[t].reshape(1)
            v = predict(latents=latents, timestep=ts)
            denoised = latents - sigmas[t] * v
            latents = stepper.step(t, latents, v, denoised)
            mx.eval(latents)
            self.step_times.append(time.perf_counter() - start)
            self.step_peaks.append(mx.get_peak_memory())
            print(f"   step {t + 1}/{config.num_inference_steps}: {self.step_times[-1]:.2f} s, peak {self.step_peaks[-1] / 1e9:.2f} GB")  # fmt: skip
            if self.args.clear_cache_each_step:
                mx.clear_cache()
        self.timings["loop"] = time.perf_counter() - loop_start
        return latents, config

    def _make_predict(self, transformer: Krea2Transformer, embeds: mx.array):
        # guidance is 1.0 for this configuration, so there is no negative pass: a step is
        # exactly one transformer call. mflux wraps this in mx.compile off M1/M2, which is
        # opt-in here because compilation is one more thing that can cost memory.
        def predict(latents: mx.array, timestep: mx.array) -> mx.array:
            return transformer(latents, timestep, embeds)

        return mx.compile(predict) if self.args.compile else predict

    def _save(self, latents: mx.array, prompt: str, config: Config, bits: int | None) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(
            str(self.out_path),
            {"latents": latents},
            metadata={
                "prompt": prompt,
                "model_path": str(self.model_path),
                "stored_bits": str(bits),
                "seed": str(self.args.seed),
                "steps": str(config.num_inference_steps),
                "scheduler": "euler",
                "height": str(config.height),
                "width": str(config.width),
                "lora": "" if self.args.no_lora else f"{self.args.lora_path}@{self.args.lora_scale}",
            },
        )

    @staticmethod
    def _report(result: dict) -> None:
        steps = result["step_times_s"]
        print("🧮 M3 DiT")
        print(f"   resolution        : {result['resolution'][0]}x{result['resolution'][1]}, {result['steps']} steps")
        print(f"   stored bits       : {result['stored_bits']}")
        print(f"   LoRA              : {result['lora']}")
        for name, value in result["timings_s"].items():
            print(f"   {name:<18}: {value:.2f} s")
        if steps:
            print(f"   per step          : {steps} s (mean {sum(steps) / len(steps):.2f})")
        print(f"   DiT resident      : {result['dit_resident_gb']:.2f} GB")
        print(f"   mx peak memory    : {result['peak_memory_gb']:.2f} GB")
        print(f"   mx active memory  : {result['active_memory_gb']:.2f} GB")
        print(f"   mx cache memory   : {result['cache_memory_gb']:.2f} GB")
        print(f"   latent            : {result['latent_shape']} -> {result['out_path']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan M3: run the Krea 2 q8 DiT alone over M2's conditioning and save the latent.",
    )
    parser.add_argument("--model", type=Path, required=True, help="path to the mflux q8 snapshot directory")
    parser.add_argument("--embeds", type=Path, required=True, help="conditioning written by tools/bench/te_encode.py")
    parser.add_argument("--out", type=Path, required=True, help="where to write the latent safetensors")
    parser.add_argument("--json", type=Path, default=None, help="also write the measurement to this JSON file")
    parser.add_argument("--steps", type=int, default=4, help="denoise steps (the 4-step LoRA wants 4)")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lora-path",
        default="lvladikov/Krea2-Turbo-Distill-4step-LoRA:krea2_turbo_4step_rank_64_lora_comfyui.safetensors",
        help="4-step distill LoRA (repo:file or a local path)",
    )
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--no-lora", action="store_true", help="skip the LoRA (isolates its 0.44 GB from the result)")
    parser.add_argument("--compile", action="store_true", help="wrap the step in mx.compile, as mflux does")
    parser.add_argument("--cache-limit-gb", type=float, default=None, help="ladder step 2: cap MLX's buffer cache")
    parser.add_argument("--clear-cache-each-step", action="store_true", help="ladder step 2: clear the cache per step")
    parser.add_argument("--wired-limit-gb", type=float, default=None, help="ladder step 3: wire the weights down")
    args = parser.parse_args()

    bench = Krea2DitBench(model_path=args.model, embeds_path=args.embeds, out_path=args.out, args=args)
    result = bench.run()

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"   json              : {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
