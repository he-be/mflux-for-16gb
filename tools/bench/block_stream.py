import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from memstat import MemStat
from mlx.utils import tree_unflatten

import mflux.models.krea2.model.krea2_scheduler  # noqa: F401 — registers the euler/er_sde schedulers
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.krea2.latent_creator.krea2_latent_creator import Krea2LatentCreator
from mflux.models.krea2.model.krea2_sampler import Krea2Sampler
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition

# Plan M5: stream the 28 transformer blocks instead of holding them. M3 settled that the
# weights are 86% of a demand this machine cannot meet; one block is 461.3 MB, so holding
# one at a time should put residency near 1 GB. The question M5 answers is whether the
# compute hides the I/O: the pass mark is compute / I/O >= 2.0, residency back to baseline
# +100 MB after each drop, and a step within +20% of M3's 28.15 s.
#
# Nothing in mflux changes. Krea2Transformer's forward does `for block in self.blocks`, so
# replacing that list with wrappers that bind, run, evaluate and drop streams the real
# forward pass exactly as written.


class BlockReader:
    def __init__(self, root: Path):
        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.by_block: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for key, shard in index.items():
            if key.startswith("blocks."):
                self.by_block[int(key.split(".")[1])][shard].append(key)

    def read(self, index: int) -> dict:
        # Fresh lazy handles every call. Reusing one loaded tree would keep the evaluated
        # arrays referenced, and then dropping a block would free nothing.
        prefix = f"blocks.{index}."
        flat = []
        for shard, keys in self.by_block[index].items():
            data = mx.load(str(self.root / shard))
            flat.extend((key[len(prefix) :], data[key]) for key in keys)
        return tree_unflatten(flat)


class StreamedBlock:
    def __init__(self, index: int, block, reader: BlockReader, stats: list):
        self.index = index
        self.block = block
        self.reader = reader
        self.stats = stats

    def __call__(self, combined: mx.array, tvec: mx.array, freqs: mx.array, mask):
        start = time.perf_counter()
        self.block.update(self.reader.read(self.index))
        mx.eval(self.block.parameters())
        bound = time.perf_counter()

        out = self.block(combined, tvec, freqs, mask)
        # MLX is lazy, so without this the weights would still be needed after the drop.
        # Streaming forces a sync per block; that cost is part of what M5 is measuring.
        mx.eval(out)
        computed = time.perf_counter()

        self.block.update(self.reader.read(self.index))
        mx.clear_cache()
        dropped = time.perf_counter()

        self.stats.append(
            {
                "block": self.index,
                "io_s": bound - start,
                "compute_s": computed - bound,
                "drop_s": dropped - computed,
                "resident_gb": mx.get_active_memory() / 1e9,
            }
        )
        return out


class Krea2StreamBench:
    GLOBAL_MODULES = ("first", "tmlp", "tproj", "txtfusion", "txtmlp", "last")

    def __init__(self, model_path: Path, embeds_path: Path, out_path: Path, args: argparse.Namespace):
        self.model_path = model_path
        self.embeds_path = embeds_path
        self.out_path = out_path
        self.args = args
        self.timings: dict[str, float] = {}
        self.stats: list[dict] = []

    def run(self) -> dict:
        if self.args.wired_limit_gb is not None:
            mx.set_wired_limit(int(self.args.wired_limit_gb * 1e9))
        before = MemStat.capture(label="stream-before")
        before.report()

        transformer, bits = self._build()
        baseline = mx.get_active_memory()
        print(f"   globals resident  : {baseline / 1e9:.3f} GB (blocks still on disk)")

        embeds = self._read_embeds()
        latents, config, step_times = self._denoise(transformer, embeds)
        self._save(latents, config)

        return self._result(bits, baseline, config, step_times, before)

    def _build(self) -> tuple[Krea2Transformer, int | None]:
        component = next(c for c in Krea2WeightDefinition.get_components() if c.name == "transformer")
        start = time.perf_counter()
        transformer = Krea2Transformer()
        weights = WeightLoader.load_single_local(component=component, root_path=self.model_path)
        bits = WeightApplier.apply_and_quantize_single(
            weights=weights,
            model=transformer,
            component=component,
            quantize_arg=None,
            quantization_predicate=Krea2WeightDefinition.quantization_predicate,
        )
        del weights
        # Materialize only what stays resident. The blocks keep the lazy handles from the
        # apply above and are never evaluated here; each wrapper rebinds its own on first use.
        for name in Krea2StreamBench.GLOBAL_MODULES:
            mx.eval(getattr(transformer, name).parameters())
        mx.clear_cache()
        self.timings["globals"] = time.perf_counter() - start

        reader = BlockReader(self.model_path)
        transformer.blocks = [StreamedBlock(i, b, reader, self.stats) for i, b in enumerate(transformer.blocks)]
        return transformer, bits

    def _read_embeds(self) -> mx.array:
        arrays = mx.load(str(self.embeds_path))
        embeds = arrays["embeds"]
        mx.eval(embeds)
        return embeds

    def _denoise(self, transformer: Krea2Transformer, embeds: mx.array) -> tuple[mx.array, Config, list[float]]:
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
        stepper = Krea2Sampler.make_stepper("euler", sigmas, self.args.seed)

        step_times = []
        for t in range(config.num_inference_steps):
            start = time.perf_counter()
            ts = sigmas[t].reshape(1)
            v = transformer(latents, ts, embeds)
            denoised = latents - sigmas[t] * v
            latents = stepper.step(t, latents, v, denoised)
            mx.eval(latents)
            step_times.append(time.perf_counter() - start)
            recent = self.stats[-28:]
            io = sum(s["io_s"] for s in recent)
            compute = sum(s["compute_s"] for s in recent)
            print(f"   step {t + 1}/{config.num_inference_steps}: {step_times[-1]:.2f} s  (I/O {io:.2f} s, compute {compute:.2f} s, ratio {compute / io:.2f})")  # fmt: skip
        return latents, config, step_times

    def _save(self, latents: mx.array, config: Config) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(
            str(self.out_path),
            {"latents": latents},
            metadata={
                "seed": str(self.args.seed),
                "steps": str(config.num_inference_steps),
                "scheduler": "euler",
                "height": str(config.height),
                "width": str(config.width),
                "mode": "block-streaming",
            },
        )

    def _result(self, bits, baseline: int, config: Config, step_times: list[float], before) -> dict:
        io = [s["io_s"] for s in self.stats]
        compute = [s["compute_s"] for s in self.stats]
        drop = [s["drop_s"] for s in self.stats]
        resident = [s["resident_gb"] for s in self.stats]
        result = {
            "stage": "M5-block-streaming",
            "model_path": str(self.model_path),
            "stored_bits": bits,
            "resolution": [config.width, config.height],
            "steps": config.num_inference_steps,
            "blocks_per_step": len(self.stats) // max(len(step_times), 1),
            "timings_s": {k: round(v, 3) for k, v in self.timings.items()},
            "step_times_s": [round(t, 3) for t in step_times],
            "io_s_mean": round(sum(io) / len(io), 4),
            "compute_s_mean": round(sum(compute) / len(compute), 4),
            "drop_s_mean": round(sum(drop) / len(drop), 4),
            "ratio_compute_over_io": round((sum(compute) / len(compute)) / (sum(io) / len(io)), 2),
            "globals_resident_gb": round(baseline / 1e9, 3),
            "resident_after_drop_gb_max": round(max(resident), 3),
            "resident_over_baseline_mb_max": round((max(resident) - baseline / 1e9) * 1000, 1),
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
            "active_memory_gb": round(mx.get_active_memory() / 1e9, 3),
            "per_block_first_step": self.stats[:28],
            "memstat_before": {"claimable_gb": round(before.claimable_gb, 3), "when": before.when},
        }
        print("🧮 M5 block streaming")
        print(f"   globals resident  : {result['globals_resident_gb']:.3f} GB")
        print(f"   I/O per block     : {result['io_s_mean'] * 1000:.1f} ms")
        print(f"   compute per block : {result['compute_s_mean'] * 1000:.1f} ms")
        print(f"   drop per block    : {result['drop_s_mean'] * 1000:.1f} ms")
        print(f"   compute / I/O     : {result['ratio_compute_over_io']:.2f}  (pass >= 2.0)")
        print(f"   resident after drop (max over baseline): {result['resident_over_baseline_mb_max']:.0f} MB")
        print(f"   mx peak memory    : {result['peak_memory_gb']:.2f} GB")
        print(f"   per step          : {result['step_times_s']} s")
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan M5: stream the Krea 2 DiT blocks from disk.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--embeds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wired-limit-gb", type=float, default=None)
    args = parser.parse_args()

    result = Krea2StreamBench(model_path=args.model, embeds_path=args.embeds, out_path=args.out, args=args).run()
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"   json              : {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
