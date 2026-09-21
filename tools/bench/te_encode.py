import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from memstat import MemStat

from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.krea2.model.krea2_text_encoder.prompt_encoder import Krea2PromptEncoder
from mflux.models.krea2.model.krea2_text_encoder.text_encoder import KREA2_TAP_LAYERS, Krea2TextEncoder
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition

# Plan M2: the text encoder alone, in a process that holds nothing else. Encodes the
# prompt, writes the conditioning to safetensors and exits, so the DiT process (M3) never
# has to pay the 8.05 GB the encoder costs. Run it under tools/swapwatch.py - a result
# that swapped is not a result.

DEFAULT_PROMPT = (
    "a photograph of a weathered brass diving helmet on a workshop bench, "
    "morning light through a dusty window, shallow depth of field"
)


class Krea2TextEncoderBench:
    def __init__(self, model_path: Path, out_path: Path):
        self.model_path = model_path
        self.out_path = out_path
        self.timings: dict[str, float] = {}

    def run(self, prompt: str) -> dict:
        before = MemStat.capture(label="te-before")
        before.report()

        encoder = self._build()
        bits = self._load(encoder)
        embeds = self._encode(encoder, prompt)
        self._save(embeds, prompt, bits)

        after = MemStat.capture(label="te-after")
        result = {
            "stage": "M2-text-encoder",
            "model_path": str(self.model_path),
            "prompt": prompt,
            "stored_bits": bits,
            "embeds_shape": list(embeds.shape),
            "embeds_dtype": str(embeds.dtype).removeprefix("mlx.core."),
            "timings_s": {k: round(v, 3) for k, v in self.timings.items()},
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
            "active_memory_gb": round(mx.get_active_memory() / 1e9, 3),
            "cache_memory_gb": round(mx.get_cache_memory() / 1e9, 3),
            "out_path": str(self.out_path),
            "memstat_before": {"claimable_gb": round(before.claimable_gb, 3), "when": before.when},
            "memstat_after": {"claimable_gb": round(after.claimable_gb, 3), "when": after.when},
        }
        self._report(result)
        return result

    def _build(self) -> Krea2TextEncoder:
        start = time.perf_counter()
        encoder = Krea2TextEncoder()
        self.timings["build"] = time.perf_counter() - start
        return encoder

    def _load(self, encoder: Krea2TextEncoder) -> int | None:
        component = next(c for c in Krea2WeightDefinition.get_components() if c.name == "text_encoder")
        start = time.perf_counter()
        weights = WeightLoader.load_single_local(component=component, root_path=self.model_path)
        self.timings["read"] = time.perf_counter() - start

        start = time.perf_counter()
        bits = WeightApplier.apply_and_quantize_single(
            weights=weights,
            model=encoder,
            component=component,
            quantize_arg=None,
        )
        del weights
        # mx.load is lazy: without the eval the read above is only a mmap and the real
        # cost lands inside the encode timing instead.
        mx.eval(encoder)
        mx.clear_cache()
        self.timings["materialize"] = time.perf_counter() - start
        return bits

    def _encode(self, encoder: Krea2TextEncoder, prompt: str) -> mx.array:
        start = time.perf_counter()
        tokenizers = TokenizerLoader.load_all(
            definitions=Krea2WeightDefinition.get_tokenizers(),
            model_path=str(self.model_path),
        )
        self.timings["tokenizer"] = time.perf_counter() - start

        start = time.perf_counter()
        embeds = Krea2PromptEncoder.encode_prompt(prompt, tokenizers["qwen3vl"], encoder)
        mx.eval(embeds)
        self.timings["encode"] = time.perf_counter() - start
        return embeds

    def _save(self, embeds: mx.array, prompt: str, bits: int | None) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        mx.save_safetensors(
            str(self.out_path),
            {"embeds": embeds},
            metadata={
                "prompt": prompt,
                "model_path": str(self.model_path),
                "stored_bits": str(bits),
                "tap_layers": ",".join(str(i) for i in KREA2_TAP_LAYERS),
            },
        )
        self.timings["save"] = time.perf_counter() - start

    @staticmethod
    def _report(result: dict) -> None:
        print("🧮 M2 text encoder")
        print(f"   prompt            : {result['prompt'][:70]}...")
        print(f"   embeds            : {result['embeds_shape']} {result['embeds_dtype']} -> {result['out_path']}")
        print(f"   stored bits       : {result['stored_bits']}")
        for name, value in result["timings_s"].items():
            print(f"   {name:<18}: {value:.2f} s")
        print(f"   mx peak memory    : {result['peak_memory_gb']:.2f} GB")
        print(f"   mx active memory  : {result['active_memory_gb']:.2f} GB")
        print(f"   mx cache memory   : {result['cache_memory_gb']:.2f} GB")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan M2: encode a prompt with the Krea 2 text encoder alone and save the conditioning.",
    )
    parser.add_argument("--model", type=Path, required=True, help="path to the mflux q8 snapshot directory")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompt to encode")
    parser.add_argument("--out", type=Path, required=True, help="where to write the embeds safetensors")
    parser.add_argument("--json", type=Path, default=None, help="also write the measurement to this JSON file")
    args = parser.parse_args()

    bench = Krea2TextEncoderBench(model_path=args.model, out_path=args.out)
    result = bench.run(args.prompt)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"   json              : {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
