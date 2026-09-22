import argparse
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from memstat import MemStat
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition

# Quantizes the Qwen3-VL-4B text encoder once and writes it out, so runs load 4.28 GB
# instead of reading 8.05 GB of bf16 and quantizing it every time (which spikes to
# 12.08 GB while both copies are live). M5b measured what this costs in quality: the
# images are visually the same, detail density unchanged.
#
# One submodule at a time, the same trick the LoRA bake uses: mx.load is lazy, so
# evaluating one layer materializes one layer. 36 decoder layers at ~200 MB and the
# 0.78 GB embedding, never more than about 1 GB at once.


class TextEncoderQuantizer:
    def __init__(self, model_path: Path, out_path: Path, bits: int):
        self.model_path = model_path
        self.root = out_path
        # Laid out like an mflux snapshot - weights under text_encoder/, the tokenizer
        # beside it - so the result is a drop-in --model for tools/bench/te_encode.py.
        self.out_path = out_path / "text_encoder"
        self.bits = bits
        self.weight_map: dict[str, str] = {}
        self.timings: dict[str, float] = {}

    def run(self) -> dict:
        MemStat.capture(label="quantize-te-before").report()
        self.out_path.mkdir(parents=True, exist_ok=True)

        self._copy_tokenizer()
        encoder = self._build()
        self._write("embed_tokens", encoder.embed_tokens)
        start = time.perf_counter()
        for index, layer in enumerate(encoder.layers):
            self._write(f"layers.{index}", layer)
            if index in (0, len(encoder.layers) - 1):
                print(f"   layer {index:>2} written  : active {mx.get_active_memory() / 1e9:.3f} GB, peak {mx.get_peak_memory() / 1e9:.3f} GB")  # fmt: skip
        self.timings["layers"] = time.perf_counter() - start
        self._write("norm", encoder.norm)
        self._write_index()

        result = {
            "stage": "quantize-te-checkpoint",
            "source": str(self.model_path),
            "out": str(self.root),
            "bits": self.bits,
            "tensors_total": len(self.weight_map),
            "timings_s": {k: round(v, 3) for k, v in self.timings.items()},
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
            "size_gb": round(sum((self.out_path / n).stat().st_size for n in set(self.weight_map.values())) / 1e9, 3),
        }
        print("🧮 quantize text encoder")
        for name, value in result["timings_s"].items():
            print(f"   {name:<18}: {value:.2f} s")
        print(f"   tensors           : {result['tensors_total']}")
        print(f"   on disk           : {result['size_gb']:.2f} GB")
        print(f"   mx peak memory    : {result['peak_memory_gb']:.2f} GB")
        print(f"   out               : {self.root}")
        return result

    def _copy_tokenizer(self) -> None:
        source = self.model_path / "tokenizer"
        if source.is_dir():
            shutil.copytree(source, self.root / "tokenizer", dirs_exist_ok=True)

    def _build(self) -> Krea2TextEncoder:
        component = next(c for c in Krea2WeightDefinition.get_components() if c.name == "text_encoder")
        start = time.perf_counter()
        encoder = Krea2TextEncoder()
        weights = WeightLoader.load_single_local(component=component, root_path=self.model_path)
        WeightApplier.apply_and_quantize_single(weights=weights, model=encoder, component=component, quantize_arg=None)
        del weights
        # Quantize the whole encoder in one call, before anything is evaluated. Doing it
        # per submodule does not work: nn.quantize replaces a module's children, so handing
        # it embed_tokens leaves embed_tokens itself in bf16 - which is exactly what the
        # first version of this script shipped, 0.36 GB too large and not matching the
        # on-the-fly path. mx.quantize is lazy, so this costs nothing until _write evals.
        nn.quantize(encoder, group_size=64, bits=self.bits, class_predicate=TextEncoderQuantizer._quantizable)
        self.timings["wire"] = time.perf_counter() - start
        print(f"   after wire (lazy) : active {mx.get_active_memory() / 1e9:.3f} GB")
        return encoder

    def _write(self, name: str, module: nn.Module) -> None:
        params = module.parameters()
        mx.eval(params)
        flat = {f"{name}.{k}": v for k, v in tree_flatten(params)}
        filename = f"{name.replace('.', '_')}.safetensors"
        mx.save_safetensors(str(self.out_path / filename), flat, metadata=self._metadata())
        self.weight_map.update({k: filename for k in flat})
        module.update(tree_unflatten([(k, mx.array([], dtype=v.dtype)) for k, v in tree_flatten(params)]))
        del params, flat
        mx.clear_cache()

    @staticmethod
    def _quantizable(path: str, module) -> bool:
        # mx.quantize needs the last dim to be a multiple of the group size.
        if not hasattr(module, "to_quantized"):
            return False
        weight = getattr(module, "weight", None)
        return weight is not None and weight.shape[-1] % 64 == 0

    def _write_index(self) -> None:
        total = sum((self.out_path / name).stat().st_size for name in set(self.weight_map.values()))
        index = {
            "metadata": {"total_size": total, "quantization_level": str(self.bits)},
            "weight_map": self.weight_map,
        }
        (self.out_path / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    def _metadata(self) -> dict[str, str]:
        return {"quantization_level": str(self.bits), "mflux_version": "16gb-bench"}

    @staticmethod
    def loadable_component():
        # The definition sets skip_quantization=True, which would leave the module
        # unquantized and then update it with packed q8 tensors. Clearing the flag lets
        # WeightApplier rebuild the quantized structure from the stored scales. Harmless on
        # a bf16 checkpoint: with no scales present its predicate matches nothing.
        component = next(c for c in Krea2WeightDefinition.get_components() if c.name == "text_encoder")
        return replace(component, skip_quantization=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantize the Krea 2 text encoder once, one submodule at a time.")
    parser.add_argument("--model", type=Path, required=True, help="source mflux snapshot (bf16 text_encoder/)")
    parser.add_argument("--out", type=Path, required=True, help="directory to write the quantized encoder into")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    result = TextEncoderQuantizer(model_path=args.model, out_path=args.out, bits=args.bits).run()
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
