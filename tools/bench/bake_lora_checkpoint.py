import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from memstat import MemStat
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.lora.mapping.lora_saver import LoRASaver
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.weights.krea2_lora_mapping import Krea2LoRAMapping
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition

# Folds the 4-step LoRA into the q8 weights once and writes a checkpoint laid out one file
# per block, so tools/bench/block_stream.py can stream it with no LoRA cost at runtime.
# M3 measured the alternative: keeping the LoRA unbaked costs +11 s on every step.
#
# The fold itself never holds the whole DiT. mx.load is lazy and so is the
# dequantize/add/requantize that baking builds, so evaluating one block at a time
# materializes one block at a time - about 1.6 GB, against the 15.23 GB peak that baking
# the resident model costs.


class LoraBaker:
    GLOBAL_MODULES = ("first", "tmlp", "tproj", "txtfusion", "txtmlp", "last")

    def __init__(self, model_path: Path, out_path: Path, args: argparse.Namespace):
        self.model_path = model_path
        self.out_path = out_path
        self.args = args
        self.weight_map: dict[str, str] = {}
        self.timings: dict[str, float] = {}

    def run(self) -> dict:
        MemStat.capture(label="bake-before").report()
        self.out_path.mkdir(parents=True, exist_ok=True)

        transformer, bits = self._build()
        blocks = self._write_blocks(transformer)
        globals_written = self._write_globals(transformer)
        self._write_index(bits)

        result = {
            "stage": "bake-lora-checkpoint",
            "source": str(self.model_path),
            "lora": self.args.lora_path,
            "lora_scale": self.args.lora_scale,
            "out": str(self.out_path),
            "stored_bits": bits,
            "blocks_written": blocks,
            "global_tensors": globals_written,
            "tensors_total": len(self.weight_map),
            "timings_s": {k: round(v, 3) for k, v in self.timings.items()},
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
            "active_memory_gb": round(mx.get_active_memory() / 1e9, 3),
        }
        print("🧮 bake")
        for name, value in result["timings_s"].items():
            print(f"   {name:<18}: {value:.2f} s")
        print(f"   tensors           : {result['tensors_total']} ({blocks} blocks + {globals_written} globals)")
        print(f"   mx peak memory    : {result['peak_memory_gb']:.2f} GB")
        print(f"   out               : {self.out_path}")
        return result

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
        # bake_lora=False only wraps the layers, which is lazy. The fold itself is not:
        # LoRASaver evaluates every layer as it bakes it (lora_saver.py:125), so baking the
        # whole transformer in one call materializes all 28 blocks - measured, it reaches
        # 13.10 GB and swaps. bake_and_strip_lora takes any module, so it is applied per
        # block below, one block resident at a time.
        LoRALoader.load_and_apply_lora(
            lora_mapping=Krea2LoRAMapping.get_mapping(),
            transformer=transformer,
            lora_paths=[self.args.lora_path],
            lora_scales=[self.args.lora_scale],
            bake_lora=False,
        )
        self.timings["wrap"] = time.perf_counter() - start
        print(f"   after wrap (lazy) : active {mx.get_active_memory() / 1e9:.3f} GB")
        return transformer, bits

    def _write_blocks(self, transformer: Krea2Transformer) -> int:
        start = time.perf_counter()
        written = 0
        for index, block in enumerate(transformer.blocks):
            LoRASaver.bake_and_strip_lora(block)
            params = block.parameters()
            mx.eval(params)
            flat = {f"blocks.{index}.{k}": v for k, v in tree_flatten(params)}
            name = f"blocks_{index:02d}.safetensors"
            mx.save_safetensors(str(self.out_path / name), flat, metadata=self._metadata())
            self.weight_map.update({k: name for k in flat})
            written += 1
            # Never needed again: swap in empty arrays so the materialized ones are freed.
            block.update(tree_unflatten([(k, mx.array([], dtype=v.dtype)) for k, v in tree_flatten(params)]))
            del params, flat
            mx.clear_cache()
            if index == 0 or index == len(transformer.blocks) - 1:
                print(f"   block {index:>2} written  : active {mx.get_active_memory() / 1e9:.3f} GB, peak {mx.get_peak_memory() / 1e9:.3f} GB")  # fmt: skip
        self.timings["blocks"] = time.perf_counter() - start
        return written

    def _write_globals(self, transformer: Krea2Transformer) -> int:
        start = time.perf_counter()
        flat: dict[str, mx.array] = {}
        for name in LoraBaker.GLOBAL_MODULES:
            module = getattr(transformer, name)
            LoRASaver.bake_and_strip_lora(module)
            params = module.parameters()
            mx.eval(params)
            flat.update({f"{name}.{k}": v for k, v in tree_flatten(params)})
        mx.save_safetensors(str(self.out_path / "globals.safetensors"), flat, metadata=self._metadata())
        self.weight_map.update({k: "globals.safetensors" for k in flat})
        self.timings["globals"] = time.perf_counter() - start
        return len(flat)

    def _write_index(self, bits: int | None) -> None:
        total = sum((self.out_path / name).stat().st_size for name in set(self.weight_map.values()))
        index = {
            "metadata": {"total_size": total, "quantization_level": str(bits)},
            "weight_map": self.weight_map,
        }
        (self.out_path / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    def _metadata(self) -> dict[str, str]:
        return {
            "quantization_level": "8",
            "mflux_version": "16gb-bench",
            "lora": f"{self.args.lora_path}@{self.args.lora_scale}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fold the 4-step LoRA into the q8 DiT, one block at a time.")
    parser.add_argument("--model", type=Path, required=True, help="source mflux q8 snapshot directory")
    parser.add_argument("--out", type=Path, required=True, help="directory to write the baked checkpoint into")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--lora-path",
        default="lvladikov/Krea2-Turbo-Distill-4step-LoRA:krea2_turbo_4step_rank_64_lora_comfyui.safetensors",
    )
    parser.add_argument("--lora-scale", type=float, default=1.0)
    args = parser.parse_args()

    result = LoraBaker(model_path=args.model, out_path=args.out, args=args).run()
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
