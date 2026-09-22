import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from memstat import MemStat

from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.utils.image_util import ImageUtil

# Plan M4: the VAE alone, in a process that holds neither the text encoder nor the DiT.
# Reads the latent M3 wrote and turns it into the actual image. The VAE is 0.51 GB, so
# this is the easy end of the pipeline - it is here to close the loop and produce
# something you can look at.


class Krea2VaeBench:
    def __init__(self, model_path: Path, latents_path: Path, out_path: Path, tile_size: int | None = None):
        self.model_path = model_path
        self.latents_path = latents_path
        self.out_path = out_path
        # Decoding 1024^2 in one piece peaks at 8.73 GB - the decoder widens 128x128x16 all
        # the way to 1024x1024x3 - so tiling is the difference between fitting and not.
        self.tiling = TilingConfig(vae_decode_tile_size=tile_size) if tile_size else None
        self.timings: dict[str, float] = {}

    def run(self) -> dict:
        before = MemStat.capture(label="vae-before")
        before.report()

        vae, bits = self._load()
        latents, metadata = self._read_latents()
        decoded = self._decode(vae, latents)
        image = self._save(decoded)

        result = {
            "stage": "M4-vae",
            "model_path": str(self.model_path),
            "latents_path": str(self.latents_path),
            "latent_shape": list(latents.shape),
            "stored_bits": bits,
            "tile_size": self.tiling.vae_decode_tile_size if self.tiling else None,
            "image_size": list(image.size),
            "out_path": str(self.out_path),
            "latent_metadata": metadata,
            "timings_s": {k: round(v, 3) for k, v in self.timings.items()},
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
            "active_memory_gb": round(mx.get_active_memory() / 1e9, 3),
        }
        print("🧮 M4 VAE")
        for name, value in result["timings_s"].items():
            print(f"   {name:<18}: {value:.2f} s")
        print(f"   mx peak memory    : {result['peak_memory_gb']:.2f} GB")
        print(f"   image             : {image.size[0]}x{image.size[1]} -> {self.out_path}")
        return result

    def _load(self) -> tuple[QwenVAE, int | None]:
        component = next(c for c in Krea2WeightDefinition.get_components() if c.name == "vae")
        start = time.perf_counter()
        vae = QwenVAE()
        weights = WeightLoader.load_single_local(component=component, root_path=self.model_path)
        bits = WeightApplier.apply_and_quantize_single(
            weights=weights,
            model=vae,
            component=component,
            quantize_arg=None,
            quantization_predicate=Krea2WeightDefinition.quantization_predicate,
        )
        del weights
        mx.eval(vae)
        self.timings["load"] = time.perf_counter() - start
        return vae, bits

    def _read_latents(self) -> tuple[mx.array, dict]:
        arrays, metadata = mx.load(str(self.latents_path), return_metadata=True)
        latents = arrays["latents"]
        mx.eval(latents)
        return latents, {k: str(v) for k, v in metadata.items()}

    def _decode(self, vae: QwenVAE, latents: mx.array) -> mx.array:
        start = time.perf_counter()
        decoded = VAEUtil.decode(vae=vae, latent=latents, tiling_config=self.tiling)
        mx.eval(decoded)
        self.timings["decode"] = time.perf_counter() - start
        return decoded

    def _save(self, decoded: mx.array):
        start = time.perf_counter()
        image = ImageUtil.to_pil(decoded)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(self.out_path)
        self.timings["save"] = time.perf_counter() - start
        return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan M4: decode M3's latent with the Krea 2 VAE alone.")
    parser.add_argument("--model", type=Path, required=True, help="path to the mflux q8 snapshot directory")
    parser.add_argument("--latents", type=Path, required=True, help="latent written by tools/bench/dit_steps.py")
    parser.add_argument("--out", type=Path, required=True, help="where to write the PNG")
    parser.add_argument("--json", type=Path, default=None, help="also write the measurement to this JSON file")
    parser.add_argument("--tile-size", type=int, default=None, help="tile the decode (omit to decode in one piece)")
    args = parser.parse_args()

    result = Krea2VaeBench(
        model_path=args.model, latents_path=args.latents, out_path=args.out, tile_size=args.tile_size
    ).run()
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"   json              : {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
