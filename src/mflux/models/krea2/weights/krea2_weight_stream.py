import json
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten

# Streams the DiT's transformer blocks from disk instead of holding them in memory.
# Measured on an 18 GB M3 Pro: holding the q8 blocks demands 15.96 GB and thrashes,
# streaming them one at a time demands 3.72 GB and does not swap, at +5.8% per step.
# The block is 461.3 MB and reads in 72 ms while it computes for 985 ms, so the I/O
# hides inside the compute with a factor of 13.6 to spare.
# See docs/16gb/measurements/2026-09-22-m5-block-streaming.md.


class Krea2StreamedBlock:
    def __init__(self, index: int, block, stream: "Krea2BlockStream"):
        self.index = index
        self.block = block
        self.stream = stream

    def __call__(self, hidden_states: mx.array, tvec: mx.array, freqs: mx.array, mask) -> mx.array:
        start = time.perf_counter()
        self.block.update(self.stream.read(self.index))
        mx.eval(self.block.parameters())
        bound = time.perf_counter()

        out = self.block(hidden_states, tvec, freqs, mask)
        # MLX is lazy: without this the weights would still be needed after the drop below,
        # and dropping them would free nothing. The forced sync per block is the cost of
        # streaming, and it is already inside the measured 29.8 s/step.
        mx.eval(out)
        computed = time.perf_counter()

        self.block.update(self.stream.read(self.index))
        mx.clear_cache()
        self.stream.record(self.index, io=bound - start, compute=computed - bound, drop=time.perf_counter() - computed)
        return out


class Krea2BlockStream:
    # Everything in the transformer that is not a block. 0.706 GB, and it stays resident.
    GLOBAL_MODULES = ("first", "tmlp", "tproj", "txtfusion", "txtmlp", "last")
    INDEX_FILE = "model.safetensors.index.json"

    def __init__(self, root: Path):
        self.root = root
        self.stats: list[dict] = []
        weight_map = json.loads((root / Krea2BlockStream.INDEX_FILE).read_text())["weight_map"]
        self.by_block: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for key, shard in weight_map.items():
            if key.startswith("blocks."):
                self.by_block[int(key.split(".")[1])][shard].append(key)
        if not self.by_block:
            raise ValueError(f"No transformer blocks in the weight index at {root / Krea2BlockStream.INDEX_FILE}.")

    @staticmethod
    def locate(model_path: Path) -> Path:
        # The transformer lives either under transformer/ (an mflux-saved snapshot, which
        # is what the block-baking tool writes) or at the root of the snapshot.
        for candidate in (model_path / "transformer", model_path):
            index = candidate / Krea2BlockStream.INDEX_FILE
            if not index.is_file():
                continue
            weight_map = json.loads(index.read_text()).get("weight_map", {})
            if any(key.startswith("blocks.") for key in weight_map):
                return candidate
        raise ValueError(
            f"Block streaming needs a sharded checkpoint with a {Krea2BlockStream.INDEX_FILE} naming "
            f"blocks.*, and {model_path} has none. Write one with tools/bench/bake_lora_checkpoint.py."
        )

    def attach(self, transformer) -> None:
        if len(self.by_block) != len(transformer.blocks):
            raise ValueError(
                f"The checkpoint at {self.root} holds {len(self.by_block)} blocks but this transformer "
                f"has {len(transformer.blocks)}."
            )
        # Materialize only what stays resident. The blocks keep the lazy handles the weight
        # apply left them and are never evaluated; each wrapper rebinds its own on first use.
        for name in Krea2BlockStream.GLOBAL_MODULES:
            mx.eval(getattr(transformer, name).parameters())
        mx.clear_cache()
        transformer.blocks = [Krea2StreamedBlock(i, b, self) for i, b in enumerate(transformer.blocks)]

    def read(self, index: int) -> dict:
        # Fresh lazy handles every call. Reusing one loaded tree keeps the evaluated arrays
        # referenced from it, and then dropping a block frees nothing.
        prefix = f"blocks.{index}."
        flat = []
        for shard, keys in self.by_block[index].items():
            data = mx.load(str(self.root / shard))
            flat.extend((key[len(prefix) :], data[key]) for key in keys)
        return tree_unflatten(flat)

    def record(self, index: int, io: float, compute: float, drop: float) -> None:
        self.stats.append({"block": index, "io_s": io, "compute_s": compute, "drop_s": drop})

    def summary(self) -> dict:
        if not self.stats:
            return {}
        blocks = len(self.by_block)
        io = sum(s["io_s"] for s in self.stats) / len(self.stats)
        compute = sum(s["compute_s"] for s in self.stats) / len(self.stats)
        drop = sum(s["drop_s"] for s in self.stats) / len(self.stats)
        return {
            "root": str(self.root),
            "blocks": blocks,
            "block_calls": len(self.stats),
            "io_ms_mean": round(io * 1000, 1),
            "compute_ms_mean": round(compute * 1000, 1),
            "drop_ms_mean": round(drop * 1000, 1),
            "ratio_compute_over_io": round(compute / io, 2) if io else None,
        }

    def report(self) -> str:
        s = self.summary()
        if not s:
            return "Block streaming: nothing streamed yet."
        return (
            f"Block streaming: {s['blocks']} blocks from {s['root']}, "
            f"I/O {s['io_ms_mean']} ms + compute {s['compute_ms_mean']} ms + drop {s['drop_ms_mean']} ms "
            f"per block (compute/IO {s['ratio_compute_over_io']})"
        )
