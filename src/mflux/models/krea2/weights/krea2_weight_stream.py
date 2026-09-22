import json
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

# Streams the DiT's transformer blocks from disk instead of holding them in memory.
# Measured on an 18 GB M3 Pro: holding the q8 blocks demands 15.96 GB and thrashes,
# streaming them one at a time demands 3.72 GB and does not swap, at +5.8% per step.
# See docs/16gb/measurements/2026-09-22-m5-block-streaming.md.
#
# The next block is read while the current one computes: the compute is dispatched
# with mx.async_eval and the read of block i+1 runs on the CPU until it is done. On the
# M6 mini the block reads in 137 ms and computes for 320 ms, and with the two overlapped
# a block costs 328 ms instead of 456. One extra block (461 MB) is resident for it.
# See docs/16gb/measurements/2026-09-22-m8b-prefetch.md.


class Krea2StreamedBlock:
    def __init__(self, index: int, block, stream: "Krea2BlockStream"):
        self.index = index
        self.block = block
        self.stream = stream

    def __call__(self, hidden_states: mx.array, tvec: mx.array, freqs: mx.array, mask) -> mx.array:
        start = time.perf_counter()
        self.block.update(self.stream.take(self.index))
        mx.eval(self.block.parameters())
        bound = time.perf_counter()

        out = self.block(hidden_states, tvec, freqs, mask)
        # Dispatch the compute, then read the next block while the GPU is busy. The eval of
        # the output is still forced per block: MLX is lazy, and without it the weights would
        # still be needed after the drop below, and dropping them would free nothing.
        mx.async_eval(out)
        prefetched = self.stream.prefetch((self.index + 1) % len(self.stream.by_block))
        mx.eval(out)
        computed = time.perf_counter()

        self.block.update(self.stream.read(self.index))
        mx.clear_cache()
        self.stream.record(
            self.index,
            io=bound - start,
            compute=computed - bound,
            drop=time.perf_counter() - computed,
            prefetch=prefetched,
        )
        return out


class Krea2BlockStream:
    # Everything in the transformer that is not a block. 0.706 GB, and it stays resident.
    GLOBAL_MODULES = ("first", "tmlp", "tproj", "txtfusion", "txtmlp", "last")
    INDEX_FILE = "model.safetensors.index.json"

    def __init__(self, root: Path):
        self.root = root
        self.stats: list[dict] = []
        self.ready: dict[int, dict] = {}
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

    def prefetch(self, index: int) -> float:
        # Reads and materializes a block now, to be taken by the next call. The evaluated
        # arrays live only in this dict and in the block that takes them, so they are freed
        # once that block rebinds its lazy handles.
        start = time.perf_counter()
        tree = self.read(index)
        mx.eval([array for _, array in tree_flatten(tree)])
        self.ready[index] = tree
        return time.perf_counter() - start

    def take(self, index: int) -> dict:
        tree = self.ready.pop(index, None)
        return tree if tree is not None else self.read(index)

    def record(self, index: int, io: float, compute: float, drop: float, prefetch: float = 0.0) -> None:
        self.stats.append({"block": index, "io_s": io, "compute_s": compute, "drop_s": drop, "prefetch_s": prefetch})

    def summary(self) -> dict:
        if not self.stats:
            return {}
        blocks = len(self.by_block)
        io = sum(s["io_s"] for s in self.stats) / len(self.stats)
        compute = sum(s["compute_s"] for s in self.stats) / len(self.stats)
        drop = sum(s["drop_s"] for s in self.stats) / len(self.stats)
        prefetch = sum(s["prefetch_s"] for s in self.stats) / len(self.stats)
        return {
            "root": str(self.root),
            "blocks": blocks,
            "block_calls": len(self.stats),
            "io_ms_mean": round(io * 1000, 1),
            "compute_ms_mean": round(compute * 1000, 1),
            "drop_ms_mean": round(drop * 1000, 1),
            "prefetch_ms_mean": round(prefetch * 1000, 1),
            "ratio_compute_over_prefetch": round(compute / prefetch, 2) if prefetch else None,
        }

    def report(self) -> str:
        s = self.summary()
        if not s:
            return "Block streaming: nothing streamed yet."
        return (
            f"Block streaming: {s['blocks']} blocks from {s['root']}, "
            f"bind {s['io_ms_mean']} ms + compute {s['compute_ms_mean']} ms + drop {s['drop_ms_mean']} ms "
            f"per block, next block read in {s['prefetch_ms_mean']} ms under the compute "
            f"(compute/read {s['ratio_compute_over_prefetch']})"
        )
