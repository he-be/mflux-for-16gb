import json
import os
import struct
import threading
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lokr_layer import LoKrLinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.krea2.model.krea2_transformer.common import Krea2RMSNorm

# Streams the DiT's transformer blocks from disk instead of holding them in memory.
# Measured on an 18 GB M3 Pro: holding the q8 blocks demands 15.96 GB and thrashes,
# streaming them one at a time demands 3.72 GB and does not swap, at +5.8% per step.
# See docs/16gb/measurements/2026-09-22-m5-block-streaming.md.
#
# The next block is read while the current one computes. A Python thread preads the file
# bytes straight into MLX arrays that were allocated up front (two block-shaped sets, used
# alternately), and it is started before the compute is dispatched: mx.async_eval blocks its
# caller for 100-250 ms, and the main thread must stay inside MLX calls or the GPU stops
# being fed. Reading through mx.load on the main thread instead cost the K-split of the down
# projection its whole gain (M8c). See docs/16gb/measurements/2026-09-22-m9c-prefetch-interference.md.


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
        nxt = (self.index + 1) % len(self.stream.by_block)
        reader = self.stream.start_read(nxt)
        # The eval of the output is forced per block: MLX is lazy, and without it the weights
        # would still be needed after the drop below, and dropping them would free nothing.
        mx.async_eval(out)
        prefetched = 0.0 if reader is not None else self.stream.prefetch(nxt)
        mx.eval(out)
        computed = time.perf_counter()
        if reader is not None:
            prefetched = self.stream.finish_read(reader)

        self.block.update(self.stream.read(self.index))
        self.block.mlp.release_down_planes()
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
    # Every block has the same tensor shapes, so the buffers a dropped block leaves in MLX's
    # cache are exactly what the next prefetch needs; clearing the cache after each block
    # cost 12-15 ms of page faults per block instead. The limit keeps the cache from holding
    # more than about two blocks' worth of dropped buffers and activations.
    CACHE_LIMIT_BYTES = 2 << 30
    INDEX_FILE = "model.safetensors.index.json"
    # An adapter keeps the real projection as a child of itself, so once one is applied the
    # checkpoint's attn.wq.weight belongs at attn.wq.linear.weight. Child attribute per kind.
    WRAPPERS = ((LoRALinear, "linear"), (LoKrLinear, "linear"), (FusedLoRALinear, "base_linear"))
    # safetensors dtype -> (numpy view dtype, mlx dtype). bf16 has no numpy dtype: it is
    # allocated as uint16 and viewed as bf16, and the two share one buffer.
    DTYPES = {
        "BF16": (np.uint16, mx.bfloat16),
        "F16": (np.float16, mx.float16),
        "F32": (np.float32, mx.float32),
        "U32": (np.uint32, mx.uint32),
        "I32": (np.int32, mx.int32),
        "U8": (np.uint8, mx.uint8),
    }

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
        self.headers = {
            shard: Krea2BlockStream._header(root / shard) for shard in {s for by in self.by_block.values() for s in by}
        }
        self.pool: list[tuple[dict, dict[str, memoryview]]] = []
        self.turn = 0
        self.wrapped: dict[str, str] = {}

    @staticmethod
    def _header(path: Path) -> tuple[int, dict]:
        with open(path, "rb") as f:
            (size,) = struct.unpack("<Q", f.read(8))
            return 8 + size, json.loads(f.read(size))

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

    def attach(self, transformer, down_splits: int = 1, native_norm: bool = False) -> None:
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
        mx.set_cache_limit(Krea2BlockStream.CACHE_LIMIT_BYTES)
        # Where the adapters sit has to be known before any layout is taken: the buffers the
        # reader fills are keyed by the position each tensor now occupies in the block.
        self.wrapped = Krea2BlockStream._wrapped_paths(transformer.blocks[0])
        for block in transformer.blocks:
            if down_splits > 1:
                block.mlp.down_splits = down_splits
                block.mlp._down_adapter = Krea2BlockStream._down_adapter(block.mlp.down)
            if native_norm:
                for module in block.modules():
                    if isinstance(module, Krea2RMSNorm):
                        module.native_dtype = True
        if self._direct_readable():
            self.pool = [self._allocate() for _ in range(2)]
        transformer.blocks = [Krea2StreamedBlock(i, b, self) for i, b in enumerate(transformer.blocks)]

    # -- reading ------------------------------------------------------------------------------

    def read(self, index: int) -> dict:
        # Fresh lazy handles every call. Reusing one loaded tree keeps the evaluated arrays
        # referenced from it, and then dropping a block frees nothing.
        prefix = f"blocks.{index}."
        flat = []
        for shard, keys in self.by_block[index].items():
            data = mx.load(str(self.root / shard))
            flat.extend((self._position(key[len(prefix) :]), data[key]) for key in keys)
        return tree_unflatten(flat)

    def _position(self, name: str) -> str:
        # Where a checkpoint tensor belongs in the live block, which is one level deeper
        # than the checkpoint says whenever an adapter wraps its layer.
        head, _, leaf = name.rpartition(".")
        child = self.wrapped.get(head)
        return f"{head}.{child}.{leaf}" if child else name

    @staticmethod
    def _wrapped_paths(block) -> dict[str, str]:
        wrapped = {}
        for path, module in block.named_modules():
            for kind, child in Krea2BlockStream.WRAPPERS:
                if isinstance(module, kind):
                    wrapped[path] = child
        return wrapped

    @staticmethod
    def _down_adapter(down):
        # (base projection, delta) for the adapters whose contribution is a side path that
        # does not need the base output, so the K-split can still run underneath them. LoKr's
        # dora variant rescales the base weight itself and has no such form, so it is left
        # out and keeps the unsplit projection.
        if isinstance(down, LoRALinear):
            return down.linear, lambda h: down.scale * mx.matmul(mx.matmul(h, down.lora_A), down.lora_B)
        if isinstance(down, FusedLoRALinear) and all(isinstance(a, LoRALinear) for a in down.loras):
            return down.base_linear, lambda h: sum(
                a.scale * mx.matmul(mx.matmul(h, a.lora_A), a.lora_B) for a in down.loras
            )
        return None

    def prefetch(self, index: int) -> float:
        # The fallback when the checkpoint cannot be read directly: reads and materializes a
        # block through mx.load on the calling thread.
        start = time.perf_counter()
        tree = self.read(index)
        mx.eval([array for _, array in tree_flatten(tree)])
        self.ready[index] = tree
        return time.perf_counter() - start

    def take(self, index: int) -> dict:
        tree = self.ready.pop(index, None)
        return tree if tree is not None else self.read(index)

    def _layout(self, index: int) -> list[tuple[str, str, int, int, str, tuple]]:
        # (name within the block, shard, byte offset, byte count, dtype, shape) per tensor.
        prefix = f"blocks.{index}."
        rows = []
        for shard, keys in self.by_block[index].items():
            base, header = self.headers[shard]
            for key in keys:
                meta = header[key]
                lo, hi = meta["data_offsets"]
                name = self._position(key[len(prefix) :])
                rows.append((name, shard, base + lo, hi - lo, meta["dtype"], tuple(meta["shape"])))
        return sorted(rows)

    def _direct_readable(self) -> bool:
        # Every block must have the same tensors, dtypes and shapes as block 0, in dtypes
        # numpy can view; otherwise the mx.load path is used.
        first = [(name, dtype, shape) for name, _, _, _, dtype, shape in self._layout(0)]
        if any(dtype not in Krea2BlockStream.DTYPES for _, dtype, _ in first):
            return False
        return all([(n, d, s) for n, _, _, _, d, s in self._layout(i)] == first for i in self.by_block)

    def _allocate(self) -> tuple[dict, dict[str, memoryview]]:
        flat, views = [], {}
        for name, _, _, _, dtype, shape in self._layout(0):
            np_dtype, mx_dtype = Krea2BlockStream.DTYPES[dtype]
            raw = mx.zeros(shape, dtype=mx.uint16 if mx_dtype is mx.bfloat16 else mx_dtype)
            array = raw.view(mx.bfloat16) if mx_dtype is mx.bfloat16 else raw
            mx.eval(raw, array)
            view = np.frombuffer(raw, dtype=np_dtype)
            view.flags.writeable = True
            views[name] = memoryview(view).cast("B")
            flat.append((name, array))
        return tree_unflatten(flat), views

    def start_read(self, index: int):
        if not self.pool:
            return None
        tree, views = self.pool[self.turn]
        self.turn ^= 1
        result: dict = {}
        thread = threading.Thread(target=lambda: result.update(seconds=self._fill(index, views)), daemon=True)
        thread.start()
        return index, tree, thread, result

    def finish_read(self, reader) -> float:
        index, tree, thread, result = reader
        thread.join()
        if "seconds" not in result:
            # The thread failed; the block will be read lazily at its bind instead.
            return 0.0
        self.ready[index] = tree
        return result["seconds"]

    def _fill(self, index: int, views: dict[str, memoryview]) -> float:
        # Runs on the reader thread. Only file descriptors and memoryviews: no MLX calls.
        start = time.perf_counter()
        fds: dict[str, int] = {}
        try:
            for name, shard, offset, nbytes, _, _ in self._layout(index):
                fd = fds.get(shard)
                if fd is None:
                    fd = fds[shard] = os.open(self.root / shard, os.O_RDONLY)
                target = views[name]
                done = 0
                while done < nbytes:
                    n = os.preadv(fd, [target[done:nbytes]], offset + done)
                    if n <= 0:
                        raise OSError(f"Short read of {name} in {shard} at byte {offset + done}.")
                    done += n
        finally:
            for fd in fds.values():
                os.close(fd)
        return time.perf_counter() - start

    # -- bookkeeping --------------------------------------------------------------------------

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
            "direct": bool(self.pool),
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
        how = "read straight into its buffers by a thread" if s["direct"] else "read through mx.load"
        return (
            f"Block streaming: {s['blocks']} blocks from {s['root']}, "
            f"bind {s['io_ms_mean']} ms + compute {s['compute_ms_mean']} ms + drop {s['drop_ms_mean']} ms "
            f"per block, next block {how} in {s['prefetch_ms_mean']} ms under the compute "
            f"(compute/read {s['ratio_compute_over_prefetch']})"
        )
