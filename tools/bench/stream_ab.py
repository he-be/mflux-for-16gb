import argparse
import fcntl
import json
import os
import struct
import subprocess
import threading
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten
from seqpatch import Patched

from mflux.models.krea2.model.krea2_transformer.rope_embedder import Krea2RopeEmbedder
from mflux.models.krea2.model.krea2_transformer.transformer_block import SingleStreamBlock
from mflux.models.krea2.weights.krea2_weight_stream import Krea2BlockStream, Krea2StreamedBlock

# The production streaming loop (bind -> dispatch -> read the next block -> wait -> drop) over
# the 28 real blocks, with the knobs plan M9c wants to turn, so a change can be judged in the
# shape it will run in. M8c found the K-split of mlp.down 30 ms faster on a block alone and
# 42 ms slower here, under the prefetch; this is for finding out why, and what to do about it.
#
#   --prefetch main    the read runs on the calling thread between async_eval and eval (production)
#   --prefetch thread  the read runs on a Python thread started after the dispatch
#   --prefetch none    no read under the compute (the block is read before it is bound)
#   --down 1|4|4c|4cs  the 16384->6144 projection whole, or as a sum of K-slices (c: cut once per
#                      bind, s: every independent matmul chained with mx.depends; see seqpatch.py)
#   --prefetch direct  arrays for the next block are allocated first and a thread preads the file
#                      bytes straight into them (no MLX call on the thread, no conversion after)
#   --prefetch threadcpu|thread|threadpread|memcpy|mlxcopy|delayed|sleeponly|sleep50|sleep100|sleep250|spin
#                      the other arrangements and stand-ins that isolate one ingredient (see the classes)
#   --read mxload      mx.load + eval (production)
#   --read pread       os.pread of each tensor's bytes into a numpy buffer, then mx.array
#   --nocache          with --read pread: F_NOCACHE, so the read bypasses the page cache
#
#   uv run python tools/bench/stream_ab.py --model ~/Library/Caches/mflux/16gb-bench/krea2-lowram \
#       --config main:1 --config main:4 --config thread:4 --steps 2


class PreadStream(Krea2BlockStream):
    # Reads a block's tensors with os.pread instead of mx.load, optionally with F_NOCACHE.
    DTYPES = {"BF16": (np.uint16, mx.bfloat16), "F16": (np.float16, mx.float16), "F32": (np.float32, mx.float32)}
    DTYPES |= {"U32": (np.uint32, mx.uint32), "I32": (np.int32, mx.int32), "U8": (np.uint8, mx.uint8)}

    def __init__(self, root: Path, nocache: bool):
        super().__init__(root)
        self.nocache = nocache
        self.headers: dict[str, tuple[int, dict]] = {}
        for shard in {s for by in self.by_block.values() for s in by}:
            with open(root / shard, "rb") as f:
                (size,) = struct.unpack("<Q", f.read(8))
                self.headers[shard] = (8 + size, json.loads(f.read(size)))

    def pread_raw(self, index: int) -> dict:
        start = time.perf_counter()
        raw = []
        for shard, keys in self.by_block[index].items():
            base, header = self.headers[shard]
            fd = os.open(self.root / shard, os.O_RDONLY)
            try:
                if self.nocache:
                    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                for key in keys:
                    lo, hi = header[key]["data_offsets"]
                    raw.append((key, header[key], os.pread(fd, hi - lo, base + lo)))
            finally:
                os.close(fd)
        return {"raw": raw, "seconds": time.perf_counter() - start}

    def arrays_from_raw(self, index: int, raw: list) -> dict:
        prefix = f"blocks.{index}."
        flat = []
        for key, meta, buf in raw:
            np_dtype, mx_dtype = self.DTYPES[meta["dtype"]]
            arr = mx.array(np.frombuffer(buf, dtype=np_dtype).reshape(meta["shape"]))
            flat.append((key[len(prefix) :], arr.view(mx_dtype) if mx_dtype is mx.bfloat16 else arr))
        tree = tree_unflatten(flat)
        mx.eval([a for _, a in tree_flatten(tree)])
        return tree

    def prefetch(self, index: int) -> float:
        start = time.perf_counter()
        self.ready[index] = self.arrays_from_raw(index, self.pread_raw(index)["raw"])
        return time.perf_counter() - start


class DirectStream(PreadStream):
    # The next block's arrays are allocated up front (main thread, a GPU fill), and a thread
    # preads the file bytes straight into their memory: no MLX call on the thread, and no
    # conversion afterwards. bf16 tensors are allocated as uint16 and viewed as bf16.
    def allocate(self, index: int) -> tuple[dict, list]:
        prefix = f"blocks.{index}."
        flat, targets = [], []
        for shard, keys in self.by_block[index].items():
            base, header = self.headers[shard]
            for key in keys:
                meta = header[key]
                np_dtype, mx_dtype = self.DTYPES[meta["dtype"]]
                raw = mx.zeros(meta["shape"], dtype=mx.uint16 if mx_dtype is mx.bfloat16 else mx_dtype)
                arr = raw.view(mx.bfloat16) if mx_dtype is mx.bfloat16 else raw
                mx.eval(raw, arr)
                view = np.frombuffer(raw, dtype=np_dtype)
                view.flags.writeable = True
                targets.append((shard, base + meta["data_offsets"][0], view))
                flat.append((key[len(prefix) :], arr))
        return tree_unflatten(flat), targets

    def fill(self, targets: list) -> float:
        start = time.perf_counter()
        fds: dict[str, int] = {}
        try:
            for shard, offset, view in targets:
                fd = fds.get(shard)
                if fd is None:
                    fd = fds[shard] = os.open(self.root / shard, os.O_RDONLY)
                    if self.nocache:
                        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                mv = memoryview(view).cast("B")
                done = 0
                while done < len(mv):
                    n = os.preadv(fd, [mv[done:]], offset + done)
                    if n <= 0:
                        raise OSError(f"short read in {shard} at {offset + done}")
                    done += n
        finally:
            for fd in fds.values():
                os.close(fd)
        return time.perf_counter() - start


class PingPongStream(DirectStream):
    # Two preallocated trees of block-shaped arrays; block i+1 is read into the tree that
    # block i-1 used, so no fill kernel runs per block. Every block has the same tensor
    # shapes, so the trees only need the header of one block.
    def __init__(self, root: Path, nocache: bool):
        super().__init__(root, nocache)
        self.pool = [self.allocate(0) for _ in range(2)]
        self.turn = 0

    def next_targets(self, index: int) -> tuple[dict, list]:
        tree, targets = self.pool[self.turn]
        self.turn ^= 1
        # Re-point the targets at block `index`'s offsets: same shard layout for every block
        # (bake_lora_checkpoint.py writes one shard per block), so only the shard name moves.
        prefix = f"blocks.{index}."
        keys = [k for ks in self.by_block[index].values() for k in ks]
        by_name = {k[len(prefix) :]: k for k in keys}
        retargeted = []
        for (shard0, _offset0, view), (name, _arr) in zip(targets, tree_flatten(tree)):
            key = by_name[name]
            shard = next(sh for sh, ks in self.by_block[index].items() if key in ks)
            base, header = self.headers[shard]
            retargeted.append((shard, base + header[key]["data_offsets"][0], view))
        return tree, retargeted


class PingPongBlock(Krea2StreamedBlock):
    def __call__(self, hidden_states, tvec, freqs, mask):
        stream: PingPongStream = self.stream
        start = time.perf_counter()
        self.block.update(stream.take(self.index))
        mx.eval(self.block.parameters())
        nxt = (self.index + 1) % len(stream.by_block)
        tree, targets = stream.next_targets(nxt)
        bound = time.perf_counter()
        out = self.block(hidden_states, tvec, freqs, mask)
        # The reader starts before the dispatch: mx.async_eval blocks for ~100 ms (K1) or more
        # (K4) before it returns, and a thread started after it reads too late.
        result = {}
        t = threading.Thread(target=lambda: result.update(p=stream.fill(targets)))
        t.start()
        t0 = time.perf_counter()
        mx.async_eval(out)
        dispatched = time.perf_counter()
        mx.eval(out)
        computed = time.perf_counter()
        t.join()
        stream.ready[nxt] = tree
        self.block.update(stream.read(self.index))
        self.stream.record(
            self.index,
            io=bound - start,
            compute=computed - bound,
            drop=time.perf_counter() - computed,
            prefetch=result.get("p", 0.0),
        )
        self.stream.stats[-1].update(async_s=dispatched - t0)
        return out


class DirectReadBlock(Krea2StreamedBlock):
    def __call__(self, hidden_states, tvec, freqs, mask):
        stream: DirectStream = self.stream
        start = time.perf_counter()
        self.block.update(stream.take(self.index))
        mx.eval(self.block.parameters())
        nxt = (self.index + 1) % len(stream.by_block)
        tree, targets = stream.allocate(nxt)
        bound = time.perf_counter()
        out = self.block(hidden_states, tvec, freqs, mask)
        built = time.perf_counter()
        result = {}

        def fill():
            result["t0"] = time.perf_counter()
            result["p"] = stream.fill(targets)
            result["t1"] = time.perf_counter()

        t = threading.Thread(target=fill)
        t.start()
        mx.async_eval(out)
        dispatched = time.perf_counter()
        mx.eval(out)
        computed = time.perf_counter()
        t.join()
        stream.ready[nxt] = tree
        self.block.update(stream.read(self.index))
        self.stream.record(
            self.index,
            io=bound - start,
            compute=computed - bound,
            drop=time.perf_counter() - computed,
            prefetch=result.get("p", 0.0),
        )
        self.stream.stats[-1].update(
            build_s=built - bound,
            async_s=dispatched - built,
            tlag_s=result["t0"] - dispatched,
            tend_s=result["t1"] - computed,
        )
        return out


class ThreadCpuBlock(Krea2StreamedBlock):
    # The production read on a thread, with the loads placed on the CPU stream.
    def __call__(self, hidden_states, tvec, freqs, mask):
        start = time.perf_counter()
        self.block.update(self.stream.take(self.index))
        mx.eval(self.block.parameters())
        bound = time.perf_counter()
        out = self.block(hidden_states, tvec, freqs, mask)
        mx.async_eval(out)
        result = {}
        nxt = (self.index + 1) % len(self.stream.by_block)

        def read():
            with mx.stream(mx.cpu):
                result["p"] = self.stream.prefetch(nxt)

        t = threading.Thread(target=read)
        t.start()
        mx.eval(out)
        computed = time.perf_counter()
        t.join()
        self.block.update(self.stream.read(self.index))
        self.stream.record(
            self.index,
            io=bound - start,
            compute=computed - bound,
            drop=time.perf_counter() - computed,
            prefetch=result.get("p", 0.0),
        )
        return out


class ThreadedBlock(Krea2StreamedBlock):
    # Same order as production, but the read of the next block runs on its own thread.
    def __call__(self, hidden_states, tvec, freqs, mask):
        start = time.perf_counter()
        self.block.update(self.stream.take(self.index))
        mx.eval(self.block.parameters())
        bound = time.perf_counter()
        out = self.block(hidden_states, tvec, freqs, mask)
        mx.async_eval(out)
        result = {}
        nxt = (self.index + 1) % len(self.stream.by_block)
        t = threading.Thread(target=lambda: result.update(p=self.stream.prefetch(nxt)))
        t.start()
        mx.eval(out)
        computed = time.perf_counter()
        t.join()
        self.block.update(self.stream.read(self.index))
        self.stream.record(
            self.index,
            io=bound - start,
            compute=computed - bound,
            drop=time.perf_counter() - computed,
            prefetch=result.get("p", 0.0),
        )
        return out


class UnderComputeBlock(Krea2StreamedBlock):
    # Production order, but what runs under the compute is swapped for a stand-in that isolates
    # one ingredient of the prefetch; the real read of the next block happens serially at the
    # next bind (so only `compute` is comparable, not the step).
    UNDER = None  # set by the subclasses below
    SRC = np.ones(461_000_000 // 4, dtype=np.float32)
    DST = np.empty_like(SRC)

    def __call__(self, hidden_states, tvec, freqs, mask):
        start = time.perf_counter()
        self.block.update(self.stream.take(self.index))
        mx.eval(self.block.parameters())
        bound = time.perf_counter()
        out = self.block(hidden_states, tvec, freqs, mask)
        mx.async_eval(out)
        t0 = time.perf_counter()
        self.under()
        under = time.perf_counter() - t0
        mx.eval(out)
        computed = time.perf_counter()
        self.block.update(self.stream.read(self.index))
        self.stream.record(
            self.index, io=bound - start, compute=computed - bound, drop=time.perf_counter() - computed, prefetch=under
        )
        return out


class MemcpyBlock(UnderComputeBlock):
    # 461 MB of numpy memcpy on the main thread: memory bandwidth only, no I/O, no MLX.
    def under(self):
        np.copyto(self.DST, self.SRC)


class MlxCopyBlock(UnderComputeBlock):
    # 461 MB copied into a fresh MLX array on the main thread: MLX allocation + eval, no I/O.
    def under(self):
        mx.eval(mx.array(self.SRC))


class SleepBlock(UnderComputeBlock):
    # The main thread only sleeps: does the GPU keep going without it? SLEEP is set per mode.
    SLEEP = 0.15

    def under(self):
        time.sleep(self.SLEEP)


class Sleep50Block(SleepBlock):
    SLEEP = 0.05


class Sleep100Block(SleepBlock):
    SLEEP = 0.10


class Sleep250Block(SleepBlock):
    SLEEP = 0.25


class SpinBlock(UnderComputeBlock):
    # The main thread holds the GIL in a busy loop for 150 ms, no MLX calls.
    def under(self):
        end = time.perf_counter() + 0.15
        while time.perf_counter() < end:
            pass


class DelayedBlock(Krea2StreamedBlock):
    # The production prefetch, started 150 ms late so it overlaps the MLP instead of the attention.
    def __call__(self, hidden_states, tvec, freqs, mask):
        start = time.perf_counter()
        self.block.update(self.stream.take(self.index))
        mx.eval(self.block.parameters())
        bound = time.perf_counter()
        out = self.block(hidden_states, tvec, freqs, mask)
        mx.async_eval(out)
        time.sleep(0.15)
        prefetched = self.stream.prefetch((self.index + 1) % len(self.stream.by_block))
        mx.eval(out)
        computed = time.perf_counter()
        self.block.update(self.stream.read(self.index))
        self.stream.record(
            self.index,
            io=bound - start,
            compute=computed - bound,
            drop=time.perf_counter() - computed,
            prefetch=prefetched,
        )
        return out


class ThreadPreadBlock(Krea2StreamedBlock):
    # A Python thread does only the os.pread of the next block (no MLX); after the compute the
    # main thread turns the bytes into arrays, and that conversion shows up in `drop`.
    def __call__(self, hidden_states, tvec, freqs, mask):
        stream: PreadStream = self.stream
        start = time.perf_counter()
        self.block.update(stream.take(self.index))
        mx.eval(self.block.parameters())
        bound = time.perf_counter()
        out = self.block(hidden_states, tvec, freqs, mask)
        mx.async_eval(out)
        nxt = (self.index + 1) % len(stream.by_block)
        result = {}
        t = threading.Thread(target=lambda: result.update(stream.pread_raw(nxt)))
        t.start()
        mx.eval(out)
        computed = time.perf_counter()
        t.join()
        stream.ready[nxt] = stream.arrays_from_raw(nxt, result["raw"])
        self.block.update(stream.read(self.index))
        self.stream.record(
            self.index,
            io=bound - start,
            compute=computed - bound,
            drop=time.perf_counter() - computed,
            prefetch=result["seconds"],
        )
        return out


class UnprefetchedBlock(Krea2StreamedBlock):
    # Reads its own weights before the compute; nothing runs under the GPU.
    def __call__(self, hidden_states, tvec, freqs, mask):
        start = time.perf_counter()
        self.stream.prefetch(self.index)
        self.block.update(self.stream.take(self.index))
        mx.eval(self.block.parameters())
        bound = time.perf_counter()
        out = self.block(hidden_states, tvec, freqs, mask)
        mx.eval(out)
        computed = time.perf_counter()
        self.block.update(self.stream.read(self.index))
        self.stream.record(self.index, io=bound - start, compute=computed - bound, drop=time.perf_counter() - computed)
        return out


class StreamAB:
    FEATURES, HEADS, KVHEADS, MULT, HEAD_DIM = 6144, 48, 12, 4, 128
    WRAPPERS = {
        "main": Krea2StreamedBlock,
        "thread": ThreadedBlock,
        "none": UnprefetchedBlock,
        "memcpy": MemcpyBlock,
        "mlxcopy": MlxCopyBlock,
        "delayed": DelayedBlock,
        "sleeponly": SleepBlock,
        "sleep50": Sleep50Block,
        "sleep100": Sleep100Block,
        "sleep250": Sleep250Block,
        "direct": DirectReadBlock,
        "pingpong": PingPongBlock,
        "threadcpu": ThreadCpuBlock,
        "spin": SpinBlock,
        "threadpread": ThreadPreadBlock,
    }

    def __init__(self, model: Path, tokens: int):
        self.root = Krea2BlockStream.locate(model)
        ids = mx.zeros((1, tokens, 3), dtype=mx.float32)
        self.freqs = Krea2RopeEmbedder(self.HEAD_DIM, 1000, [32, 48, 48])(ids)
        self.x = mx.random.normal((1, tokens, self.FEATURES)).astype(mx.bfloat16)
        self.vec = mx.random.normal((1, 1, 6 * self.FEATURES)).astype(mx.bfloat16)
        mx.eval(self.freqs, self.x, self.vec)

    def run(self, prefetch: str, down: str, read: str, nocache: bool, steps: int) -> dict:
        if prefetch == "pingpong":
            stream = PingPongStream(self.root, nocache)
        elif prefetch == "direct":
            stream = DirectStream(self.root, nocache)
        elif prefetch == "threadpread" or read == "pread":
            stream = PreadStream(self.root, nocache)
        else:
            stream = Krea2BlockStream(self.root)
        blocks = []
        for i in range(len(stream.by_block)):
            b = SingleStreamBlock(self.FEATURES, self.HEADS, self.MULT, False, self.KVHEADS)
            b.set_dtype(mx.bfloat16)
            nn.quantize(b, group_size=64, bits=8)
            b.update(stream.read(i))
            blocks.append(self.WRAPPERS[prefetch](i, b, stream))
        mx.clear_cache()
        mx.set_cache_limit(Krea2BlockStream.CACHE_LIMIT_BYTES)
        with Patched(down):
            # One warm-up step, then the measured ones; the stats of the warm-up are dropped.
            for step in range(steps + 1):
                x = self.x
                for block in blocks:
                    x = block(x, self.vec, self.freqs, None)
                if step == 0:
                    stream.stats.clear()
            summary = stream.summary()
            extra_keys = [k for k in ("build_s", "async_s", "tlag_s", "tend_s") if k in stream.stats[-1]]
            if extra_keys:
                summary["extra"] = {
                    k[:-2] + "_ms": 1000 * sum(st[k] for st in stream.stats) / len(stream.stats) for k in extra_keys
                }
        stream.ready.clear()
        del blocks
        mx.clear_cache()
        return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=4126)
    parser.add_argument("--steps", type=int, default=2, help="measured steps per configuration (plus one warm-up)")
    parser.add_argument("--rounds", type=int, default=2, help="how many times to cycle through the configurations")
    parser.add_argument("--purge", action="store_true", help="run `purge` before each configuration so reads are cold")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="prefetch:down[:read[:nocache]] e.g. main:1, main:4, main:4c (pre-cut slices), thread:4, main:4:pread:nocache",
    )
    args = parser.parse_args()
    configs = args.config or ["main:1", "main:4", "thread:4"]
    print(mx.device_info(), "mlx", mx.__version__, f"tokens={args.tokens} steps={args.steps}")
    ab = StreamAB(args.model, args.tokens)
    print(f"\n  {'config':28s} {'compute':>9s} {'prefetch':>9s} {'bind':>7s} {'drop':>6s}   {'step (28 blocks)':>16s}")
    can_purge = args.purge and subprocess.run(["purge"], capture_output=True).returncode == 0
    print(f"  page cache purge between configs: {'yes' if can_purge else 'no'}")
    for _ in range(args.rounds):
        for cfg in configs:
            if can_purge:
                subprocess.run(["purge"], capture_output=True)
            parts = cfg.split(":")
            prefetch, down = parts[0], parts[1]
            read = parts[2] if len(parts) > 2 else "mxload"
            nocache = len(parts) > 3 and parts[3] == "nocache"
            s = ab.run(prefetch, down, read, nocache, args.steps)
            step = 28 * (s["compute_ms_mean"] + s["io_ms_mean"] + s["drop_ms_mean"]) / 1000
            extra = "".join(f"  {k}={v:.1f}" for k, v in s.get("extra", {}).items())
            print(
                f"  {cfg:28s} {s['compute_ms_mean']:9.1f} {s['prefetch_ms_mean']:9.1f} {s['io_ms_mean']:7.1f} "
                f"{s['drop_ms_mean']:6.1f}   {step:13.2f} s{extra}"
            )
    print(f"\npeak memory {mx.get_peak_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
