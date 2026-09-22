import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ane_probe import SHAPES, Block0, Converter, Worker  # noqa: E402

# M11b-lite: the real block 0 with its MLP split between the GPU and the Neural Engine, timed
# against the same block on the GPU alone. The ANE takes a share of gate/up's columns and
# silu(g)*u over them (the gateup57 / gateup25 packages of ane_probe.py); the GPU does the
# other columns, then the down projection as two K-slices: its own columns first, without
# waiting, and the ANE's columns once their h comes back through shared memory. The down
# projection never leaves the GPU (M11a-3: a co-tenant that holds it costs the GPU 60%).
#
#   UV_PROJECT_ENVIRONMENT=.venv313 uv run --python 3.13 --with coremltools \
#       python tools/bench/ane_block.py --configs alone4,alone1,ane57,ane25
#
# See docs/16gb/measurements/2026-09-23-m11a3-ane-coscan.md, section 6.


class SplitMlp:
    # Stands in for Krea2SwiGLU.__call__ on one block. Column slices of the q8 weights are cut
    # once (uint32 packs 4 int8 along K; scales/biases go by groups of 64).
    PACK, GROUP = 4, 64

    def __init__(self, mlp, worker: Worker, ane_cols: int, ane_k_splits: int, chunks: int = 1):
        import mlx.core as mx

        self.mx = mx
        self.worker = worker
        self.ane_cols = ane_cols
        self.ane_k_splits = ane_k_splits
        self.chunks = chunks
        self.gate = tuple(a[ane_cols:] for a in (mlp.gate.weight, mlp.gate.scales, mlp.gate.biases))
        self.up = tuple(a[ane_cols:] for a in (mlp.up.weight, mlp.up.scales, mlp.up.biases))
        d = mlp.down
        self.down_gpu = (
            d.weight[:, ane_cols // self.PACK :],
            d.scales[:, ane_cols // self.GROUP :],
            d.biases[:, ane_cols // self.GROUP :],
        )
        self.down_ane = (
            d.weight[:, : ane_cols // self.PACK],
            d.scales[:, : ane_cols // self.GROUP],
            d.biases[:, : ane_cols // self.GROUP],
        )
        mx.eval(*self.gate, *self.up, *self.down_gpu, *self.down_ane)
        self.last: dict = {}

    def qmm(self, x, w):
        return self.mx.quantized_matmul(x, *w, transpose=True, group_size=self.GROUP, bits=8)

    def __call__(self, x):
        from mlx import nn

        mx = self.mx
        # The handoff: x is the MLP input the block just built, so this eval waits for the
        # attention; then 51 MB go to the worker as fp16 and its predict starts.
        t0 = time.perf_counter()
        np.copyto(self.worker.x, np.array(x[0].astype(mx.float16)))
        tokens = x.shape[1]
        step = tokens // self.chunks
        ranges = [(i * step, tokens if i == self.chunks - 1 else (i + 1) * step) for i in range(self.chunks)]
        for r0, r1 in ranges:
            self.worker.send("step", r0, r1)
        t1 = time.perf_counter()
        # The GPU's own columns, and the down projection over them, dispatched while the ANE works.
        h = nn.silu(self.qmm(x, self.gate)) * self.qmm(x, self.up)
        y = self.qmm(h, self.down_gpu)
        mx.async_eval(y)
        # The ANE's columns come back a chunk of rows at a time; each chunk's share of the down
        # projection is dispatched as soon as it lands, so the tail after the last chunk is short.
        parts_y = []
        predict = 0.0
        t2 = None
        for r0, r1 in ranges:
            info = self.worker.recv()
            predict += info["predict_s"]
            t2 = t2 or time.perf_counter()
            h_ane = mx.array(self.worker.y[r0:r1])[None].astype(mx.bfloat16)
            parts_y.append(self.down_over_ane(h_ane))
            mx.async_eval(parts_y[-1])
        t3 = time.perf_counter()
        y = y + (parts_y[0] if len(parts_y) == 1 else mx.concatenate(parts_y, axis=1))
        self.last = {
            "send_ms": (t1 - t0) * 1e3,
            "wait_ms": (t2 - t1) * 1e3,
            "last_ms": (t3 - t1) * 1e3,
            "predict_ms": predict * 1e3,
        }
        return y

    def down_over_ane(self, h_ane):
        mx = self.mx
        if self.ane_k_splits > 1:
            parts = self.ane_k_splits
            w, s, b = (
                mx.contiguous(a.reshape(a.shape[0], parts, a.shape[1] // parts).transpose(1, 0, 2))
                for a in self.down_ane
            )
            hs = h_ane.reshape(h_ane.shape[1], parts, self.ane_cols // parts).transpose(1, 0, 2)
            return mx.quantized_matmul(hs, w, s, b, transpose=True, group_size=self.GROUP, bits=8).sum(axis=0)[None]
        return self.qmm(h_ane, self.down_ane)


class BlockBench:
    HEADS, KVHEADS, MULT, HEAD_DIM, FEATURES = 48, 12, 4, 128, 6144

    def __init__(self, model: Path, tokens: int, cache: Path):
        import mlx.core as mx

        from mflux.models.krea2.model.krea2_transformer.rope_embedder import Krea2RopeEmbedder

        self.mx = mx
        self.tokens = tokens
        self.cache = cache
        self.block0 = Block0(model)
        self.block = self.block0.block()
        ids = mx.zeros((1, tokens, 3), dtype=mx.float32)
        self.freqs = Krea2RopeEmbedder(self.HEAD_DIM, 1000, [32, 48, 48])(ids)
        mx.random.seed(0)
        self.x = mx.random.normal((1, tokens, self.FEATURES)).astype(mx.bfloat16)
        self.vec = mx.random.normal((1, 1, 6 * self.FEATURES)).astype(mx.bfloat16)
        mx.eval(self.freqs, self.x, self.vec)
        self.original_call = type(self.block.mlp).__call__

    def run(self, config: str, warmup: int, iters: int) -> dict:
        mx = self.mx
        mlp = self.block.mlp
        worker = None
        split = None
        if config.startswith("alone"):
            type(mlp).__call__ = self.original_call
            mlp.down_splits = int(config[len("alone") :] or 1)
        else:
            # ane57 / ane25: the share; a trailing "a8" takes the int8-activation package
            name = config.split(":")[0]
            variant = "a8" if name.endswith("a8") else "row"
            shape = {"ane57": "gateup57", "ane25": "gateup25"}[name.removesuffix("a8")]
            # name[:k-splits of the ANE columns' down[:row chunks]], e.g. ane57a8:2:2
            opts = config.split(":")[1:]
            ksplit = int(opts[0]) if opts else 1
            chunks = int(opts[1]) if len(opts) > 1 else 1
            _, _, cols = SHAPES[shape]
            if self.tokens % chunks:
                return {"error": f"{self.tokens} tokens do not split into {chunks} chunks"}
            path = Converter(self.cache, "linear").path(shape, self.tokens // chunks, variant)
            if not path.exists():
                return {"error": f"{path.name} is not converted (run ane_probe.py --only convert --shapes {shape})"}
            worker = Worker((self.tokens, self.FEATURES), (self.tokens, cols), "all")
            worker.load(path)
            split = SplitMlp(mlp, worker, cols, ksplit, chunks)
            type(mlp).__call__ = lambda _self, x: split(x)
        times, details = [], []
        try:
            for i in range(warmup + iters):
                mx.synchronize()
                t0 = time.perf_counter()
                out = self.block(self.x, self.vec, self.freqs, None)
                mx.eval(out)
                sec = time.perf_counter() - t0
                if i >= warmup:
                    times.append(sec)
                    if split is not None:
                        details.append(dict(split.last))
            ref = out
        finally:
            type(mlp).__call__ = self.original_call
            mlp.release_down_planes()
            if worker is not None:
                worker.close()
        entry = {"block_ms": 1e3 * float(np.mean(times)), "block_ms_min": 1e3 * float(np.min(times)), "iters": iters}
        if details:
            for key in ("send_ms", "wait_ms", "last_ms", "predict_ms"):
                entry[key] = float(np.mean([d[key] for d in details]))
        entry["_out"] = ref
        return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path, default=Path("~/Library/Caches/mflux/16gb-bench/krea2-lowram").expanduser()
    )
    parser.add_argument(
        "--cache", type=Path, default=Path("~/Library/Caches/mflux/16gb-bench/krea2-lowram/ane/probe").expanduser()
    )
    parser.add_argument("--tokens", type=int, default=4126)
    parser.add_argument("--configs", default="alone4,alone1,ane57,ane57:2,ane25,alone4")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    bench = BlockBench(args.model, args.tokens, args.cache)
    print(f"\n== ane_block: block 0 real weights, tokens={args.tokens}, mean of {args.iters} (warmup {args.warmup}) ==")
    print(
        f"  {'config':12s} {'block ms':>9s} {'min':>8s} {'send':>7s} {'wait':>7s} {'last':>7s} {'predict':>8s}   vs alone4"
    )
    results: dict = {}
    ref = None
    base = None
    for config in args.configs.split(","):
        entry = bench.run(config, args.warmup, args.iters)
        out = entry.pop("_out", None)
        if "error" in entry:
            print(f"  {config:12s} FAILED: {entry['error']}")
            results[config] = entry
            continue
        if config == "alone4" and ref is None:
            ref, base = out, entry["block_ms"]
        elif ref is not None and out is not None:
            d = bench.mx.abs(out.astype(bench.mx.float32) - ref.astype(bench.mx.float32))
            entry["max_abs_vs_alone4"] = float(d.max())
            entry["mean_abs_vs_alone4"] = float(d.mean())
        rel = f"{entry['block_ms'] / base * 100 - 100:+.1f}%" if base else ""
        ane = (
            f"{entry['send_ms']:7.1f} {entry['wait_ms']:7.1f} {entry['last_ms']:7.1f} {entry['predict_ms']:8.1f}"
            if "send_ms" in entry
            else " " * 32
        )
        num = (
            f"  |Δ| max {entry['max_abs_vs_alone4']:.3f} mean {entry['mean_abs_vs_alone4']:.4f}"
            if "max_abs_vs_alone4" in entry
            else ""
        )
        print(f"  {config:12s} {entry['block_ms']:9.1f} {entry['block_ms_min']:8.1f} {ane}   {rel}{num}", flush=True)
        results[config] = entry
    print(f"\npeak memory {bench.mx.get_peak_memory() / 1e9:.2f} GB")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
