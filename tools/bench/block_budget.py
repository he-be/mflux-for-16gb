import argparse
import time
from pathlib import Path

import mlx.core as mx
from mlx import nn
from seqpatch import Patched

from mflux.models.krea2.model.krea2_transformer.common import Krea2RMSNorm
from mflux.models.krea2.model.krea2_transformer.feed_forward import Krea2SwiGLU
from mflux.models.krea2.model.krea2_transformer.rope_embedder import Krea2RopeEmbedder
from mflux.models.krea2.model.krea2_transformer.transformer_block import SingleStreamBlock
from mflux.models.krea2.weights.krea2_weight_stream import Krea2BlockStream

# Where do the 323 ms of one production block go, measured in one process so the parts add
# up? M8 (block_profile.py) put the eight matmuls at 255 ms apart and 296 ms chained, and
# left the 40 ms gap unexplained. This grows the chain one matmul at a time to find where
# it opens, then times the block with the candidate fixes of plan M9d/M9c applied one by
# one: norms without the float32 round trip, the block under mx.compile, the down
# projection split along K. It also checks how far each norm variant moves the numbers.
#
#   uv run python tools/bench/block_budget.py --model ~/Library/Caches/mflux/16gb-bench/krea2-lowram
#   uv run python tools/bench/block_budget.py --model ... --tokens 6430    # 1280^2


class KSplitDown:
    # The 16384->6144 projection as a sum of K-slices: each slice is a K=4096 matmul that
    # MLX runs at 17 TFLOPS instead of 9 (M9 section 2). Slices are cut once, contiguously.
    def __init__(self, linear, parts: int):
        self.group_size, self.bits = linear.group_size, linear.bits
        n, packed = linear.weight.shape
        kk, sk = packed // parts, linear.scales.shape[1] // parts
        self.k = kk * 32 // self.bits
        self.slices = []
        for i in range(parts):
            w = mx.contiguous(linear.weight[:, i * kk : (i + 1) * kk])
            s = mx.contiguous(linear.scales[:, i * sk : (i + 1) * sk])
            b = mx.contiguous(linear.biases[:, i * sk : (i + 1) * sk])
            self.slices.append((w, s, b))
        mx.eval(*[a for sl in self.slices for a in sl])

    def __call__(self, x: mx.array) -> mx.array:
        acc = None
        for i, (w, s, b) in enumerate(self.slices):
            y = mx.quantized_matmul(
                x[..., i * self.k : (i + 1) * self.k],
                w,
                s,
                b,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
            )
            acc = y if acc is None else acc + y
        return acc


class BlockBudget:
    FEATURES, HEADS, KVHEADS, MULT, HEAD_DIM = 6144, 48, 12, 4, 128

    def __init__(self, tokens: int, model: Path, warmup: int, iters: int):
        self.tokens, self.warmup, self.iters = tokens, warmup, iters
        self.block = SingleStreamBlock(self.FEATURES, self.HEADS, self.MULT, False, self.KVHEADS)
        self.block.set_dtype(mx.bfloat16)
        nn.quantize(self.block, group_size=64, bits=8)
        self.block.update(Krea2BlockStream(Krea2BlockStream.locate(model)).read(0))
        mx.eval(self.block.parameters())
        ids = mx.zeros((1, tokens, 3), dtype=mx.float32)
        self.freqs = Krea2RopeEmbedder(self.HEAD_DIM, 1000, [32, 48, 48])(ids)
        self.x = mx.random.normal((1, tokens, self.FEATURES)).astype(mx.bfloat16)
        self.vec = mx.random.normal((1, 1, 6 * self.FEATURES)).astype(mx.bfloat16)
        mx.eval(self.freqs, self.x, self.vec)
        self.down = self.block.mlp.down

    def compiled(self):
        c = mx.compile(lambda x, v: self.block(x, v, self.freqs, None))
        return lambda: c(self.x, self.vec)

    def time(self, fn) -> float:
        for _ in range(self.warmup):
            mx.eval(fn())
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(self.iters):
            mx.eval(fn())
        mx.synchronize()
        return (time.perf_counter() - t0) / self.iters

    @staticmethod
    def row(label: str, seconds: float, note: str = "") -> None:
        print(f"  {label:46s} {seconds * 1e3:8.1f} ms  {note}")

    # -- 1. the chain, one matmul at a time --------------------------------------------------

    def chain(self) -> None:
        b, x = self.block, self.x
        a = b.attn.wq(x) * mx.sigmoid(b.attn.gate(x)) + b.attn.wk(x).sum() + b.attn.wv(x).sum()
        o = b.attn.wo(a)
        h = nn.silu(b.mlp.gate(o)) * b.mlp.up(o)
        mx.eval(a, o, h)
        alone = [
            ("wq", lambda: b.attn.wq(x)),
            ("gate", lambda: b.attn.gate(x)),
            ("wk", lambda: b.attn.wk(x)),
            ("wv", lambda: b.attn.wv(x)),
            ("wo", lambda: b.attn.wo(a)),
            ("mlp.gate", lambda: b.mlp.gate(o)),
            ("mlp.up", lambda: b.mlp.up(o)),
            ("mlp.down", lambda: b.mlp.down(h)),
        ]

        def chained(n: int):
            def run():
                s = {}
                s["q"] = b.attn.wq(x)
                if n >= 2:
                    s["g"] = b.attn.gate(x)
                if n >= 3:
                    s["k"] = b.attn.wk(x)
                if n >= 4:
                    s["v"] = b.attn.wv(x)
                if n >= 5:
                    s["o"] = b.attn.wo(s["q"] * mx.sigmoid(s["g"]) + s["k"].sum() + s["v"].sum())
                if n >= 6:
                    s["mg"] = b.mlp.gate(s["o"])
                if n >= 7:
                    s["mu"] = b.mlp.up(s["o"])
                if n >= 8:
                    s["d"] = b.mlp.down(nn.silu(s["mg"]) * s["mu"])
                return list(s.values())

            return run

        print("\n== the eight matmuls: alone, and chained one at a time ==")
        print(f"  {'':46s} {'alone':>8s}   {'sum':>8s}   {'chain':>8s}   {'gap':>6s}")
        total = 0.0
        for n, (name, fn) in enumerate(alone, start=1):
            t_alone = self.time(fn)
            total += t_alone
            t_chain = self.time(chained(n))
            print(
                f"  {f'+ {name}':46s} {t_alone * 1e3:8.1f}   {total * 1e3:8.1f}   {t_chain * 1e3:8.1f}   "
                f"{(t_chain - total) * 1e3:+6.1f}"
            )

    # -- 1b. what opens the gap: a second kernel, or a second large output alive ------------

    def gap(self) -> None:
        b, x = self.block, self.x
        o = b.attn.wo(x)
        mx.eval(o)
        print("\n== the gap: two matmuls in one eval, in different arrangements ==")
        self.row("wq(x)", self.time(lambda: b.attn.wq(x)))
        self.row("[wq(x), gate(x)]  two outputs alive", self.time(lambda: [b.attn.wq(x), b.attn.gate(x)]))
        self.row("gate(wq(x))       dependent, one output", self.time(lambda: b.attn.gate(b.attn.wq(x))))
        self.row("wq(x) + gate(x)   summed, one output", self.time(lambda: b.attn.wq(x) + b.attn.gate(x)))
        self.row("mlp.gate(o)", self.time(lambda: b.mlp.gate(o)))
        self.row("[mlp.gate(o), mlp.up(o)]  two 135 MB alive", self.time(lambda: [b.mlp.gate(o), b.mlp.up(o)]))
        self.row("mlp.gate(o) + mlp.up(o)   summed", self.time(lambda: b.mlp.gate(o) + b.mlp.up(o)))
        self.row("silu(mlp.gate(o)) * mlp.up(o)  as the block", self.time(lambda: nn.silu(b.mlp.gate(o)) * b.mlp.up(o)))
        limit = mx.set_cache_limit(0)
        self.row("[mlp.gate(o), mlp.up(o)]  cache limit 0", self.time(lambda: [b.mlp.gate(o), b.mlp.up(o)]))
        self.row("mlp.gate(o)               cache limit 0", self.time(lambda: b.mlp.gate(o)))
        mx.set_cache_limit(limit)

        # Synchronizing after every eval removes the overlap between iterations that the
        # timing loop otherwise allows, so a single kernel's true wall time shows.
        def strict(fn):
            for _ in range(self.warmup):
                mx.eval(fn())
                mx.synchronize()
            t0 = time.perf_counter()
            for _ in range(self.iters):
                mx.eval(fn())
                mx.synchronize()
            return (time.perf_counter() - t0) / self.iters

        self.row("mlp.gate(o)               sync each", strict(lambda: b.mlp.gate(o)))
        self.row("[mlp.gate(o), mlp.up(o)]  sync each", strict(lambda: [b.mlp.gate(o), b.mlp.up(o)]))

    # -- 1b'. the same pair, chained by a real data dependency and by an eval boundary ------

    def gap2(self) -> None:
        b = self.block
        o = b.attn.wo(self.x)
        o2 = o + 0
        mx.eval(o, o2)
        print("\n== mlp.gate + mlp.up: independent, chained by data, split by an eval boundary ==")
        self.row("[gate(o), up(o)]  independent", self.time(lambda: [b.mlp.gate(o), b.mlp.up(o)]))
        self.row(
            "[gate(o), up(o2)]  independent, separate input copy", self.time(lambda: [b.mlp.gate(o), b.mlp.up(o2)])
        )

        def data_chained():
            g = b.mlp.gate(o)
            return [g, b.mlp.up(o + (g[..., :1] * 0).astype(o.dtype))]

        self.row("up(o + 0*g[..., :1])  chained by data (+1 elementwise)", self.time(data_chained))

        def depends_chained():
            g = b.mlp.gate(o)
            return [g, b.mlp.up(mx.depends(o, g))]

        self.row("up(depends(o, g))  chained by mx.depends", self.time(depends_chained))

        def boundary():
            g = b.mlp.gate(o)
            mx.async_eval(g)
            return [g, b.mlp.up(o)]

        self.row("async_eval(g) then up(o)  eval boundary", self.time(boundary))
        self.row("o + 0*g[..., :1] alone (the cost of the link)", self.time(lambda: o + (o[..., :1] * 0)))

    # -- 1c. independent matmuls chained with mx.depends, and the K-slices chained too -------

    def seq(self) -> None:
        b = self.block
        print("\n== the block with its independent matmuls run in sequence (mx.depends) ==")
        self.row("block", self.time(lambda: b(self.x, self.vec, self.freqs, None)))
        for spec, note in (
            ("1s", "attention + mlp chained by mx.depends"),
            ("1d", "attention + mlp chained by data"),
            ("4c", "K4 pre-cut, slices concurrent"),
            ("4cs", "K4 pre-cut, everything chained by mx.depends"),
            ("4cd", "K4 pre-cut, everything chained by data"),
        ):
            with Patched(spec):
                self.row(f"block, {spec}", self.time(lambda: b(self.x, self.vec, self.freqs, None)), note)
        original = Krea2RMSNorm.__call__
        Krea2RMSNorm.__call__ = BlockBudget.rms_bf16_weight
        with Patched("4cd"):
            self.row("block, 4cd, norm bf16 weight", self.time(lambda: b(self.x, self.vec, self.freqs, None)), "all")
        Krea2RMSNorm.__call__ = original

    # -- 1d. fusing the elementwise work piece by piece with small compiled functions ---------

    def fuse(self) -> None:
        b = self.block
        print("\n== elementwise work: the block with small pieces compiled, one at a time ==")
        base = self.time(lambda: b(self.x, self.vec, self.freqs, None))
        self.row("block", base)

        saved_rope = Krea2RopeEmbedder.apply_rope
        saved_swiglu = Krea2SwiGLU.__call__
        saved_block = SingleStreamBlock.__call__
        saved_norm = Krea2RMSNorm.__call__

        rope_c = mx.compile(saved_rope)
        Krea2RopeEmbedder.apply_rope = staticmethod(lambda q, k, f: rope_c(q, k, f))
        self.row("block, apply_rope compiled", self.time(lambda: b(self.x, self.vec, self.freqs, None)))
        Krea2RopeEmbedder.apply_rope = saved_rope

        glu = mx.compile(lambda g, u: nn.silu(g) * u)

        def swiglu(self_, x):
            return self_.down(glu(self_.gate(x), self_.up(x)))

        Krea2SwiGLU.__call__ = swiglu
        self.row("block, silu*up compiled", self.time(lambda: b(self.x, self.vec, self.freqs, None)))
        Krea2SwiGLU.__call__ = saved_swiglu

        modn = mx.compile(lambda n, scale, shift: (1 + scale) * n + shift)
        resid = mx.compile(lambda x, gate, y: x + gate * y)

        def block_call(self_, x, vec, freqs, mask=None):
            prescale, preshift, pregate, postscale, postshift, postgate = self_.mod(vec)
            x = resid(x, pregate, self_.attn(modn(self_.prenorm(x), prescale, preshift), freqs=freqs, mask=mask))
            return resid(x, postgate, self_.mlp(modn(self_.postnorm(x), postscale, postshift)))

        SingleStreamBlock.__call__ = block_call
        self.row("block, modulation + residual compiled", self.time(lambda: b(self.x, self.vec, self.freqs, None)))

        def norm_mod(self_, x, vec, freqs, mask=None):
            prescale, preshift, pregate, postscale, postshift, postgate = self_.mod(vec)
            x = resid(
                x, pregate, self_.attn(normmod(self_.prenorm.scale, x, prescale, preshift), freqs=freqs, mask=mask)
            )
            return resid(x, postgate, self_.mlp(normmod(self_.postnorm.scale, x, postscale, postshift)))

        normmod = mx.compile(
            lambda scale, x, s, sh: (
                (1 + s) * mx.fast.rms_norm(x.astype(mx.float32), scale.astype(mx.float32) + 1.0, 1e-5).astype(x.dtype)
                + sh
            )
        )
        SingleStreamBlock.__call__ = norm_mod
        self.row(
            "block, norm + modulation + residual compiled", self.time(lambda: b(self.x, self.vec, self.freqs, None))
        )
        Krea2RopeEmbedder.apply_rope = staticmethod(lambda q, k, f: rope_c(q, k, f))
        Krea2SwiGLU.__call__ = swiglu
        self.row("block, all pieces compiled", self.time(lambda: b(self.x, self.vec, self.freqs, None)))
        Krea2RMSNorm.__call__ = BlockBudget.rms_bf16_weight
        normmod_bf16 = mx.compile(
            lambda scale, x, s, sh: (
                (1 + s) * mx.fast.rms_norm(x, (scale.astype(mx.float32) + 1.0).astype(x.dtype), 1e-5) + sh
            )
        )

        def norm_mod_bf16(self_, x, vec, freqs, mask=None):
            prescale, preshift, pregate, postscale, postshift, postgate = self_.mod(vec)
            x = resid(
                x, pregate, self_.attn(normmod_bf16(self_.prenorm.scale, x, prescale, preshift), freqs=freqs, mask=mask)
            )
            return resid(x, postgate, self_.mlp(normmod_bf16(self_.postnorm.scale, x, postscale, postshift)))

        SingleStreamBlock.__call__ = norm_mod_bf16
        self.row("block, all pieces compiled, norm bf16", self.time(lambda: b(self.x, self.vec, self.freqs, None)))

        Krea2RopeEmbedder.apply_rope = saved_rope
        Krea2SwiGLU.__call__ = saved_swiglu
        SingleStreamBlock.__call__ = saved_block
        Krea2RMSNorm.__call__ = saved_norm

    # -- 1e. the down projection: whole, as a loop of K-slices, as one batched qmm ------------

    def ksplit(self) -> None:
        b = self.block
        o = b.attn.wo(self.x)
        h = nn.silu(b.mlp.gate(o)) * b.mlp.up(o)
        mx.eval(h)
        d = b.mlp.down
        ref = d(h)
        mx.eval(ref)
        print("\n== mlp.down 16384->6144: whole, 4 K-slices in a loop, 4 K-slices as one batched qmm ==")
        self.row("down whole (K=16384)", self.time(lambda: d(h)))
        loop = KSplitDown(d, 4)
        self.row("down as 4 slices, loop + 3 adds", self.time(lambda: loop(h)))
        out = loop(h)
        mx.eval(out)
        print(f"    max |diff| vs whole {mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item():.4f}")
        from seqpatch import batched_down

        with Patched("4b"):
            w, sc, bi = batched_down(b.mlp, 4)

            def batched():
                bb, m, kk = h.shape
                xs = h.reshape(bb * m, 4, kk // 4).transpose(1, 0, 2)
                y = mx.quantized_matmul(xs, w, sc, bi, transpose=True, group_size=d.group_size, bits=d.bits)
                return y.sum(axis=0).reshape(bb, m, -1)

            self.row("down as one batched qmm (4, M, 4096) + sum", self.time(batched))
            out = batched()
            mx.eval(out)
            print(f"    max |diff| vs whole {mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item():.4f}")
        print("\n== the block with each down form (and the bf16 norm) ==")
        self.row("block", self.time(lambda: b(self.x, self.vec, self.freqs, None)))
        for spec in ("4c", "4b", "4bn", "1n"):
            with Patched(spec):
                self.row(f"block, {spec}", self.time(lambda: b(self.x, self.vec, self.freqs, None)))

    # -- 2. the block with each candidate fix ----------------------------------------------

    def block_variants(self) -> None:
        b = self.block
        print("\n== the block, and the block with each fix (M9c / M9d) ==")
        base = self.time(lambda: b(self.x, self.vec, self.freqs, None))
        self.row("block", base)
        self.row("block, compiled", self.time(self.compiled()), "M9d-2")

        original = Krea2RMSNorm.__call__
        for name, impl in (
            ("norm bf16 weight", BlockBudget.rms_bf16_weight),
            ("norm f32 weight", BlockBudget.rms_f32_weight),
        ):
            Krea2RMSNorm.__call__ = impl
            self.row(f"block, {name}", self.time(lambda: b(self.x, self.vec, self.freqs, None)), "M9d-1")
            self.row(f"block, {name}, compiled", self.time(self.compiled()))
            Krea2RMSNorm.__call__ = original

        b.mlp.down = KSplitDown(self.down, 4)
        self.row("block, down K4", self.time(lambda: b(self.x, self.vec, self.freqs, None)), "M9c")
        self.row("block, down K4, compiled", self.time(self.compiled()))
        Krea2RMSNorm.__call__ = BlockBudget.rms_f32_weight
        self.row(
            "block, down K4, norm f32 weight, compiled",
            self.time(self.compiled()),
            "all",
        )
        Krea2RMSNorm.__call__ = original
        b.mlp.down = self.down

    # -- 3. how far the norm variants move the numbers -------------------------------------

    @staticmethod
    def rms_bf16_weight(self, x: mx.array) -> mx.array:
        # No float32 round trip at all: the kernel accumulates in float32 internally, but the
        # weight (1 + scale) is rounded to bf16 first, which is the precision question.
        return mx.fast.rms_norm(x, (self.scale.astype(mx.float32) + 1.0).astype(x.dtype), self.eps)

    @staticmethod
    def rms_f32_weight(self, x: mx.array) -> mx.array:
        # Normalize in the input dtype, apply the float32 weight afterwards, round once.
        n = mx.fast.rms_norm(x, None, self.eps)
        return (n * (self.scale.astype(mx.float32) + 1.0)).astype(x.dtype)

    def norm_numerics(self) -> None:
        print("\n== the norm variants against the current float32 round trip (block 0's prenorm, real scale) ==")
        norm = self.block.prenorm
        x = self.x
        ref = Krea2RMSNorm.__call__(norm, x).astype(mx.float32)
        # The residual stream at this norm has a bf16 ULP of about 2^-8 relative.
        for name, impl in (("bf16 weight", BlockBudget.rms_bf16_weight), ("f32 weight", BlockBudget.rms_f32_weight)):
            out = impl(norm, x).astype(mx.float32)
            diff = mx.abs(out - ref)
            rel = diff / mx.maximum(mx.abs(ref), 1e-3)
            mx.eval(diff, rel)
            print(
                f"  {name:46s} max abs {diff.max().item():.5f}  mean abs {diff.mean().item():.6f}  "
                f"max rel {rel.max().item():.4f}  mean rel {rel.mean().item():.6f}  "
                f"differing {(diff > 0).sum().item() / diff.size:.3f}"
            )
        scale = self.block.prenorm.scale.astype(mx.float32)
        w = scale + 1.0
        w_bf16 = w.astype(mx.bfloat16).astype(mx.float32)
        mx.eval(w, w_bf16)
        print(
            f"  (1 + scale) itself: min {w.min().item():.4f} max {w.max().item():.4f}, "
            f"rounding it to bf16 moves it by at most {mx.abs(w - w_bf16).max().item():.5f} "
            f"({(mx.abs(w - w_bf16) / mx.abs(w)).max().item() * 100:.3f}%)"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="streamed checkpoint; uses its block 0")
    parser.add_argument("--tokens", type=int, default=4126)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--only", default="chain,gap,gap2,seq,fuse,ksplit,block_variants,norm_numerics")
    args = parser.parse_args()
    print(mx.device_info(), "mlx", mx.__version__, f"tokens={args.tokens}")
    budget = BlockBudget(args.tokens, args.model, args.warmup, args.iters)
    for name in args.only.split(","):
        getattr(budget, name)()
    print(f"\npeak memory {mx.get_peak_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
