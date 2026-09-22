import argparse
import time
from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.krea2.model.krea2_transformer.rope_embedder import Krea2RopeEmbedder
from mflux.models.krea2.model.krea2_transformer.transformer_block import SingleStreamBlock
from mflux.models.krea2.weights.krea2_weight_stream import Krea2BlockStream

# M10 measured a runtime LoRA at +81.7 ms on a 282 ms block, while the adapter's own
# arithmetic is about 1.3% of the block's FLOPs. This says where the rest of it goes and
# what each candidate fix returns, on one real block in one process so the parts add up.
#
#   uv run python tools/bench/lora_budget.py --model ~/Library/Caches/mflux/16gb-bench/krea2-lowram
#   uv run python tools/bench/lora_budget.py --model ... --tokens 6430    # 1280^2
#
# The factors are random: only their shapes and dtype decide the time, and a file would
# only tie the probe to one adapter.


class ScaledLoRA(nn.Module):
    # The shipped layer is base(x) + scale * (x @ A @ B), which walks the full-size output
    # twice more than it has to. Folding the scalar into B once at load leaves one add.
    def __init__(self, linear, lora_A: mx.array, lora_B: mx.array, scale: float):
        super().__init__()
        self.linear = linear
        self.lora_A = lora_A
        self.lora_B = (scale * lora_B).astype(lora_B.dtype)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x) + mx.matmul(mx.matmul(x, self.lora_A), self.lora_B)


class NarrowScaledLoRA(nn.Module):
    # The same product with the scalar moved onto the rank-r intermediate, which is a few
    # hundred KB instead of the full-size output. Nothing is precomputed and the dtype is
    # unchanged, so this is available to every adapter without a load-time pass.
    def __init__(self, linear, lora_A: mx.array, lora_B: mx.array, scale: float):
        super().__init__()
        self.linear = linear
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.scale = scale

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x) + mx.matmul(self.scale * mx.matmul(x, self.lora_A), self.lora_B)


class LoraBudget:
    FEATURES, HEADS, KVHEADS, MULT, HEAD_DIM = 6144, 48, 12, 4, 128
    # Every layer the Krea 2 mapping touches inside a block, with the module that holds it.
    TARGETS = (("attn", "wq"), ("attn", "wk"), ("attn", "wv"), ("attn", "gate"), ("attn", "wo"),
               ("mlp", "gate"), ("mlp", "up"), ("mlp", "down"))  # fmt: skip

    def __init__(self, tokens: int, model: Path, rank: int, scale: float, splits: int, warmup: int, iters: int):
        self.tokens, self.rank, self.scale = tokens, rank, scale
        self.warmup, self.iters = warmup, iters
        self.block = SingleStreamBlock(self.FEATURES, self.HEADS, self.MULT, False, self.KVHEADS)
        self.block.set_dtype(mx.bfloat16)
        nn.quantize(self.block, group_size=64, bits=8)
        self.block.update(Krea2BlockStream(Krea2BlockStream.locate(model)).read(0))
        # The production shape: block streaming splits the down projection, and M8c showed a
        # change that wins without it can lose with it.
        self.block.mlp.down_splits = splits
        mx.eval(self.block.parameters())
        ids = mx.zeros((1, tokens, 3), dtype=mx.float32)
        self.freqs = Krea2RopeEmbedder(self.HEAD_DIM, 1000, [32, 48, 48])(ids)
        self.x = mx.random.normal((1, tokens, self.FEATURES)).astype(mx.bfloat16)
        self.vec = mx.random.normal((1, 1, 6 * self.FEATURES)).astype(mx.bfloat16)
        mx.eval(self.freqs, self.x, self.vec)
        self.bare = {(h, n): getattr(getattr(self.block, h), n) for h, n in self.TARGETS}
        self.factors = {key: self._factors(linear) for key, linear in self.bare.items()}

    def run(self) -> None:
        print(f"\n== one production block, {self.tokens} tokens, rank {self.rank} on {len(self.TARGETS)} layers ==")
        base = self.time(self.forward)
        LoraBudget.row("block, no adapter", base)

        self.wrap(LoraBudget._shipped)
        shipped = self.time(self.forward)
        LoraBudget.row("block + LoRALinear as shipped", shipped, f"{(shipped - base) * 1e3:+.1f} ms")

        self.wrap(LoraBudget._scaled)
        folded = self.time(self.forward)
        LoraBudget.row("block + scale folded into B", folded, f"{(folded - base) * 1e3:+.1f} ms")

        self.wrap(LoraBudget._narrow)
        narrow = self.time(self.forward)
        LoraBudget.row("block + scale on the rank-r intermediate", narrow, f"{(narrow - base) * 1e3:+.1f} ms")
        self.restore()

        self.arithmetic(base)
        self.dtypes()

    # -- the adapter's own pieces, outside the block -----------------------------------------

    def arithmetic(self, base: float) -> None:
        print("\n== the adapter's arithmetic alone, by piece ==")
        inputs = self.inputs()
        totals = {"A only": 0.0, "A then B": 0.0, "+ scalar": 0.0, "+ add": 0.0, "narrow": 0.0}
        for key in self.TARGETS:
            a, b = self.factors[key]
            src = inputs[key]
            out = self.bare[key](src)
            mx.eval(out)
            times = {
                "A only": self.time(lambda a=a, s=src: mx.matmul(s, a)),
                "A then B": self.time(lambda a=a, b=b, s=src: mx.matmul(mx.matmul(s, a), b)),
                "+ scalar": self.time(lambda a=a, b=b, s=src: self.scale * mx.matmul(mx.matmul(s, a), b)),
                "+ add": self.time(lambda a=a, b=b, s=src, o=out: o + self.scale * mx.matmul(mx.matmul(s, a), b)),
                "narrow": self.time(lambda a=a, b=b, s=src, o=out: o + mx.matmul(self.scale * mx.matmul(s, a), b)),
            }
            for name, seconds in times.items():
                totals[name] += seconds
            LoraBudget.row(
                f"{key[0]}.{key[1]:<6s} out {out.shape[-1]:>5d}",
                times["+ add"],
                "  ".join(f"{n} {t * 1e3:5.1f}" for n, t in times.items() if n != "+ add"),
            )
        print()
        for name, seconds in totals.items():
            LoraBudget.row(f"sum over the block, {name}", seconds, f"{seconds / base:5.1%} of the block")

    def inputs(self) -> dict:
        # What each layer actually sees: the block's hidden for the attention projections,
        # the attention output for the mlp, and the 16384-wide activation for mlp.down.
        wide = mx.random.normal((1, self.tokens, self.block.mlp.down.weight.shape[1] * 32 // 8)).astype(mx.bfloat16)
        mx.eval(wide)
        return {key: (wide if key == ("mlp", "down") else self.x) for key in self.TARGETS}

    def dtypes(self) -> None:
        print("\n== dtypes (a float32 factor would push the whole block back to float32) ==")
        key = ("attn", "wq")
        a, b = self.factors[key]
        out = self.bare[key](self.x) + self.scale * mx.matmul(mx.matmul(self.x, a), b)
        mx.eval(out)
        print(f"  x {self.x.dtype}, A {a.dtype}, B {b.dtype}, base {self.bare[key](self.x).dtype}, sum {out.dtype}")

    # -- plumbing ----------------------------------------------------------------------------

    def _factors(self, linear) -> tuple:
        out_dims, packed = linear.weight.shape
        in_dims = packed * 32 // linear.bits
        a = mx.random.normal((in_dims, self.rank)).astype(mx.bfloat16) * 0.02
        b = mx.random.normal((self.rank, out_dims)).astype(mx.bfloat16) * 0.02
        mx.eval(a, b)
        return a, b

    def wrap(self, make) -> None:
        for (holder, name), linear in self.bare.items():
            a, b = self.factors[(holder, name)]
            setattr(getattr(self.block, holder), name, make(linear, a, b, self.scale))
        self.block.mlp._down_adapter = LoraBudget._adapter(self.block.mlp.down)
        mx.eval(self.block.parameters())

    @staticmethod
    def _adapter(down):
        # The shipped layer is the only one Krea2BlockStream knows; the probe's two variants
        # need the same treatment or they silently lose the K-split and pay the M9 cliff on
        # top of the adapter, which is not what is being compared.
        if isinstance(down, LoRALinear):
            return Krea2BlockStream._down_adapter(down)
        if isinstance(down, ScaledLoRA):
            return down.linear, lambda h: mx.matmul(mx.matmul(h, down.lora_A), down.lora_B)
        if isinstance(down, NarrowScaledLoRA):
            return down.linear, lambda h: mx.matmul(down.scale * mx.matmul(h, down.lora_A), down.lora_B)
        return None

    def restore(self) -> None:
        for (holder, name), linear in self.bare.items():
            setattr(getattr(self.block, holder), name, linear)
        self.block.mlp._down_adapter = None

    @staticmethod
    def _shipped(linear, a, b, scale):
        layer = LoRALinear.from_linear(linear, r=a.shape[1], scale=scale)
        layer.lora_A, layer.lora_B = a, b
        return layer

    @staticmethod
    def _scaled(linear, a, b, scale):
        return ScaledLoRA(linear, a, b, scale)

    @staticmethod
    def _narrow(linear, a, b, scale):
        return NarrowScaledLoRA(linear, a, b, scale)

    def forward(self):
        return self.block(self.x, self.vec, self.freqs, None)

    def time(self, fn) -> float:
        for _ in range(self.warmup):
            mx.eval(fn())
        mx.synchronize()
        start = time.perf_counter()
        for _ in range(self.iters):
            mx.eval(fn())
        mx.synchronize()
        return (time.perf_counter() - start) / self.iters

    @staticmethod
    def row(label: str, seconds: float, note: str = "") -> None:
        print(f"  {label:44s} {seconds * 1e3:8.1f} ms  {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Where a runtime LoRA's time goes inside one block.")
    parser.add_argument("--model", type=Path, required=True, help="streamed checkpoint; uses its block 0")
    parser.add_argument("--tokens", type=int, default=4126)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--down-splits", type=int, default=4)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=8)
    args = parser.parse_args()

    mx.set_cache_limit(Krea2BlockStream.CACHE_LIMIT_BYTES)
    LoraBudget(args.tokens, args.model, args.rank, args.scale, args.down_splits, args.warmup, args.iters).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
