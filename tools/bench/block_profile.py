import argparse
import time
from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.krea2.model.krea2_transformer.rope_embedder import Krea2RopeEmbedder
from mflux.models.krea2.model.krea2_transformer.transformer_block import SingleStreamBlock
from mflux.models.krea2.weights.krea2_weight_stream import Krea2BlockStream

# Where does the time inside one DiT block go? M7 measured 610 ms of compute per block on
# the M6 mini against a matmul budget that the nax kernels should finish in ~350 ms. This
# splits one block at the real 1024^2 shape into its matmuls, its attention and the rest,
# in float32 (what the pipeline runs today, because the latent noise is float32) and in
# bfloat16 (what the q8 checkpoint's bf16 scales and norms suggest it was meant to run in).
#
# With --model it loads block 0 of a real streamed checkpoint (bf16 scales, like production);
# without it the block is synthetic and --scales picks the dtype of its scales and norms.
#
#   uv run python tools/bench/block_profile.py --model ~/Library/Caches/mflux/16gb-bench/krea2-lowram
#   uv run python tools/bench/block_profile.py --tokens 6430   # 1280^2, synthetic weights


class BlockProfile:
    FEATURES, HEADS, KVHEADS, MULT, HEAD_DIM = 6144, 48, 12, 4, 128
    MLP_DIM = 16384

    def __init__(self, tokens: int, warmup: int, iters: int, scales, model: Path | None):
        self.tokens = tokens
        self.warmup = warmup
        self.iters = iters
        self.block = SingleStreamBlock(self.FEATURES, self.HEADS, self.MULT, False, self.KVHEADS)
        self.block.set_dtype(scales)
        nn.quantize(self.block, group_size=64, bits=8)
        if model is not None:
            self.block.update(Krea2BlockStream(Krea2BlockStream.locate(model)).read(0))
        mx.eval(self.block.parameters())
        self.weights = "block 0 of " + str(model) if model else f"synthetic, scales {scales}"
        ids = mx.zeros((1, tokens, 3), dtype=mx.float32)
        self.freqs = Krea2RopeEmbedder(self.HEAD_DIM, 1000, [32, 48, 48])(ids)
        mx.eval(self.freqs)

    def time(self, fn, clear_cache: bool = False) -> float:
        for _ in range(self.warmup):
            mx.eval(fn())
        mx.synchronize()
        total = 0.0
        for _ in range(self.iters):
            if clear_cache:
                mx.clear_cache()
            t0 = time.perf_counter()
            mx.eval(fn())
            mx.synchronize()
            total += time.perf_counter() - t0
        return total / self.iters

    def matmul_flops(self) -> float:
        L, F, M, KV = self.tokens, self.FEATURES, self.MLP_DIM, self.KVHEADS * self.HEAD_DIM
        return 2 * L * (3 * F * F + 2 * F * KV + 2 * F * M + M * F)

    def attention_flops(self) -> float:
        return 4 * self.HEADS * self.tokens * self.tokens * self.HEAD_DIM

    def run(self, dtype) -> dict:
        L, F = self.tokens, self.FEATURES
        x = mx.random.normal((1, L, F)).astype(dtype)
        vec = mx.random.normal((1, 1, 6 * F)).astype(dtype)
        mx.eval(x, vec)
        b = self.block
        rows: dict[str, float] = {}

        # The eight matmuls of one block, chained the way the block chains them, attention left out.
        def matmuls():
            a = b.attn.wq(x) * mx.sigmoid(b.attn.gate(x)) + b.attn.wk(x).sum() + b.attn.wv(x).sum()
            o = b.attn.wo(a)
            return b.mlp.down(nn.silu(b.mlp.gate(o)) * b.mlp.up(o))

        rows["matmuls"] = self.time(matmuls)

        # Attention as the block issues it (k/v repeated to 48 heads) and as GQA (12 kv heads).
        q = mx.random.normal((1, self.HEADS, L, self.HEAD_DIM)).astype(dtype)
        kv = mx.random.normal((1, self.KVHEADS, L, self.HEAD_DIM)).astype(dtype)
        mx.eval(q, kv)
        scale = self.HEAD_DIM**-0.5
        rep = self.HEADS // self.KVHEADS
        rows["sdpa repeat"] = self.time(
            lambda: mx.fast.scaled_dot_product_attention(
                q, mx.repeat(kv, rep, axis=1), mx.repeat(kv, rep, axis=1), scale=scale
            )
        )
        rows["sdpa gqa"] = self.time(lambda: mx.fast.scaled_dot_product_attention(q, kv, kv, scale=scale))

        rows["block"] = self.time(lambda: b(x, vec, self.freqs, None))
        rows["block+clear_cache"] = self.time(lambda: b(x, vec, self.freqs, None), clear_cache=True)
        rows["matmuls again"] = self.time(matmuls)
        # Everything that is neither a matmul nor attention: norms, modulation, rope, residuals.
        rows["rest (block - matmuls - sdpa repeat)"] = rows["block"] - rows["matmuls"] - rows["sdpa repeat"]
        return rows

    def report(self, dtype, rows: dict) -> None:
        mm, at = self.matmul_flops(), self.attention_flops()
        print(f"\n{dtype} activations, {self.weights}, tokens={self.tokens}")
        print(f"  matmul {mm / 1e12:.2f} TFLOP  attention {at / 1e12:.2f} TFLOP")
        for name, t in rows.items():
            flops = (
                mm + at
                if name.startswith("block")
                else mm
                if name.startswith("matmuls")
                else at
                if name.startswith("sdpa")
                else None
            )
            tf = f"{flops / t / 1e12:6.2f} TFLOPS" if flops else ""
            print(f"  {name:38s} {t * 1e3:8.1f} ms  {tf}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4126)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--dtypes", default="float32,bfloat16")
    parser.add_argument("--scales", default="bfloat16", help="dtype of the synthetic block's scales and norms")
    parser.add_argument("--model", type=Path, default=None, help="streamed checkpoint; profiles its block 0")
    args = parser.parse_args()
    print(mx.device_info(), "mlx", mx.__version__)
    profile = BlockProfile(args.tokens, args.warmup, args.iters, getattr(mx, args.scales), args.model)
    for name in args.dtypes.split(","):
        dtype = getattr(mx, name)
        profile.report(dtype, profile.run(dtype))
        print(f"  peak memory {mx.get_peak_memory() / 1e9:.2f} GB")
        mx.reset_peak_memory()


if __name__ == "__main__":
    main()
