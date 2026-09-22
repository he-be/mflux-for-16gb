import argparse
import time

import mlx.core as mx

# Each of the eight q8 matmuls of one DiT block on its own, so a slow shape cannot hide in
# the block total. block_profile.py measured the eight together at 7.3 TFLOPS in bfloat16
# while nax_probe.py saw 17 TFLOPS on the largest of them; this finds out which shapes
# fall short, and whether the token count (4126 is not a multiple of 64) matters. The
# 16384->6144 projection is the slow one, so it is also timed as a sum of K-slices.
#
#   uv run python tools/bench/qmm_shapes.py --tokens 4126,4096,5120


class QmmShapes:
    SHAPES = {
        "wq/gate/wo  6144->6144": (6144, 6144),
        "wk/wv       6144->1536": (6144, 1536),
        "mlp gate/up 6144->16384": (6144, 16384),
        "mlp down   16384->6144": (16384, 6144),
    }

    @staticmethod
    def time(fn, warmup: int, iters: int) -> float:
        for _ in range(warmup):
            mx.eval(fn())
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        mx.synchronize()
        return (time.perf_counter() - t0) / iters

    @classmethod
    def run(cls, tokens: int, dtype, warmup: int, iters: int) -> None:
        print(f"\n{dtype}  tokens={tokens}")
        for name, (k, n) in cls.SHAPES.items():
            x = mx.random.normal((1, tokens, k)).astype(dtype)
            wq, scales, biases = mx.quantize(mx.random.normal((n, k)).astype(mx.bfloat16), group_size=64, bits=8)
            mx.eval(x, wq, scales, biases)
            t = cls.time(
                lambda: mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=64, bits=8),
                warmup,
                iters,
            )
            flops = 2 * tokens * k * n
            print(f"  {name:26s} {t * 1e3:7.1f} ms  {flops / t / 1e12:6.2f} TFLOPS")
            if k != 16384:
                continue
            for parts in (2, 4):
                kk = k // parts
                slices = [
                    mx.quantize(mx.random.normal((n, kk)).astype(mx.bfloat16), group_size=64, bits=8)
                    for _ in range(parts)
                ]
                mx.eval(*[a for sl in slices for a in sl])

                def split():
                    acc = None
                    for i, (wq_i, s_i, b_i) in enumerate(slices):
                        y = mx.quantized_matmul(
                            x[..., i * kk : (i + 1) * kk], wq_i, s_i, b_i, transpose=True, group_size=64, bits=8
                        )
                        acc = y if acc is None else acc + y
                    return acc

                t = cls.time(split, warmup, iters)
                print(f"  {f'  same as {parts} K-slices':26s} {t * 1e3:7.1f} ms  {flops / t / 1e12:6.2f} TFLOPS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="4126,4096,5120")
    parser.add_argument("--dtypes", default="bfloat16,float32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    print(mx.device_info(), "mlx", mx.__version__)
    for tokens in (int(t) for t in args.tokens.split(",")):
        for name in args.dtypes.split(","):
            QmmShapes.run(tokens, getattr(mx, name), args.warmup, args.iters)


if __name__ == "__main__":
    main()
