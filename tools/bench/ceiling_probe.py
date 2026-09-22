import argparse
import time

import mlx.core as mx

# How fast can this GPU multiply matrices at all, and where do MLX's kernels fall short of
# that? Measured on the M6 mini (docs/16gb/measurements/2026-09-22-m9-ceiling-probe.md):
# dense bf16 peaks at 19.2 TFLOPS at 4096^3, and both the dense gemm and the q8 qmm lose
# half of it once K passes 8192 (16384->6144: 9.1 TFLOPS). PyTorch MPS holds 17.9 on the
# same shape (tools/bench/mps_probe.py), so the cliff is MLX's, not the silicon's.
#
#   uv run python tools/bench/ceiling_probe.py             # everything, ~2 minutes
#   uv run python tools/bench/ceiling_probe.py --only ksweep,bits


class CeilingProbe:
    M, N, K_DOWN = 4126, 6144, 16384  # Krea2 DiT at 1024^2, and its 16384->6144 down projection

    @staticmethod
    def time(fn, warmup: int = 3, iters: int = 8) -> float:
        for _ in range(warmup):
            mx.eval(fn())
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        mx.synchronize()
        return (time.perf_counter() - t0) / iters

    @staticmethod
    def row(label: str, seconds: float, flops: float, extra: str = "") -> None:
        print(f"  {label:44s} {seconds * 1e3:8.2f} ms {flops / seconds / 1e12:6.2f} TFLOPS  {extra}")

    @classmethod
    def dense(cls) -> None:
        print("\n== dense square matmul: the ceiling as MLX exposes it ==")
        for n in (2048, 4096, 6144, 8192):
            for dt in (mx.bfloat16, mx.float16):
                a = mx.random.normal((n, n)).astype(dt)
                b = mx.random.normal((n, n)).astype(dt)
                mx.eval(a, b)
                cls.row(f"{n}^3 {dt}", cls.time(lambda: a @ b), 2 * n**3)

    @classmethod
    def qmm(cls, x, w, bits: int, gs: int, mode: str = "affine"):
        q = mx.quantize(w, group_size=gs, bits=bits, mode=mode)
        mx.eval(x, *q)
        return q, cls.time(lambda: mx.quantized_matmul(x, *q, transpose=True, group_size=gs, bits=bits, mode=mode))

    @classmethod
    def ksweep(cls) -> None:
        print(f"\n== q8 qmm, K sweep at N={cls.N}, M={cls.M}: where the cliff starts ==")
        for k in (2048, 4096, 6144, 8192, 12288, 16384):
            x = mx.random.normal((1, cls.M, k)).astype(mx.bfloat16)
            _, s = cls.qmm(x, mx.random.normal((cls.N, k)).astype(mx.bfloat16), 8, 64)
            cls.row(f"q8 K={k}->{cls.N}", s, 2 * cls.M * k * cls.N)
        print(f"\n== q8 qmm {cls.K_DOWN}->{cls.N}: does group size, dtype or M move the cliff? ==")
        for gs in (32, 64, 128):
            for dt in (mx.bfloat16, mx.float16):
                x = mx.random.normal((1, cls.M, cls.K_DOWN)).astype(dt)
                _, s = cls.qmm(x, mx.random.normal((cls.N, cls.K_DOWN)).astype(dt), 8, gs)
                cls.row(f"gs={gs} {dt}", s, 2 * cls.M * cls.K_DOWN * cls.N)
        for m in (2048, 8192):
            x = mx.random.normal((1, m, cls.K_DOWN)).astype(mx.bfloat16)
            _, s = cls.qmm(x, mx.random.normal((cls.N, cls.K_DOWN)).astype(mx.bfloat16), 8, 64)
            cls.row(f"M={m} gs=64 bf16", s, 2 * m * cls.K_DOWN * cls.N)
        print("\n== the same two MLP shapes dequantized to dense bf16 ==")
        for k, n in ((6144, 16384), (16384, 6144)):
            x = mx.random.normal((1, cls.M, k)).astype(mx.bfloat16)
            q, s = cls.qmm(x, mx.random.normal((n, k)).astype(mx.bfloat16), 8, 64)
            cls.row(f"qmm q8 {k}->{n}", s, 2 * cls.M * k * n)
            w = mx.dequantize(*q, group_size=64, bits=8)
            mx.eval(w)
            cls.row(f"dense bf16 {k}->{n}", cls.time(lambda: x @ w.T), 2 * cls.M * k * n)

    @classmethod
    def bits(cls) -> None:
        print("\n== fewer bits: what q6 / q4 / mxfp8 buy in compute at the Krea2 shapes ==")
        for k, n in ((6144, 6144), (6144, 16384), (16384, 6144)):
            x = mx.random.normal((1, cls.M, k)).astype(mx.bfloat16)
            w = mx.random.normal((n, k)).astype(mx.bfloat16)
            for bits, gs, mode in (
                (8, 64, "affine"),
                (6, 64, "affine"),
                (4, 64, "affine"),
                (4, 32, "affine"),
                (8, 32, "mxfp8"),
            ):
                q, s = cls.qmm(x, w, bits, gs, mode)
                nbytes = sum(a.nbytes for a in q)
                cls.row(f"{k}->{n} {mode} b{bits} gs{gs}", s, 2 * cls.M * k * n, f"weight {nbytes / 1e6:6.1f} MB")

    @classmethod
    def sdpa(cls) -> None:
        print(f"\n== sdpa 48 heads / 12 kv heads, L={cls.M}, D=128 bf16 ==")
        q = mx.random.normal((1, 48, cls.M, 128)).astype(mx.bfloat16)
        kv = mx.random.normal((1, 12, cls.M, 128)).astype(mx.bfloat16)
        mx.eval(q, kv)
        s = cls.time(lambda: mx.fast.scaled_dot_product_attention(q, kv, kv, scale=128**-0.5))
        cls.row("sdpa gqa", s, 4 * 48 * cls.M * cls.M * 128)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="dense,ksweep,bits,sdpa")
    args = parser.parse_args()
    print(mx.device_info(), "mlx", mx.__version__)
    for name in args.only.split(","):
        getattr(CeilingProbe, name)()
    print(f"\npeak memory {mx.get_peak_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
