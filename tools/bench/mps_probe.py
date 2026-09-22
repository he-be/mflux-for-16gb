import time

import torch
import torch.nn.functional as F

# The same shapes as tools/bench/ceiling_probe.py, through PyTorch's MPS backend instead of
# MLX, to tell an MLX kernel limit from a silicon limit. On the M6 mini MPS holds 17.9-19.0
# TFLOPS at every shape, including the 16384->6144 projection that MLX runs at 9.1; its
# attention matches MLX at 17; and dequantizing a q8 weight to bf16 before a dense matmul
# (what a GGUF-style path would do) costs 11 ms per 100 MB.
# See docs/16gb/measurements/2026-09-22-m9-ceiling-probe.md.
#
# torch is not a project dependency; run it in a throwaway environment:
#   uv run --no-project --isolated --with torch --python 3.12 python tools/bench/mps_probe.py


class MpsProbe:
    M, N, K_DOWN = 4126, 6144, 16384
    dev = torch.device("mps")

    @staticmethod
    def time(fn, warmup: int = 3, iters: int = 8) -> float:
        for _ in range(warmup):
            fn()
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.mps.synchronize()
        return (time.perf_counter() - t0) / iters

    @staticmethod
    def row(label: str, seconds: float, flops: float) -> None:
        print(f"  {label:44s} {seconds * 1e3:8.2f} ms {flops / seconds / 1e12:6.2f} TFLOPS")

    @classmethod
    def matmul(cls) -> None:
        print("\n== dense matmul ==")
        shapes = (
            (4096, 4096, 4096),
            (6144, 6144, 6144),
            (8192, 8192, 8192),
            (cls.M, 6144, 16384),
            (cls.M, 16384, 6144),
        )
        for m, k, n in shapes:
            for dt in (torch.float16, torch.bfloat16):
                a = torch.randn(m, k, device=cls.dev, dtype=dt)
                b = torch.randn(k, n, device=cls.dev, dtype=dt)
                cls.row(f"{m}x{k}@{k}x{n} {dt}", cls.time(lambda: a @ b), 2 * m * k * n)

    @classmethod
    def sdpa(cls) -> None:
        print("\n== sdpa 48 heads / 12 kv heads, D=128 ==")
        L, H, KV, D = cls.M, 48, 12, 128
        for dt in (torch.float16, torch.bfloat16):
            q = torch.randn(1, H, L, D, device=cls.dev, dtype=dt)
            k = torch.randn(1, H, L, D, device=cls.dev, dtype=dt)
            v = torch.randn_like(k)
            cls.row(
                f"sdpa k/v repeated {dt}", cls.time(lambda: F.scaled_dot_product_attention(q, k, v)), 4 * H * L * L * D
            )
            kv = torch.randn(1, KV, L, D, device=cls.dev, dtype=dt)
            s = cls.time(lambda: F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True))
            cls.row(f"sdpa gqa {dt}", s, 4 * H * L * L * D)

    @classmethod
    def dequant(cls) -> None:
        print("\n== q8 dequantized to bf16 on the fly, then dense (a GGUF-style path) ==")
        M, K, N = cls.M, cls.K_DOWN, cls.N
        x = torch.randn(1, M, K, device=cls.dev, dtype=torch.bfloat16)
        w8 = torch.randint(-128, 127, (N, K), device=cls.dev, dtype=torch.int8)
        scales = torch.randn(N, K // 64, 1, device=cls.dev, dtype=torch.bfloat16)

        def dequantize():
            return (w8.view(N, K // 64, 64).to(torch.bfloat16) * scales).view(N, K)

        cls.row(f"dequant + dense {K}->{N}", cls.time(lambda: x @ dequantize().T), 2 * M * K * N)
        print(f"  {'dequant alone (100 MB of q8)':44s} {cls.time(dequantize) * 1e3:8.2f} ms")


def main() -> None:
    assert torch.backends.mps.is_available()
    print("torch", torch.__version__)
    MpsProbe.matmul()
    MpsProbe.sdpa()
    MpsProbe.dequant()
    print(f"\nmps driver memory {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
