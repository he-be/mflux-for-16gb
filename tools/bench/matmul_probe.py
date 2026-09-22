import time

import mlx.core as mx


class MatmulProbe:
    M, K, N = 5120, 6144, 6144
    B, H, L, D = 1, 24, 4096, 128

    @staticmethod
    def time(fn, warmup=3, iters=20):
        for _ in range(warmup):
            mx.eval(fn())
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        mx.synchronize()
        return (time.perf_counter() - t0) / iters

    @classmethod
    def run(cls):
        M, K, N = cls.M, cls.K, cls.N
        flops = 2 * M * K * N
        print(mx.device_info())
        print(f"shape {M}x{K} @ {K}x{N}  ({flops / 1e9:.1f} GFLOP)")

        for dt in (mx.bfloat16, mx.float16, mx.float32):
            a = mx.random.normal((M, K)).astype(dt)
            b = mx.random.normal((K, N)).astype(dt)
            mx.eval(a, b)
            t = cls.time(lambda: a @ b)
            print(f"  {str(dt):>18}  {t * 1e3:7.2f} ms  {flops / t / 1e12:6.2f} TFLOPS")

        for bits in (8, 4):
            a = mx.random.normal((M, K)).astype(mx.bfloat16)
            w = mx.random.normal((N, K)).astype(mx.bfloat16)
            wq, scales, biases = mx.quantize(w, group_size=64, bits=bits)
            mx.eval(a, wq, scales, biases)
            t = cls.time(lambda: mx.quantized_matmul(a, wq, scales, biases, transpose=True, group_size=64, bits=bits))
            print(f"  {f'q{bits} matmul':>18}  {t * 1e3:7.2f} ms  {flops / t / 1e12:6.2f} TFLOPS")

        B, H, L, D = cls.B, cls.H, cls.L, cls.D
        q = mx.random.normal((B, H, L, D)).astype(mx.bfloat16)
        k = mx.random.normal((B, H, L, D)).astype(mx.bfloat16)
        v = mx.random.normal((B, H, L, D)).astype(mx.bfloat16)
        mx.eval(q, k, v)
        t = cls.time(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5))
        print(f"  {f'sdpa {L}x{D}x{H}':>18}  {t * 1e3:7.2f} ms  {4 * B * H * L * L * D / t / 1e12:6.2f} TFLOPS")


if __name__ == "__main__":
    MatmulProbe.run()
