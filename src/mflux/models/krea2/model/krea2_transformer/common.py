import mlx.core as mx
from mlx import nn


class Krea2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.scale = mx.zeros((dim,))
        self.eps = eps
        # False normalizes in float32 and casts back. True hands the kernel the input as is:
        # mx.fast.rms_norm accumulates in float32 either way, so what changes is only that
        # (1 + scale) is rounded to the input dtype first (at most 0.39% on the Krea 2
        # checkpoint), and the two casts of the 50 MB residual stream go away (-10 ms a block
        # on an M6). Only the block-streaming path sets this; the resident path keeps its
        # reference images. See docs/16gb/measurements/2026-09-22-m9b-block-budget.md, section 5.
        self.native_dtype = False

    def __call__(self, x: mx.array) -> mx.array:
        weight = self.scale.astype(mx.float32) + 1.0
        if self.native_dtype:
            return mx.fast.rms_norm(x, weight.astype(x.dtype), self.eps)
        return mx.fast.rms_norm(x.astype(mx.float32), weight, self.eps).astype(x.dtype)


class Krea2QKNorm(nn.Module):
    def __init__(self, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.qnorm = Krea2RMSNorm(head_dim, eps=eps)
        self.knorm = Krea2RMSNorm(head_dim, eps=eps)

    def __call__(self, q: mx.array, k: mx.array) -> tuple[mx.array, mx.array]:
        return self.qnorm(q), self.knorm(k)
