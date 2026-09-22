import mlx.core as mx
from mlx import nn


class Krea2SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)
        # 1 runs the down projection as one matmul. On GPUs with neural accelerators MLX's
        # q8 matmul halves its speed once K passes 8192 (16384->6144: 9 TFLOPS against 17 for
        # the other shapes), and issuing it as one batched matmul over K-slices gets it back
        # (92 -> 51 ms a block on an M6). Only the block-streaming path sets this; the sum
        # changes the accumulation order, so the resident path keeps its reference images.
        # See docs/16gb/measurements/2026-09-22-m9-ceiling-probe.md, section 2.
        self.down_splits = 1

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.silu(self.gate(x)) * self.up(x)
        if self.down_splits > 1 and isinstance(self.down, nn.QuantizedLinear):
            return self._down_in_slices(h)
        return self.down(h)

    def _down_in_slices(self, h: mx.array) -> mx.array:
        d, parts = self.down, self.down_splits
        cache = getattr(self, "_down_planes", None)
        if cache is None or cache[0] is not d.weight:
            # (N, K/pack) -> (parts, N, K/pack/parts), cut once per bound weight: the
            # streamed blocks rebind every call, the resident ones never.
            planes = tuple(
                mx.contiguous(a.reshape(a.shape[0], parts, a.shape[1] // parts).transpose(1, 0, 2))
                for a in (d.weight, d.scales, d.biases)
            )
            self._down_planes = cache = (d.weight, planes)
        w, scales, biases = cache[1]
        b, m, k = h.shape
        xs = h.reshape(b * m, parts, k // parts).transpose(1, 0, 2)
        y = mx.quantized_matmul(xs, w, scales, biases, transpose=True, group_size=d.group_size, bits=d.bits)
        y = y.sum(axis=0).reshape(b, m, -1)
        return y + d["bias"] if "bias" in d else y

    def release_down_planes(self) -> None:
        # The planes are copies of a bound weight; a streamed block drops them with it, or
        # 28 blocks' worth (about 3 GB) would stay resident.
        self._down_planes = None
