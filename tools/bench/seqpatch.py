import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.krea2.model.krea2_transformer.attention import Krea2Attention
from mflux.models.krea2.model.krea2_transformer.common import Krea2RMSNorm
from mflux.models.krea2.model.krea2_transformer.feed_forward import Krea2SwiGLU
from mflux.models.krea2.model.krea2_transformer.rope_embedder import Krea2RopeEmbedder

# MLX dispatches independent kernels concurrently, and two of this block's q8 matmuls running
# at once cost more than the two in sequence (mlp.gate + mlp.up: 97.6 ms in sequence, 132.7
# concurrent — M9b). These replacements for Krea2Attention.__call__ and Krea2SwiGLU.__call__
# chain the independent matmuls with mx.depends so they run one after another, and cut the
# down projection into K-slices that are chained the same way. Used by block_budget.py and
# stream_ab.py to measure before anything is changed in src/.


def after(x: mx.array, prev: mx.array, data: bool) -> mx.array:
    # x, made to run after prev. mx.depends is a graph-only edge and leaves the Metal dispatch
    # concurrent (M9b 3b); a real data dependency through one tiny elementwise kernel (0.9 ms
    # on a 50 MB x) does put a barrier between the two matmuls.
    if data:
        return x + (prev[..., :1] * 0).astype(x.dtype)
    return mx.depends(x, prev)


def make_attention(data: bool):
    def call(self: Krea2Attention, x: mx.array, freqs=None, mask=None) -> mx.array:
        B, L, _ = x.shape
        q = self.wq(x)
        k = self.wk(after(x, q, data))
        v = self.wv(after(x, k, data))
        gate = self.gate(after(x, v, data))
        return finish_attention(self, x, q, k, v, gate, freqs, mask)

    return call


def finish_attention(self: Krea2Attention, x, q, k, v, gate, freqs, mask) -> mx.array:
    B, L, _ = x.shape
    q = q.reshape(B, L, self.heads, self.head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, L, self.kvheads, self.head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, L, self.kvheads, self.head_dim).transpose(0, 2, 1, 3)
    q, k = self.qknorm(q, k)
    if freqs is not None:
        q, k = Krea2RopeEmbedder.apply_rope(q, k, freqs)
    out = scaled_dot_product_attention(q.astype(v.dtype), k.astype(v.dtype), v, scale=self.scale, mask=mask)
    out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.wo(out * mx.sigmoid(gate))


def down_slices(mlp: Krea2SwiGLU, parts: int, precut: bool) -> list:
    d = mlp.down
    _, packed = d.weight.shape
    kk, sk = packed // parts, d.scales.shape[1] // parts
    views = [
        (d.weight[:, i * kk : (i + 1) * kk], d.scales[:, i * sk : (i + 1) * sk], d.biases[:, i * sk : (i + 1) * sk])
        for i in range(parts)
    ]
    if not precut:
        return views
    # Cut once per bound weight and evaluate: a small GPU copy before the block is dispatched.
    if getattr(mlp, "_split_for", None) is not d.weight:
        mlp._split = [tuple(mx.contiguous(a) for a in sl) for sl in views]
        mx.eval(*[a for sl in mlp._split for a in sl])
        mlp._split_for = d.weight
    return mlp._split


def batched_down(mlp: Krea2SwiGLU, parts: int) -> tuple:
    # The K-split as one batched qmm: weight (N, K/4 packed) -> (parts, N, K/4/parts), cut and
    # evaluated once per bound weight. Zero cost if the checkpoint stores it this way.
    d = mlp.down
    if getattr(mlp, "_batched_for", None) is not d.weight:
        n, packed = d.weight.shape
        sk = d.scales.shape[1]
        mlp._batched = (
            mx.contiguous(d.weight.reshape(n, parts, packed // parts).transpose(1, 0, 2)),
            mx.contiguous(d.scales.reshape(n, parts, sk // parts).transpose(1, 0, 2)),
            mx.contiguous(d.biases.reshape(n, parts, sk // parts).transpose(1, 0, 2)),
        )
        mx.eval(*mlp._batched)
        mlp._batched_for = d.weight
    return mlp._batched


def rms_norm_bf16(self: Krea2RMSNorm, x: mx.array) -> mx.array:
    # No float32 round trip: the kernel accumulates in float32 anyway; (1 + scale) is rounded
    # to the activation dtype (at most 0.39% on this checkpoint, M9b section 5).
    return mx.fast.rms_norm(x, (self.scale.astype(mx.float32) + 1.0).astype(x.dtype), self.eps)


def make_swiglu(parts: int, precut: bool, sequential: bool, data: bool = False, batched: bool = False):
    def call(self: Krea2SwiGLU, x: mx.array) -> mx.array:
        g = self.gate(x)
        u = self.up(after(x, g, data) if sequential else x)
        h = nn.silu(g) * u
        if parts == 1:
            return self.down(h)
        d = self.down
        if batched:
            w, sc, bi = batched_down(self, parts)
            b, m, kk = h.shape
            xs = h.reshape(b * m, parts, kk // parts).transpose(1, 0, 2)
            y = mx.quantized_matmul(xs, w, sc, bi, transpose=True, group_size=d.group_size, bits=d.bits)
            return y.sum(axis=0).reshape(b, m, -1)
        k = h.shape[-1] // parts
        acc = None
        for i, (w, sc, bi) in enumerate(down_slices(self, parts, precut)):
            xs = h[..., i * k : (i + 1) * k]
            if sequential and acc is not None:
                xs = after(xs, acc, data)
            y = mx.quantized_matmul(xs, w, sc, bi, transpose=True, group_size=d.group_size, bits=d.bits)
            acc = y if acc is None else acc + y
        return acc

    return call


class Patched:
    # Context manager: `with Patched("4bn"): ...` applies a spec — digits for the number of
    # K-slices of mlp.down, "c" to pre-cut them, "b" to issue them as one batched qmm, "s" to
    # chain every independent matmul with mx.depends, "d" to chain them with a real data
    # dependency, "n" to run the RMSNorms without the float32 round trip.
    def __init__(self, spec: str):
        digits = "".join(ch for ch in spec if ch.isdigit()) or "1"
        self.parts = int(digits)
        self.precut = "c" in spec
        self.data = "d" in spec
        self.batched = "b" in spec
        self.norm_bf16 = "n" in spec
        self.sequential = "s" in spec or self.data

    def __enter__(self):
        self.saved = (Krea2Attention.__call__, Krea2SwiGLU.__call__, Krea2RMSNorm.__call__)
        if self.sequential:
            Krea2Attention.__call__ = make_attention(self.data)
        if self.parts > 1 or self.sequential:
            Krea2SwiGLU.__call__ = make_swiglu(self.parts, self.precut, self.sequential, self.data, self.batched)
        if self.norm_bf16:
            Krea2RMSNorm.__call__ = rms_norm_bf16
        return self

    def __exit__(self, *exc):
        Krea2Attention.__call__, Krea2SwiGLU.__call__, Krea2RMSNorm.__call__ = self.saved
