import math

import mlx.core as mx
from mlx import nn


class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(
        linear: nn.Linear | nn.QuantizedLinear,
        r: int = 16,
        scale: float = 1.0,
    ):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            scale=scale,
        )
        lora_lin.linear = linear
        return lora_lin

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        scale: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.scale = scale
        scale = 1 / math.sqrt(input_dims)

        self.lora_A = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, r),
        )
        self.lora_B = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(r, output_dims),
        )

    def __call__(self, x):
        # The scalar rides the rank-r intermediate, not the full-size product: scaling after
        # lora_B walks an output as wide as the base layer's, which on Krea 2's 16384-wide
        # mlp is 135 MB a layer. Same value, and measured at -34.5 ms on a 275 ms block
        # (docs/16gb/measurements/2026-09-22-m10b-lora-budget.md).
        return self.linear(x) + mx.matmul(self.scale * mx.matmul(x, self.lora_A), self.lora_B)
