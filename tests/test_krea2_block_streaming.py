import json

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.krea2.krea2_initializer import Krea2Initializer
from mflux.models.krea2.model.krea2_transformer.common import Krea2RMSNorm
from mflux.models.krea2.model.krea2_transformer.feed_forward import Krea2SwiGLU
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.weights.krea2_weight_stream import Krea2BlockStream, Krea2StreamedBlock

pytestmark = pytest.mark.fast


class _StreamFixture:
    # head_dim = features // heads = 16 keeps the RoPE axis split valid; tiny everywhere else.
    @staticmethod
    def transformer(layers: int = 2) -> Krea2Transformer:
        return Krea2Transformer(
            features=32,
            tdim=16,
            txtdim=16,
            heads=2,
            kvheads=1,
            multiplier=2,
            layers=layers,
            patch=2,
            channels=16,
            theta=1000,
            txtlayers=12,
            txtheads=2,
            txtkvheads=1,
        )

    @staticmethod
    def write_checkpoint(transformer: Krea2Transformer, path) -> None:
        # One file per block plus an index naming them, the layout
        # tools/bench/bake_lora_checkpoint.py writes.
        path.mkdir(parents=True, exist_ok=True)
        weight_map = {}
        for index, block in enumerate(transformer.blocks):
            flat = {f"blocks.{index}.{k}": v for k, v in tree_flatten(block.parameters())}
            name = f"blocks_{index:02d}.safetensors"
            mx.save_safetensors(str(path / name), flat)
            weight_map.update({k: name for k in flat})
        (path / Krea2BlockStream.INDEX_FILE).write_text(json.dumps({"weight_map": weight_map}))

    @staticmethod
    def zero_blocks(transformer: Krea2Transformer, keep: tuple = ()) -> None:
        for block in transformer.blocks:
            flat = tree_flatten(block.parameters())
            kept = [(k, v if any(t in k for t in keep) else mx.zeros_like(v)) for k, v in flat]
            block.update(tree_unflatten(kept))
        mx.eval(transformer.parameters())

    @staticmethod
    def wrap_with_lora(transformer: Krea2Transformer, r: int = 4, scale: float = 0.5) -> None:
        # Stands in for a runtime adapter without needing a file on disk. The wrappers are
        # what the streamed bind has to see through: the checkpoint still names attn.wq.weight
        # while the live block keeps that tensor at attn.wq.linear.weight.
        for block in transformer.blocks:
            for holder, name in ((block.attn, "wq"), (block.mlp, "down")):
                setattr(holder, name, LoRALinear.from_linear(getattr(holder, name), r=r, scale=scale))
        mx.eval(transformer.parameters())

    @staticmethod
    def inputs(transformer: Krea2Transformer) -> tuple[mx.array, mx.array, mx.array]:
        hidden = mx.random.normal((1, 16, 8, 8), key=mx.random.key(0))
        timestep = mx.array([1.0])
        context = mx.random.normal((1, 4, transformer.txtlayers * transformer.txtdim), key=mx.random.key(1))
        return hidden, timestep, context


def test_streamed_forward_matches_the_resident_one_bit_for_bit(tmp_path):
    resident = _StreamFixture.transformer()
    mx.eval(resident.parameters())
    _StreamFixture.write_checkpoint(resident, tmp_path / "transformer")
    hidden, timestep, context = _StreamFixture.inputs(resident)
    expected = resident(hidden, timestep, context)

    streamed = _StreamFixture.transformer()
    streamed.update(resident.parameters())
    # Zeroed so a block that never binds, or binds the wrong file, cannot pass.
    _StreamFixture.zero_blocks(streamed)
    stream = Krea2BlockStream(Krea2BlockStream.locate(tmp_path))
    stream.attach(streamed)

    assert all(isinstance(block, Krea2StreamedBlock) for block in streamed.blocks)
    # Two passes: the second one binds block 0 from the buffers the last block of the first
    # pass read into, and every block from the alternate buffer set.
    assert mx.array_equal(streamed(hidden, timestep, context), expected)
    assert mx.array_equal(streamed(hidden, timestep, context), expected)
    assert stream.summary()["block_calls"] == 2 * len(resident.blocks)
    assert stream.summary()["direct"]


def test_attach_hands_the_streaming_speedups_to_every_block(tmp_path):
    transformer = _StreamFixture.transformer()
    mx.eval(transformer.parameters())
    _StreamFixture.write_checkpoint(transformer, tmp_path / "transformer")
    stream = Krea2BlockStream(Krea2BlockStream.locate(tmp_path))

    stream.attach(transformer, down_splits=4, native_norm=True)

    for wrapper in transformer.blocks:
        assert wrapper.block.mlp.down_splits == 4
        norms = [m for m in wrapper.block.modules() if isinstance(m, Krea2RMSNorm)]
        assert len(norms) == 4 and all(n.native_dtype for n in norms)


def test_down_projection_in_slices_matches_the_whole_one():
    mlp = Krea2SwiGLU(features=64, multiplier=2)
    mx.eval(mlp.parameters())
    nn.quantize(mlp, group_size=64, bits=8)
    x = mx.random.normal((1, 8, 64), key=mx.random.key(0))
    whole = mlp(x)
    mlp.down_splits = 2

    sliced = mlp(x)

    assert sliced.shape == whole.shape
    assert mx.abs(sliced - whole).max().item() < 1e-3 * mx.abs(whole).max().item()
    mlp.release_down_planes()
    assert mlp._down_planes is None


def test_native_dtype_norm_stays_within_bf16_rounding_of_the_float32_one():
    norm = Krea2RMSNorm(32)
    norm.scale = mx.random.normal((32,), key=mx.random.key(1)) * 0.5
    x = mx.random.normal((4, 32), key=mx.random.key(2)).astype(mx.bfloat16)
    reference = norm(x).astype(mx.float32)
    norm.native_dtype = True

    out = norm(x).astype(mx.float32)

    assert out.dtype == mx.float32 and norm(x).dtype == mx.bfloat16
    relative = mx.abs(out - reference) / mx.maximum(mx.abs(reference), 1e-2)
    assert relative.max().item() < 2**-6


def test_locate_prefers_the_transformer_subdir_over_the_root(tmp_path):
    transformer = _StreamFixture.transformer(layers=1)
    mx.eval(transformer.parameters())
    _StreamFixture.write_checkpoint(transformer, tmp_path / "transformer")
    (tmp_path / Krea2BlockStream.INDEX_FILE).write_text(json.dumps({"weight_map": {"first.weight": "x.safetensors"}}))

    assert Krea2BlockStream.locate(tmp_path) == tmp_path / "transformer"


def test_locate_rejects_a_checkpoint_whose_index_names_no_blocks(tmp_path):
    (tmp_path / Krea2BlockStream.INDEX_FILE).write_text(json.dumps({"weight_map": {"first.weight": "x.safetensors"}}))

    with pytest.raises(ValueError, match="naming blocks"):
        Krea2BlockStream.locate(tmp_path)


def test_attach_rejects_a_checkpoint_with_the_wrong_block_count(tmp_path):
    written = _StreamFixture.transformer(layers=1)
    mx.eval(written.parameters())
    _StreamFixture.write_checkpoint(written, tmp_path / "transformer")
    stream = Krea2BlockStream(Krea2BlockStream.locate(tmp_path))

    with pytest.raises(ValueError, match="holds 1 blocks"):
        stream.attach(_StreamFixture.transformer(layers=2))


def test_streamed_forward_matches_the_resident_one_through_a_lora(tmp_path):
    resident = _StreamFixture.transformer()
    mx.eval(resident.parameters())
    # Written before the adapters go on: a checkpoint never holds them.
    _StreamFixture.write_checkpoint(resident, tmp_path / "transformer")
    streamed = _StreamFixture.transformer()
    _StreamFixture.wrap_with_lora(resident)
    _StreamFixture.wrap_with_lora(streamed)
    streamed.update(resident.parameters())
    mx.eval(streamed.parameters())
    hidden, timestep, context = _StreamFixture.inputs(resident)
    expected = resident(hidden, timestep, context)

    # Only the tensors the checkpoint supplies are zeroed, so a bind that misses the adapter's
    # position fails and an adapter that never fires fails too.
    _StreamFixture.zero_blocks(streamed, keep=("lora_A", "lora_B"))
    stream = Krea2BlockStream(Krea2BlockStream.locate(tmp_path))
    stream.attach(streamed)

    assert stream.wrapped == {"attn.wq": "linear", "mlp.down": "linear"}
    assert mx.array_equal(streamed(hidden, timestep, context), expected)
    assert mx.array_equal(streamed(hidden, timestep, context), expected)


def test_down_projection_in_slices_matches_the_whole_one_through_a_lora():
    mlp = Krea2SwiGLU(features=64, multiplier=2)
    mx.eval(mlp.parameters())
    nn.quantize(mlp, group_size=64, bits=8)
    mlp.down = LoRALinear.from_linear(mlp.down, r=4, scale=0.5)
    mx.eval(mlp.parameters())
    x = mx.random.normal((1, 8, 64), key=mx.random.key(0))
    whole = mlp(x)

    mlp.down_splits = 2
    mlp._down_adapter = Krea2BlockStream._down_adapter(mlp.down)

    sliced = mlp(x)
    assert mlp._down_adapter is not None
    assert mx.abs(sliced - whole).max().item() < 1e-3 * mx.abs(whole).max().item()


def test_block_streaming_hands_the_lora_to_the_staged_loader(tmp_path):
    model = type("Stub", (), {})()

    Krea2Initializer._init_staged(model, str(tmp_path), ModelConfig.krea2(), None, ["a.safetensors"], [0.7])

    assert model.staged_loader.lora_paths == ["a.safetensors"]
    assert model.staged_loader.lora_scales == [0.7]
    assert (model.lora_paths, model.lora_scales) == (["a.safetensors"], [0.7])
