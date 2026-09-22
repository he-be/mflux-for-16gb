import json

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.config import ModelConfig
from mflux.models.krea2.krea2_initializer import Krea2Initializer
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
    def zero_blocks(transformer: Krea2Transformer) -> None:
        for block in transformer.blocks:
            flat = tree_flatten(block.parameters())
            block.update(tree_unflatten([(k, mx.zeros_like(v)) for k, v in flat]))
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
    assert mx.array_equal(streamed(hidden, timestep, context), expected)
    assert stream.summary()["block_calls"] == len(resident.blocks)


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


def test_block_streaming_refuses_a_runtime_lora_before_touching_the_disk():
    model = type("Stub", (), {})()

    with pytest.raises(ValueError, match="bake_lora_checkpoint"):
        Krea2Initializer.init(
            model=model,
            model_config=ModelConfig.krea2(),
            quantize=None,
            model_path="/nonexistent",
            lora_paths=["some-lora.safetensors"],
            block_streaming=True,
        )
