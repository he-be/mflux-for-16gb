import gc
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.weights.krea2_lora_mapping import Krea2LoRAMapping
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
from mflux.models.krea2.weights.krea2_weight_stream import Krea2BlockStream
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE

# Builds one Krea 2 component at a time so the pipeline never holds two of them.
# The three together are 22.2 GB of q8 weights, which does not fit in 18 GB; taken one at a
# time with the DiT streamed no stage exceeds 5 GB. The caller decides when to let go — see
# Krea2._component — because a component can only be released after the arrays that depend
# on it have been evaluated.


class Krea2StagedLoader:
    # Two block-level speedups that change the numbers slightly (accumulation order, one bf16
    # rounding of the norm weights), so they ride only with block streaming, whose images are
    # judged against each other and not against the resident references. M6 mini, 1024^2:
    # 9.0-9.5 -> 8.5-8.6 s a step. See docs/16gb/measurements/2026-09-22-m9c-prefetch-interference.md.
    DOWN_SPLITS = 4
    NATIVE_NORM = True

    def __init__(
        self,
        model_path: str,
        model_config: ModelConfig,
        quantize: int | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        root = PathResolution.resolve(
            path=model_path,
            patterns=Krea2WeightDefinition.get_download_patterns(model_config.model_name),
        )
        if root is None:
            raise ValueError("Block streaming needs a snapshot on disk to stream from: name one with --model.")
        self.root: Path = root
        self.model_config = model_config
        self.quantize = quantize
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales
        self.bits: int | None = None
        self.stream: Krea2BlockStream | None = None

    def build(self, name: str) -> nn.Module:
        if name == "text_encoder":
            return self._build(Krea2TextEncoder(), Krea2StagedLoader._text_encoder_component())
        if name == "vae":
            return self._build(QwenVAE(), self._component("vae"))
        if name == "transformer":
            return self._build_transformer()
        raise ValueError(f"Unknown Krea 2 component: {name!r}")

    @staticmethod
    def release() -> None:
        gc.collect()
        mx.clear_cache()

    def _build_transformer(self) -> Krea2Transformer:
        # Locate the blocks before loading anything, so a checkpoint that cannot be streamed
        # fails before 13.62 GB of handles are wired up.
        stream = Krea2BlockStream(Krea2BlockStream.locate(self.root))
        transformer = Krea2Transformer(**(self.model_config.transformer_overrides or {}))
        self._apply(transformer, self._component("transformer"), materialize=False)
        # Never baked: a streamed block's weights arrive from disk on every step, so there is
        # nothing stable to fold into. The adapters ride as a side path instead, and their
        # factors are the one part of a block that stays resident between steps.
        self.lora_paths, self.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=Krea2LoRAMapping.get_mapping(),
            transformer=transformer,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            bake_lora=False,
        )
        stream.attach(transformer, down_splits=self.DOWN_SPLITS, native_norm=self.NATIVE_NORM)
        self.stream = stream
        return transformer

    def _build(self, module: nn.Module, component: ComponentDefinition) -> nn.Module:
        self._apply(module, component, materialize=True)
        return module

    def _apply(self, module: nn.Module, component: ComponentDefinition, materialize: bool) -> None:
        weights = WeightLoader.load_single_local(component=component, root_path=self.root)
        bits = WeightApplier.apply_and_quantize_single(
            weights=weights,
            model=module,
            component=component,
            quantize_arg=self.quantize,
            quantization_predicate=Krea2WeightDefinition.quantization_predicate,
        )
        del weights
        if bits is not None:
            self.bits = bits
        if materialize:
            # mx.load is lazy: without the eval the read above is only a mmap, and the real
            # cost lands in the middle of the next stage instead.
            mx.eval(module)
            mx.clear_cache()

    @staticmethod
    def _component(name: str) -> ComponentDefinition:
        # Passed through untouched: WeightLoader resolves hf_subdir and, for the transformer,
        # runs the variant selector that tells a single-file checkpoint from a sharded one.
        return next(c for c in Krea2WeightDefinition.get_components() if c.name == name)

    @staticmethod
    def _text_encoder_component() -> ComponentDefinition:
        # The definition sets skip_quantization=True, which for a pre-quantized encoder would
        # leave the module unquantized and then update it with packed q8 tensors. Clearing the
        # flag lets WeightApplier rebuild the quantized structure from the stored scales.
        # Harmless on a bf16 encoder: with no scales present the predicate matches nothing.
        return replace(Krea2StagedLoader._component("text_encoder"), skip_quantization=False)
