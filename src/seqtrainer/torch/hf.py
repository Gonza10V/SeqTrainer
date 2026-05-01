"""Hugging Face helpers for sequence model fine-tuning workflows."""

from __future__ import annotations

from dataclasses import dataclass

from seqtrainer.models import BackboneSpec, ModelRegistry


@dataclass(frozen=True)
class NucleotideTransformerV2Spec:
    """Preset metadata for InstaDeep Nucleotide Transformer v2 backbone."""

    name: str = "nucleotide-transformer-v2"
    model_name: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    tokenizer_name: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    max_length: int = 256


def register_default_torch_backbones(registry: ModelRegistry) -> ModelRegistry:
    """Register built-in torch backbones used by SeqTrainer notebooks."""
    ntv2 = NucleotideTransformerV2Spec()
    registry.register_backbone(
        BackboneSpec(
            name=ntv2.name,
            framework="torch",
            config={
                "model_name": ntv2.model_name,
                "tokenizer_name": ntv2.tokenizer_name,
                "max_length": ntv2.max_length,
            },
        )
    )
    return registry


def get_nucleotide_transformer_v2_backbone() -> BackboneSpec:
    """Return the pre-registered Nucleotide Transformer v2 backbone spec."""
    registry = register_default_torch_backbones(ModelRegistry())
    return registry.get_backbone("nucleotide-transformer-v2")
