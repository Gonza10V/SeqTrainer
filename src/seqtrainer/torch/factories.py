"""Factories that compose torch backbones with standard task heads."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from .backbones import HFBackboneConfig, build_hf_backbone
from .heads import (
    ClassificationHeadConfig,
    RegressionHeadConfig,
    build_classification_head,
    build_regression_head,
)


@dataclass(slots=True)
class TorchSequenceModelConfig:
    """Configuration for backbone+head torch sequence models."""

    backbone_model_name: str
    task: str = "regression"  # regression | classification
    num_classes: int = 2
    pooling: str = "cls"
    trainable_backbone: bool = True
    head_hidden_dim: int | None = None
    head_dropout: float = 0.1


def _import_torch():
    try:
        return importlib.import_module("torch.nn")
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install seqtrainer[torch] to use torch model factories") from exc


def build_sequence_model(config: TorchSequenceModelConfig):
    """Compose HF/DNABERT backbone + regression/classification head."""
    nn = _import_torch()

    backbone = build_hf_backbone(
        HFBackboneConfig(
            model_name=config.backbone_model_name,
            pooling=config.pooling,
            trainable=config.trainable_backbone,
        )
    )
    output_dim = int(backbone.output_dim)

    if config.task == "regression":
        head = build_regression_head(
            RegressionHeadConfig(
                input_dim=output_dim,
                hidden_dim=config.head_hidden_dim,
                dropout=config.head_dropout,
            )
        )
    elif config.task == "classification":
        head = build_classification_head(
            ClassificationHeadConfig(
                input_dim=output_dim,
                num_classes=config.num_classes,
                hidden_dim=config.head_hidden_dim,
                dropout=config.head_dropout,
            )
        )
    else:
        raise ValueError(f"Unsupported task: {config.task}")

    class SequenceModel(nn.Module):
        def __init__(self, backbone_module, head_module):
            super().__init__()
            self.backbone = backbone_module
            self.head = head_module

        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            features = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            return self.head(features)

    return SequenceModel(backbone, head)
