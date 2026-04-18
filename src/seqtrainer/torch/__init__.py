"""PyTorch integration points for SeqTrainer."""

from .adapters import TorchAdapterConfig, to_torch_dataloader, to_torch_dataset
from .backbones import HFBackboneConfig, build_hf_backbone, dnabert2_backbone
from .factories import TorchSequenceModelConfig, build_sequence_model
from .finetune import build_finetune_config, default_loss_for_task
from .heads import (
    ClassificationHeadConfig,
    RegressionHeadConfig,
    build_classification_head,
    build_regression_head,
)

__all__ = [
    "TorchAdapterConfig",
    "to_torch_dataset",
    "to_torch_dataloader",
    "HFBackboneConfig",
    "build_hf_backbone",
    "dnabert2_backbone",
    "RegressionHeadConfig",
    "ClassificationHeadConfig",
    "build_regression_head",
    "build_classification_head",
    "TorchSequenceModelConfig",
    "build_sequence_model",
    "default_loss_for_task",
    "build_finetune_config",
]
