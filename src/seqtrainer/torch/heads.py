"""Standard PyTorch heads for regression and classification tasks."""

from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(slots=True)
class RegressionHeadConfig:
    input_dim: int
    hidden_dim: int | None = None
    dropout: float = 0.1


@dataclass(slots=True)
class ClassificationHeadConfig:
    input_dim: int
    num_classes: int
    hidden_dim: int | None = None
    dropout: float = 0.1


def _import_nn():
    try:
        return importlib.import_module("torch.nn")
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install seqtrainer[torch] to use torch heads") from exc


def build_regression_head(config: RegressionHeadConfig):
    """Build a standard MLP regression head."""
    nn = _import_nn()
    if config.hidden_dim is None:
        return nn.Linear(config.input_dim, 1)

    return nn.Sequential(
        nn.Linear(config.input_dim, config.hidden_dim),
        nn.GELU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_dim, 1),
    )


def build_classification_head(config: ClassificationHeadConfig):
    """Build a standard MLP classification head."""
    nn = _import_nn()
    if config.hidden_dim is None:
        return nn.Linear(config.input_dim, config.num_classes)

    return nn.Sequential(
        nn.Linear(config.input_dim, config.hidden_dim),
        nn.GELU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_dim, config.num_classes),
    )
