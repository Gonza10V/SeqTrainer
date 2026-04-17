"""Framework-neutral model registry for pluggable backbone/head definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BackboneSpec:
    name: str
    framework: str
    factory_path: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HeadSpec:
    name: str
    task_type: str
    framework: str
    factory_path: str
    config: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Registry for framework-neutral backbone/head specs."""

    def __init__(self) -> None:
        self._backbones: dict[str, BackboneSpec] = {}
        self._heads: dict[str, HeadSpec] = {}

    def register_backbone(self, spec: BackboneSpec) -> None:
        self._backbones[spec.name] = spec

    def register_head(self, spec: HeadSpec) -> None:
        self._heads[spec.name] = spec

    def get_backbone(self, name: str) -> BackboneSpec:
        return self._backbones[name]

    def get_head(self, name: str) -> HeadSpec:
        return self._heads[name]

    def list_backbones(self) -> list[str]:
        return sorted(self._backbones.keys())

    def list_heads(self) -> list[str]:
        return sorted(self._heads.keys())


def default_registry() -> ModelRegistry:
    """Return a registry preloaded with initial torch specs."""
    registry = ModelRegistry()

    registry.register_backbone(
        BackboneSpec(
            name="dnabert2",
            framework="torch",
            factory_path="seqtrainer.torch.backbones.dnabert2_backbone",
            config={"model_name": "zhihan1996/DNABERT-2-117M", "pooling": "cls"},
        )
    )
    registry.register_backbone(
        BackboneSpec(
            name="hf-sequence",
            framework="torch",
            factory_path="seqtrainer.torch.backbones.build_hf_backbone",
            config={"pooling": "cls"},
        )
    )

    registry.register_head(
        HeadSpec(
            name="regression-mlp",
            task_type="regression",
            framework="torch",
            factory_path="seqtrainer.torch.heads.build_regression_head",
            config={"hidden_dim": 256, "dropout": 0.1},
        )
    )
    registry.register_head(
        HeadSpec(
            name="classification-mlp",
            task_type="classification",
            framework="torch",
            factory_path="seqtrainer.torch.heads.build_classification_head",
            config={"hidden_dim": 256, "dropout": 0.1},
        )
    )
    return registry
