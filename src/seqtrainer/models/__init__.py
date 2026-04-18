"""Framework-neutral model abstractions."""

from .registry import BackboneSpec, HeadSpec, ModelRegistry, default_registry

__all__ = ["BackboneSpec", "HeadSpec", "ModelRegistry", "default_registry"]
