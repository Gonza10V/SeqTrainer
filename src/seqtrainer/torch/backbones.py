"""PyTorch backbone wrappers for sequence models (HuggingFace/DNABERT)."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class HostedModelPreset:
    """A pinned hosted-model reference that performs no I/O on construction."""

    name: str
    model_id: str
    revision: str
    tokenizer_id: str | None = None
    max_length: int = 512
    trust_remote_code: bool = False


# Revisions are immutable commit IDs, rather than moving hub tags.  Loading is
# deliberately separate so listing or selecting presets never downloads files.
HOSTED_MODEL_PRESETS: dict[str, HostedModelPreset] = {
    "nucleotide-transformer-v2": HostedModelPreset(
        name="nucleotide-transformer-v2",
        model_id="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        revision="e6a28f5a7c4b1f47e6027f3e6a9d83c5e4db4a89",
        max_length=1024,
        trust_remote_code=True,
    ),
    "hyenadna-tiny": HostedModelPreset(
        name="hyenadna-tiny",
        model_id="LongSafari/hyenadna-tiny-1k-seqlen-hf",
        revision="d3c3a0af2a6e4a2c8935e3bb7df67db4fddf7ab5",
        max_length=1024,
        trust_remote_code=True,
    ),
    "evo2-1b": HostedModelPreset(
        name="evo2-1b",
        model_id="arcinstitute/evo2_1b_base",
        revision="83e58b9e0dc0386ac2c29d4ad82d3714fc603761",
        max_length=8192,
        trust_remote_code=True,
    ),
    "gemma-3-4b": HostedModelPreset(
        name="gemma-3-4b",
        model_id="google/gemma-3-4b-pt",
        revision="c0f50f5d5b4ed72a7f6390a34e570d9c2f312ad4",
        max_length=2048,
    ),
}


def get_hosted_model_preset(name: str) -> HostedModelPreset:
    """Return a pinned preset without importing optional dependencies."""
    try:
        return HOSTED_MODEL_PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(HOSTED_MODEL_PRESETS))
        raise ValueError(f"Unknown hosted-model preset {name!r}. Available presets: {choices}") from exc


def build_hosted_model_preset(name: str, *, trainable: bool = True, pooling: str = "cls"):
    """Lazily construct a Hugging Face wrapper for a pinned preset.

    This is the explicit point at which the optional torch/transformers stack
    may access a local Hugging Face cache or, if configured by that stack, the
    network. Tests can replace ``build_hf_backbone`` with an offline fake.
    """
    preset = get_hosted_model_preset(name)
    return build_hf_backbone(
        HFBackboneConfig(
            model_name=preset.model_id,
            pooling=pooling,
            trainable=trainable,
            trust_remote_code=preset.trust_remote_code,
            model_kwargs={"revision": preset.revision},
        )
    )


@dataclass(slots=True)
class HFBackboneConfig:
    """Configuration for HuggingFace backbone wrappers."""

    model_name: str
    pooling: str = "cls"
    trainable: bool = True
    trust_remote_code: bool = False
    model_kwargs: dict[str, Any] = field(default_factory=dict)


def _import_torch_hf_stack():
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install seqtrainer[torch] to use torch backbone wrappers") from exc

    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install seqtrainer[torch] for HuggingFace backbone wrappers") from exc

    return torch, nn, transformers.AutoModel


def build_hf_backbone(config: HFBackboneConfig):
    """Build a HuggingFace sequence backbone with configurable pooling."""
    torch, nn, auto_model_cls = _import_torch_hf_stack()

    class HFBackbone(nn.Module):  # type: ignore[name-defined]
        def __init__(self):
            super().__init__()
            self.model = auto_model_cls.from_pretrained(
                config.model_name,
                trust_remote_code=config.trust_remote_code,
                **config.model_kwargs,
            )
            self.pooling = config.pooling
            if not config.trainable:
                for param in self.model.parameters():
                    param.requires_grad = False

        @property
        def output_dim(self) -> int:
            return int(self.model.config.hidden_size)

        def _pool(self, last_hidden_state, attention_mask=None):
            if self.pooling == "cls":
                return last_hidden_state[:, 0, :]
            if self.pooling == "mean":
                if attention_mask is None:
                    return last_hidden_state.mean(dim=1)
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
                summed = (last_hidden_state * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-8)
                return summed / denom
            if self.pooling == "max":
                return torch.max(last_hidden_state, dim=1).values
            raise ValueError(f"Unsupported pooling strategy: {self.pooling}")

        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            hidden = outputs.last_hidden_state
            pooled = self._pool(hidden, attention_mask=attention_mask)
            return pooled

    return HFBackbone()


def dnabert2_backbone(
    *,
    model_name: str = "zhihan1996/DNABERT-2-117M",
    pooling: str = "cls",
    trainable: bool = True,
):
    """Build the default DNABERT-2 backbone wrapper."""
    return build_hf_backbone(
        HFBackboneConfig(
            model_name=model_name,
            pooling=pooling,
            trainable=trainable,
        )
    )
