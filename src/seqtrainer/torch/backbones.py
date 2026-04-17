"""PyTorch backbone wrappers for sequence models (HuggingFace/DNABERT)."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any


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

    class HFBackbone(nn.Module):
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
