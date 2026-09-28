"""Small causal Memory-as-Context reference model.

The implementation deliberately favors readable, deterministic semantics over
accelerated kernels. It is experimental research code, not a stable model API.
"""

from __future__ import annotations

from dataclasses import dataclass

from .backends import ExactMemoryBackend, MemoryBackend


@dataclass(frozen=True)
class TitansConfig:
    vocab_size: int
    hidden_size: int = 64
    memory_momentum: float = 0.9

    def __post_init__(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")


@dataclass
class MemoryState:
    value: object

    def detach(self) -> "MemoryState":
        return MemoryState(self.value.detach())


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise ImportError("Install seqtrainer[torch] to use experimental Titans") from exc
    return torch, nn


class PaperMAC:  # constructed as a torch module dynamically to keep base imports light
    """Causal token model with an explicit recurrent memory state."""

    def __new__(cls, config: TitansConfig, *, backend: MemoryBackend | None = None):
        torch, nn = _torch()

        class _PaperMAC(nn.Module):
            checkpoint_format = "experimental.titans.v1"

            def __init__(self) -> None:
                super().__init__()
                self.config = config
                self.backend = backend or ExactMemoryBackend(config.memory_momentum)
                self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
                self.input_norm = nn.LayerNorm(config.hidden_size)
                self.output = nn.Linear(config.hidden_size, config.vocab_size)

            def initial_state(self, batch_size: int, *, device=None, dtype=None) -> MemoryState:
                if batch_size <= 0:
                    raise ValueError("batch_size must be positive")
                return MemoryState(torch.zeros(batch_size, config.hidden_size, device=device, dtype=dtype))

            def forward(self, tokens, *, state: MemoryState | None = None, return_state: bool = False):
                if tokens.ndim != 2:
                    raise ValueError("tokens must have shape [batch, sequence]")
                embedded = self.embedding(tokens)
                memory = (
                    self.initial_state(tokens.shape[0], device=embedded.device, dtype=embedded.dtype).value
                    if state is None
                    else state.value
                )
                if memory.shape != (tokens.shape[0], config.hidden_size):
                    raise ValueError("state shape does not match batch size and hidden_size")
                outputs = []
                # Updating after emitting each position ensures strict causality:
                # position t cannot consume token t as context.
                for token_embedding in embedded.unbind(dim=1):
                    outputs.append(self.input_norm(token_embedding + memory))
                    memory = self.backend.update(memory, token_embedding)
                logits = self.output(torch.stack(outputs, dim=1))
                final_state = MemoryState(memory)
                return (logits, final_state) if return_state else logits

        instance = _PaperMAC()
        instance.__class__.__name__ = cls.__name__
        return instance
