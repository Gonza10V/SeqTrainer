"""Exact backend boundary for experimental Titans memory."""

from __future__ import annotations

from typing import Protocol


class MemoryBackend(Protocol):
    """A causal memory update backend.

    Backends receive and return tensors shaped ``[batch, hidden]``.  The first
    release exposes only the reference path; approximate/accelerated backends
    belong in opt-in evidence tooling, not in this contract.
    """

    def update(self, state, values): ...


class ExactMemoryBackend:
    """Reference exponential-memory update with portable torch operations."""

    def __init__(self, momentum: float = 0.9) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.momentum = float(momentum)

    def update(self, state, values):
        return state * self.momentum + values * (1.0 - self.momentum)
