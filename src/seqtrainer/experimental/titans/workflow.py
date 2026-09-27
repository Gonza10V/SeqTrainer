"""Deterministic CPU train/checkpoint/resume/evaluate workflow for Titans."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .model import PaperMAC, TitansConfig


@dataclass(frozen=True)
class CpuWorkflowResult:
    checkpoint: Path
    steps: int
    mean_loss: float


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install seqtrainer[torch] to use experimental Titans") from exc
    return torch


def _seed(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_cpu_workflow(
    tokens,
    *,
    config: TitansConfig,
    checkpoint: str | Path,
    steps: int,
    seed: int = 0,
    resume: bool = False,
) -> CpuWorkflowResult:
    """Train deterministically on CPU and save model, optimizer, and RNG state."""
    torch = _torch()
    if steps <= 0:
        raise ValueError("steps must be positive")
    tokens = torch.as_tensor(tokens, dtype=torch.long, device="cpu")
    if tokens.ndim != 2 or tokens.shape[1] < 2:
        raise ValueError("tokens must have shape [batch, sequence >= 2]")
    _seed(seed, torch)
    model = PaperMAC(config).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = Path(checkpoint)
    completed = 0
    if resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("format") != "experimental.titans.v1":
            raise ValueError("checkpoint is not an experimental.titans.v1 checkpoint")
        if saved.get("config") != asdict(config):
            raise ValueError("checkpoint config does not match requested config")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        completed = int(saved["steps"])
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        logits = model(tokens[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, config.vocab_size), tokens[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "experimental.titans.v1", "config": asdict(config), "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "steps": completed + steps, "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
    }, checkpoint)
    return CpuWorkflowResult(checkpoint=checkpoint, steps=completed + steps, mean_loss=float(np.mean(losses)))


def evaluate_checkpoint(tokens, *, config: TitansConfig, checkpoint: str | Path) -> float:
    """Return deterministic next-token loss from an experimental checkpoint."""
    torch = _torch()
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if saved.get("format") != "experimental.titans.v1" or saved.get("config") != asdict(config):
        raise ValueError("checkpoint is incompatible with this experimental Titans config")
    model = PaperMAC(config).to("cpu")
    model.load_state_dict(saved["model"])
    model.eval()
    values = torch.as_tensor(tokens, dtype=torch.long, device="cpu")
    with torch.no_grad():
        logits = model(values[:, :-1])
        return float(torch.nn.functional.cross_entropy(logits.reshape(-1, config.vocab_size), values[:, 1:].reshape(-1)))
