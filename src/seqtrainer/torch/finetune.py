"""Fine-tuning helpers for PyTorch workflows."""

from __future__ import annotations


def default_loss_for_task(task: str) -> str:
    """Return default objective name for a given task."""
    if task == "regression":
        return "mse"
    if task == "classification":
        return "cross_entropy"
    raise ValueError(f"Unsupported task: {task}")


def build_finetune_config(
    *,
    backbone: str,
    head: str,
    task: str = "regression",
    learning_rate: float = 1e-4,
    epochs: int = 5,
    weight_decay: float = 0.0,
) -> dict:
    """Return a fine-tuning configuration dictionary."""
    return {
        "framework": "torch",
        "backbone": backbone,
        "head": head,
        "task": task,
        "loss": default_loss_for_task(task),
        "learning_rate": learning_rate,
        "epochs": epochs,
        "weight_decay": weight_decay,
    }
