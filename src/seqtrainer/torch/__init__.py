"""PyTorch integration points for SeqTrainer."""

from .adapters import to_torch_dataset
from .finetune import build_finetune_config

__all__ = ["to_torch_dataset", "build_finetune_config", "ProkBERTClassifier"]


def __getattr__(name: str):
    """Load optional ProkBERT symbols without making torch extras mandatory."""
    if name == "ProkBERTClassifier":
        from .prokbert_benchmark import ProkBERTClassifier

        return ProkBERTClassifier
    raise AttributeError(name)
