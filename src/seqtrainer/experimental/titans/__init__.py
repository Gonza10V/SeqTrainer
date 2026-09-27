"""Experimental, paper-oriented Titans Memory-as-Context reference code.

This namespace is intentionally separate from the stable SeqTrainer API.
Checkpoints written by this package use the ``experimental.titans.v1`` format
and are not promised to be compatible with historical Titans implementations.
"""

from .backends import ExactMemoryBackend, MemoryBackend
from .model import MemoryState, PaperMAC, TitansConfig
from .workflow import CpuWorkflowResult, evaluate_checkpoint, train_cpu_workflow

__all__ = [
    "CpuWorkflowResult",
    "ExactMemoryBackend",
    "MemoryBackend",
    "MemoryState",
    "PaperMAC",
    "TitansConfig",
    "evaluate_checkpoint",
    "train_cpu_workflow",
]
