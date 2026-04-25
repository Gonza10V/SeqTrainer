"""Deprecated module for prototype GNN experiments.

Prototype graph code has moved under ``seqtrainer.graph`` and framework-specific
adapters under ``seqtrainer.torch``.
"""

from __future__ import annotations

from seqtrainer._deprecation import warn_deprecated

warn_deprecated(old="seqtrainer.gnn", new="seqtrainer.graph + seqtrainer.torch", kind="module", stacklevel=2)

__all__ = []
