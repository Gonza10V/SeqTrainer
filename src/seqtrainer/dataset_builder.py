"""Backward-compatible SBOL dataset builder wrappers.

Deprecated: import from ``seqtrainer.data.sbol`` instead.
"""

from __future__ import annotations

from seqtrainer._deprecation import warn_deprecated
from seqtrainer.data.sbol import build_dataset_from_files as build_dataset
from seqtrainer.data.sbol import get_sequence_from_sbol, get_y_label


def _warn() -> None:
    warn_deprecated(old="seqtrainer.dataset_builder", new="seqtrainer.data.sbol", kind="module", stacklevel=3)


def get_sequences_from_sbol(file_paths):
    _warn()
    return [get_sequence_from_sbol(path) for path in file_paths]


def get_y_labels_from_sbol(file_paths, uri):
    _warn()
    return [get_y_label(path, uri) for path in file_paths]


def find_possible_y_uris(file_path):
    """Placeholder for compatibility.

    TODO: add richer predicate/type inspection in ``seqtrainer.sparql.recipes``.
    """
    _warn()
    return []


__all__ = ["build_dataset", "get_sequence_from_sbol", "get_y_label", "get_sequences_from_sbol", "get_y_labels_from_sbol", "find_possible_y_uris"]
