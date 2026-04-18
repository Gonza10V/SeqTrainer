"""Framework-neutral tensorization/tokenization for sequence datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from seqtrainer.data.materialized import MaterializedDataset
from seqtrainer.transforms.dna import normalize_sequence, one_hot_encode, pad_or_trim

DEFAULT_VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}


@dataclass(slots=True)
class SequenceTensorizationConfig:
    """Configuration for converting sequence examples into numeric arrays."""

    sequence_field: str = "sequence"
    label_field: str | None = "target"
    encoding: str = "one_hot"  # one_hot | token_ids
    max_length: int | None = None
    pad_char: str = "N"
    vocab: dict[str, int] | None = None
    label_dtype: str = "float32"


def _resolve_length(sequences: list[str], max_length: int | None) -> int:
    if max_length is not None:
        return max_length
    if not sequences:
        return 0
    return max(len(seq) for seq in sequences)


def _normalize_and_pad(sequences: list[str], *, max_length: int | None, pad_char: str) -> list[str]:
    length = _resolve_length(sequences, max_length)
    return [pad_or_trim(normalize_sequence(seq), length, pad_char=pad_char) for seq in sequences]


def _tokenize_nucleotides(sequences: list[str], *, vocab: dict[str, int], max_length: int | None, pad_char: str) -> tuple[np.ndarray, np.ndarray]:
    normalized = [normalize_sequence(seq) for seq in sequences]
    length = _resolve_length(normalized, max_length)
    processed = [pad_or_trim(seq, length, pad_char=pad_char) for seq in normalized]

    pad_id = 0
    token_ids = np.full((len(processed), length), pad_id, dtype=np.int32)
    attention_mask = np.zeros_like(token_ids, dtype=np.int32)

    for i, seq in enumerate(processed):
        content_length = min(len(normalized[i]), length)
        for j, token in enumerate(seq):
            token_ids[i, j] = vocab.get(token, vocab.get("N", 0))
            attention_mask[i, j] = 1 if j < content_length else 0

    return token_ids, attention_mask


def tensorize_materialized_dataset(
    dataset: MaterializedDataset,
    config: SequenceTensorizationConfig | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Convert `MaterializedDataset` into framework-neutral numpy tensors."""
    cfg = config or SequenceTensorizationConfig()
    rows = dataset.examples
    sequences = [str(row.get(cfg.sequence_field, "")) for row in rows]

    features: dict[str, np.ndarray]
    if cfg.encoding == "one_hot":
        processed = _normalize_and_pad(sequences, max_length=cfg.max_length, pad_char=cfg.pad_char)
        features = {"sequence": one_hot_encode(processed)}
    elif cfg.encoding == "token_ids":
        vocab = cfg.vocab or DEFAULT_VOCAB
        input_ids, attention_mask = _tokenize_nucleotides(
            sequences,
            vocab=vocab,
            max_length=cfg.max_length,
            pad_char=cfg.pad_char,
        )
        features = {"input_ids": input_ids, "attention_mask": attention_mask}
    else:
        raise ValueError(f"Unsupported encoding: {cfg.encoding}")

    labels = None
    if cfg.label_field is not None:
        label_values: list[Any] = [row.get(cfg.label_field) for row in rows]
        if any(value is not None for value in label_values):
            labels = np.asarray(label_values, dtype=cfg.label_dtype)

    return features, labels
