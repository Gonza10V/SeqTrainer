"""Shared split loading and summary helpers for benchmark runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import BenchmarkConfig

REQUIRED_SPLITS = ("train", "validation", "test")


def resolve_split_paths(config: BenchmarkConfig, base_dir: str | Path | None = None) -> dict[str, Path]:
    """Resolve configured split paths for predefined split benchmarks."""
    if config.split.strategy != "predefined":
        raise ValueError("resolve_split_paths only supports split.strategy='predefined'")

    root = Path(base_dir) if base_dir is not None else Path.cwd()
    paths = {}
    for split in REQUIRED_SPLITS:
        raw_path = config.dataset.split_files[split]
        path = Path(raw_path)
        paths[split] = path if path.is_absolute() else root / path
    return paths


def load_predefined_split_frames(
    config: BenchmarkConfig,
    base_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load train/validation/test CSV split files with basic schema checks."""
    paths = resolve_split_paths(config, base_dir=base_dir)
    frames: dict[str, pd.DataFrame] = {}
    required_columns = {config.dataset.sequence_field, config.dataset.label_field}
    if config.dataset.id_field:
        required_columns.add(config.dataset.id_field)

    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {split} split file: {path}")
        frame = pd.read_csv(path)
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"{split} split is missing required columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{split} split must contain at least one row")
        nullable_columns = [
            column
            for column in (config.dataset.sequence_field, config.dataset.label_field)
            if frame[column].isna().any()
        ]
        if nullable_columns:
            raise ValueError(
                f"{split} split contains null values in required columns: {nullable_columns}"
            )
        frames[split] = frame

    _reject_cross_split_duplicates(config, frames)
    return frames


def _reject_cross_split_duplicates(
    config: BenchmarkConfig,
    frames: dict[str, pd.DataFrame],
) -> None:
    """Reject normalized sequences shared by different predefined splits."""
    sequence_field = config.dataset.sequence_field
    seen: dict[str, tuple[str, Any]] = {}
    duplicates: list[tuple[str, str, str]] = []
    conflicting_labels: list[dict[str, Any]] = []
    label_field = config.dataset.label_field
    for split, frame in frames.items():
        normalized = (
            frame[sequence_field]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace("U", "T", regex=False)
            .str.replace(r"\s+", "", regex=True)
        )
        for sequence, label in zip(normalized, frame[label_field]):
            previous = seen.get(sequence)
            if previous is None:
                seen[sequence] = (split, label)
                continue
            previous_split, previous_label = previous
            if previous_label != label:
                conflicting_labels.append(
                    {
                        "sequence": sequence,
                        "splits": [previous_split, split],
                        "labels": [previous_label, label],
                    }
                )
            elif previous_split != split:
                duplicates.append((sequence, previous_split, split))

    if conflicting_labels:
        raise ValueError(
            "Predefined splits contain conflicting labels for the same normalized sequence; "
            f"examples: {conflicting_labels[:3]}"
        )
    if duplicates:
        examples = [
            {"sequence": sequence, "splits": [left, right]}
            for sequence, left, right in duplicates[:3]
        ]
        raise ValueError(
            "Predefined splits contain duplicate normalized sequences across splits; "
            f"examples: {examples}"
        )


def _split_content_sha256(config: BenchmarkConfig, frame: pd.DataFrame) -> str:
    """Hash the ordered benchmark rows used for split comparison."""
    columns = [config.dataset.sequence_field, config.dataset.label_field]
    if config.dataset.id_field and config.dataset.id_field in frame:
        columns.append(config.dataset.id_field)
    rows = [
        ["" if pd.isna(value) else str(value) for value in row]
        for row in frame[columns].itertuples(index=False, name=None)
    ]
    payload = json.dumps(
        {"columns": columns, "rows": rows},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize_split_frames(config: BenchmarkConfig, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Summarize split sizes, class balance, and source files for manifests."""
    summary: dict[str, Any] = {}
    for split in REQUIRED_SPLITS:
        frame = frames[split]
        labels = frame[config.dataset.label_field]
        class_counts = labels.value_counts().sort_index().to_dict()
        split_summary: dict[str, Any] = {
            "rows": int(len(frame)),
            "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        }
        if config.dataset.sequence_field in frame:
            lengths = frame[config.dataset.sequence_field].astype(str).str.len()
            split_summary["sequence_length"] = {
                "min": int(lengths.min()) if len(lengths) else None,
                "median": float(lengths.median()) if len(lengths) else None,
                "max": int(lengths.max()) if len(lengths) else None,
            }
        if config.dataset.id_field and config.dataset.id_field in frame:
            split_summary["unique_ids"] = int(frame[config.dataset.id_field].nunique())
        split_summary["content_sha256"] = _split_content_sha256(config, frame)
        summary[split] = split_summary
    return summary
