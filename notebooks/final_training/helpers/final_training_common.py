"""Shared utilities for the final Colab-T4 training notebooks."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

EXPECTED_ROWS = {"train": 136484, "validation": 19498, "test": 38996}
SPLIT_FILES = {
    "train": "train_EP_DNA_BERT2_genomic_order.csv",
    "validation": "eval_EP_DNA_BERT2_genomic_order.csv",
    "test": "test_EP_DNA_BERT2_genomic_order.csv",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json_atomic(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    os.replace(temporary, path)


def atomic_copy(source: str | Path, destination: str | Path) -> None:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def audit_and_stage(drive_data_dir: str | Path, local_data_dir: str | Path, audit_path: str | Path):
    """Audit the explicit Drive directory, then atomically stage local copies."""
    drive_data_dir = Path(drive_data_dir)
    local_data_dir = Path(local_data_dir)
    if not drive_data_dir.is_dir():
        raise FileNotFoundError(
            f"DRIVE_DATA_DIR does not exist: {drive_data_dir}\n"
            f"Required files: {', '.join(SPLIT_FILES.values())}"
        )

    frames = {}
    audit = {
        "dataset": "GSE144621 EP_DNA_BERT2_genomic_order",
        "seed": 42,
        "drive_data_dir": str(drive_data_dir),
        "files": {},
        "duplicates": {},
    }
    normalized_sequences = {}
    for split, filename in SPLIT_FILES.items():
        source = drive_data_dir / filename
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing {split} split: {source}\n"
                f"Required files: {', '.join(SPLIT_FILES.values())}"
            )
        frame = pd.read_csv(source)
        missing = {"sequence", "label"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{source} is missing required columns: {sorted(missing)}")
        if len(frame) != EXPECTED_ROWS[split]:
            raise ValueError(
                f"{split} row count mismatch: expected {EXPECTED_ROWS[split]}, got {len(frame)}. "
                "Refusing to continue with a different split."
            )
        labels = pd.to_numeric(frame["label"], errors="raise").astype(int)
        if sorted(labels.unique().tolist()) != [0, 1]:
            raise ValueError(f"{split} labels must be exactly 0 and 1.")
        frame = frame.copy()
        frame["label"] = labels
        frame["sequence"] = frame["sequence"].astype(str).str.upper().str.replace("U", "T", regex=False)
        normalized_sequences[split] = set(frame["sequence"])
        audit["files"][split] = {
            "name": filename,
            "path": str(source),
            "sha256": sha256_file(source),
            "rows": int(len(frame)),
            "label_counts": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
            "duplicate_sequences_within_split": int(frame["sequence"].duplicated().sum()),
        }
        frames[split] = frame

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        audit["duplicates"][f"{left}_vs_{right}"] = len(
            normalized_sequences[left].intersection(normalized_sequences[right])
        )
    audit["duplicate_check_passed"] = all(
        value == 0 for value in audit["duplicates"].values()
    ) and all(
        item["duplicate_sequences_within_split"] == 0
        for item in audit["files"].values()
    )
    write_json_atomic(audit, audit_path)
    if not audit["duplicate_check_passed"]:
        raise ValueError(
            "Duplicate sequences were found within or across the shared splits. "
            "The final benchmark will not continue with possible leakage."
        )

    local_data_dir.mkdir(parents=True, exist_ok=True)
    for filename in SPLIT_FILES.values():
        local_target = local_data_dir / filename
        temporary = local_target.with_name(f".{filename}.{uuid.uuid4().hex}.tmp")
        shutil.copy2(drive_data_dir / filename, temporary)
        os.replace(temporary, local_target)
    return audit, frames


def exact_mcc_threshold(labels: Any, probabilities: Any):
    """Return threshold, MCC, and AUPRC using all unique validation scores."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    best = None
    selected = (0.5, 0.0, 0.0)
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        mcc = float(matthews_corrcoef(labels, predictions))
        auprc = float(average_precision_score(labels, probabilities))
        current = (mcc, auprc, -float(threshold))
        if best is None or current > best:
            best = current
            selected = (float(threshold), mcc, auprc)
    return selected


def _rank_metric(function: Any, labels: np.ndarray, probabilities: np.ndarray):
    try:
        return float(function(labels, probabilities))
    except ValueError:
        return None


def classification_metrics(labels: Any, probabilities: Any | None, threshold: float | None, *, loss: float | None = None, hard_predictions: Any | None = None):
    labels = np.asarray(labels, dtype=int)
    has_scores = probabilities is not None
    if has_scores:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        predictions = (probabilities >= float(threshold)).astype(int)
    else:
        predictions = np.asarray(hard_predictions, dtype=int)
        probabilities = None
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auroc": _rank_metric(roc_auc_score, labels, probabilities) if has_scores else None,
        "auprc": _rank_metric(average_precision_score, labels, probabilities) if has_scores else None,
        "threshold": None if threshold is None else float(threshold),
        "loss": None if loss is None else float(loss),
    }


def flatten_metrics(metrics):
    return pd.DataFrame([{"split": split, **values} for split, values in metrics.items()])


def append_history(row: dict[str, Any], local_path: str | Path, drive_path: str | Path) -> None:
    local_path = Path(local_path)
    if local_path.exists() and local_path.stat().st_size:
        history = pd.read_csv(local_path)
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    else:
        history = pd.DataFrame([row])
    temporary = local_path.with_name(f".{local_path.name}.{uuid.uuid4().hex}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(temporary, index=False)
    os.replace(temporary, local_path)
    atomic_copy(local_path, drive_path)


def save_checkpoint(payload: dict[str, Any], local_path: str | Path, drive_path: str | Path, torch_module: Any) -> None:
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = local_path.with_name(f".{local_path.name}.{uuid.uuid4().hex}.tmp")
    torch_module.save(payload, temporary)
    os.replace(temporary, local_path)
    atomic_copy(local_path, drive_path)


def restore_checkpoint(drive_path: str | Path, local_path: str | Path, torch_module: Any):
    atomic_copy(drive_path, local_path)
    try:
        return torch_module.load(local_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(local_path, map_location="cpu")


def rng_state(torch_module: Any):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch_module.get_rng_state(),
    }
    if torch_module.cuda.is_available():
        state["torch_cuda"] = torch_module.cuda.get_rng_state_all()
    return state


def restore_rng_state(state, torch_module: Any) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_module.set_rng_state(state["torch_cpu"])
    if torch_module.cuda.is_available() and "torch_cuda" in state:
        torch_module.cuda.set_rng_state_all(state["torch_cuda"])


def masked_mean_max(hidden: Any, attention_mask: Any, torch_module: Any) -> Any:
    mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    count = mask.sum(dim=1).clamp_min(1.0)
    mean = (hidden * mask).sum(dim=1) / count
    masked_hidden = hidden.masked_fill(mask == 0, torch_module.finfo(hidden.dtype).min)
    maximum = masked_hidden.max(dim=1).values
    return torch_module.cat([mean, maximum], dim=-1)


def verify_required_outputs(root: str | Path, *, include_embeddings: bool = False):
    root = Path(root)
    required = [
        "config.json", "input_split_audit.json", "environment.json", "history.csv",
        "metrics.csv", "metrics.json", "predictions.csv", "manifest.json",
        "checkpoints", "plots", "logs",
    ]
    if include_embeddings:
        required.append("fold_predictions")
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing final-training artifacts: {missing}")
    return [str(root / name) for name in required]
