"""ProkBERT-mini fine-tuning for the shared SeqTrainer benchmark contract.

The module intentionally contains the ProkBERT-specific loading, tokenization,
pooling, and training code while using SeqTrainer's split, metric, manifest,
and artifact helpers for the benchmark surface.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from seqtrainer.benchmarks.artifacts import write_benchmark_outputs, write_json
from seqtrainer.benchmarks.config import BenchmarkConfig
from seqtrainer.benchmarks.manifest import build_run_manifest, git_metadata, to_plain_data
from seqtrainer.benchmarks.policy import decide_imbalance_policy
from seqtrainer.benchmarks.runner import BenchmarkRunResult, BenchmarkSkipped
from seqtrainer.benchmarks.splits import (
    load_predefined_split_frames,
    resolve_split_paths,
    summarize_split_frames,
)
from seqtrainer.metrics import best_threshold_by_metric, binary_classification_metrics


HF_MODEL_REVISION = "feb2520a43cd9cdb5b3d8477e47209dbcb55d1dc"
OFFICIAL_PROKBERT_REPOSITORY = "https://github.com/nbrg-ppcu/prokbert.git"
OFFICIAL_PROKBERT_COMMIT = "8670ae92b816cff158a0b85647a8dea122e251eb"
MODEL_LICENSE = "CC-BY-NC-4.0"


@dataclass(frozen=True)
class TokenizationAudit:
    """Summary of the tokenizer/encoder preflight audit."""

    rows: int
    raw_sequence_lengths: dict[str, int]
    token_lengths: dict[str, int]
    attention_mask_present: bool
    output_shape: tuple[int, ...]
    truncated_rows: int


class ProkBERTClassifier(nn.Module):
    """A binary classifier around a ProkBERT encoder.

    The encoder output is pooled with either attention-mask-aware mean pooling
    or first-token (CLS) pooling, followed by dropout and one linear logit.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        hidden_size: int | None = None,
        dropout: float = 0.1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        normalized_pooling = pooling.lower()
        if normalized_pooling not in {"mean", "cls"}:
            raise ValueError("ProkBERT pooling must be either 'mean' or 'cls'")
        self.encoder = encoder
        self.pooling = normalized_pooling
        size = hidden_size or _encoder_hidden_size(encoder)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = _last_hidden_state(outputs)
        if self.pooling == "cls":
            pooled = hidden[:, 0, :]
        else:
            pooled = mean_pool_hidden_states(hidden, attention_mask)
        return self.classifier(self.dropout(pooled)).squeeze(-1)


def mean_pool_hidden_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token states while excluding padding tokens."""
    if hidden_states.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("hidden_states must be [batch, tokens, hidden] and attention_mask [batch, tokens]")
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError("hidden_states and attention_mask have incompatible batch/token dimensions")
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def load_prokbert_backbone(
    model_name: str,
    *,
    revision: str,
    tokenizer_revision: str,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    """Load the generic ProkBERT tokenizer and encoder through Transformers."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise BenchmarkSkipped("ProkBERT requires the optional transformers dependency.") from exc

    if not revision or revision == "main" or not tokenizer_revision or tokenizer_revision == "main":
        raise ValueError("ProkBERT model and tokenizer revisions must be pinned commit SHAs, not 'main'")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=tokenizer_revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        encoder = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except OSError as exc:
        raise BenchmarkSkipped(
            "ProkBERT tokenizer/model files are unavailable. Set model.params.allow_download=true "
            "only in an environment where the pinned Hugging Face revision may be downloaded."
        ) from exc
    return tokenizer, encoder


def clean_sequence_frame(frame: pd.DataFrame, sequence_field: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize sequence case/whitespace and report invalid characters.

    Invalid characters are retained so that no row disappears silently. The
    tokenizer or a later validation step can then produce the useful failure.
    """
    if sequence_field not in frame.columns:
        raise ValueError(f"Dataset is missing sequence column {sequence_field!r}")
    cleaned = frame.copy()
    modified_rows = 0
    invalid_rows = 0
    invalid_counts: dict[str, int] = {}
    values: list[str] = []
    for raw in cleaned[sequence_field].tolist():
        value = "" if pd.isna(raw) else str(raw)
        normalized = value.strip().upper()
        modified_rows += int(normalized != value)
        invalid = sorted(set(normalized) - set("ACGTN"))
        if invalid:
            invalid_rows += 1
            for character in invalid:
                invalid_counts[character] = invalid_counts.get(character, 0) + normalized.count(character)
        values.append(normalized)
    cleaned[sequence_field] = values
    return cleaned, {
        "rows": int(len(cleaned)),
        "modified_rows": modified_rows,
        "invalid_rows": invalid_rows,
        "invalid_character_counts": invalid_counts,
    }


def audit_prokbert_tokenizer(
    tokenizer: Any,
    encoder: Any,
    sequences: Sequence[str],
    *,
    model_max_length: int = 512,
) -> dict[str, Any]:
    """Audit raw lengths, custom-tokenizer lengths, masks, and encoder shape."""
    if len(sequences) < 5:
        raise ValueError("The ProkBERT tokenization audit requires at least five real sequences")
    sample = [str(sequence) for sequence in sequences[:5]]
    encoded = _tokenize(tokenizer, sample, max_length=model_max_length)
    attention_mask = encoded["attention_mask"]
    if attention_mask is None:
        raise ValueError("The ProkBERT tokenizer did not return attention_mask")
    token_lengths = [int(value) for value in attention_mask.sum(dim=1).tolist()]
    truncated_rows = 0
    try:
        untruncated = tokenizer(
            sample,
            padding="longest",
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        untruncated_ids = _as_tensor(untruncated["input_ids"], dtype=torch.long)
        untruncated_lengths = [int(value) for value in untruncated_ids.shape[1:2]] * len(sample)
        truncated_rows = sum(length > model_max_length for length in untruncated_lengths)
    except (TypeError, ValueError, KeyError):
        # Some old custom-tokenizer versions do not expose a no-truncation mode;
        # the normal bounded call and explicit max-length check still apply.
        truncated_rows = sum(length > model_max_length for length in token_lengths)
    if any(length > model_max_length for length in token_lengths):
        raise ValueError("ProkBERT tokenization exceeded model_max_length despite truncation=True")
    with torch.no_grad():
        outputs = encoder(**encoded)
    hidden = _last_hidden_state(outputs)
    if hidden.ndim != 3 or hidden.shape[0] != len(sample):
        raise ValueError(f"Unexpected ProkBERT encoder output dimensions: {tuple(hidden.shape)}")
    audit = TokenizationAudit(
        rows=len(sample),
        raw_sequence_lengths={str(index): len(sequence) for index, sequence in enumerate(sample)},
        token_lengths={str(index): length for index, length in enumerate(token_lengths)},
        attention_mask_present=True,
        output_shape=tuple(int(value) for value in hidden.shape),
        truncated_rows=int(truncated_rows),
    )
    print("ProkBERT tokenization audit:", json.dumps(to_plain_data(audit), sort_keys=True))
    return to_plain_data(audit)


def run_prokbert_csv_splits(
    config: BenchmarkConfig,
    *,
    base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    tokenizer: Any | None = None,
    encoder: Any | None = None,
) -> BenchmarkRunResult:
    """Fine-tune ProkBERT on the three predefined CSV splits."""
    _seed_everything(config.training.seed)
    frames = load_predefined_split_frames(config, base_dir=base_dir)
    cleaned_frames: dict[str, pd.DataFrame] = {}
    cleaning: dict[str, Any] = {}
    for split, frame in frames.items():
        cleaned_frames[split], cleaning[split] = clean_sequence_frame(frame, config.dataset.sequence_field)
        _validate_binary_labels(cleaned_frames[split], config)

    params = dict(config.model.params)
    train_params = dict(config.training.params)
    model_revision = str(params.get("revision") or config.model.version or HF_MODEL_REVISION)
    tokenizer_revision = str(params.get("tokenizer_revision") or model_revision)
    model_name = config.model.name
    trust_remote_code = bool(params.get("trust_remote_code", True))
    allow_download = bool(params.get("allow_download", False))
    if tokenizer is None or encoder is None:
        tokenizer, encoder = load_prokbert_backbone(
            model_name,
            revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            trust_remote_code=trust_remote_code,
            local_files_only=not allow_download,
        )
    if tokenizer is None or encoder is None:  # for type checkers
        raise RuntimeError("ProkBERT tokenizer and encoder must both be available")

    model_max_length = int(config.preprocessing.params.get("model_max_length", 512))
    audit = audit_prokbert_tokenizer(
        tokenizer,
        encoder,
        cleaned_frames["train"][config.dataset.sequence_field].tolist(),
        model_max_length=model_max_length,
    )
    encoded = {
        split: _encode_split(cleaned_frames[split], config, tokenizer)
        for split in ("train", "validation", "test")
    }
    device = _resolve_device(config.environment.device)
    pooling = str(params.get("pooling", "mean")).lower()
    model = ProkBERTClassifier(
        encoder,
        dropout=float(params.get("classifier_dropout", params.get("dropout", 0.1))),
        pooling=pooling,
    )
    freeze_encoder = bool(params.get("freeze_encoder", False)) or str(params.get("mode", "full_finetune")) == "frozen"
    if freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    gradient_checkpointing_requested = bool(train_params.get("gradient_checkpointing", False))
    gradient_checkpointing_enabled = False
    if gradient_checkpointing_requested and not freeze_encoder:
        enable = getattr(model.encoder, "gradient_checkpointing_enable", None)
        if callable(enable):
            enable()
            if hasattr(getattr(model.encoder, "config", None), "use_cache"):
                model.encoder.config.use_cache = False
            gradient_checkpointing_enabled = True

    requested_batch_size = int(config.training.batch_size or 16)
    requested_accumulation = int(train_params.get("gradient_accumulation_steps", 2))
    effective_batch_size = requested_batch_size * requested_accumulation
    physical_batch_size = requested_batch_size
    gradient_accumulation_steps = requested_accumulation
    out_dir = Path(output_dir or config.outputs.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    try:
        _preflight_batch(model, encoded["train"], criterion, device, precision=config.environment.precision)
    except RuntimeError as exc:
        if not _is_cuda_oom(exc, device) or requested_batch_size != 16:
            raise
        physical_batch_size, gradient_accumulation_steps, fallback_effective = resolve_batch_profile(
            requested_batch_size,
            requested_accumulation,
            cuda_oom=True,
        )
        if fallback_effective != effective_batch_size:
            raise AssertionError("ProkBERT OOM fallback changed the effective batch size")
        if physical_batch_size * gradient_accumulation_steps != effective_batch_size:
            raise AssertionError("ProkBERT OOM fallback changed the effective batch size")
        torch.cuda.empty_cache()
        _preflight_batch(model, encoded["train"], criterion, device, precision=config.environment.precision)
    oom_fallback = physical_batch_size != requested_batch_size

    loaders = {
        split: DataLoader(
            TensorDataset(value["input_ids"], value["attention_mask"], value["labels"]),
            batch_size=physical_batch_size,
            shuffle=split == "train",
        )
        for split, value in encoded.items()
    }
    prediction_loaders = {
        split: DataLoader(
            TensorDataset(value["input_ids"], value["attention_mask"], value["labels"]),
            batch_size=physical_batch_size,
            shuffle=False,
        )
        for split, value in encoded.items()
    }
    learning_rate = float(config.training.learning_rate or 2e-5)
    weight_decay = float(train_params.get("weight_decay", 0.01))
    max_epochs = int(config.training.max_epochs or 3)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    optimizer_steps_per_epoch = max(1, ceil(len(loaders["train"]) / gradient_accumulation_steps))
    total_steps = max(1, max_epochs * optimizer_steps_per_epoch)
    warmup_ratio = float(train_params.get("warmup_ratio", 0.10))
    scheduler = _linear_warmup_scheduler(
        optimizer,
        warmup_steps=int(total_steps * warmup_ratio),
        total_steps=total_steps,
    )
    precision = str(config.environment.precision).lower()
    scaler = _make_grad_scaler(device, precision)
    max_grad_norm = float(train_params.get("max_grad_norm", 1.0))
    patience = int(train_params.get("early_stopping_patience", 1))
    split_hashes = _split_hashes(config, base_dir)
    seqtrainer_commit = _seqtrainer_commit()
    config_hash = _config_hash(config)
    resume_path = train_params.get("resume_from_checkpoint") or params.get("resume_from_checkpoint")
    start_epoch = 1
    best_mcc = float("-inf")
    best_threshold = 0.5
    history: list[dict[str, Any]] = []
    checkpoint_path = out_dir / "best_checkpoint.pt"
    if resume_path:
        resume_state = load_checkpoint_for_resume(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected={
                "config_hash": config_hash,
                "split_hashes": split_hashes,
                "model_revision": model_revision,
                "seqtrainer_commit": seqtrainer_commit,
            },
            map_location=device,
        )
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        best_mcc = float(resume_state.get("best_validation_mcc", float("-inf")))
        best_threshold = float(resume_state.get("selected_validation_threshold", 0.5))
        history = list(resume_state.get("history", []))

    bad_epochs = 0
    run_started = time.perf_counter()
    for epoch in range(start_epoch, max_epochs + 1):
        train_loss = _train_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            precision=precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
        )
        validation = _predict(model, prediction_loaders["validation"], criterion, device, precision=precision)
        threshold, validation_mcc = best_threshold_by_metric(
            validation["label"], validation["probability"], metric="mcc"
        )
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation_loss": float(validation["loss"]),
            "validation_mcc": float(validation_mcc),
            "validation_threshold": float(threshold),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "stopped_early": 0,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        if validation_mcc > best_mcc:
            best_mcc = float(validation_mcc)
            best_threshold = float(threshold)
            bad_epochs = 0
            _save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_mcc=best_mcc,
                threshold=best_threshold,
                history=history,
                metadata={
                    "config_hash": config_hash,
                    "split_hashes": split_hashes,
                    "model_revision": model_revision,
                    "seqtrainer_commit": seqtrainer_commit,
                },
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                history[-1]["stopped_early"] = 1
                break
    if not checkpoint_path.exists():
        raise RuntimeError("ProkBERT training did not produce best_checkpoint.pt")

    _load_state_dict(checkpoint_path, model, device)
    # The test split is touched only in this final post-checkpoint prediction pass.
    predictions = {
        split: _predict(model, prediction_loaders[split], criterion, device, precision=precision)
        for split in ("train", "validation", "test")
    }
    metrics: dict[str, dict[str, Any]] = {}
    prediction_frames: list[pd.DataFrame] = []
    for split in ("train", "validation", "test"):
        prediction = predictions[split]
        split_metrics = binary_classification_metrics(
            prediction["label"], prediction["probability"], threshold=best_threshold
        )
        split_metrics["loss"] = prediction["loss"]
        metrics[split] = split_metrics
        frame = cleaned_frames[split].reset_index(drop=True)
        result_frame: dict[str, Any] = {
            "split": split,
            "row_index": np.arange(len(frame), dtype=int),
            "label": frame[config.dataset.label_field].map(lambda value: _label_to_int(value, config)).to_numpy(dtype=int),
            "probability": prediction["probability"],
            "prediction": (prediction["probability"] >= best_threshold).astype(int),
            "threshold": best_threshold,
        }
        id_field = _existing_id_field(frame, config)
        if id_field is not None:
            result_frame[id_field] = frame[id_field].to_numpy()
        result_frame[config.dataset.sequence_field] = frame[config.dataset.sequence_field].to_numpy()
        prediction_frames.append(pd.DataFrame(result_frame))

    _save_tokenizer(tokenizer, out_dir / "tokenizer", metadata={"model_revision": model_revision})
    test_timestamp = datetime.now(timezone.utc).isoformat()
    split_summary = summarize_split_frames(config, cleaned_frames)
    split_summary["file_hashes"] = split_hashes
    split_summary["token_lengths"] = _token_length_summary(encoded)
    imbalance_policy = decide_imbalance_policy(split_summary)
    manifest = build_run_manifest(
        config,
        split_summary=split_summary,
        threshold=best_threshold,
        model_metadata={
            "base_model": model_name,
            "pretraining_task": "masked_language_modeling",
            "promoter_finetuned_checkpoint_used": False,
            "trust_remote_code": trust_remote_code,
            "model_license": MODEL_LICENSE,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "prokbert_package_commit": str(params.get("prokbert_package_commit", OFFICIAL_PROKBERT_COMMIT)),
            "pooling": pooling,
            "dropout": float(params.get("classifier_dropout", params.get("dropout", 0.1))),
            "freeze_encoder": freeze_encoder,
            "gradient_checkpointing_enabled": gradient_checkpointing_enabled,
            "best_checkpoint": str(checkpoint_path),
            "runtime_seconds": float(time.perf_counter() - run_started),
        },
        extra={
            "status": "completed",
            "seqtrainer_commit": seqtrainer_commit,
            "split_file_paths": {split: str(path) for split, path in resolve_split_paths(config, base_dir).items()},
            "split_file_sha256": split_hashes,
            "sequence_cleaning": cleaning,
            "tokenization_audit": audit,
            "requested_batch_size": requested_batch_size,
            "effective_batch_size": effective_batch_size,
            "physical_batch_size": physical_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "maximum_epochs": max_epochs,
            "precision": precision,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "transformers_version": _package_version("transformers"),
            "cuda_version": torch.version.cuda,
            "best_epoch": int(max(history, key=lambda row: row["validation_mcc"])["epoch"]),
            "best_validation_mcc": best_mcc,
            "selected_validation_threshold": best_threshold,
            "test_evaluation_timestamp": test_timestamp,
            "oom_fallback_occurred": oom_fallback,
            "gradient_checkpointing_enabled": gradient_checkpointing_enabled,
            "encoder_frozen": freeze_encoder,
            "promoter_finetuned_checkpoint_used": False,
            "imbalance_policy": to_plain_data(imbalance_policy),
            "runtime_seconds": float(time.perf_counter() - run_started),
        },
    )
    write_benchmark_outputs(
        out_dir,
        manifest=manifest,
        metrics=metrics,
        predictions=pd.concat(prediction_frames, ignore_index=True),
        history=pd.DataFrame(history),
        config=config,
    )
    (out_dir / "run_summary.txt").write_text(
        "SeqTrainer ProkBERT-mini benchmark\n"
        f"status=completed\nmodel={model_name}\nmodel_revision={model_revision}\n"
        f"best_epoch={manifest['extra']['best_epoch']}\n"
        f"best_validation_mcc={best_mcc:.8f}\nselected_validation_threshold={best_threshold:.8f}\n"
        f"test_evaluation_timestamp={test_timestamp}\n",
        encoding="utf-8",
    )
    return BenchmarkRunResult(output_dir=out_dir, status="completed", metrics=metrics, manifest=manifest)


def validate_resume_checkpoint(checkpoint: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Reject a checkpoint whose reproducibility contract does not match."""
    for key in ("config_hash", "split_hashes", "model_revision", "seqtrainer_commit"):
        if checkpoint.get(key) != expected.get(key):
            raise ValueError(
                f"Incompatible ProkBERT checkpoint: {key} does not match the current run "
                f"(checkpoint={checkpoint.get(key)!r}, current={expected.get(key)!r})"
            )


def resolve_batch_profile(
    requested_batch_size: int,
    requested_gradient_accumulation: int,
    *,
    cuda_oom: bool = False,
) -> tuple[int, int, int]:
    """Return physical/effective batch settings, with the documented T4 fallback."""
    physical = int(requested_batch_size)
    accumulation = int(requested_gradient_accumulation)
    if cuda_oom and physical == 16:
        physical, accumulation = 8, 4
    return physical, accumulation, physical * accumulation


def load_checkpoint_for_resume(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    optimizer: Any,
    scheduler: Any,
    scaler: Any | None,
    expected: Mapping[str, Any],
    map_location: Any = "cpu",
) -> dict[str, Any]:
    """Validate and restore a resumable training checkpoint."""
    state = _torch_load(Path(checkpoint_path), map_location=map_location)
    validate_resume_checkpoint(state, expected)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and state.get("scaler_state_dict"):
        scaler.load_state_dict(state["scaler_state_dict"])
    return state


def _encode_split(frame: pd.DataFrame, config: BenchmarkConfig, tokenizer: Any) -> dict[str, torch.Tensor]:
    encoded = _tokenize(
        tokenizer,
        frame[config.dataset.sequence_field].tolist(),
        max_length=int(config.preprocessing.params.get("model_max_length", 512)),
    )
    labels = torch.tensor(
        [_label_to_int(value, config) for value in frame[config.dataset.label_field].tolist()],
        dtype=torch.float32,
    )
    encoded["labels"] = labels
    return encoded


def _tokenize(tokenizer: Any, sequences: Sequence[str], *, max_length: int) -> dict[str, torch.Tensor]:
    kwargs = {
        "padding": "longest",
        "truncation": True,
        "max_length": max_length,
        "return_attention_mask": True,
        "return_tensors": "pt",
    }
    encoded = tokenizer(list(sequences), **kwargs)
    if "attention_mask" not in encoded:
        raise ValueError("The ProkBERT tokenizer must return attention_mask")
    return {
        "input_ids": _as_tensor(encoded["input_ids"], dtype=torch.long),
        "attention_mask": _as_tensor(encoded["attention_mask"], dtype=torch.long),
    }


def _as_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    return tensor.to(dtype=dtype)


def _preflight_batch(model: nn.Module, encoded: Mapping[str, torch.Tensor], criterion: Any, device: torch.device, *, precision: str) -> None:
    model.eval()
    with torch.no_grad(), _autocast_context(device, precision):
        input_ids = encoded["input_ids"][:16].to(device)
        attention_mask = encoded["attention_mask"][:16].to(device)
        labels = encoded["labels"][:16].to(device)
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
    if logits.ndim != 1 or logits.shape[0] != input_ids.shape[0] or not torch.isfinite(loss):
        raise ValueError("ProkBERT preflight produced invalid logits or loss dimensions")


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any | None,
    device: torch.device,
    *,
    precision: str,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_rows = 0
    accumulation = max(1, int(gradient_accumulation_steps))
    for index, (input_ids, attention_mask, labels) in enumerate(loader, start=1):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        with _autocast_context(device, precision):
            raw_loss = criterion(model(input_ids, attention_mask), labels)
            loss = raw_loss / accumulation
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
        if index % accumulation == 0 or index == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(raw_loss.detach().item()) * int(labels.shape[0])
        total_rows += int(labels.shape[0])
    return total_loss / max(total_rows, 1)


def _predict(model: nn.Module, loader: DataLoader, criterion: Any, device: torch.device, *, precision: str) -> dict[str, Any]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    total_loss = 0.0
    total_rows = 0
    with torch.no_grad():
        for input_ids, attention_mask, batch_labels in loader:
            input_ids, attention_mask, batch_labels = input_ids.to(device), attention_mask.to(device), batch_labels.to(device)
            with _autocast_context(device, precision):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, batch_labels)
            labels.append(batch_labels.cpu().numpy().astype(int))
            probabilities.append(torch.sigmoid(logits).cpu().numpy().astype(float))
            total_loss += float(loss.item()) * int(batch_labels.shape[0])
            total_rows += int(batch_labels.shape[0])
    return {
        "label": np.concatenate(labels) if labels else np.empty(0, dtype=int),
        "probability": np.concatenate(probabilities) if probabilities else np.empty(0, dtype=float),
        "loss": total_loss / max(total_rows, 1),
    }


def _save_checkpoint(path: Path, *, model: nn.Module, optimizer: Any, scheduler: Any, scaler: Any | None, epoch: int, best_mcc: float, threshold: float, history: list[dict[str, Any]], metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **dict(metadata),
            "epoch": epoch,
            "best_validation_mcc": best_mcc,
            "selected_validation_threshold": threshold,
            "history": history,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def _load_state_dict(path: Path, model: nn.Module, device: torch.device) -> None:
    state = _torch_load(path, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))


def _torch_load(path: Path, *, map_location: Any) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - older torch compatibility
        return torch.load(path, map_location=map_location)


def _save_tokenizer(tokenizer: Any, path: Path, *, metadata: Mapping[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    save_pretrained = getattr(tokenizer, "save_pretrained", None)
    if callable(save_pretrained):
        save_pretrained(str(path))
    else:
        write_json(path / "tokenizer_info.json", dict(metadata))


def _linear_warmup_scheduler(optimizer: Any, *, warmup_steps: int, total_steps: int) -> Any:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        return float(max(0, total_steps - step)) / float(max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _autocast_context(device: torch.device, precision: str) -> Any:
    if device.type != "cuda":
        return nullcontext()
    if precision in {"fp16", "float16"}:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision in {"bf16", "bfloat16"}:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _make_grad_scaler(device: torch.device, precision: str) -> Any | None:
    if device.type != "cuda" or precision not in {"fp16", "float16"}:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):  # pragma: no cover - older torch compatibility
        return torch.cuda.amp.GradScaler()


def _is_cuda_oom(exc: RuntimeError, device: torch.device) -> bool:
    return device.type == "cuda" and "out of memory" in str(exc).lower()


def _last_hidden_state(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    raise AttributeError("ProkBERT encoder output did not contain last_hidden_state")


def _encoder_hidden_size(encoder: Any) -> int:
    config = getattr(encoder, "config", None)
    for name in ("hidden_size", "d_model", "embedding_size"):
        value = getattr(config, name, None)
        if value:
            return int(value)
    raise ValueError("Could not infer ProkBERT encoder hidden size from encoder.config")


def _validate_binary_labels(frame: pd.DataFrame, config: BenchmarkConfig) -> None:
    values = set(frame[config.dataset.label_field].tolist())
    allowed = {config.label.positive_label, config.label.negative_label}
    if not values.issubset(allowed):
        raise ValueError(f"Labels must be {sorted(allowed, key=str)}; observed unexpected labels {sorted(values - allowed, key=str)}")


def _label_to_int(value: Any, config: BenchmarkConfig) -> int:
    if value == config.label.positive_label:
        return 1
    if value == config.label.negative_label:
        return 0
    raise ValueError(f"Unexpected binary label {value!r}")


def _split_hashes(config: BenchmarkConfig, base_dir: str | Path | None) -> dict[str, str]:
    paths = resolve_split_paths(config, base_dir)
    return {split: _sha256(path) for split, path in paths.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_length_summary(encoded: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split, values in encoded.items():
        lengths = values["attention_mask"].sum(dim=1).cpu().numpy().astype(int)
        summary[split] = {
            "min": int(lengths.min()) if len(lengths) else None,
            "median": float(np.median(lengths)) if len(lengths) else None,
            "max": int(lengths.max()) if len(lengths) else None,
        }
    return summary


def _config_hash(config: BenchmarkConfig) -> str:
    payload = json.dumps(to_plain_data(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _seqtrainer_commit() -> str | None:
    repo = Path(__file__).resolve().parents[3]
    return git_metadata(repo).get("commit")


def _package_version(name: str) -> str | None:
    try:
        from importlib import metadata

        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _existing_id_field(frame: pd.DataFrame, config: BenchmarkConfig) -> str | None:
    """Find the configured or conventional identifier column without inventing one."""
    if config.dataset.id_field and config.dataset.id_field in frame.columns:
        return config.dataset.id_field
    for candidate in ("id", "sequence_id", "example_id"):
        if candidate in frame.columns:
            return candidate
    return None
