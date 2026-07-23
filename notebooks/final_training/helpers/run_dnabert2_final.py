"""Final DNABERT2 staged fine-tuning runner used by the Colab notebook.

The active data loader always reads local /content files. Drive is used only
for durable artifacts, model cache, history, and resumable checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from final_training_common import (
    EXPECTED_ROWS,
    SPLIT_FILES,
    append_history,
    atomic_copy,
    audit_and_stage,
    classification_metrics,
    exact_mcc_threshold,
    flatten_metrics,
    masked_mean_max,
    restore_checkpoint,
    restore_rng_state,
    rng_state,
    save_checkpoint,
    sha256_file,
    verify_required_outputs,
    write_json_atomic,
)


def seed_everything(seed: int, torch: Any) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def token_audit(tokenizer: Any, frames: dict[str, pd.DataFrame], output_dir: Path, batch_size: int = 256) -> dict[str, Any]:
    lengths = []
    for frame in frames.values():
        sequences = frame["sequence"].tolist()
        for start in range(0, len(sequences), batch_size):
            encoded = tokenizer(
                sequences[start : start + batch_size],
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
            lengths.extend(len(item) for item in encoded["input_ids"])
    values = np.asarray(lengths, dtype=np.int64)
    report = {
        "minimum": int(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": int(values.max()),
        "thresholds": {
            str(limit): {
                "truncated_rows": int((values > limit).sum()),
                "truncated_percent": float((values > limit).mean() * 100.0),
            }
            for limit in (70, 104, 128)
        },
    }
    write_json_atomic(report, output_dir / "token_length_audit.json")
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 4))
        plt.hist(values, bins=40, color="#2f6f8f")
        plt.axvline(70, color="red", label="70")
        plt.axvline(104, color="orange", label="104")
        plt.axvline(128, color="green", label="128")
        plt.xlabel("DNABERT2 token length")
        plt.ylabel("sequences")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "token_length_histogram.png", dpi=160)
        plt.close()
    except Exception as exc:
        print("Token histogram skipped:", exc)
    return report


class SequenceDataset:
    def __init__(self, frame: pd.DataFrame):
        self.sequences = frame["sequence"].tolist()
        self.labels = frame["label"].astype(int).to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequences[index], int(self.labels[index])


class DnaBertHead:
    def __init__(self, encoder: Any, torch: Any, nn: Any, dropout: float = 0.20):
        self.torch = torch
        self.encoder = encoder
        hidden = int(getattr(encoder.config, "hidden_size", 768))
        self.norm = nn.LayerNorm(hidden * 2)
        self.projection = nn.Linear(hidden * 2, 256)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(256, 1)
        self.module = nn.ModuleList([self.encoder, self.norm, self.projection, self.activation, self.dropout, self.output])

    def parameters(self):
        return self.module.parameters()

    def named_parameters(self):
        for name, parameter in self.encoder.named_parameters():
            yield f"encoder.{name}", parameter
        for name, parameter in self.norm.named_parameters():
            yield f"norm.{name}", parameter
        for name, parameter in self.projection.named_parameters():
            yield f"projection.{name}", parameter
        for name, parameter in self.output.named_parameters():
            yield f"output.{name}", parameter

    def train(self, mode=True):
        self.module.train(mode)
        return self

    def eval(self):
        self.module.eval()
        return self

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        return self.module.load_state_dict(state)

    def to(self, device):
        self.module.to(device)
        return self

    def __call__(self, batch):
        output = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        pooled = masked_mean_max(output.last_hidden_state, batch["attention_mask"], self.torch)
        pooled = self.norm(pooled)
        return self.output(self.dropout(self.activation(self.projection(pooled)))).squeeze(-1)


def encoder_layers(encoder: Any) -> list[Any]:
    candidates = [
        getattr(getattr(encoder, "encoder", None), "layer", None),
        getattr(getattr(getattr(encoder, "bert", None), "encoder", None), "layer", None),
        getattr(getattr(getattr(encoder, "base_model", None), "encoder", None), "layer", None),
    ]
    for layers in candidates:
        if layers is not None:
            return list(layers)
    raise RuntimeError("Could not locate DNABERT2 transformer layers for staged unfreezing.")


def set_stage(encoder: Any, epoch: int, head_only_epochs: int, unfreeze_top_layers: int) -> str:
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    if epoch <= head_only_epochs:
        return "classifier_head_only"
    layers = encoder_layers(encoder)
    for layer in layers[-unfreeze_top_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    return f"top_{unfreeze_top_layers}_layers"


def collate_factory(tokenizer: Any, max_length: int, torch: Any):
    def collate(items):
        sequences, labels = zip(*items)
        encoded = tokenizer(
            list(sequences),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.float32)
        return encoded
    return collate


def make_loader(frame, tokenizer, max_length, batch_size, torch, shuffle, workers=2):
    from torch.utils.data import DataLoader

    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_factory(tokenizer, max_length, torch),
        "pin_memory": torch.cuda.is_available(),
        "num_workers": workers,
    }
    if workers:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(SequenceDataset(frame), **kwargs)


def loss_value(logits, labels, criterion, torch):
    return criterion(logits, labels)


def run_epoch(model, loader, optimizer, scheduler, scaler, criterion, device, torch, use_amp, accumulation, max_grad_norm):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    optimizer_steps = 0
    pending_microbatches = 0

    def step_optimizer(microbatches: int) -> None:
        """Apply one logical batch, correcting the final short accumulation group."""
        nonlocal optimizer_steps
        scaler.unscale_(optimizer)
        if microbatches < accumulation:
            correction = accumulation / microbatches
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

    for batch_index, batch in enumerate(loader):
        labels = batch.pop("labels").to(device, non_blocking=True)
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            logits = model(batch)
            loss = loss_value(logits, labels, criterion, torch) / accumulation
        scaler.scale(loss).backward()
        pending_microbatches += 1
        if pending_microbatches == accumulation:
            step_optimizer(pending_microbatches)
            pending_microbatches = 0
        total_loss += float(loss.detach().item() * accumulation)
    if pending_microbatches:
        step_optimizer(pending_microbatches)
    return total_loss / max(len(loader), 1), optimizer_steps


def evaluate(model, loader, criterion, device, torch, use_amp):
    model.eval()
    labels = []
    probabilities = []
    losses = []
    with torch.inference_mode():
        for batch in loader:
            batch_labels = batch.pop("labels").to(device, non_blocking=True)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                logits = model(batch)
                loss = criterion(logits, batch_labels)
            labels.extend(batch_labels.detach().cpu().numpy().astype(int).tolist())
            probabilities.extend(torch.sigmoid(logits.float()).detach().cpu().numpy().tolist())
            losses.append(float(loss.float().item()))
    return np.asarray(labels), np.asarray(probabilities, dtype=np.float64), float(np.mean(losses))


def focal_loss(logits, labels, gamma, torch):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probability = torch.sigmoid(logits)
    pt = probability * labels + (1.0 - probability) * (1.0 - labels)
    return ((1.0 - pt).pow(gamma) * bce).mean()


def build_manifest(config, audit, env, best, threshold, drive_output, resume_source, token_report):
    training = config["training"]
    return {
        "repository": "https://github.com/simplyshree/SeqTrainer",
        "branch": config["branch"],
        "commit": config["commit"],
        "notebook": "notebooks/final_training/dnabert2_final_training_t4_colab.ipynb",
        "dataset": audit,
        "seed": 42,
        "model": config["model"],
        "tokenization": {"max_length": config["max_length"], "pooling": "attention-mask mean + max"},
        "classifier": "LayerNorm(1536) -> Linear(256) -> GELU -> Dropout(0.20) -> Linear(1)",
        "optimizer": "AdamW",
        "scheduler": "linear warmup then cosine decay",
        "training": training,
        "learning_rates": {
            "encoder": training["encoder_learning_rate"],
            "classifier_head": training["head_learning_rate"],
        },
        "weight_decay": training["weight_decay"],
        "loss": config["loss"],
        "gradient_accumulation": training["gradient_accumulation"],
        "physical_batch_size": training["batch_size"],
        "precision": "fp16 training, fp32 validation/test probabilities",
        "best_epoch": best["epoch"],
        "best_validation_mcc": best["mcc"],
        "best_validation_auprc": best["auprc"],
        "threshold": threshold,
        "threshold_source": "validation exact unique-probability MCC search",
        "checkpoint_path": str(drive_output / "checkpoints" / "best_validation_mcc.pt"),
        "resume_source": resume_source,
        "environment": env,
        "token_length_audit": token_report,
        "test_evaluation_policy": "test evaluated once after validation candidate selection",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--drive-data-dir", type=Path, required=True)
    parser.add_argument("--local-data-dir", type=Path, required=True)
    parser.add_argument("--drive-output-dir", type=Path, required=True)
    parser.add_argument("--local-output-dir", type=Path, required=True)
    parser.add_argument("--resume", choices=("latest", "best", "none"), default="latest")
    parser.add_argument("--max-length", type=int, default=104)
    parser.add_argument("--run-max-length-128", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--loss-mode", choices=("bce", "focal"), default="bce")
    parser.add_argument("--max-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--head-only-epochs", type=int, default=1)
    parser.add_argument("--unfreeze-top-layers", type=int, default=4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    args = parser.parse_args()

    if args.max_epochs < 1 or args.patience < 1 or args.head_only_epochs < 0:
        raise ValueError("max_epochs and patience must be positive; head_only_epochs cannot be negative.")
    if args.unfreeze_top_layers < 1 or args.batch_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("unfreeze_top_layers, batch_size, and gradient_accumulation must be positive.")
    if not 0.0 <= args.dropout < 1.0 or not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("dropout and warmup_ratio must be in [0, 1).")
    if args.head_learning_rate <= 0.0 or args.encoder_learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("learning rates must be positive and weight_decay cannot be negative.")

    sys.path.insert(0, str(args.repo_dir / "src"))
    helper_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(helper_dir))
    import torch
    from torch import nn
    from torch.optim.lr_scheduler import LambdaLR

    seed_everything(42, torch)
    run_started = time.perf_counter()
    args.local_output_dir.mkdir(parents=True, exist_ok=True)
    args.drive_output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoints", "plots", "logs"):
        (args.local_output_dir / name).mkdir(parents=True, exist_ok=True)
        (args.drive_output_dir / name).mkdir(parents=True, exist_ok=True)

    audit, frames = audit_and_stage(
        args.drive_data_dir,
        args.local_data_dir,
        args.drive_output_dir / "input_split_audit.json",
    )
    atomic_copy(args.drive_output_dir / "input_split_audit.json", args.local_output_dir / "input_split_audit.json")

    commit = os.popen(f"git -C {args.repo_dir} rev-parse HEAD").read().strip()
    env = {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    write_json_atomic(env, args.local_output_dir / "environment.json")
    atomic_copy(args.local_output_dir / "environment.json", args.drive_output_dir / "environment.json")
    config = {
        "branch": "issue-3-all-model-baselines",
        "commit": commit,
        "max_length": args.max_length,
        "run_max_length_128_candidate": bool(args.run_max_length_128),
        "model": "zhihan1996/DNABERT-2-117M",
        "revision": "7bce263b15377fc15361f52cfab88f8b586abda0",
        "loss": args.loss_mode,
        "seed": 42,
        "training": {
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "head_only_epochs": args.head_only_epochs,
            "unfreeze_top_layers": args.unfreeze_top_layers,
            "head_learning_rate": args.head_learning_rate,
            "encoder_learning_rate": args.encoder_learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "warmup_ratio": args.warmup_ratio,
        },
    }
    write_json_atomic(config, args.local_output_dir / "config.json")
    atomic_copy(args.local_output_dir / "config.json", args.drive_output_dir / "config.json")

    from seqtrainer.torch.dnabert2_benchmark import _enable_gradient_checkpointing, _load_huggingface_dnabert2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, encoder = _load_huggingface_dnabert2(
        "zhihan1996/DNABERT-2-117M",
        device=device,
        trust_remote_code=True,
        local_files_only=False,
        disable_flash_attention=True,
        revision="7bce263b15377fc15361f52cfab88f8b586abda0",
    )
    token_report = token_audit(tokenizer, frames, args.local_output_dir)
    atomic_copy(args.local_output_dir / "token_length_audit.json", args.drive_output_dir / "token_length_audit.json")
    if (args.local_output_dir / "token_length_histogram.png").exists():
        atomic_copy(args.local_output_dir / "token_length_histogram.png", args.drive_output_dir / "token_length_histogram.png")

    max_length = 128 if args.run_max_length_128 else args.max_length
    if args.run_max_length_128:
        print("Optional 128-token candidate enabled. It is compared by validation MCC/AUPRC only.")
    model = DnaBertHead(encoder, torch, nn, dropout=args.dropout).to(device)
    _enable_gradient_checkpointing(encoder)
    train_ratio = float(frames["train"]["label"].sum()) / len(frames["train"])
    negative_positive_ratio = (1.0 - train_ratio) / max(train_ratio, 1e-12)
    use_pos_weight = negative_positive_ratio > 1.20 or negative_positive_ratio < 0.83
    pos_weight = torch.tensor([negative_positive_ratio], device=device, dtype=torch.float32)
    if args.loss_mode == "focal":
        criterion = lambda logits, labels: focal_loss(logits, labels, 1.5, torch)
        loss_description = "focal gamma=1.5"
    elif use_pos_weight:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_description = f"BCEWithLogitsLoss pos_weight={negative_positive_ratio:.6f}"
    else:
        criterion = nn.BCEWithLogitsLoss()
        loss_description = "BCEWithLogitsLoss unweighted; measured ratio within balance limits"
    config["loss"] = loss_description
    write_json_atomic(config, args.local_output_dir / "config.json")
    atomic_copy(args.local_output_dir / "config.json", args.drive_output_dir / "config.json")

    loaders = {
        split: make_loader(
            frame,
            tokenizer,
            max_length,
            batch_size=args.batch_size,
            torch=torch,
            shuffle=(split == "train"),
            workers=2,
        )
        for split, frame in frames.items()
    }
    prediction_loaders = {
        split: make_loader(
            frame,
            tokenizer,
            max_length,
            batch_size=args.batch_size,
            torch=torch,
            shuffle=False,
            workers=2,
        )
        for split, frame in frames.items()
    }
    accumulation = args.gradient_accumulation
    max_epochs = args.max_epochs
    patience = args.patience
    train_steps = max(1, math.ceil(len(loaders["train"]) / accumulation))
    total_steps = max_epochs * train_steps
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    parameters = [
        {"params": [p for p in encoder.parameters()], "lr": args.encoder_learning_rate},
        {"params": [p for name, p in model.named_parameters() if not name.startswith("encoder.")], "lr": args.head_learning_rate},
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=args.weight_decay)
    scheduler = LambdaLR(
        optimizer,
        lambda step: min(1.0, step / max(warmup_steps, 1))
        if step < warmup_steps
        else 0.5 * (1.0 + np.cos(np.pi * min(1.0, (step - warmup_steps) / max(total_steps - warmup_steps, 1)))),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    use_amp = torch.cuda.is_available()
    local_latest = args.local_output_dir / "checkpoints" / "latest.pt"
    drive_latest = args.drive_output_dir / "checkpoints" / "latest.pt"
    local_best = args.local_output_dir / "checkpoints" / "best_validation_mcc.pt"
    drive_best = args.drive_output_dir / "checkpoints" / "best_validation_mcc.pt"
    history_path = args.local_output_dir / "history.csv"
    drive_history = args.drive_output_dir / "history.csv"
    if drive_history.exists() and not history_path.exists():
        atomic_copy(drive_history, history_path)
    start_epoch = 1
    global_step = 0
    best = {"epoch": 0, "mcc": float("-inf"), "auprc": float("-inf"), "threshold": 0.5}
    resume_source = "none"
    if args.resume == "none":
        for stale_path in (local_latest, local_best, history_path):
            stale_path.unlink(missing_ok=True)
    selected_resume = drive_latest if args.resume == "latest" else drive_best
    local_resume = local_latest if args.resume == "latest" else local_best
    if args.resume != "none" and selected_resume.exists():
        state = restore_checkpoint(selected_resume, local_resume, torch)
        saved_config = state.get("config", {})
        resume_keys = ("max_length", "model", "revision", "loss", "training")
        if any(saved_config.get(key) != config.get(key) for key in resume_keys):
            raise RuntimeError(
                "The requested profile differs from the checkpoint profile. "
                "Use --resume none or choose a new output directory instead of mixing runs."
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best = state["best"]
        if "rng" in state:
            restore_rng_state(state["rng"], torch)
        resume_source = str(selected_resume)
        print("Resuming from", selected_resume)

    bad_epochs = 0
    for epoch in range(start_epoch, max_epochs + 1):
        stage = set_stage(encoder, epoch, args.head_only_epochs, args.unfreeze_top_layers)
        started = time.perf_counter()
        try:
            train_loss, optimizer_steps = run_epoch(model, loaders["train"], optimizer, scheduler, scaler, criterion, device, torch, use_amp, accumulation, 1.0)
        except RuntimeError as exc:
            if "worker" not in str(exc).lower():
                raise
            print("DataLoader workers failed; retrying this epoch with num_workers=0.")
            loaders = {
                split: make_loader(frame, tokenizer, max_length, args.batch_size, torch, split == "train", workers=0)
                for split, frame in frames.items()
            }
            prediction_loaders = {
                split: make_loader(frame, tokenizer, max_length, args.batch_size, torch, False, workers=0)
                for split, frame in frames.items()
            }
            train_loss, optimizer_steps = run_epoch(model, loaders["train"], optimizer, scheduler, scaler, criterion, device, torch, use_amp, accumulation, 1.0)
        global_step += optimizer_steps
        labels, probabilities, validation_loss = evaluate(model, prediction_loaders["validation"], criterion, device, torch, use_amp)
        threshold, validation_mcc, validation_auprc = exact_mcc_threshold(labels, probabilities)
        improved = validation_mcc > best["mcc"] or (
            validation_mcc == best["mcc"] and validation_auprc > best["auprc"]
        )
        if improved:
            best = {"epoch": epoch, "mcc": validation_mcc, "auprc": validation_auprc, "threshold": threshold}
            bad_epochs = 0
        else:
            bad_epochs += 1
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "stage": stage,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_mcc": validation_mcc,
            "validation_auprc": validation_auprc,
            "validation_threshold": threshold,
            "learning_rate_encoder": optimizer.param_groups[0]["lr"],
            "learning_rate_head": optimizer.param_groups[1]["lr"],
            "epoch_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if use_amp else None,
            "is_best": int(improved),
        }
        append_history(row, history_path, drive_history)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best": best,
            "config": config,
            "rng": rng_state(torch),
        }
        save_checkpoint(state, local_latest, drive_latest, torch)
        if improved:
            save_checkpoint(state, local_best, drive_best, torch)
        print(f"epoch={epoch} stage={stage} val_mcc={validation_mcc:.6f} val_auprc={validation_auprc:.6f} threshold={threshold:.6f}")
        if bad_epochs >= patience:
            print("Early stopping after validation MCC patience", patience)
            break

    if not local_best.exists():
        if not drive_best.exists():
            raise RuntimeError("No validation-selected DNABERT2 checkpoint was produced.")
        state = restore_checkpoint(drive_best, local_best, torch)
    else:
        state = torch.load(local_best, map_location="cpu")
    model.load_state_dict(state["model"])
    threshold = float(state["best"]["threshold"])
    metrics = {}
    prediction_rows = []
    report_splits = ("train", "validation") if args.selection_only else ("train", "validation", "test")
    for split in report_splits:
        labels, probabilities, loss = evaluate(model, prediction_loaders[split], criterion, device, torch, use_amp)
        metrics[split] = classification_metrics(labels, probabilities, threshold, loss=loss)
        for index, (label, probability) in enumerate(zip(labels, probabilities)):
            prediction_rows.append({
                "split": split,
                "idx": index,
                "sequence": frames[split].iloc[index]["sequence"],
                "label": int(label),
                "probability": float(probability),
                "threshold": threshold,
                "prediction": int(probability >= threshold),
            })
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(args.local_output_dir / "predictions.csv", index=False)
    flatten_metrics(metrics).to_csv(args.local_output_dir / "metrics.csv", index=False)
    write_json_atomic(metrics, args.local_output_dir / "metrics.json")
    manifest = build_manifest(config, audit, env, state["best"], threshold, args.drive_output_dir, resume_source, token_report)
    manifest["loss_decision"] = {
        "loss_mode": args.loss_mode,
        "negative_positive_ratio": negative_positive_ratio,
        "class_weighting_applied": use_pos_weight,
    }
    manifest["selection_only"] = args.selection_only
    manifest["runtime_seconds"] = time.perf_counter() - run_started
    manifest["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2 if use_amp else None
    write_json_atomic(manifest, args.local_output_dir / "manifest.json")
    history_source = args.local_output_dir / "history.csv"
    for name in ("predictions.csv", "metrics.csv", "metrics.json", "manifest.json"):
        atomic_copy(args.local_output_dir / name, args.drive_output_dir / name)
    if history_source.exists():
        atomic_copy(history_source, args.drive_output_dir / "history.csv")
    try:
        import matplotlib.pyplot as plt

        history = pd.read_csv(history_source) if history_source.stat().st_size else pd.DataFrame()
        if not history.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(history["epoch"], history["train_loss"], label="train")
            axes[0].plot(history["epoch"], history["validation_loss"], label="validation")
            axes[0].legend()
            axes[0].set_title("Loss")
            axes[1].plot(history["epoch"], history["validation_mcc"], label="validation MCC")
            axes[1].plot(history["epoch"], history["validation_auprc"], label="validation AUPRC")
            axes[1].legend()
            axes[1].set_title("Validation selection metrics")
            fig.tight_layout()
            fig.savefig(args.local_output_dir / "plots" / "learning_curves.png", dpi=160)
            plt.close(fig)
            atomic_copy(args.local_output_dir / "plots" / "learning_curves.png", args.drive_output_dir / "plots" / "learning_curves.png")
    except Exception as exc:
        print("Learning-curve plot skipped:", exc)
    verify_required_outputs(args.local_output_dir)
    print("Final DNABERT2 artifacts:", args.drive_output_dir)


if __name__ == "__main__":
    main()
