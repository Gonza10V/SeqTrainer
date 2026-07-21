"""Final T4 iPro-MP five-fold inference and train-fitted ensemble runner."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from final_training_common import (
    SPLIT_FILES,
    append_history,
    atomic_copy,
    audit_and_stage,
    classification_metrics,
    exact_mcc_threshold,
    flatten_metrics,
    sha256_file,
    verify_required_outputs,
    write_json_atomic,
)


def write_fasta_and_mapping(frames, output_dir):
    fasta_dir = output_dir / "ipromp_fasta"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    mapping_rows = []
    for split, frame in frames.items():
        fasta_path = fasta_dir / f"{split}.fasta"
        with fasta_path.open("w", encoding="utf-8") as handle:
            for idx, row in frame.reset_index(drop=True).iterrows():
                sequence_id = f"{split}_{idx:06d}"
                sequence = str(row["sequence"]).upper().replace("U", "T")
                handle.write(f">{sequence_id}|split={split}|sequence_id={sequence_id}\n{sequence}\n")
                mapping_rows.append({
                    "split": split,
                    "idx": idx,
                    "row_index": idx,
                    "sequence_id": sequence_id,
                    "label": int(row["label"]),
                    "sequence": sequence,
                })
    mapping = pd.DataFrame(mapping_rows)
    mapping.to_csv(output_dir / "ipromp_id_mapping.csv", index=False)
    return fasta_dir, mapping


def load_records(path, split):
    from seqtrainer.adapters.ipromp_inference import read_seqtrainer_fasta

    return read_seqtrainer_fasta(path, expected_split=split)


def build_fold_model(dnabert_dir, torch, nn):
    from transformers import BertModel

    class PromoterClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = BertModel.from_pretrained(str(dnabert_dir))
            self.dropout = nn.Dropout(p=0.3)
            self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)
            self.layer_norm1 = nn.LayerNorm(512)
            self.fc2 = nn.Linear(512, 256)
            self.layer_norm2 = nn.LayerNorm(256)
            self.classifier = nn.Linear(256, 2)
            self.activation = nn.GELU()

        def forward(self, input_ids, attention_mask):
            hidden = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state[:, 0, :]
            hidden = self.dropout(hidden)
            hidden = self.layer_norm1(self.activation(self.fc1(hidden)))
            hidden = self.dropout(hidden)
            hidden = self.layer_norm2(self.activation(self.fc2(hidden)))
            return self.classifier(self.dropout(hidden))

    return PromoterClassifier()


def normalize_state_dict(raw_state):
    from seqtrainer.adapters.ipromp_inference import normalize_state_dict as normalize

    return normalize(raw_state)


def run_one_fold(records, tokenizer, checkpoint, dnabert_dir, batch_size, max_length, device, torch, nn):
    from torch.utils.data import DataLoader, Dataset

    class SequenceDataset(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            sequence = records[index].sequence
            kmers = [sequence[i : i + 6] for i in range(len(sequence) - 5)]
            encoded = tokenizer(
                kmers,
                is_split_into_words=True,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }

    loader = DataLoader(
        SequenceDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = build_fold_model(dnabert_dir, torch, nn)
    try:
        raw_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        raw_state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(normalize_state_dict(raw_state), strict=True)
    model.to(device).eval()
    logits = []
    with torch.inference_mode():
        for batch in loader:
            output = model(
                input_ids=batch["input_ids"].to(device, non_blocking=True),
                attention_mask=batch["attention_mask"].to(device, non_blocking=True),
            )
            logits.extend((output[:, 1] - output[:, 0]).float().cpu().numpy().tolist())
    logits = np.asarray(logits, dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    del model, raw_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return logits, probabilities


def valid_cache(logits_path, metadata_path, expected_rows, input_sha):
    if not logits_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        values = np.load(logits_path)
        return (
            int(metadata["rows"]) == expected_rows
            and metadata["input_sha256"] == input_sha
            and len(values) == expected_rows
        )
    except Exception:
        return False


def candidate_table(fold_probabilities, fold_logits, labels):
    probabilities = np.column_stack(fold_probabilities)
    logits = np.column_stack(fold_logits)
    features = np.column_stack(
        [
            logits,
            logits.mean(axis=1),
            np.median(logits, axis=1),
            logits.std(axis=1),
            logits.min(axis=1),
            logits.max(axis=1),
        ]
    )
    rows = []
    candidates = {
        "official_five_fold_mean": probabilities.mean(axis=1),
        "median": np.median(probabilities, axis=1),
        "trimmed_mean": np.sort(probabilities, axis=1)[:, 1:-1].mean(axis=1),
    }
    for name, values in candidates.items():
        threshold, mcc, auprc = exact_mcc_threshold(labels, values)
        rows.append({
            "strategy": name,
            "C": None,
            "validation_mcc": mcc,
            "validation_auprc": auprc,
            "validation_threshold": threshold,
            "complexity_rank": {"official_five_fold_mean": 0, "median": 1, "trimmed_mean": 2}[name],
        })
    return pd.DataFrame(rows), candidates, features, {}


def probabilities_for_strategy(name, candidates, features, models):
    if name in candidates:
        return candidates[name]
    return models[name].predict_proba(features)[:, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--drive-data-dir", type=Path, required=True)
    parser.add_argument("--local-data-dir", type=Path, required=True)
    parser.add_argument("--drive-output-dir", type=Path, required=True)
    parser.add_argument("--local-output-dir", type=Path, required=True)
    parser.add_argument("--drive-model-cache", type=Path, required=True)
    parser.add_argument("--local-model-root", type=Path, required=True)
    parser.add_argument("--resume", choices=("latest", "best", "none"), default="latest")
    parser.add_argument("--run-train-split", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_dir / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    from torch import nn
    from transformers import BertTokenizer

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    args.local_output_dir.mkdir(parents=True, exist_ok=True)
    args.drive_output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("fold_predictions", "checkpoints", "plots", "logs"):
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
        "dataset_files": SPLIT_FILES,
        "seed": 42,
        "species_id": 10,
        "folds": 5,
        "dnabert6": "local model cache",
        "kmer_size": 6,
        "max_length": 300,
        "inference_batch_size": 4,
        "selection": "validation MCC, then validation AUPRC, then simpler strategy",
        "test_policy": "evaluate selected ensemble once",
    }
    write_json_atomic(config, args.local_output_dir / "config.json")
    atomic_copy(args.local_output_dir / "config.json", args.drive_output_dir / "config.json")

    model_root = args.local_model_root
    ipromp_model_dir = model_root / "ipromp_ecoli"
    dnabert_dir = model_root / "DNABERT-6"
    checkpoints = [ipromp_model_dir / f"10_fold_{fold}.pth" for fold in range(1, 6)]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    missing += [str(dnabert_dir / name) for name in ("config.json", "pytorch_model.bin", "vocab.txt") if not (dnabert_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "iPro-MP local model cache is incomplete. Restore the official five folds and "
            f"DNABERT-6 from Drive first. Missing: {missing}"
        )

    fasta_dir, mapping = write_fasta_and_mapping(frames, args.local_output_dir)
    mapping.to_csv(args.drive_output_dir / "ipromp_id_mapping.csv", index=False)
    tokenizer = BertTokenizer.from_pretrained(str(dnabert_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Train and validation are materialized before candidate selection. Test
    # inference begins only after the ensemble and threshold are locked.
    splits = ["train", "validation"]
    all_logits = {split: [] for split in splits}
    all_probabilities = {split: [] for split in splits}
    history_path = args.local_output_dir / "history.csv"
    drive_history = args.drive_output_dir / "history.csv"

    for split in splits:
        records = load_records(fasta_dir / f"{split}.fasta", split)
        input_sha = sha256_file(fasta_dir / f"{split}.fasta")
        for fold, checkpoint in enumerate(checkpoints, start=1):
            local_logits = args.local_output_dir / "fold_predictions" / f"fold_{fold}_{split}_logits.npy"
            drive_logits = args.drive_output_dir / "fold_predictions" / local_logits.name
            local_metadata = local_logits.with_suffix(".metadata.json")
            drive_metadata = drive_logits.with_suffix(".metadata.json")
            elapsed = 0.0
            if valid_cache(local_logits, local_metadata, len(records), input_sha):
                logits = np.load(local_logits)
                cache_status = "local_restored"
            elif valid_cache(drive_logits, drive_metadata, len(records), input_sha):
                atomic_copy(drive_logits, local_logits)
                atomic_copy(drive_metadata, local_metadata)
                logits = np.load(local_logits)
                cache_status = "drive_restored"
            else:
                started = time.perf_counter()
                logits, _ = run_one_fold(
                    records, tokenizer, checkpoint, dnabert_dir, 4, 300, device, torch, nn
                )
                np.save(local_logits, logits)
                metadata = {
                    "rows": len(records),
                    "input_sha256": input_sha,
                    "fold": fold,
                    "split": split,
                    "model_checkpoint": str(checkpoint),
                }
                write_json_atomic(metadata, local_metadata)
                atomic_copy(local_logits, drive_logits)
                atomic_copy(local_metadata, drive_metadata)
                cache_status = "computed"
                elapsed = time.perf_counter() - started
            all_logits[split].append(np.asarray(logits, dtype=np.float64))
            all_probabilities[split].append(1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64))))
            append_history({
                "stage": "fold_inference",
                "split": split,
                "fold": fold,
                "rows": len(records),
                "runtime_seconds": elapsed,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
                "cache_status": cache_status,
            }, history_path, drive_history)
        fold_frame = pd.DataFrame({"split": split, "idx": np.arange(len(records))})
        for fold, values in enumerate(all_probabilities[split], start=1):
            fold_frame[f"fold_{fold}_probability"] = values
            fold_frame[f"fold_{fold}_logit"] = all_logits[split][fold - 1]
        fold_frame.to_csv(args.local_output_dir / "fold_predictions" / f"{split}_fold_predictions.csv", index=False)
        atomic_copy(
            args.local_output_dir / "fold_predictions" / f"{split}_fold_predictions.csv",
            args.drive_output_dir / "fold_predictions" / f"{split}_fold_predictions.csv",
        )

    labels = {split: frames[split]["label"].to_numpy(dtype=int) for split in splits}
    candidate_input = {
        split: (
            all_probabilities[split],
            all_logits[split],
            labels[split],
        )
        for split in splits
    }
    # Candidate selection sees train and validation only. Test probabilities are
    # not materialized until the selected ensemble is locked.
    train_candidate = candidate_table(*candidate_input["train"]) if "train" in candidate_input else None
    validation_candidate = candidate_table(
        all_probabilities["validation"],
        all_logits["validation"],
        labels["validation"],
    )
    # Fit stackers on train and score validation in one deterministic block.
    train_logits = np.column_stack(all_logits.get("train", all_logits["validation"]))
    train_probabilities = np.column_stack(all_probabilities.get("train", all_probabilities["validation"]))
    validation_logits = np.column_stack(all_logits["validation"])
    validation_probabilities = np.column_stack(all_probabilities["validation"])
    feature_sets = {}
    for split_name, logits, probabilities in (
        ("train", train_logits, train_probabilities),
        ("validation", validation_logits, validation_probabilities),
    ):
        feature_sets[split_name] = np.column_stack([
            logits,
            logits.mean(axis=1),
            np.median(logits, axis=1),
            logits.std(axis=1),
            logits.min(axis=1),
            logits.max(axis=1),
        ])
    rows = []
    strategies = {
        "official_five_fold_mean": validation_probabilities.mean(axis=1),
        "median": np.median(validation_probabilities, axis=1),
        "trimmed_mean": np.sort(validation_probabilities, axis=1)[:, 1:-1].mean(axis=1),
    }
    models = {}
    for name, values in strategies.items():
        threshold, mcc, auprc = exact_mcc_threshold(labels["validation"], values)
        rows.append({"strategy": name, "C": None, "validation_mcc": mcc, "validation_auprc": auprc, "validation_threshold": threshold, "complexity_rank": {"official_five_fold_mean": 0, "median": 1, "trimmed_mean": 2}[name]})
    for c_value in (0.01, 0.1, 1.0, 10.0):
        stacker = LogisticRegression(C=c_value, solver="liblinear", random_state=42, max_iter=2000)
        stacker.fit(feature_sets["train"], labels.get("train", labels["validation"]))
        values = stacker.predict_proba(feature_sets["validation"])[:, 1]
        threshold, mcc, auprc = exact_mcc_threshold(labels["validation"], values)
        name = f"logistic_stacker_C_{c_value:g}"
        models[name] = stacker
        rows.append({"strategy": name, "C": c_value, "validation_mcc": mcc, "validation_auprc": auprc, "validation_threshold": threshold, "complexity_rank": 3})
    candidate_frame = pd.DataFrame(rows).sort_values(
        ["validation_mcc", "validation_auprc", "complexity_rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    candidate_frame.to_csv(args.local_output_dir / "ensemble_candidates.csv", index=False)
    atomic_copy(args.local_output_dir / "ensemble_candidates.csv", args.drive_output_dir / "ensemble_candidates.csv")
    for candidate in candidate_frame.to_dict(orient="records"):
        append_history({"stage": "ensemble_candidate", "split": "validation", "fold": None, "rows": len(labels["validation"]), "runtime_seconds": None, "peak_gpu_memory_mb": None, "cache_status": "evaluated", **candidate}, history_path, drive_history)
    selected = candidate_frame.iloc[0].to_dict()
    selected_name = str(selected["strategy"])
    selected_threshold = float(selected["validation_threshold"])

    # Test fold inference is deliberately delayed until after validation-only
    # ensemble and threshold selection. Fold-level files remain resumable.
    split = "test"
    records = load_records(fasta_dir / "test.fasta", split)
    input_sha = sha256_file(fasta_dir / "test.fasta")
    all_logits[split] = []
    all_probabilities[split] = []
    for fold, checkpoint in enumerate(checkpoints, start=1):
        local_logits = args.local_output_dir / "fold_predictions" / f"fold_{fold}_{split}_logits.npy"
        drive_logits = args.drive_output_dir / "fold_predictions" / local_logits.name
        local_metadata = local_logits.with_suffix(".metadata.json")
        drive_metadata = drive_logits.with_suffix(".metadata.json")
        elapsed = 0.0
        if valid_cache(local_logits, local_metadata, len(records), input_sha):
            logits = np.load(local_logits)
            cache_status = "local_restored"
        elif valid_cache(drive_logits, drive_metadata, len(records), input_sha):
            atomic_copy(drive_logits, local_logits)
            atomic_copy(drive_metadata, local_metadata)
            logits = np.load(local_logits)
            cache_status = "drive_restored"
        else:
            started = time.perf_counter()
            logits, _ = run_one_fold(records, tokenizer, checkpoint, dnabert_dir, 4, 300, device, torch, nn)
            np.save(local_logits, logits)
            metadata = {"rows": len(records), "input_sha256": input_sha, "fold": fold, "split": split, "model_checkpoint": str(checkpoint)}
            write_json_atomic(metadata, local_metadata)
            atomic_copy(local_logits, drive_logits)
            atomic_copy(local_metadata, drive_metadata)
            cache_status = "computed"
            elapsed = time.perf_counter() - started
        all_logits[split].append(np.asarray(logits, dtype=np.float64))
        all_probabilities[split].append(1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64))))
        append_history({"stage": "fold_inference", "split": split, "fold": fold, "rows": len(records), "runtime_seconds": elapsed, "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None, "cache_status": cache_status}, history_path, drive_history)
    test_fold_frame = pd.DataFrame({"split": split, "idx": np.arange(len(records))})
    for fold, values in enumerate(all_probabilities[split], start=1):
        test_fold_frame[f"fold_{fold}_probability"] = values
        test_fold_frame[f"fold_{fold}_logit"] = all_logits[split][fold - 1]
    test_fold_frame.to_csv(args.local_output_dir / "fold_predictions" / "test_fold_predictions.csv", index=False)
    atomic_copy(args.local_output_dir / "fold_predictions" / "test_fold_predictions.csv", args.drive_output_dir / "fold_predictions" / "test_fold_predictions.csv")
    if selected_name.startswith("logistic_stacker"):
        import joblib

        joblib.dump(models[selected_name], args.local_output_dir / "stacker.joblib")
        atomic_copy(args.local_output_dir / "stacker.joblib", args.drive_output_dir / "stacker.joblib")

    def chosen_probability(split, probability_values, logit_values):
        probability_matrix = np.column_stack(probability_values)
        logit_matrix = np.column_stack(logit_values)
        if selected_name == "official_five_fold_mean":
            return probability_matrix.mean(axis=1)
        if selected_name == "median":
            return np.median(probability_matrix, axis=1)
        if selected_name == "trimmed_mean":
            return np.sort(probability_matrix, axis=1)[:, 1:-1].mean(axis=1)
        features = np.column_stack([
            logit_matrix,
            logit_matrix.mean(axis=1),
            np.median(logit_matrix, axis=1),
            logit_matrix.std(axis=1),
            logit_matrix.min(axis=1),
            logit_matrix.max(axis=1),
        ])
        if split == "test":
            return models[selected_name].predict_proba(features)[:, 1]
        return models[selected_name].predict_proba(features)[:, 1]

    metrics = {}
    predictions = []
    for split in ("train", "validation", "test"):
        if split not in all_probabilities:
            # Train inference is only needed for the stacker; use the
            # validation model if RUN_TRAIN_SPLIT=False and report no train row.
            continue
        values = chosen_probability(split, all_probabilities[split], all_logits[split])
        metrics[split] = classification_metrics(
            labels[split],
            values,
            selected_threshold,
        )
        for idx, probability in enumerate(values):
            predictions.append({
                "split": split,
                "idx": idx,
                "sequence_id": f"{split}_{idx:06d}",
                "label": int(labels[split][idx]),
                "probability": float(probability),
                "threshold": selected_threshold,
                "prediction": int(probability >= selected_threshold),
            })
    predictions_frame = pd.DataFrame(predictions)
    predictions_frame.to_csv(args.local_output_dir / "predictions.csv", index=False)
    flatten_metrics(metrics).to_csv(args.local_output_dir / "metrics.csv", index=False)
    write_json_atomic(metrics, args.local_output_dir / "metrics.json")
    write_json_atomic({
        "strategy": selected_name,
        "C": selected.get("C"),
        "validation_mcc": selected["validation_mcc"],
        "validation_auprc": selected["validation_auprc"],
        "validation_threshold": selected_threshold,
        "test_metrics_were_not_used_for_selection": True,
    }, args.local_output_dir / "selected_ensemble.json")
    manifest = {
        "repository": "https://github.com/simplyshree/SeqTrainer",
        "branch": "issue-3-all-model-baselines",
        "commit": commit,
        "notebook": "notebooks/final_training/ipromp_final_training_t4_colab.ipynb",
        "dataset": audit,
        "model": {
            "official_model": "iPro-MP E. coli species ID 10",
            "dnabert_backbone": "DNABERT-6",
            "folds": 5,
            "kmer_size": 6,
            "max_length": 300,
            "batch_size": 4,
            "inference_only_official_baseline": True,
            "ensemble": selected_name,
        },
        "seed": 42,
        "selection": "validation MCC, then validation AUPRC, then simpler strategy",
        "threshold_source": "validation exact unique-probability MCC search",
        "test_policy": "selected ensemble evaluated once after selection",
        "environment": env,
        "fold_checkpoints": [str(path) for path in checkpoints],
    }
    write_json_atomic(manifest, args.local_output_dir / "manifest.json")
    try:
        import matplotlib.pyplot as plt

        plot_frame = candidate_frame.sort_values("validation_mcc", ascending=True)
        plt.figure(figsize=(9, 5))
        plt.barh(plot_frame["strategy"], plot_frame["validation_mcc"], color="#39708f")
        plt.xlabel("Validation MCC")
        plt.title("iPro-MP ensemble candidates selected without test metrics")
        plt.tight_layout()
        plt.savefig(args.local_output_dir / "plots" / "ensemble_validation_mcc.png", dpi=160)
        plt.close()
        atomic_copy(args.local_output_dir / "plots" / "ensemble_validation_mcc.png", args.drive_output_dir / "plots" / "ensemble_validation_mcc.png")
    except Exception as exc:
        print("Ensemble plot skipped:", exc)
    for name in ("predictions.csv", "metrics.csv", "metrics.json", "manifest.json", "selected_ensemble.json", "ipromp_id_mapping.csv"):
        atomic_copy(args.local_output_dir / name, args.drive_output_dir / name)
    verify_required_outputs(args.local_output_dir, include_embeddings=True)
    print("Final iPro-MP artifacts:", args.drive_output_dir)


if __name__ == "__main__":
    main()
