"""Offline iPro-MP/iPromoter E. coli ensemble inference.

This module intentionally performs no downloads. It expects a staged DNABERT-6
directory and five local E. coli fold checkpoints named ``10_fold_1.pth`` ...
``10_fold_5.pth``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class FastaRecord:
    """A FASTA sequence with its original identifier preserved."""

    id: str
    sequence: str


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_ipromp_offline_inference(
        input_fasta=args.input_fasta,
        output_csv=args.output_csv,
        dnabert_dir=args.dnabert_dir,
        model_dir=args.model_dir,
        species_id=args.species_id,
        token_max_length=args.token_max_length,
        batch_size=args.batch_size,
        device=args.device,
        metadata_json=args.metadata_json,
    )
    return 0


def run_ipromp_offline_inference(
    *,
    input_fasta: str | Path,
    output_csv: str | Path,
    dnabert_dir: str | Path,
    model_dir: str | Path,
    species_id: int = 10,
    token_max_length: int = 128,
    batch_size: int = 16,
    device: str = "auto",
    metadata_json: str | Path | None = None,
) -> dict[str, Any]:
    """Run the five-fold iPro-MP E. coli ensemble and write probabilities."""
    try:
        import pandas as pd
        import torch
        from torch import nn
        from transformers import AutoModel, BertTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - optional heavy deps
        raise RuntimeError(
            "iPro-MP offline inference requires pandas, torch, and transformers in the container."
        ) from exc

    records = read_fasta(input_fasta)
    fold_paths = expected_fold_paths(model_dir, species_id=species_id)
    validate_ipromp_assets(dnabert_dir=dnabert_dir, fold_paths=fold_paths)

    resolved_device = _resolve_device(device, torch)
    tokenizer = BertTokenizer.from_pretrained(str(dnabert_dir), do_lower_case=False, local_files_only=True)
    probability_by_fold: list[np.ndarray] = []
    checkpoint_names: list[str] = []

    for fold_path in fold_paths:
        checkpoint_names.append(fold_path.name)
        encoder = AutoModel.from_pretrained(str(dnabert_dir), local_files_only=True)
        classifier = IProMPClassifier(encoder, hidden_size=int(getattr(encoder.config, "hidden_size", 768)))
        _load_fold_state(classifier, fold_path, torch)
        classifier.to(resolved_device)
        classifier.eval()
        probability_by_fold.append(
            _predict_fold(
                classifier,
                tokenizer,
                records,
                token_max_length=token_max_length,
                batch_size=batch_size,
                device=resolved_device,
                torch=torch,
            )
        )
        del classifier

    probabilities = average_fold_probabilities(probability_by_fold)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": [record.id for record in records],
            "probability": probabilities,
            "prediction_at_0_5": (probabilities >= 0.5).astype(int),
        }
    ).to_csv(output_path, index=False)

    metadata = truncation_metadata(
        records,
        token_max_length=token_max_length,
        species_id=species_id,
        checkpoint_names=checkpoint_names,
    )
    metadata["output_csv"] = str(output_path)
    if metadata_json is not None:
        metadata_path = Path(metadata_json)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


class IProMPClassifier:
    """DNABERT-6 encoder plus the iPro-MP binary classification head."""

    def __init__(self, encoder: Any, hidden_size: int, dropout: float = 0.3) -> None:
        from torch import nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = encoder
                self.dropout = nn.Dropout(dropout)
                self.fc1 = nn.Linear(hidden_size, 512)
                self.act1 = nn.GELU()
                self.norm1 = nn.LayerNorm(512)
                self.fc2 = nn.Linear(512, 256)
                self.act2 = nn.GELU()
                self.norm2 = nn.LayerNorm(256)
                self.classifier = nn.Linear(256, 2)

            def forward(self, input_ids: Any, attention_mask: Any) -> Any:
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                x = self.dropout(pooled)
                x = self.norm1(self.act1(self.fc1(x)))
                x = self.dropout(x)
                x = self.norm2(self.act2(self.fc2(x)))
                x = self.dropout(x)
                return self.classifier(x)

        self._model = _Model()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)


def read_fasta(path: str | Path) -> list[FastaRecord]:
    """Read FASTA records without reordering them."""
    records: list[FastaRecord] = []
    current_id: str | None = None
    chunks: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append(FastaRecord(current_id, _normalise_sequence("".join(chunks))))
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        records.append(FastaRecord(current_id, _normalise_sequence("".join(chunks))))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def sequence_to_kmer_text(sequence: str, *, k: int = 6) -> str:
    """Convert a DNA sequence to overlapping k-mer tokens for DNABERT-6."""
    sequence = _normalise_sequence(sequence)
    if len(sequence) < k:
        return sequence
    return " ".join(sequence[index : index + k] for index in range(len(sequence) - k + 1))


def truncation_metadata(
    records: Iterable[FastaRecord],
    *,
    token_max_length: int,
    species_id: int,
    checkpoint_names: list[str],
) -> dict[str, Any]:
    lengths = [len(record.sequence) for record in records]
    token_counts = [max(1, len(record.sequence) - 6 + 1) for record in records]
    rows_truncated = sum(count > token_max_length for count in token_counts)
    total = len(token_counts)
    return {
        "species_id": int(species_id),
        "fold_checkpoint_names": checkpoint_names,
        "ensemble_method": "mean_positive_class_probability",
        "tokenizer": "DNABERT-6 overlapping 6-mer",
        "token_max_length": int(token_max_length),
        "rows": int(total),
        "rows_truncated": int(rows_truncated),
        "percent_truncated": float(100.0 * rows_truncated / total) if total else 0.0,
        "nucleotide_sequence_length": _summary(lengths),
        "generated_6mer_token_count": _summary(token_counts),
    }


def expected_fold_paths(model_dir: str | Path, *, species_id: int = 10) -> list[Path]:
    return [Path(model_dir) / f"{species_id}_fold_{fold}.pth" for fold in range(1, 6)]


def validate_ipromp_assets(*, dnabert_dir: str | Path, fold_paths: list[Path]) -> None:
    dnabert = Path(dnabert_dir)
    required_dnabert = ["config.json", "pytorch_model.bin", "vocab.txt"]
    missing = [str(dnabert / name) for name in required_dnabert if not (dnabert / name).is_file()]
    missing.extend(str(path) for path in fold_paths if not path.is_file())
    if missing:
        raise FileNotFoundError("Missing offline iPro-MP/DNABERT-6 assets:\n- " + "\n- ".join(missing))


def average_fold_probabilities(probabilities: list[np.ndarray]) -> np.ndarray:
    if len(probabilities) != 5:
        raise ValueError(f"Expected five iPro-MP fold probability arrays, got {len(probabilities)}")
    lengths = {len(item) for item in probabilities}
    if len(lengths) != 1:
        raise ValueError("All fold probability arrays must have the same length")
    return np.mean(np.vstack(probabilities), axis=0)


def _predict_fold(
    model: Any,
    tokenizer: Any,
    records: list[FastaRecord],
    *,
    token_max_length: int,
    batch_size: int,
    device: Any,
    torch: Any,
) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(records), max(1, batch_size)):
            batch_records = records[start : start + max(1, batch_size)]
            texts = [sequence_to_kmer_text(record.sequence) for record in batch_records]
            batch = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=token_max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            probs = torch.softmax(logits, dim=-1)[:, 1]
            probabilities.append(probs.detach().cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def _load_fold_state(model: Any, checkpoint_path: Path, torch: Any) -> None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported iPro-MP checkpoint format: {checkpoint_path}")
    cleaned = {}
    for key, value in state.items():
        name = str(key)
        for prefix in ("module.", "model.", "_model."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        cleaned[name] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if len(missing) > 12:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not match the SeqTrainer iPro-MP architecture. "
            f"Missing keys include: {list(missing)[:8]}; unexpected keys include: {list(unexpected)[:8]}"
        )


def _resolve_device(device: str, torch: Any) -> Any:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _normalise_sequence(sequence: str) -> str:
    return "".join(base if base in {"A", "C", "G", "T", "N"} else "N" for base in str(sequence).upper().replace("U", "T"))


def _summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": int(np.min(values)),
        "median": float(np.median(values)),
        "max": int(np.max(values)),
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fasta", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--dnabert-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--species-id", type=int, default=10)
    parser.add_argument("--token-max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metadata-json", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
