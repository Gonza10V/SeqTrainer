"""Validate a SeqTrainer offline bundle before Alpine submission.

The validator performs only local filesystem checks. It never downloads models,
packages, code, or data. Exit code 2 means the bundle is incomplete or invalid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path


SPLIT_FILES = {
    "train": "train_EP_DNA_BERT2_genomic_order.csv",
    "validation": "eval_EP_DNA_BERT2_genomic_order.csv",
    "test": "test_EP_DNA_BERT2_genomic_order.csv",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    errors = validate_bundle(args.bundle_root, model=args.model)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Offline bundle OK for {args.model}: {args.bundle_root}")
    return 0


def validate_bundle(bundle_root: str | Path, *, model: str) -> list[str]:
    root = Path(bundle_root)
    errors: list[str] = []
    _require_file(root / "image" / "seqtrainer-alpine-gpu.sif", errors)
    _require_dir(root / "repository" / "SeqTrainer", errors)
    _require_dir(root / "models", errors)
    _require_file(root / "manifests" / "offline_bundle_manifest.json", errors)
    _require_file(root / "manifests" / "repository_revision.txt", errors)

    repo = root / "repository" / "SeqTrainer"
    if model == "dnabert2":
        _validate_dnabert2(root, repo, errors)
    elif model == "ipromp":
        _validate_ipromp(root, repo, errors)
    else:
        errors.append(f"Unsupported model {model!r}")
    _validate_data(root / "data" / "promoter_classification", errors)
    _validate_checksums(root, errors)
    return errors


def _validate_dnabert2(root: Path, repo: Path, errors: list[str]) -> None:
    _require_file(repo / "config-examples" / "benchmarks" / "dnabert2_finetune_alpine_offline.toml", errors)
    model_dir = root / "models" / "DNABERT-2-117M"
    for name in ("config.json", "configuration_bert.py", "bert_layers.py", "bert_padding.py"):
        _require_file(model_dir / name, errors)
    if not ((model_dir / "tokenizer.json").is_file() or (model_dir / "vocab.txt").is_file()):
        errors.append(f"Missing DNABERT2 tokenizer file in {model_dir}: tokenizer.json or vocab.txt")
    if not ((model_dir / "pytorch_model.bin").is_file() or (model_dir / "model.safetensors").is_file()):
        errors.append(f"Missing DNABERT2 weights in {model_dir}: pytorch_model.bin or model.safetensors")


def _validate_ipromp(root: Path, repo: Path, errors: list[str]) -> None:
    _require_file(repo / "config-examples" / "benchmarks" / "ipromp_external_alpine_offline.toml", errors)
    dnabert_dir = root / "models" / "DNABERT-6"
    for name in ("config.json", "pytorch_model.bin", "vocab.txt"):
        _require_file(dnabert_dir / name, errors)
    fold_dir = root / "models" / "ipromp_ecoli"
    for fold in range(1, 6):
        _require_file(fold_dir / f"10_fold_{fold}.pth", errors)


def _validate_data(data_dir: Path, errors: list[str]) -> None:
    for split, filename in SPLIT_FILES.items():
        path = data_dir / filename
        _require_file(path, errors)
        if path.is_file():
            _validate_csv_split(path, split, errors)


def _validate_csv_split(path: Path, split: str, errors: list[str]) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing = {"sequence", "label"}.difference(fields)
            if missing:
                errors.append(f"{split} split is missing columns {sorted(missing)}: {path}")
                return
            rows = 0
            labels = set()
            for row in reader:
                rows += 1
                labels.add(str(row["label"]))
            if rows == 0:
                errors.append(f"{split} split has no rows: {path}")
            if not labels.issubset({"0", "1"}):
                errors.append(f"{split} labels must be binary 0/1, found {sorted(labels)}: {path}")
    except OSError as exc:
        errors.append(f"Could not read {split} split {path}: {exc}")


def _validate_checksums(root: Path, errors: list[str]) -> None:
    checksum_path = root / "manifests" / "SHA256SUMS"
    if not checksum_path.exists():
        return
    for line_no, raw_line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"Invalid SHA256SUMS line {line_no}: {raw_line!r}")
            continue
        expected, relative = parts
        relative = relative.lstrip("*")
        path = root / relative
        if not path.exists():
            errors.append(f"Checksum target is missing: {relative}")
            continue
        actual = _sha256(path)
        if actual.lower() != expected.lower():
            errors.append(f"Checksum mismatch for {relative}: expected {expected}, got {actual}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"Missing required file: {path}")


def _require_dir(path: Path, errors: list[str]) -> None:
    if not path.is_dir():
        errors.append(f"Missing required directory: {path}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=("dnabert2", "ipromp"))
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
