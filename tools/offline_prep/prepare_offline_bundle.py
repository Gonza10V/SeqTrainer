"""Prepare a SeqTrainer offline bundle skeleton outside Alpine.

This helper is intentionally conservative: it stages files you already have and
computes checksums. Downloading model snapshots and building the Apptainer image
should be done on an internet-enabled machine before transfer to Alpine.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    bundle = args.bundle_root
    for relative in (
        "image",
        "repository",
        "models/DNABERT-2-117M",
        "models/DNABERT-6",
        "models/ipromp_ecoli",
        "data/promoter_classification",
        "manifests",
    ):
        (bundle / relative).mkdir(parents=True, exist_ok=True)
    if args.repo_dir:
        target = bundle / "repository" / "SeqTrainer"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(args.repo_dir, target, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    _write_checksums(bundle)
    print(f"Offline bundle skeleton prepared at {bundle}")
    return 0


def _write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and "SHA256SUMS" not in path.name:
            rows.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "manifests" / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--repo-dir", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
