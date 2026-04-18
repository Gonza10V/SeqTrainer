"""Dataset snapshot/cache helpers for local reproducible materialization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .materialized import MaterializedDataset

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "seqtrainer"
MANIFEST_VERSION = "1"


@dataclass(slots=True)
class DatasetManifest:
    """Versioned manifest for a materialized dataset snapshot."""

    manifest_version: str
    dataset_name: str
    dataset_version: str
    created_at: str
    row_count: int
    fingerprint_sha256: str
    files: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)
    recipe: dict[str, Any] = field(default_factory=dict)

    @property
    def snapshot_key(self) -> str:
        return f"{self.dataset_name}:{self.dataset_version}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetManifest":
        return cls(**payload)


def _stable_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_examples_fingerprint(examples: list[dict[str, Any]]) -> str:
    """Compute deterministic SHA256 fingerprint for example rows."""
    serialized = _stable_json_dumps(examples)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def resolve_snapshot_dir(
    dataset_name: str,
    dataset_version: str,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Resolve cache path for a dataset snapshot."""
    root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    safe_name = dataset_name.replace("/", "_")
    safe_version = dataset_version.replace("/", "_")
    return root / safe_name / safe_version


def write_snapshot(
    dataset: MaterializedDataset,
    *,
    dataset_name: str,
    dataset_version: str | None = None,
    recipe: dict[str, Any] | None = None,
    cache_dir: str | Path | None = None,
) -> DatasetManifest:
    """Write dataset examples + manifest to local cache."""
    fingerprint = compute_examples_fingerprint(dataset.examples)
    version = dataset_version or fingerprint[:12]
    snapshot_dir = resolve_snapshot_dir(dataset_name, version, cache_dir=cache_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    data_file = snapshot_dir / "examples.jsonl"
    metadata_file = snapshot_dir / "metadata.json"
    manifest_file = snapshot_dir / "manifest.json"

    with data_file.open("w", encoding="utf-8") as fh:
        for row in dataset.examples:
            fh.write(_stable_json_dumps(row) + "\n")

    with metadata_file.open("w", encoding="utf-8") as fh:
        json.dump(dataset.metadata, fh, indent=2, sort_keys=True)

    manifest = DatasetManifest(
        manifest_version=MANIFEST_VERSION,
        dataset_name=dataset_name,
        dataset_version=version,
        created_at=datetime.now(timezone.utc).isoformat(),
        row_count=len(dataset.examples),
        fingerprint_sha256=fingerprint,
        files={
            "examples": str(data_file.name),
            "metadata": str(metadata_file.name),
            "manifest": str(manifest_file.name),
        },
        metadata={"snapshot_dir": str(snapshot_dir)},
        recipe=recipe or {},
    )

    with manifest_file.open("w", encoding="utf-8") as fh:
        json.dump(manifest.to_dict(), fh, indent=2, sort_keys=True)

    return manifest


def load_snapshot(
    *,
    dataset_name: str,
    dataset_version: str,
    cache_dir: str | Path | None = None,
) -> tuple[MaterializedDataset, DatasetManifest]:
    """Load a dataset snapshot and its manifest from local cache."""
    snapshot_dir = resolve_snapshot_dir(dataset_name, dataset_version, cache_dir=cache_dir)
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found for snapshot: {snapshot_dir}")

    manifest = DatasetManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))

    examples_file = snapshot_dir / manifest.files["examples"]
    metadata_file = snapshot_dir / manifest.files["metadata"]

    examples = [json.loads(line) for line in examples_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    dataset = MaterializedDataset(examples=examples, metadata=metadata)

    fingerprint = compute_examples_fingerprint(dataset.examples)
    if fingerprint != manifest.fingerprint_sha256:
        raise ValueError(
            "Snapshot fingerprint mismatch. "
            f"Expected {manifest.fingerprint_sha256}, got {fingerprint}."
        )

    return dataset, manifest


def list_snapshot_versions(dataset_name: str, *, cache_dir: str | Path | None = None) -> list[str]:
    """List versions available for a cached dataset."""
    root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    dataset_dir = root / dataset_name.replace("/", "_")
    if not dataset_dir.exists():
        return []
    return sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])
