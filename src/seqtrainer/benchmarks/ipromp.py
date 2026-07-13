"""iPro-MP external-prediction benchmark helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seqtrainer.metrics import best_threshold_by_metric, binary_classification_metrics

from .artifacts import write_benchmark_outputs
from .config import BenchmarkConfig, load_benchmark_config
from .manifest import build_run_manifest
from .policy import threshold_metric_from_strategy
from .runner import BenchmarkRunResult, BenchmarkSkipped
from .splits import REQUIRED_SPLITS, load_predefined_split_frames, summarize_split_frames


def prepare_ipromp_inputs(
    config_or_path: BenchmarkConfig | str | Path,
    *,
    base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Write FASTA files and an ID mapping table for external iPro-MP inference."""
    config = load_benchmark_config(config_or_path) if not isinstance(config_or_path, BenchmarkConfig) else config_or_path
    if config.model.family != "ipromp":
        raise ValueError(f"prepare_ipromp_inputs requires model.family='ipromp', got {config.model.family!r}")

    frames = load_predefined_split_frames(config, base_dir=base_dir)
    out_dir = _resolve_output_dir(config, base_dir=base_dir, output_dir=output_dir)
    fasta_dir = out_dir / "ipromp_fasta"
    fasta_dir.mkdir(parents=True, exist_ok=True)

    mapping_rows: list[dict[str, Any]] = []
    written: dict[str, Path] = {}
    for split in REQUIRED_SPLITS:
        frame = frames[split].reset_index(drop=True)
        fasta_path = fasta_dir / f"{split}.fasta"
        with fasta_path.open("w", encoding="utf-8") as handle:
            for idx, row in frame.iterrows():
                seq_id = _row_id(split, idx, row, config)
                sequence = _normalise_sequence(row[config.dataset.sequence_field])
                label = _normalise_label(row[config.dataset.label_field], config)
                handle.write(f">{seq_id}\n{sequence}\n")
                mapping_rows.append(
                    {
                        "id": seq_id,
                        "split": split,
                        "idx": int(idx),
                        "label": int(label),
                        "sequence": sequence,
                    }
                )
        written[split] = fasta_path

    mapping_path = out_dir / "ipromp_id_mapping.csv"
    pd.DataFrame(mapping_rows).to_csv(mapping_path, index=False)
    written["mapping"] = mapping_path
    return written


def run_ipromp_external_predictions(
    config: BenchmarkConfig,
    *,
    base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> BenchmarkRunResult:
    """Evaluate externally generated iPro-MP prediction CSVs with shared metrics."""
    frames = load_predefined_split_frames(config, base_dir=base_dir)
    split_summary = summarize_split_frames(config, frames)
    out_dir = _resolve_output_dir(config, base_dir=base_dir, output_dir=output_dir)
    prediction_paths = _prediction_paths(config, base_dir=base_dir, out_dir=out_dir)

    missing = [str(path) for path in prediction_paths.values() if not path.exists()]
    if missing:
        raise BenchmarkSkipped(
            "Missing external iPro-MP prediction files. Run prepare-ipromp plus the official "
            f"iPro-MP inference script first. Missing: {missing}"
        )

    predictions: dict[str, pd.DataFrame] = {}
    for split in REQUIRED_SPLITS:
        predictions[split] = _aligned_prediction_frame(
            split,
            frames[split],
            prediction_paths[split],
            config,
        )

    validation = predictions["validation"]
    if config.evaluation.threshold_strategy == "fixed_0_5":
        threshold = 0.5
    else:
        metric = threshold_metric_from_strategy(config.evaluation.threshold_strategy)
        if metric is None:
            threshold = 0.5
        else:
            threshold, _ = best_threshold_by_metric(
                validation["label"].to_numpy(dtype=int),
                validation["probability"].to_numpy(dtype=float),
                metric=metric,
            )

    metrics = {
        split: binary_classification_metrics(
            frame["label"].to_numpy(dtype=int),
            frame["probability"].to_numpy(dtype=float),
            threshold=threshold,
        )
        for split, frame in predictions.items()
    }

    for frame in predictions.values():
        frame["threshold"] = float(threshold)
        frame["prediction"] = (frame["probability"].to_numpy(dtype=float) >= threshold).astype(int)

    manifest = build_run_manifest(
        config,
        split_summary=split_summary,
        threshold=threshold,
        model_metadata={
            "mode": "external_iProMP_predictions",
            "prediction_files": {split: str(path) for split, path in prediction_paths.items()},
        },
    )
    write_benchmark_outputs(
        out_dir,
        manifest=manifest,
        metrics=metrics,
        predictions=pd.concat([predictions[split] for split in REQUIRED_SPLITS], ignore_index=True),
        config=config,
    )
    return BenchmarkRunResult(output_dir=out_dir, status="completed", metrics=metrics, manifest=manifest)


def _aligned_prediction_frame(
    split: str,
    frame: pd.DataFrame,
    prediction_path: Path,
    config: BenchmarkConfig,
) -> pd.DataFrame:
    raw = pd.read_csv(prediction_path, sep=None, engine="python")
    probability_col = _first_existing(raw, ("probability", "score", "positive_probability", "promoter_probability"))
    if probability_col is None:
        raise ValueError(f"{prediction_path} must contain a probability or score column")

    source = frame.reset_index(drop=True)
    labels = [_normalise_label(value, config) for value in source[config.dataset.label_field]]
    rows = pd.DataFrame(
        {
            "split": split,
            "idx": np.arange(len(source)),
            "sequence": source[config.dataset.sequence_field].astype(str),
            "label": labels,
        }
    )

    id_col = _first_existing(raw, ("id", "sequence_id", "seq_id"))
    if id_col is not None:
        expected_ids = [_row_id(split, idx, row, config) for idx, row in source.iterrows()]
        rows["id"] = expected_ids
        raw = raw.rename(columns={id_col: "id", probability_col: "probability"})
        merged = rows.merge(raw[["id", "probability"]], on="id", how="left", validate="one_to_one")
        if merged["probability"].isna().any():
            raise ValueError(f"{prediction_path} is missing predictions for some {split} rows")
        return merged

    if len(raw) != len(rows):
        raise ValueError(
            f"{prediction_path} has {len(raw)} rows, but {split} split has {len(rows)} rows. "
            "Include an id column or preserve split row order."
        )
    rows["probability"] = raw[probability_col].to_numpy(dtype=float)
    return rows


def _prediction_paths(
    config: BenchmarkConfig,
    *,
    base_dir: str | Path | None,
    out_dir: Path,
) -> dict[str, Path]:
    params = dict(config.model.params)
    keys = {
        "train": "train_predictions_csv",
        "validation": "validation_predictions_csv",
        "test": "test_predictions_csv",
    }
    paths = {}
    for split, key in keys.items():
        raw = params.get(key) or str(out_dir / "external_predictions" / f"{split}_predictions.csv")
        path = Path(str(raw))
        if not path.is_absolute():
            root = Path(base_dir) if base_dir is not None else Path.cwd()
            path = root / path
        paths[split] = path
    return paths


def _resolve_output_dir(
    config: BenchmarkConfig,
    *,
    base_dir: str | Path | None,
    output_dir: str | Path | None,
) -> Path:
    raw = Path(output_dir or config.outputs.output_dir)
    if raw.is_absolute():
        return raw
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    return root / raw


def _row_id(split: str, idx: int, row: pd.Series, config: BenchmarkConfig) -> str:
    if config.dataset.id_field and config.dataset.id_field in row:
        return str(row[config.dataset.id_field])
    return f"{split}_{idx}"


def _normalise_sequence(value: Any) -> str:
    return str(value).upper().replace("U", "T")


def _normalise_label(value: Any, config: BenchmarkConfig) -> int:
    if value == config.label.positive_label or str(value) == str(config.label.positive_label):
        return 1
    if value == config.label.negative_label or str(value) == str(config.label.negative_label):
        return 0
    raise ValueError(f"Label {value!r} does not match configured positive/negative labels")


def _first_existing(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None
