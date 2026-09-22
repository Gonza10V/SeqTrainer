"""End-to-end promoter annotation MVP."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .genbank_io import read_genbank, record_topology, write_genbank
from .predictors import PromoterPredictor, build_predictor
from .provenance import file_sha256
from .windows import SequenceWindow, generate_sliding_windows
from .write_features import PromoterRegion, add_predicted_promoter_features


@dataclass(frozen=True)
class PromoterAnnotationConfig:
    input_file: Path
    output_file: Path | None = None
    predictions_csv: Path | None = None
    manifest: Path | None = None
    model_family: str = "dummy"
    model_bundle: Path | None = None
    checkpoint: Path | None = None
    benchmark_manifest: Path | None = None
    threshold: float | None = None
    window_size: int | None = None
    step_size: int = 25
    scan_both_strands: bool = True
    merge_distance: int = 25
    evaluation_dir: Path | None = None
    sbol_output: Path | None = None
    sbol2_output: Path | None = None
    sbol_namespace: str = "https://seqtrainer.org/designs"
    promoter_label_mode: str = "labelled"
    annotation_completeness: str = "unknown"
    iou_threshold: float = 0.50
    source_url: str | None = None


def run_promoter_annotation(
    config: PromoterAnnotationConfig,
    *,
    predictor: PromoterPredictor | None = None,
) -> dict[str, Any]:
    checkpoint, benchmark_manifest = _resolve_model_bundle(
        config.model_bundle,
        checkpoint=config.checkpoint,
        benchmark_manifest=config.benchmark_manifest,
    )
    if checkpoint != config.checkpoint or benchmark_manifest != config.benchmark_manifest:
        config = replace(
            config,
            checkpoint=checkpoint,
            benchmark_manifest=benchmark_manifest,
        )
    if config.model_family == "dnabert2" and (config.checkpoint is None or config.benchmark_manifest is None):
        raise ValueError("DNABERT2 annotation requires a checkpoint and matching benchmark manifest.")
    if config.model_family == "dummy" and (config.threshold is None or config.window_size is None):
        raise ValueError("Dummy annotation requires explicit threshold and window_size values.")

    record = read_genbank(config.input_file)
    output_file, predictions_csv, manifest_path = _resolve_outputs(config)
    original_feature_count = len(record.features)

    gold_promoters = None
    if config.evaluation_dir is not None or config.sbol_output is not None or config.sbol2_output is not None:
        from .ground_truth import extract_ground_truth_promoters

        gold_promoters = extract_ground_truth_promoters(
            record,
            plasmid_id=str(record.id),
            source_url=config.source_url,
            label_mode=config.promoter_label_mode,
        )

    manifest_data = _load_manifest(config.benchmark_manifest) if config.model_family == "dnabert2" else {}
    threshold, threshold_source = _resolve_threshold(config.threshold, manifest_data)
    window_size = _resolve_window_size(config.window_size, manifest_data)
    step_size = config.step_size

    predictor = predictor or build_predictor(
        config.model_family,
        checkpoint=config.checkpoint,
        benchmark_manifest=config.benchmark_manifest,
    )

    windows = generate_sliding_windows(
        str(record.seq),
        window_size=window_size,
        step_size=step_size,
        circular=record_topology(record) == "circular",
        scan_both_strands=config.scan_both_strands,
    )
    scores = predictor.predict_proba([window.sequence for window in windows])
    if len(scores) != len(windows):
        raise ValueError("Predictor returned a different number of scores than input windows")

    rows = []
    passing: list[tuple[SequenceWindow, float]] = []
    for window, raw_score in zip(windows, scores):
        score = float(raw_score)
        passed = score >= threshold
        overlaps = _overlap_summary(record, window.start, window.end, window.is_circular_boundary_window)
        row = {
            "sequence_id": record.id,
            "window_id": window.window_id,
            "start": int(window.start),
            "end": int(window.end % len(record.seq) if window.is_circular_boundary_window else window.end),
            "strand": window.strand,
            "score": score,
            "threshold": threshold,
            "passed_threshold": bool(passed),
            "merged_region_id": "",
            "overlaps_existing_feature": overlaps["overlaps_existing_feature"],
            "overlaps_existing_promoter": overlaps["overlaps_existing_promoter"],
            "overlapping_feature_labels": ";".join(overlaps["overlapping_feature_labels"]),
            "is_circular_boundary_window": window.is_circular_boundary_window,
            "window_sequence": window.sequence,
        }
        rows.append(row)
        if passed:
            passing.append((window, score))

    regions = _merge_passing_windows(
        passing,
        merge_distance=config.merge_distance,
        sequence_length=len(record.seq),
        circular=record_topology(record) == "circular",
    )
    region_by_window = {
        window_id: region.region_id
        for region in regions
        for window_id in region.source_window_ids
    }
    for row in rows:
        row["merged_region_id"] = region_by_window.get(row["window_id"], "")

    added, boundary_written = add_predicted_promoter_features(
        record,
        regions,
        model_family=config.model_family,
        threshold=threshold,
        window_size=window_size,
        step_size=step_size,
    )

    write_genbank(record, output_file)
    prediction_frame = pd.DataFrame(rows)
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(predictions_csv, index=False)

    evaluation_artifacts = _write_external_evaluation(
        record,
        config=config,
        windows=windows,
        scores=scores,
        regions=regions,
        threshold=threshold,
        threshold_source=threshold_source,
        predictor=predictor,
        gold_promoters=gold_promoters,
    )

    warnings = []
    if config.model_family == "dummy":
        warnings.append("Dummy predictor used for smoke testing only; do not treat scores as biological evidence.")
    if config.model_family == "dnabert2" and threshold_source == "cli_override":
        warnings.append("Threshold was overridden on the command line; this result is not the benchmark-selected operating point.")
    manifest = {
        "input_file": str(config.input_file),
        "input_sha256": file_sha256(config.input_file),
        "output_file": str(output_file),
        "predictions_csv": str(predictions_csv),
        "sequence_id": record.id,
        "sequence_length": len(record.seq),
        "topology": record_topology(record),
        "model_family": config.model_family,
        "model_bundle": str(config.model_bundle) if config.model_bundle else None,
        "checkpoint": str(config.checkpoint) if config.checkpoint else None,
        "checkpoint_sha256": file_sha256(config.checkpoint) if config.checkpoint else None,
        "benchmark_manifest": str(config.benchmark_manifest) if config.benchmark_manifest else None,
        "benchmark_manifest_sha256": file_sha256(config.benchmark_manifest) if config.benchmark_manifest else None,
        "source_url": config.source_url,
        "window_size": window_size,
        "step_size": step_size,
        "scan_both_strands": config.scan_both_strands,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "merge_distance": config.merge_distance,
        "total_windows_scanned": len(windows),
        "windows_above_threshold": len(passing),
        "predicted_promoters_added": added,
        "existing_features_preserved": original_feature_count,
        "overlaps_existing_promoters_count": int(prediction_frame["overlaps_existing_promoter"].sum()) if "overlaps_existing_promoter" in prediction_frame else 0,
        "circular_boundary_windows_scanned": int(prediction_frame["is_circular_boundary_window"].sum()) if "is_circular_boundary_window" in prediction_frame else 0,
        "circular_boundary_features_written": boundary_written,
        "warnings": warnings,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "predictor_metadata": predictor.metadata(),
        "annotation_completeness": config.annotation_completeness,
        "sbol_namespace": config.sbol_namespace if config.sbol_output or config.sbol2_output else None,
        "evaluation": evaluation_artifacts,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**manifest, "manifest_file": str(manifest_path)}


def _write_external_evaluation(
    record: Any,
    *,
    config: PromoterAnnotationConfig,
    windows: list[SequenceWindow],
    scores: list[float],
    regions: list[PromoterRegion],
    threshold: float,
    threshold_source: str,
    predictor: PromoterPredictor,
    gold_promoters: list[Any] | None,
) -> dict[str, Any]:
    """Write labelled-plasmid evaluation artifacts when evaluation is requested."""
    if config.evaluation_dir is None and config.sbol_output is None and config.sbol2_output is None:
        return {}
    from .evaluation import evaluate_merged_features, evaluate_windows
    from .ground_truth import write_gold_promoters
    from .provenance import model_provenance

    evaluation_dir = Path(config.evaluation_dir or Path(config.predictions_csv or "outputs/annotations").parent)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    gold = list(gold_promoters or [])
    gold_path = write_gold_promoters(gold, evaluation_dir / "gold_promoters.csv")
    window_frame, window_metrics = evaluate_windows(
        windows,
        scores,
        gold,
        threshold=threshold,
        sequence_length=len(record.seq),
        plasmid_id=str(record.id),
        predictor_method=config.model_family,
        model_version=str(predictor.metadata().get("model_name", config.model_family)),
        completeness=config.annotation_completeness,
    )
    merged_frame, merged_metrics = evaluate_merged_features(
        regions,
        gold,
        sequence_length=len(record.seq),
        plasmid_id=str(record.id),
        circular=record_topology(record) == "circular",
        iou_threshold=config.iou_threshold,
    )
    window_path = evaluation_dir / "window_predictions.csv"
    matches_path = evaluation_dir / "promoter_matches.csv"
    window_frame.to_csv(window_path, index=False)
    merged_frame.to_csv(matches_path, index=False)
    metrics = {"window": window_metrics, "merged": merged_metrics, "annotation_completeness": config.annotation_completeness}
    (evaluation_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default) + "\n", encoding="utf-8")
    pd.DataFrame([{"scope": "window", **_flat_metrics(window_metrics)}, {"scope": "merged", **_flat_metrics(merged_metrics)}]).to_csv(evaluation_dir / "metrics.csv", index=False)

    sbol_artifacts = {}
    provenance = {}
    if config.sbol_output or config.sbol2_output:
        provenance = {
            **model_provenance(
                checkpoint=config.checkpoint,
                benchmark_manifest=config.benchmark_manifest,
                model_family=config.model_family,
                threshold=threshold,
                threshold_source=threshold_source,
            ),
            **predictor.metadata(),
        }
    if config.sbol_output:
        from .sbol3_export import export_sbol3

        sbol_artifacts = export_sbol3(
            record,
            gold_promoters=gold,
            predicted_regions=regions,
            output_path=config.sbol_output,
            validation_path=evaluation_dir / "sbol_validation.json",
            namespace=config.sbol_namespace,
            source_url=config.source_url,
            provenance=provenance,
        )
    if config.sbol2_output:
        from .sbol2_export import export_sbol2

        sbol_artifacts["sbol2"] = export_sbol2(
            record,
            gold_promoters=gold,
            predicted_regions=regions,
            output_path=config.sbol2_output,
            namespace=config.sbol_namespace,
            source_url=config.source_url,
            provenance=provenance,
        )
    return {"evaluation_dir": str(evaluation_dir), "gold_csv": str(gold_path), "metrics_json": str(evaluation_dir / "metrics.json"), "sbol": sbol_artifacts}


def _flat_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flat = {}
    for key, value in metrics.items():
        flat[key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
    return flat


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark manifest not found: {path}")
    # ``utf-8-sig`` accepts standard UTF-8 and the BOM emitted by Windows
    # PowerShell, which makes copied benchmark manifests portable.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve_threshold(explicit: float | None, manifest: dict[str, Any]) -> tuple[float, str]:
    if explicit is not None:
        return float(explicit), "cli_override"
    candidates = [
        manifest.get("evaluation", {}).get("selected_threshold"),
        manifest.get("threshold_selection", {}).get("threshold"),
    ]
    for value in candidates:
        if value is not None:
            return float(value), "benchmark_manifest"
    raise ValueError("Benchmark manifest does not contain a validation-selected threshold.")


def _resolve_window_size(explicit: int | None, manifest: dict[str, Any]) -> int:
    if explicit is not None:
        return int(explicit)
    value = manifest.get("preprocessing", {}).get("sequence_length")
    if value is None:
        raise ValueError("Benchmark manifest does not contain preprocessing.sequence_length.")
    return int(value)


def _resolve_outputs(config: PromoterAnnotationConfig) -> tuple[Path, Path, Path]:
    out_dir = Path("outputs") / "annotations"
    stem = config.input_file.stem
    output_file = config.output_file or out_dir / f"{stem}_{config.model_family}_annotated.gb"
    predictions_csv = config.predictions_csv or out_dir / f"{stem}_{config.model_family}_predictions.csv"
    manifest = config.manifest or out_dir / f"{stem}_{config.model_family}_manifest.json"
    return output_file, predictions_csv, manifest


def _resolve_model_bundle(
    bundle: Path | None,
    *,
    checkpoint: Path | None,
    benchmark_manifest: Path | None,
) -> tuple[Path | None, Path | None]:
    """Resolve a trained checkpoint and matching benchmark manifest from one folder."""
    if bundle is None:
        return checkpoint, benchmark_manifest

    bundle = Path(bundle).expanduser()
    if not bundle.is_dir():
        raise FileNotFoundError(f"Model bundle directory not found: {bundle}")

    resolved_checkpoint = checkpoint or bundle / "checkpoints" / "best_model.pt"
    resolved_manifest = benchmark_manifest or bundle / "manifest.json"
    missing = []
    if not resolved_checkpoint.is_file():
        missing.append("checkpoint (expected checkpoints/best_model.pt)")
    if not resolved_manifest.is_file():
        missing.append("manifest.json")
    if missing:
        raise FileNotFoundError(
            f"Model bundle {bundle} is incomplete; missing " + ", ".join(missing)
        )
    return resolved_checkpoint, resolved_manifest


def _merge_passing_windows(
    passing: list[tuple[SequenceWindow, float]],
    *,
    merge_distance: int,
    sequence_length: int | None = None,
    circular: bool = False,
) -> list[PromoterRegion]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for window, score in sorted(passing, key=lambda item: (item[0].strand, item[0].start, item[0].end)):
        states = grouped.setdefault(window.strand, [])
        if not states or window.start > states[-1]["end"] + merge_distance:
            states.append(
                {
                    "start": window.start,
                    "end": window.end,
                    "strand": window.strand,
                    "score": score,
                    "windows": [window.window_id],
                    "crosses_boundary": window.is_circular_boundary_window,
                }
            )
            continue
        current = states[-1]
        current["end"] = max(current["end"], window.end)
        current["score"] = max(current["score"], score)
        current["windows"].append(window.window_id)
        current["crosses_boundary"] = current["crosses_boundary"] or window.is_circular_boundary_window

    states: list[dict[str, Any]] = []
    for strand in sorted(grouped):
        strand_states = grouped[strand]
        if (
            circular
            and sequence_length is not None
            and len(strand_states) > 1
            and strand_states[-1]["crosses_boundary"]
            and strand_states[0]["start"] + sequence_length <= strand_states[-1]["end"] + merge_distance
        ):
            first = strand_states.pop(0)
            last = strand_states.pop()
            strand_states.insert(
                0,
                {
                    "start": last["start"],
                    "end": first["end"] + sequence_length,
                    "strand": strand,
                    "score": max(last["score"], first["score"]),
                    "windows": last["windows"] + first["windows"],
                    "crosses_boundary": True,
                },
            )
        states.extend(strand_states)
    return [_region_from_state(state, index) for index, state in enumerate(states)]


def _region_from_state(state: dict[str, Any], idx: int) -> PromoterRegion:
    return PromoterRegion(
        region_id=f"predicted_promoter_{idx}",
        start=int(state["start"]),
        end=int(state["end"]),
        strand=str(state["strand"]),
        score=float(state["score"]),
        source_window_ids=tuple(state["windows"]),
        crosses_boundary=bool(state["crosses_boundary"]),
    )


def _overlap_summary(record: Any, start: int, end: int, crosses_boundary: bool) -> dict[str, Any]:
    intervals = _window_intervals(start, end, len(record.seq), crosses_boundary)
    labels: list[str] = []
    promoter_overlap = False
    for feature in record.features:
        feature_intervals = _feature_intervals(feature)
        if not any(_intervals_overlap(a, b) for a in intervals for b in feature_intervals):
            continue
        label = _feature_label(feature)
        if label:
            labels.append(label)
        if feature.type == "promoter" or "promoter" in label.lower():
            promoter_overlap = True
    return {
        "overlaps_existing_feature": bool(labels),
        "overlaps_existing_promoter": promoter_overlap,
        "overlapping_feature_labels": sorted(set(labels)),
    }


def _window_intervals(start: int, end: int, seq_len: int, crosses_boundary: bool) -> list[tuple[int, int]]:
    if crosses_boundary:
        return [(start, seq_len), (0, end % seq_len)]
    return [(start, min(end, seq_len))]


def _feature_intervals(feature: Any) -> list[tuple[int, int]]:
    parts = getattr(feature.location, "parts", None)
    locations = list(parts) if parts is not None else [feature.location]
    return [(int(location.start), int(location.end)) for location in locations]


def _intervals_overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def _feature_label(feature: Any) -> str:
    for key in ("label", "gene", "product", "note", "locus_tag"):
        values = feature.qualifiers.get(key)
        if values:
            return str(values[0])
    return feature.type


def _git_sha() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None
