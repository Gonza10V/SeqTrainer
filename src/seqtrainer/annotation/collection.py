from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .genbank_io import read_genbank
from .ground_truth import extract_ground_truth_promoters
from .predictors import build_predictor
from .promoter_inference import PromoterAnnotationConfig, run_promoter_annotation
from .provenance import file_sha256


def run_promoter_collection(
    manifest_path: str | Path,
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    model_family: str = "dummy",
    checkpoint: str | Path | None = None,
    benchmark_manifest: str | Path | None = None,
    threshold: float | None = None,
    window_size: int | None = None,
    sbol_namespace: str = "https://seqtrainer.org/designs",
    promoter_label_mode: str = "labelled",
    write_sbol3: bool = False,
    continue_on_error: bool = False,
    annotation_completeness: str = "unknown",
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    table = pd.read_csv(manifest_path)
    required = {"addgene_id", "plasmid_name", "expected_local_filename"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Collection manifest missing columns: {sorted(missing)}")

    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    predictor = build_predictor(
        model_family,
        checkpoint=Path(checkpoint) if checkpoint else None,
        benchmark_manifest=Path(benchmark_manifest) if benchmark_manifest else None,
    )
    for row in table.to_dict(orient="records"):
        filename = str(row["expected_local_filename"])
        source = input_dir / filename
        base = _safe_id(str(row["plasmid_name"]))
        if not source.exists():
            excluded.append({**row, "inclusion_status": "excluded", "exclusion_reason": "local GenBank file unavailable"})
            continue
        try:
            record = read_genbank(source)
            gold = extract_ground_truth_promoters(record, plasmid_id=str(row["addgene_id"]), label_mode=promoter_label_mode)
            if not gold:
                excluded.append({**row, "inclusion_status": "excluded", "exclusion_reason": "no explicitly labelled promoter"})
                continue
            plasmid_dir = output_dir / "plasmids" / base
            result = run_promoter_annotation(
                PromoterAnnotationConfig(
                    input_file=source,
                    output_file=plasmid_dir / "annotated.gb",
                    predictions_csv=plasmid_dir / "predictions.csv",
                    manifest=plasmid_dir / "annotation_manifest.json",
                    model_family=model_family,
                    checkpoint=Path(checkpoint) if checkpoint else None,
                    benchmark_manifest=Path(benchmark_manifest) if benchmark_manifest else None,
                    threshold=threshold,
                    window_size=window_size,
                    evaluation_dir=plasmid_dir,
                    sbol_output=plasmid_dir / "annotated.nt" if write_sbol3 else None,
                    sbol_namespace=sbol_namespace,
                    promoter_label_mode=promoter_label_mode,
                    annotation_completeness=annotation_completeness,
                    source_url=str(row.get("plasmid_url", "")) or None,
                ),
                predictor=predictor,
            )
            included.append({**row, "inclusion_status": "included", "exclusion_reason": "", "sha256": file_sha256(source), "promoter_count": len(gold)})
            window_metrics = result.get("evaluation", {}).get("metrics_json")
            if window_metrics and Path(window_metrics).exists():
                payload = json.loads(Path(window_metrics).read_text(encoding="utf-8"))
                aggregate_rows.append({"addgene_id": row["addgene_id"], **_flatten(payload.get("merged", {}))})
        except (OSError, ValueError) as exc:
            excluded.append({**row, "inclusion_status": "excluded", "exclusion_reason": f"processing error: {type(exc).__name__}: {exc}"})
            if not continue_on_error:
                raise

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / "included_plasmids.csv", included)
    _write_rows(output_dir / "excluded_plasmids.csv", excluded)
    pd.DataFrame(aggregate_rows).to_csv(output_dir / "aggregate_metrics.csv", index=False)
    aggregate = {"included_count": len(included), "excluded_count": len(excluded), "metrics": aggregate_rows}
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2, default=str) + "\n", encoding="utf-8")
    collection_manifest = {
        "source_manifest": str(manifest_path), "input_dir": str(input_dir), "output_dir": str(output_dir),
        "model_family": model_family, "checkpoint": str(checkpoint) if checkpoint else None,
        "benchmark_manifest": str(benchmark_manifest) if benchmark_manifest else None,
        "promoter_label_mode": promoter_label_mode, "annotation_completeness": annotation_completeness,
        "write_sbol3": write_sbol3, "included_count": len(included), "excluded_count": len(excluded),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "collection_manifest.json").write_text(json.dumps(collection_manifest, indent=2) + "\n", encoding="utf-8")
    return collection_manifest


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        path.write_text("\n", encoding="utf-8")


def _flatten(values: dict[str, Any]) -> dict[str, Any]:
    return {key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in values.items()}


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value) or "plasmid"
