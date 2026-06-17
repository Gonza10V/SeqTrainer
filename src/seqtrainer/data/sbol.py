"""SBOL loading and dataset materialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from rdflib import Graph
from rdflib.query import ResultRow

DEFAULT_Y_URI = "http://www.ontology-of-units-of-measure.org/resource/om-2/hasNumericalValue"
VALID_DNA_BASES = set("ACGTN")


def _load_graph(file_path: str | Path) -> Graph:
    graph = Graph()
    graph.parse(str(file_path), format="xml")
    return graph


def get_sequence_from_sbol(file_path: str | Path) -> str | None:
    """Extract the first SBOL sequence from a local SBOL XML file."""
    graph = _load_graph(file_path)
    record = _first_sequence_record(graph)
    if record:
        return str(record["sequence"])
    return None


def get_y_label(file_path: str | Path, uri: str = DEFAULT_Y_URI) -> float | None:
    """Extract the first numeric value mapped by the provided predicate URI."""
    graph = _load_graph(file_path)
    query = f"SELECT ?value WHERE {{ ?s <{uri}> ?value . }}"
    for row in graph.query(query):
        if isinstance(row, ResultRow):
            return float(row.value)
    return None


def build_dataset_from_files(
    file_paths: list[str | Path],
    y_uri: str = DEFAULT_Y_URI,
    *,
    label_threshold: float | None = None,
    include_dropped: bool = False,
    summary_path: str | Path | None = None,
    warnings_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build a provenance-aware tabular sequence/target dataset from SBOL files.

    By default, invalid or incomplete records are omitted from the returned
    frame to preserve the historical API behavior. Set ``include_dropped=True``
    to retain them with ``dropped=True`` and ``parsing_warnings`` populated.
    """
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    seen_sequence_ids: dict[str, str] = {}
    for path in file_paths:
        row = _build_dataset_row(path, y_uri=y_uri, label_threshold=label_threshold)
        sequence_id = row.get("sequence_id")
        if sequence_id:
            previous = seen_sequence_ids.get(str(sequence_id))
            if previous is not None:
                row["parsing_warnings"].append(f"duplicate_sequence_id:{sequence_id}")
                row["validation_status"] = "warning"
            else:
                seen_sequence_ids[str(sequence_id)] = str(row["source_path"])

        if row["parsing_warnings"]:
            warnings.append(
                {
                    "source_path": row["source_path"],
                    "sequence_id": row.get("sequence_id"),
                    "warnings": ";".join(row["parsing_warnings"]),
                }
            )

        row["dropped"] = bool(row["drop_reason"])
        row["parsing_warnings"] = ";".join(row["parsing_warnings"])
        if row["dropped"] and not include_dropped:
            continue
        rows.append(row)

    frame = pd.DataFrame(rows)
    summary = dataset_validation_summary(frame, total_files=len(file_paths), warnings=warnings)
    if summary_path is not None:
        _write_json(summary_path, summary)
    if warnings_path is not None:
        _write_warnings(warnings_path, warnings)
    return frame


def dataset_validation_summary(
    frame: pd.DataFrame,
    *,
    total_files: int | None = None,
    warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize validation and provenance fields from an SBOL dataset frame."""
    total_rows = int(len(frame))
    dropped = int(frame["dropped"].sum()) if "dropped" in frame else 0
    kept = total_rows - dropped
    warning_rows = int(frame["parsing_warnings"].astype(bool).sum()) if "parsing_warnings" in frame else 0
    return {
        "total_files": int(total_files) if total_files is not None else total_rows,
        "rows_returned": total_rows,
        "kept_rows": kept,
        "dropped_rows": dropped,
        "warning_rows": warning_rows,
        "warning_records": len(warnings or []),
        "label_threshold": _first_present(frame, "label_threshold"),
        "label_rule": _first_present(frame, "label_rule"),
        "columns": list(frame.columns),
    }


def _build_dataset_row(path: str | Path, *, y_uri: str, label_threshold: float | None) -> dict[str, Any]:
    source_path = Path(path)
    warnings: list[str] = []
    sequence_record: dict[str, str] | None = None
    target: float | None = None
    drop_reason = ""
    validation_status = "ok"
    try:
        graph = _load_graph(source_path)
        sequence_record = _first_sequence_record(graph)
        target = _first_numeric_value(graph, y_uri)
    except Exception as exc:  # noqa: BLE001 - preserve parse failure context in the dataset row.
        warnings.append(f"parse_error:{type(exc).__name__}:{exc}")
        validation_status = "error"

    sequence = sequence_record["sequence"] if sequence_record else None
    sequence_id = sequence_record.get("sequence_id") if sequence_record else None
    sequence_uri = sequence_record.get("sequence_uri") if sequence_record else None
    if sequence is None:
        warnings.append("missing_sequence")
        drop_reason = "missing_sequence"
    else:
        invalid_bases = sorted(set(sequence.upper()).difference(VALID_DNA_BASES))
        if invalid_bases:
            warnings.append(f"invalid_bases:{''.join(invalid_bases)}")
            validation_status = "warning"

    if target is None:
        warnings.append("missing_target")
        drop_reason = drop_reason or "missing_target"

    label = None
    label_rule = "provided_numeric_target"
    if label_threshold is not None:
        label_rule = "numeric_target_threshold"
        if target is not None:
            label = int(target >= label_threshold)

    if drop_reason and validation_status != "error":
        validation_status = "dropped"

    return {
        "sequence": sequence,
        "target": target,
        "label": label,
        "source": str(source_path),
        "source_path": str(source_path),
        "source_file": source_path.name,
        "source_stem": source_path.stem,
        "sequence_id": sequence_id,
        "sequence_uri": sequence_uri,
        "target_uri": y_uri,
        "target_derivation": f"first value for predicate {y_uri}",
        "label_rule": label_rule,
        "label_threshold": label_threshold,
        "validation_status": validation_status,
        "drop_reason": drop_reason,
        "parsing_warnings": warnings,
    }


def _first_sequence_record(graph: Graph) -> dict[str, str] | None:
    query = """
    SELECT ?sequence_node ?sequence ?display_id WHERE {
      ?sequence_node <http://sbols.org/v2#elements> ?sequence .
      OPTIONAL { ?sequence_node <http://sbols.org/v2#displayId> ?display_id . }
    } LIMIT 1
    """
    for row in graph.query(query):
        if isinstance(row, ResultRow):
            sequence_uri = str(row.sequence_node)
            return {
                "sequence": str(row.sequence).upper(),
                "sequence_uri": sequence_uri,
                "sequence_id": str(row.display_id) if row.display_id else sequence_uri.rsplit("/", 2)[-2],
            }
    return None


def _first_numeric_value(graph: Graph, uri: str) -> float | None:
    query = f"SELECT ?value WHERE {{ ?s <{uri}> ?value . }} LIMIT 1"
    for row in graph.query(query):
        if isinstance(row, ResultRow):
            return float(row.value)
    return None


def _first_present(frame: pd.DataFrame, column: str) -> Any:
    if column not in frame or frame.empty:
        return None
    values = frame[column].dropna().unique()
    if len(values) == 0:
        return None
    return values[0].item() if hasattr(values[0], "item") else values[0]


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_warnings(path: str | Path, warnings: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(warnings, columns=["source_path", "sequence_id", "warnings"]).to_csv(out, index=False)
