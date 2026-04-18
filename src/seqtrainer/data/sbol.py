"""SBOL loading and dataset materialization helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdflib import Graph
from rdflib.query import ResultRow

from seqtrainer.data.materialized import MaterializedDataset
from seqtrainer.data.recipes import DatasetRecipe
from seqtrainer.sparql.recipes import sequence_query

DEFAULT_Y_URI = "http://www.ontology-of-units-of-measure.org/resource/om-2/hasNumericalValue"


def _load_graph(file_path: str | Path) -> Graph:
    graph = Graph()
    graph.parse(str(file_path), format="xml")
    return graph


def get_sequence_from_sbol(file_path: str | Path) -> str | None:
    """Extract the first SBOL sequence from a local SBOL XML file."""
    graph = _load_graph(file_path)
    results = graph.query(sequence_query())
    for row in results:
        if isinstance(row, ResultRow):
            return str(row.sequence)
    return None


def get_y_label(file_path: str | Path, uri: str = DEFAULT_Y_URI) -> float | None:
    """Extract the first numeric value mapped by the provided predicate URI."""
    graph = _load_graph(file_path)
    query = f"SELECT ?value WHERE {{ ?s <{uri}> ?value . }}"
    for row in graph.query(query):
        if isinstance(row, ResultRow):
            return float(row.value)
    return None


def build_dataset_from_files(file_paths: list[str | Path], y_uri: str = DEFAULT_Y_URI) -> pd.DataFrame:
    """Build tabular sequence/target dataset from SBOL files."""
    rows: list[dict[str, str | float]] = []
    for path in file_paths:
        sequence = get_sequence_from_sbol(path)
        label = get_y_label(path, y_uri)
        if sequence is None or label is None:
            continue
        rows.append({"sequence": sequence, "target": label, "source": str(path)})
    return pd.DataFrame(rows)


def materialize_dataset_from_sbol(
    file_paths: list[str | Path],
    *,
    y_uri: str = DEFAULT_Y_URI,
    dataset_name: str = "sbol-local",
    dataset_version: str | None = None,
    cache_dir: str | Path | None = None,
    write_cache: bool = False,
) -> tuple[MaterializedDataset, object | None]:
    """Materialize local SBOL files into :class:`MaterializedDataset`.

    Optionally writes a snapshot manifest + jsonl payload into the local cache.
    """
    frame = build_dataset_from_files(file_paths, y_uri=y_uri)
    dataset = MaterializedDataset(
        examples=frame.to_dict(orient="records"),
        metadata={
            "source": "local_sbol",
            "file_count": len(file_paths),
            "y_uri": y_uri,
        },
    )

    manifest = None
    if write_cache:
        recipe = DatasetRecipe(
            name=dataset_name,
            query=sequence_query(),
            sequence_field="sequence",
            label_field="target",
            metadata_fields=("source",),
            provenance={"source": "local_sbol", "y_uri": y_uri},
        )
        manifest = dataset.save_snapshot(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            recipe=recipe.as_manifest_payload(),
            cache_dir=str(cache_dir) if cache_dir else None,
        )

    return dataset, manifest
