"""SBOL loading and dataset materialization helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdflib import Graph
from rdflib.query import ResultRow

from seqtrainer.data.materialized import MaterializedDataset
from seqtrainer.data.recipes import DatasetRecipe, get_builtin_dataset_recipe
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


def build_dataset_from_files(
    file_paths: list[str | Path],
    y_uri: str = DEFAULT_Y_URI,
    *,
    include_target: bool = True,
) -> pd.DataFrame:
    """Build tabular sequence dataset from SBOL files.

    When ``include_target=False``, target extraction is skipped.
    """
    rows: list[dict[str, str | float]] = []
    for path in file_paths:
        sequence = get_sequence_from_sbol(path)
        if sequence is None:
            continue

        row: dict[str, str | float] = {"sequence": sequence, "source": str(path)}
        if include_target:
            label = get_y_label(path, y_uri)
            if label is None:
                continue
            row["target"] = label

        rows.append(row)
    return pd.DataFrame(rows)


def materialize_dataset_from_sbol(
    file_paths: list[str | Path],
    *,
    y_uri: str = DEFAULT_Y_URI,
    recipe_name: str = "local-sbol-regression",
    dataset_name: str = "sbol-local",
    dataset_version: str | None = None,
    cache_dir: str | Path | None = None,
    write_cache: bool = False,
) -> tuple[MaterializedDataset, object | None]:
    """Materialize local SBOL files into :class:`MaterializedDataset`.

    Optionally writes a snapshot manifest + jsonl payload into the local cache.
    """
    recipe: DatasetRecipe = get_builtin_dataset_recipe(recipe_name, y_uri=y_uri)
    include_target = recipe.label_field is not None

    frame = build_dataset_from_files(file_paths, y_uri=y_uri, include_target=include_target)
    dataset = MaterializedDataset(
        examples=frame.to_dict(orient="records"),
        metadata={
            "source": "local_sbol",
            "file_count": len(file_paths),
            "y_uri": y_uri,
            "recipe_name": recipe_name,
        },
    )

    manifest = None
    if write_cache:
        manifest = dataset.save_snapshot(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            recipe=recipe.as_manifest_payload(),
            cache_dir=str(cache_dir) if cache_dir else None,
        )

    return dataset, manifest
