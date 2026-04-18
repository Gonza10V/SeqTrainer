"""Dataset recipe declarations and built-in recipe presets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from seqtrainer.sparql.recipes import sequence_query

DEFAULT_Y_URI = "http://www.ontology-of-units-of-measure.org/resource/om-2/hasNumericalValue"


LabelExtractor = Callable[[dict[str, Any]], Any]


@dataclass(slots=True)
class DatasetRecipe:
    """Declarative description of how a synbio dataset should be materialized."""

    name: str
    query: str
    sequence_field: str = "sequence"
    label_field: str | None = None
    label_extractor: LabelExtractor | None = None
    metadata_fields: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict[str, Any] = field(default_factory=dict)

    def extract_label(self, row: dict[str, Any]) -> Any:
        """Get row label using either label field or extractor callback."""
        if self.label_extractor is not None:
            return self.label_extractor(row)
        if self.label_field is None:
            return None
        return row.get(self.label_field)

    def as_manifest_payload(self) -> dict[str, Any]:
        """Return stable, serializable recipe metadata for snapshot manifests."""
        return {
            "name": self.name,
            "query": self.query,
            "sequence_field": self.sequence_field,
            "label_field": self.label_field,
            "metadata_fields": list(self.metadata_fields),
            "provenance": self.provenance,
            "recipe_hash": self.recipe_hash,
        }

    @property
    def recipe_hash(self) -> str:
        """Stable hash identifier for this recipe's serializable shape."""
        payload = {
            "name": self.name,
            "query": self.query,
            "sequence_field": self.sequence_field,
            "label_field": self.label_field,
            "metadata_fields": list(self.metadata_fields),
            "provenance": self.provenance,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---- Built-in dataset recipes -------------------------------------------------


def local_sbol_regression_recipe(*, y_uri: str = DEFAULT_Y_URI) -> DatasetRecipe:
    """Built-in local SBOL recipe for sequence-to-regression datasets."""
    return DatasetRecipe(
        name="local-sbol-regression",
        query=sequence_query(),
        sequence_field="sequence",
        label_field="target",
        metadata_fields=("source",),
        provenance={"source": "local_sbol", "y_uri": y_uri, "task": "regression"},
    )


def local_sbol_sequence_only_recipe() -> DatasetRecipe:
    """Built-in local SBOL recipe when labels are not required."""
    return DatasetRecipe(
        name="local-sbol-sequence-only",
        query=sequence_query(),
        sequence_field="sequence",
        label_field=None,
        metadata_fields=("source",),
        provenance={"source": "local_sbol", "task": "sequence_only"},
    )


def list_builtin_dataset_recipes() -> dict[str, DatasetRecipe]:
    """Return built-in dataset recipe presets."""
    return {
        "local-sbol-regression": local_sbol_regression_recipe(),
        "local-sbol-sequence-only": local_sbol_sequence_only_recipe(),
    }


def get_builtin_dataset_recipe(name: str, *, y_uri: str = DEFAULT_Y_URI) -> DatasetRecipe:
    """Resolve one built-in recipe by name."""
    if name == "local-sbol-regression":
        return local_sbol_regression_recipe(y_uri=y_uri)
    if name == "local-sbol-sequence-only":
        return local_sbol_sequence_only_recipe()
    raise KeyError(f"Unknown dataset recipe: {name}")
