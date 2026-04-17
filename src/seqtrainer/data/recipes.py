"""Dataset recipe declarations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable


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
