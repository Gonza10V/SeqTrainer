"""Typed normalization for SPARQL JSON results."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

SparqlType = Literal["auto", "str", "uri", "int", "float", "bool", "datetime"]


def _coerce_auto(value: str, datatype: str | None) -> Any:
    dt = (datatype or "").lower()
    if dt.endswith("#integer") or dt.endswith("#int"):
        return int(value)
    if dt.endswith("#decimal") or dt.endswith("#double") or dt.endswith("#float"):
        return float(value)
    if dt.endswith("#boolean"):
        return value.lower() == "true"
    if dt.endswith("#datetime"):
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    return value


def normalize_binding_value(binding_value: dict[str, str], expected_type: SparqlType = "auto") -> Any:
    """Normalize one SPARQL binding value into a typed Python object."""
    value = binding_value.get("value")
    datatype = binding_value.get("datatype")
    if value is None:
        return None

    if expected_type in {"str", "uri"}:
        return value
    if expected_type == "int":
        return int(value)
    if expected_type == "float":
        return float(value)
    if expected_type == "bool":
        return value.lower() == "true"
    if expected_type == "datetime":
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _coerce_auto(value, datatype)


def normalize_bindings(
    payload: dict[str, Any],
    *,
    schema: dict[str, SparqlType] | None = None,
    include_missing: bool = True,
) -> list[dict[str, Any]]:
    """Normalize SPARQL JSON payload rows according to a schema.

    Parameters
    ----------
    payload:
        SPARQL JSON payload with ``results.bindings``.
    schema:
        Optional map from variable name to expected output type.
    include_missing:
        If true, declared schema fields missing in a row are returned as None.
    """
    schema = schema or {}
    rows = payload.get("results", {}).get("bindings", [])
    normalized: list[dict[str, Any]] = []
    for row in rows:
        out: dict[str, Any] = {}
        for key, value in row.items():
            out[key] = normalize_binding_value(value, schema.get(key, "auto"))
        if include_missing:
            for required_key in schema:
                out.setdefault(required_key, None)
        normalized.append(out)
    return normalized
