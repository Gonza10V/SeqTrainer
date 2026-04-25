"""Composable SPARQL query builder helpers."""

from __future__ import annotations

from .prefixes import format_prefixes


def build_select_query(
    *,
    fields: list[str],
    where_lines: list[str],
    prefixes: dict[str, str] | None = None,
    distinct: bool = False,
    order_by: list[str] | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> str:
    """Build a SELECT query from structured parts."""
    modifier = "DISTINCT " if distinct else ""
    fields_str = " ".join(fields)
    where = "\n  ".join(where_lines)
    query = f"{format_prefixes(prefixes)}\n\nSELECT {modifier}{fields_str}\nWHERE {{\n  {where}\n}}"
    if order_by:
        query += f"\nORDER BY {' '.join(order_by)}"
    if limit is not None:
        query += f"\nLIMIT {limit}"
    if offset is not None:
        query += f"\nOFFSET {offset}"
    return query
