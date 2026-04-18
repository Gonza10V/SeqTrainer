"""Canonical SBOL/SynBioHub SPARQL query recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .builder import build_select_query
from .normalize import SparqlType, normalize_bindings


@dataclass(frozen=True, slots=True)
class SparqlRecipe:
    """Declarative SPARQL recipe with typed result schema."""

    name: str
    description: str
    query: str
    schema: dict[str, SparqlType] = field(default_factory=dict)

    def normalize(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize payload rows using the recipe's schema."""
        return normalize_bindings(payload, schema=self.schema)


DEFAULT_NUMERIC_VALUE_URI = "http://www.ontology-of-units-of-measure.org/resource/om-2/hasNumericalValue"


def sequence_query(*, limit: int | None = None, offset: int | None = None) -> str:
    """Return a query that extracts SBOL sequence strings."""
    return build_select_query(
        fields=["?sequence"],
        where_lines=["?s sbol:elements ?sequence ."],
        limit=limit,
        offset=offset,
    )


def sequence_elements_recipe(*, limit: int | None = None) -> SparqlRecipe:
    return SparqlRecipe(
        name="sequence-elements",
        description="Extract SBOL sequence element strings.",
        query=sequence_query(limit=limit),
        schema={"sequence": "str"},
    )


def component_definitions_recipe(*, limit: int | None = None) -> SparqlRecipe:
    query = build_select_query(
        fields=["?component", "?displayId", "?name", "?description"],
        where_lines=[
            "?component rdf:type sbol:ComponentDefinition .",
            "OPTIONAL { ?component sbol:displayId ?displayId . }",
            "OPTIONAL { ?component sbol:name ?name . }",
            "OPTIONAL { ?component sbol:description ?description . }",
        ],
        distinct=True,
        order_by=["?displayId"],
        limit=limit,
    )
    return SparqlRecipe(
        name="component-definitions",
        description="Return component definitions with common metadata.",
        query=query,
        schema={
            "component": "uri",
            "displayId": "str",
            "name": "str",
            "description": "str",
        },
    )


def component_sequences_recipe(*, limit: int | None = None) -> SparqlRecipe:
    query = build_select_query(
        fields=["?component", "?sequenceUri", "?sequence"],
        where_lines=[
            "?component rdf:type sbol:ComponentDefinition .",
            "?component sbol:sequence ?sequenceUri .",
            "?sequenceUri sbol:elements ?sequence .",
        ],
        distinct=True,
        order_by=["?component"],
        limit=limit,
    )
    return SparqlRecipe(
        name="component-sequences",
        description="Map component definitions to sequence URIs and sequence strings.",
        query=query,
        schema={"component": "uri", "sequenceUri": "uri", "sequence": "str"},
    )


def numerical_measurements_recipe(
    *,
    value_predicate_uri: str = DEFAULT_NUMERIC_VALUE_URI,
    limit: int | None = None,
) -> SparqlRecipe:
    query = build_select_query(
        fields=["?subject", "?value"],
        where_lines=[f"?subject <{value_predicate_uri}> ?value ."],
        distinct=True,
        limit=limit,
    )
    return SparqlRecipe(
        name="numerical-measurements",
        description="Extract numeric values associated with SBOL subjects.",
        query=query,
        schema={"subject": "uri", "value": "float"},
    )


def sequence_annotations_recipe(*, limit: int | None = None) -> SparqlRecipe:
    query = build_select_query(
        fields=["?component", "?annotation", "?start", "?end"],
        where_lines=[
            "?component rdf:type sbol:ComponentDefinition .",
            "?component sbol:sequenceAnnotation ?annotation .",
            "?annotation sbol:location ?location .",
            "?location sbol:start ?start .",
            "?location sbol:end ?end .",
        ],
        distinct=True,
        order_by=["?component", "?start"],
        limit=limit,
    )
    return SparqlRecipe(
        name="sequence-annotations",
        description="Extract sequence annotation ranges for component definitions.",
        query=query,
        schema={
            "component": "uri",
            "annotation": "uri",
            "start": "int",
            "end": "int",
        },
    )


def list_recipes() -> dict[str, SparqlRecipe]:
    """Return the built-in canonical recipe registry."""
    registry = {
        "sequence-elements": sequence_elements_recipe(),
        "component-definitions": component_definitions_recipe(),
        "component-sequences": component_sequences_recipe(),
        "numerical-measurements": numerical_measurements_recipe(),
        "sequence-annotations": sequence_annotations_recipe(),
    }
    return registry
