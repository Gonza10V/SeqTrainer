"""RDF/SBOL graph extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rdflib import Graph
from rdflib.query import ResultRow


@dataclass(frozen=True, slots=True)
class TypedEdge:
    """Typed edge relation discovered in an RDF graph."""

    source_type: str
    predicate: str
    target_type: str


def load_graph(path: str | Path, *, format: str = "xml") -> Graph:
    """Load an RDF graph from disk."""
    graph = Graph()
    graph.parse(str(path), format=format)
    return graph


def xml_to_nt(input_path: str | Path, output_path: str | Path) -> None:
    """Convert SBOL XML to N-Triples for graph tooling."""
    graph = load_graph(input_path, format="xml")
    graph.serialize(destination=str(output_path), format="nt")


def extract_node_types(
    graph: Graph,
    *,
    exclude_uri: str | None = None,
    rdf_type_uri: str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
) -> list[str]:
    """Extract distinct node (rdf:type) URIs from an RDF graph."""
    filter_clause = f"FILTER(?value != <{exclude_uri}>)" if exclude_uri else ""
    query = f"""
    SELECT DISTINCT ?value
    WHERE {{
      ?s <{rdf_type_uri}> ?value .
      {filter_clause}
    }}
    """
    node_types: list[str] = []
    for row in graph.query(query):
        if isinstance(row, ResultRow):
            node_types.append(str(row.value))
    return sorted(set(node_types))


def extract_typed_edges(
    graph: Graph,
    *,
    allowed_types: Iterable[str] | None = None,
    rdf_type_uri: str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
) -> list[TypedEdge]:
    """Extract typed edges `(source_type, predicate, target_type)` from graph."""
    filter_clause = ""
    if allowed_types:
        formatted = ",\n".join(f"<{uri}>" for uri in allowed_types)
        filter_clause = f"""
        FILTER(?stype IN ({formatted}))
        FILTER(?vtype IN ({formatted}))
        """

    query = f"""
    SELECT DISTINCT ?stype ?prop ?vtype
    WHERE {{
      ?s ?prop ?value .
      ?s <{rdf_type_uri}> ?stype .
      ?value <{rdf_type_uri}> ?vtype .
      {filter_clause}
    }}
    """

    edges: list[TypedEdge] = []
    for row in graph.query(query):
        if isinstance(row, ResultRow):
            edges.append(
                TypedEdge(
                    source_type=str(row.stype),
                    predicate=str(row.prop),
                    target_type=str(row.vtype),
                )
            )
    return edges


def extract_schema_from_sbol(
    file_path: str | Path,
    *,
    exclude_type_uri: str | None = None,
) -> tuple[list[str], list[TypedEdge]]:
    """Convenience loader for node types + typed edges from an SBOL XML file."""
    graph = load_graph(file_path, format="xml")
    node_types = extract_node_types(graph, exclude_uri=exclude_type_uri)
    edges = extract_typed_edges(graph, allowed_types=node_types)
    return node_types, edges
