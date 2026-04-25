"""Config helpers for AutoRDF2GML-style graph pipelines."""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path

from .rdf import TypedEdge


@dataclass(slots=True)
class GraphConfigSpec:
    """Specification for writing graph conversion configs."""

    input_path: str
    save_path_numeric_graph: str
    save_path_mapping: str
    node_uris: list[str]
    edges: list[TypedEdge]
    nld_class: str | None = None
    embedding_model: str = "allenai/scibert_scivocab_uncased"
    kge_model: str = "distmult"
    n_hop_edges: list[list[str]] = field(default_factory=list)


def _uri_tail(uri: str) -> str:
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rstrip("/").split("/")[-1]


def build_config(spec: GraphConfigSpec) -> configparser.ConfigParser:
    """Build an in-memory config parser from graph spec."""
    cfg = configparser.ConfigParser()
    cfg.optionxform = str  # type: ignore[assignment]

    classes = {_uri_tail(uri): uri for uri in spec.node_uris}
    class_names = list(classes.keys())

    cfg["InputPath"] = {"input_path": spec.input_path}
    cfg["SavePath"] = {
        "save_path_numeric_graph": spec.save_path_numeric_graph,
        "save_path_mapping": spec.save_path_mapping,
    }
    cfg["MODEL"] = {"kge_model": spec.kge_model}
    cfg["EMBEDDING"] = {"embedding_model": spec.embedding_model}
    cfg["NLD"] = {"nld_class": spec.nld_class or class_names[0]}

    cfg["Nodes"] = {"classes": ", ".join(class_names), **classes}

    edge_names: list[str] = []
    simple_edges: dict[str, str] = {}
    for edge in spec.edges:
        start_node = _uri_tail(edge.source_type)
        end_node = _uri_tail(edge.target_type)
        edge_name = f"{start_node}_{end_node}"
        edge_names.append(edge_name)
        simple_edges[f"{edge_name}_start_node"] = start_node
        simple_edges[f"{edge_name}_properties"] = edge.predicate
        simple_edges[f"{edge_name}_end_node"] = end_node

    cfg["SimpleEdges"] = {"edge_names": ", ".join(edge_names), **simple_edges}
    cfg["N-HopEdges"] = {}

    if spec.n_hop_edges:
        apply_n_hop_edges(cfg, spec.n_hop_edges)

    return cfg


def apply_n_hop_edges(cfg: configparser.ConfigParser, n_hop_edges: list[list[str]]) -> None:
    """Apply n-hop edge definitions to existing config in-place."""
    simple_edge_names = [edge for edge in cfg["SimpleEdges"].get("edge_names", "").split(", ") if edge]
    n_hop_dict: dict[str, str] = {}
    consumed_edges: set[str] = set()

    for chain in n_hop_edges:
        if len(chain) < 2:
            raise ValueError("n-hop chain must contain at least two simple edges")

        first, last = chain[0], chain[-1]
        final_name = f"{first.split('_')[0]}_{last.split('_')[-1]}"
        n_hop_dict[f"{final_name}_start_node"] = cfg["SimpleEdges"][f"{first}_start_node"]
        n_hop_dict[f"{final_name}_end_node"] = cfg["SimpleEdges"][f"{last}_end_node"]

        for hop_idx, edge_name in enumerate(chain, start=1):
            if edge_name not in simple_edge_names:
                raise ValueError(f"Edge {edge_name} is not defined in SimpleEdges")
            n_hop_dict[f"{final_name}_hop{hop_idx}_properties"] = cfg["SimpleEdges"][f"{edge_name}_properties"]
            consumed_edges.add(edge_name)

    new_simple = [edge for edge in simple_edge_names if edge not in consumed_edges]
    cfg["SimpleEdges"]["edge_names"] = ", ".join(new_simple)

    n_hop_names = [f"{chain[0].split('_')[0]}_{chain[-1].split('_')[-1]}" for chain in n_hop_edges]
    cfg["N-HopEdges"] = {"edge_names": ", ".join(n_hop_names), **n_hop_dict}


def write_config(spec: GraphConfigSpec, output_path: str | Path) -> Path:
    """Write graph config to disk and return output path."""
    cfg = build_config(spec)
    path = Path(output_path)
    with path.open("w", encoding="utf-8") as fh:
        cfg.write(fh)
    return path
