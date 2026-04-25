from pathlib import Path

from seqtrainer.graph import (
    GraphConfigSpec,
    apply_n_hop_edges,
    build_config,
    extract_schema_from_sbol,
    write_config,
)


def test_extract_schema_from_sbol_fixture():
    fixture = Path("data/sbol_data/sample_design_0.xml")
    node_types, edges = extract_schema_from_sbol(fixture)

    assert len(node_types) > 0
    assert len(edges) > 0
    assert all(edge.source_type for edge in edges)


def test_build_config_has_sections_and_edges(tmp_path: Path):
    fixture = Path("data/sbol_data/sample_design_0.xml")
    node_types, edges = extract_schema_from_sbol(fixture)

    spec = GraphConfigSpec(
        input_path=str(fixture),
        save_path_numeric_graph=str(tmp_path / "numeric"),
        save_path_mapping=str(tmp_path / "mapping"),
        node_uris=node_types[:3],
        edges=edges[:3],
        nld_class="ComponentDefinition",
    )

    cfg = build_config(spec)
    assert "InputPath" in cfg
    assert "Nodes" in cfg
    assert "SimpleEdges" in cfg


def test_apply_n_hop_edges_updates_sections(tmp_path: Path):
    fixture = Path("data/sbol_data/sample_design_0.xml")
    node_types, edges = extract_schema_from_sbol(fixture)

    spec = GraphConfigSpec(
        input_path=str(fixture),
        save_path_numeric_graph=str(tmp_path / "numeric"),
        save_path_mapping=str(tmp_path / "mapping"),
        node_uris=node_types,
        edges=edges,
    )
    cfg = build_config(spec)

    edge_names = [edge for edge in cfg["SimpleEdges"]["edge_names"].split(", ") if edge]
    if len(edge_names) >= 2:
        apply_n_hop_edges(cfg, [edge_names[:2]])
        assert "edge_names" in cfg["N-HopEdges"]


def test_write_config_roundtrip(tmp_path: Path):
    fixture = Path("data/sbol_data/sample_design_0.xml")
    node_types, edges = extract_schema_from_sbol(fixture)

    spec = GraphConfigSpec(
        input_path=str(fixture),
        save_path_numeric_graph=str(tmp_path / "numeric"),
        save_path_mapping=str(tmp_path / "mapping"),
        node_uris=node_types[:4],
        edges=edges[:4],
    )

    output = write_config(spec, tmp_path / "graph_config.ini")
    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "[InputPath]" in content
    assert "[SimpleEdges]" in content
