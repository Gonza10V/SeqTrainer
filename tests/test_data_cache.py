from pathlib import Path

from seqtrainer.data.cache import list_snapshot_versions, load_snapshot
from seqtrainer.data.materialized import MaterializedDataset
from seqtrainer.data.recipes import DatasetRecipe
from seqtrainer.data.sbol import materialize_dataset_from_sbol


def test_snapshot_roundtrip(tmp_path: Path):
    dataset = MaterializedDataset(
        examples=[{"sequence": "ACGT", "target": 1.0}, {"sequence": "TGCA", "target": 2.0}],
        metadata={"source": "unit-test"},
    )
    recipe = DatasetRecipe(name="unit-recipe", query="SELECT ?s WHERE {?s ?p ?o}", label_field="target")

    manifest = dataset.save_snapshot(
        dataset_name="unit-dataset",
        dataset_version="v1",
        recipe=recipe.as_manifest_payload(),
        cache_dir=str(tmp_path),
    )

    loaded, loaded_manifest = load_snapshot(
        dataset_name="unit-dataset",
        dataset_version="v1",
        cache_dir=str(tmp_path),
    )

    assert manifest.dataset_version == "v1"
    assert loaded.examples == dataset.examples
    assert loaded.metadata == dataset.metadata
    assert loaded_manifest.fingerprint_sha256 == manifest.fingerprint_sha256


def test_snapshot_versions_listing(tmp_path: Path):
    dataset = MaterializedDataset(examples=[{"sequence": "ACGT", "target": 1.0}])
    dataset.save_snapshot(dataset_name="versioned", dataset_version="v1", cache_dir=str(tmp_path))
    dataset.save_snapshot(dataset_name="versioned", dataset_version="v2", cache_dir=str(tmp_path))

    versions = list_snapshot_versions("versioned", cache_dir=str(tmp_path))
    assert versions == ["v1", "v2"]


def test_materialize_from_sbol_and_cache(tmp_path: Path):
    fixture = Path("data/sbol_data/sample_design_0.xml")

    dataset, manifest = materialize_dataset_from_sbol(
        [fixture],
        dataset_name="sbol-fixture",
        dataset_version="fixture-v1",
        cache_dir=tmp_path,
        write_cache=True,
    )

    assert len(dataset.examples) == 1
    assert manifest is not None

    loaded, _ = MaterializedDataset.load_snapshot(
        dataset_name="sbol-fixture",
        dataset_version="fixture-v1",
        cache_dir=str(tmp_path),
    )
    assert loaded.examples[0]["sequence"] == dataset.examples[0]["sequence"]
