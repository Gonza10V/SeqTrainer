"""Tests for Alpine offline benchmark workflows."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from seqtrainer.benchmarks.config import load_benchmark_config
from seqtrainer.adapters.ipromp_offline import (
    FastaRecord,
    average_fold_probabilities,
    expected_fold_paths,
    read_fasta,
    sequence_to_kmer_text,
    truncation_metadata,
)
from seqtrainer.torch.dnabert2_benchmark import _validate_local_dnabert2_dir
from tools.offline_prep.validate_offline_bundle import validate_bundle


def test_alpine_offline_configs_disable_downloads() -> None:
    dnabert = load_benchmark_config(ROOT / "config-examples/benchmarks/dnabert2_finetune_alpine_offline.toml")
    ipromp = load_benchmark_config(ROOT / "config-examples/benchmarks/ipromp_external_alpine_offline.toml")

    assert dnabert.model.family == "dnabert2"
    assert dnabert.model.params["allow_download"] is False
    assert dnabert.model.params["require_model_files"] is True
    assert dnabert.model.params["local_model_dir"] == "/models/DNABERT-2-117M"
    assert dnabert.evaluation.threshold_strategy == "validation_mcc"

    assert ipromp.model.family == "ipromp"
    assert ipromp.model.params["species_id"] == 10
    assert ipromp.model.params["token_max_length"] == 128
    assert len(ipromp.model.params["fold_checkpoint_names"]) == 5
    assert ipromp.evaluation.threshold_strategy == "validation_mcc"


def test_dnabert2_local_dir_validation_accepts_bin_or_safetensors(tmp_path: Path) -> None:
    model_dir = tmp_path / "DNABERT-2-117M"
    model_dir.mkdir()
    for name in ("config.json", "configuration_bert.py", "bert_layers.py", "bert_padding.py", "tokenizer.json"):
        (model_dir / name).write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"fake")

    _validate_local_dnabert2_dir(model_dir, disable_flash_attention=True, require_model_files=True)


def test_dnabert2_local_dir_validation_reports_missing_files(tmp_path: Path) -> None:
    model_dir = tmp_path / "DNABERT-2-117M"
    model_dir.mkdir()
    with pytest.raises(Exception, match="Missing"):
        _validate_local_dnabert2_dir(model_dir, disable_flash_attention=False, require_model_files=True)


def test_ipromp_fasta_and_kmer_helpers_preserve_ids(tmp_path: Path) -> None:
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">row_a extra text\nACGTU\n>row_b\nNNACGT\n", encoding="utf-8")
    records = read_fasta(fasta)

    assert [record.id for record in records] == ["row_a", "row_b"]
    assert records[0].sequence == "ACGTT"
    assert sequence_to_kmer_text("ACGTACGT", k=6) == "ACGTAC CGTACG GTACGT"


def test_ipromp_fold_paths_average_and_truncation_metadata() -> None:
    paths = expected_fold_paths("/models/ipromp_ecoli", species_id=10)
    assert [path.name for path in paths] == [f"10_fold_{fold}.pth" for fold in range(1, 6)]

    averaged = average_fold_probabilities(
        [
            __import__("numpy").array([0.1, 0.9]),
            __import__("numpy").array([0.2, 0.8]),
            __import__("numpy").array([0.3, 0.7]),
            __import__("numpy").array([0.4, 0.6]),
            __import__("numpy").array([0.5, 0.5]),
        ]
    )
    assert averaged.tolist() == pytest.approx([0.3, 0.7])

    metadata = truncation_metadata(
        [FastaRecord("short", "ACGTAC"), FastaRecord("long", "A" * 200)],
        token_max_length=128,
        species_id=10,
        checkpoint_names=[path.name for path in paths],
    )
    assert metadata["rows_truncated"] == 1
    assert metadata["species_id"] == 10
    assert metadata["token_max_length"] == 128


def test_offline_bundle_validator_reports_missing_bundle(tmp_path: Path) -> None:
    errors = validate_bundle(tmp_path / "missing", model="dnabert2")
    assert errors
    assert any("Missing required" in error for error in errors)


def test_offline_bundle_validator_accepts_minimal_ipromp_bundle(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _touch(root / "image/seqtrainer-alpine-gpu.sif")
    _touch(root / "repository/SeqTrainer/config-examples/benchmarks/ipromp_external_alpine_offline.toml")
    _touch(root / "manifests/offline_bundle_manifest.json")
    _touch(root / "manifests/repository_revision.txt")
    for name in ("config.json", "pytorch_model.bin", "vocab.txt"):
        _touch(root / f"models/DNABERT-6/{name}")
    for fold in range(1, 6):
        _touch(root / f"models/ipromp_ecoli/10_fold_{fold}.pth")
    for filename in (
        "train_EP_DNA_BERT2_genomic_order.csv",
        "eval_EP_DNA_BERT2_genomic_order.csv",
        "test_EP_DNA_BERT2_genomic_order.csv",
    ):
        _write_split(root / f"data/promoter_classification/{filename}")

    assert validate_bundle(root, model="ipromp") == []


def test_production_slurm_scripts_are_offline_only() -> None:
    scripts = [
        ROOT / "notebooks/benchmarks_sg/sg_hpc/dnabert2_alpine/run_dnabert2_finetune_alpine.sbatch",
        ROOT / "notebooks/benchmarks_sg/sg_hpc/ipromp_alpine/run_ipromp_alpine.sbatch",
    ]
    forbidden = (" git ", "git clone", "git pull", "pip install", "conda ", "wget ", "curl ", "http://", "https://")
    for script in scripts:
        text = f" {script.read_text(encoding='utf-8')} "
        assert "HF_HUB_OFFLINE=1" in text
        assert "apptainer exec" in text
        for token in forbidden:
            assert token not in text


def test_slurm_scripts_parse_with_bash_when_available() -> None:
    bash = __import__("shutil").which("bash")
    if bash is None:
        pytest.skip("bash is not available")
    for script in (
        ROOT / "notebooks/benchmarks_sg/sg_hpc/dnabert2_alpine/run_dnabert2_finetune_alpine.sbatch",
        ROOT / "notebooks/benchmarks_sg/sg_hpc/ipromp_alpine/run_ipromp_alpine.sbatch",
    ):
        try:
            subprocess.run([bash, "-n", str(script)], check=True)
        except OSError as exc:
            pytest.skip(f"bash launcher is present but not usable: {exc}")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def _write_split(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sequence", "label"))
        writer.writeheader()
        writer.writerow({"sequence": "ACGT", "label": "0"})
        writer.writerow({"sequence": "TGCA", "label": "1"})
