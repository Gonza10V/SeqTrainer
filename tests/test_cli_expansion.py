import json
from pathlib import Path

from seqtrainer.cli.main import main


def test_dataset_recipes_command(capsys):
    code = main(["dataset", "recipes"])
    assert code == 0
    out = capsys.readouterr().out
    assert "local-sbol-regression" in out


def test_dataset_build_with_output_and_cache(tmp_path: Path):
    fixture = Path("data/sbol_data/sample_design_0.xml")
    output = tmp_path / "dataset.csv"

    code = main(
        [
            "dataset",
            "build",
            str(fixture),
            "--output",
            str(output),
            "--cache",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--dataset-name",
            "cli-test",
            "--dataset-version",
            "v1",
        ]
    )
    assert code == 0
    assert output.exists()
    header = output.read_text(encoding="utf-8").splitlines()[0]
    assert set(header.split(",")) == {"sequence", "target", "source"}


def test_dataset_build_sequence_only_jsonl(tmp_path: Path):
    fixture = Path("data/sbol_data/sample_design_0.xml")
    output = tmp_path / "dataset.jsonl"

    code = main(
        [
            "dataset",
            "build",
            str(fixture),
            "--recipe",
            "local-sbol-sequence-only",
            "--output",
            str(output),
            "--output-format",
            "jsonl",
        ]
    )
    assert code == 0
    payload = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert "sequence" in payload[0]
    assert "target" not in payload[0]


def test_model_build_torch_json(capsys):
    code = main(["model", "build", "--framework", "torch", "--task", "regression"])
    assert code == 0
    out = capsys.readouterr().out
    assert '"framework": "torch"' in out


def test_model_build_keras_to_file(tmp_path: Path):
    output = tmp_path / "keras_model.json"
    code = main(["model", "build", "--framework", "keras", "--output", str(output)])
    assert code == 0
    assert output.exists()
    assert '"framework": "keras"' in output.read_text(encoding="utf-8")
