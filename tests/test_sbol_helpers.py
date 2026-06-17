from pathlib import Path

import pandas as pd

from seqtrainer.data.sbol import build_dataset_from_files, dataset_validation_summary, get_sequence_from_sbol
from seqtrainer.cli.main import main


def test_get_sequence_from_sbol_fixture():
    fixture = Path("data/sbol_data/sample_design_0.xml")
    sequence = get_sequence_from_sbol(fixture)
    assert isinstance(sequence, str)
    assert len(sequence) > 0


def test_build_dataset_records_sbol_provenance_and_label_rule(tmp_path):
    fixture = Path("data/sbol_data/sample_design_0.xml")
    summary_path = tmp_path / "summary.json"
    warnings_path = tmp_path / "warnings.csv"

    frame = build_dataset_from_files(
        [fixture],
        label_threshold=50.0,
        summary_path=summary_path,
        warnings_path=warnings_path,
    )

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["source_file"] == "sample_design_0.xml"
    assert row["sequence_id"] == "promoter_seq_0"
    assert row["label_rule"] == "numeric_target_threshold"
    assert row["label_threshold"] == 50.0
    assert bool(row["dropped"]) is False
    assert summary_path.exists()
    assert warnings_path.exists()


def test_build_dataset_can_include_dropped_parse_failures(tmp_path):
    bad_file = tmp_path / "bad.xml"
    bad_file.write_text("<not-valid", encoding="utf-8")

    frame = build_dataset_from_files([bad_file], include_dropped=True)

    assert len(frame) == 1
    assert bool(frame.iloc[0]["dropped"]) is True
    assert "parse_error" in frame.iloc[0]["parsing_warnings"]


def test_dataset_validation_summary_counts_dropped_rows():
    frame = pd.DataFrame(
        {
            "dropped": [False, True],
            "parsing_warnings": ["", "missing_sequence"],
            "label_threshold": [0.5, 0.5],
            "label_rule": ["numeric_target_threshold", "numeric_target_threshold"],
        }
    )

    summary = dataset_validation_summary(frame, total_files=2)

    assert summary["total_files"] == 2
    assert summary["kept_rows"] == 1
    assert summary["dropped_rows"] == 1


def test_build_dataset_cli_writes_validation_artifacts(tmp_path, capsys):
    fixture = Path("data/sbol_data/sample_design_0.xml")
    summary_path = tmp_path / "summary.json"
    warnings_path = tmp_path / "warnings.csv"

    exit_code = main(
        [
            "build-dataset",
            str(fixture),
            "--label-threshold",
            "50",
            "--summary-path",
            str(summary_path),
            "--warnings-path",
            str(warnings_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "sequence_id" in captured.out
    assert summary_path.exists()
    assert warnings_path.exists()
