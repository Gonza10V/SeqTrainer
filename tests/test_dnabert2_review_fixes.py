from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from seqtrainer.benchmarks.ai_x_bio import split_ai_x_bio_frame, standardize_ai_x_bio_frame
from seqtrainer.adapters.ipromp import (
    build_ipromp_mapping,
    normalize_ipromp_predictions,
    _normalize_official_predictions,
    write_ipromp_fastas,
    write_ipromp_run_commands,
)
from seqtrainer.benchmarks.config import load_benchmark_config
from seqtrainer.torch.dnabert2_benchmark import (
    _normalize_binary_labels,
    _select_dnabert2_threshold,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config-examples"
    / "benchmarks"
    / "ipromp_external.toml"
)


def test_dnabert2_maps_configured_string_labels():
    config = load_benchmark_config(CONFIG_PATH)
    config = replace(
        config,
        label=replace(
            config.label,
            negative_label="background",
            positive_label="promoter",
        ),
    )
    frame = pd.DataFrame(
        {
            "sequence": ["ACGT", "TGCA", "AAAA"],
            "label": ["background", "promoter", "promoter"],
        }
    )

    labels = _normalize_binary_labels(config, frame)

    assert labels.tolist() == [0.0, 1.0, 1.0]


def test_dnabert2_respects_reversed_numeric_label_configuration():
    config = load_benchmark_config(CONFIG_PATH)
    config = replace(
        config,
        label=replace(config.label, negative_label=1, positive_label=0),
    )
    frame = pd.DataFrame(
        {
            "sequence": ["ACGT", "TGCA"],
            "label": [1, 0],
        }
    )

    labels = _normalize_binary_labels(config, frame)

    assert labels.tolist() == [0.0, 1.0]


def test_dnabert2_rejects_unconfigured_labels():
    config = load_benchmark_config(CONFIG_PATH)
    frame = pd.DataFrame(
        {
            "sequence": ["ACGT"],
            "label": ["unknown"],
        }
    )

    with pytest.raises(ValueError, match="do not match configured"):
        _normalize_binary_labels(config, frame)


def test_ai_x_bio_rejects_missing_sequences():
    frame = pd.DataFrame(
        {
            "sequence": ["ACGT", None],
            "label": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="missing or empty sequences"):
        standardize_ai_x_bio_frame(frame)


def test_dnabert2_honors_configured_threshold_strategy():
    config = load_benchmark_config(CONFIG_PATH)
    config = replace(
        config,
        evaluation=replace(config.evaluation, threshold_strategy="validation_f1"),
    )

    threshold, score, metric = _select_dnabert2_threshold(
        config,
        labels=pd.Series([0, 1, 1, 0]).to_numpy(),
        probabilities=pd.Series([0.1, 0.7, 0.8, 0.2]).to_numpy(),
    )

    assert metric == "f1"
    assert threshold >= 0.0
    assert score >= 0.0


def test_ipromp_mapping_rejects_duplicate_configured_ids():
    config = load_benchmark_config(CONFIG_PATH)
    config = replace(
        config,
        dataset=replace(config.dataset, id_field="id"),
    )
    frames = {
        split: pd.DataFrame(
            {
                "sequence": ["ACGT", "TGCA"],
                "label": [0, 1],
                "id": ["same-id", "same-id"],
            }
        )
        for split in ("train", "validation", "test")
    }

    with pytest.raises(ValueError, match="duplicate split/sequence_id"):
        build_ipromp_mapping(config, frames)


def test_ipromp_rejects_cached_mapping_from_different_splits(tmp_path):
    config = load_benchmark_config(CONFIG_PATH)
    original_frames = {
        split: pd.DataFrame(
            {"sequence": ["ACGT", "TGCA"], "label": [0, 1]}
        )
        for split in ("train", "validation", "test")
    }
    mapping_path = tmp_path / "mapping.csv"
    build_ipromp_mapping(config, original_frames).to_csv(mapping_path, index=False)

    changed_frames = {
        split: frame.copy()
        for split, frame in original_frames.items()
    }
    changed_frames["test"].loc[0, "sequence"] = "AAAA"
    predictions = pd.DataFrame(
        {
            "split": [row.split for row in build_ipromp_mapping(config, original_frames).itertuples()],
            "sequence_id": [row.sequence_id for row in build_ipromp_mapping(config, original_frames).itertuples()],
            "probability": 0.5,
        }
    )
    predictions_path = tmp_path / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    with pytest.raises(ValueError, match="does not match"):
        normalize_ipromp_predictions(
            config,
            mapping_csv=mapping_path,
            predictions_csv=predictions_path,
            frames=changed_frames,
        )


def test_ipromp_rejects_reordered_idless_labels(tmp_path):
    config = load_benchmark_config(CONFIG_PATH)
    frames = {
        split: pd.DataFrame(
            {"sequence": ["ACGT", "TGCA"], "label": [0, 1]}
        )
        for split in ("train", "validation", "test")
    }
    mapping = build_ipromp_mapping(config, frames)
    mapping_path = tmp_path / "mapping.csv"
    mapping.to_csv(mapping_path, index=False)
    predictions = mapping[["split", "label"]].copy()
    validation_rows = predictions["split"] == "validation"
    predictions.loc[validation_rows, "label"] = predictions.loc[validation_rows, "label"].iloc[::-1].to_numpy()
    predictions["probability"] = 0.5
    predictions_path = tmp_path / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    with pytest.raises(ValueError, match="labels do not match"):
        normalize_ipromp_predictions(
            config,
            mapping_csv=mapping_path,
            predictions_csv=predictions_path,
            frames=frames,
        )


def test_ipromp_official_predictions_with_ids_join_by_id_and_require_coverage():
    config = load_benchmark_config(CONFIG_PATH)
    frames = {
        split: pd.DataFrame({"sequence": ["ACGT", "TGCA"], "label": [0, 1]})
        for split in ("train", "validation", "test")
    }
    mapping = build_ipromp_mapping(config, frames)
    validation_mapping = mapping[mapping["split"] == "validation"].sort_values("row_index")
    predictions = pd.DataFrame(
        {
            "Sequence": ["TGCA", "ACGT"],
            "sequence_id": validation_mapping["sequence_id"].tolist()[::-1],
            "Probability": [0.8, 0.2],
        }
    )

    normalized = _normalize_official_predictions(config, predictions, mapping, "validation")

    assert normalized["sequence_id"].tolist() == validation_mapping["sequence_id"].tolist()
    assert normalized["probability"].tolist() == [0.2, 0.8]

    with pytest.raises(ValueError, match="exactly one row"):
        _normalize_official_predictions(config, predictions.iloc[:1], mapping, "validation")


def test_dnabert2_scales_final_partial_accumulation_window():
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset

    from seqtrainer.torch.dnabert2_benchmark import _run_epoch

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[0.0]]))

        def forward(self, input_ids, attention_mask):
            return input_ids.float() @ self.weight

    class NoOpScheduler:
        def step(self):
            return None

    model = TinyModel()
    loader = DataLoader(
        TensorDataset(
            torch.ones(3, 1),
            torch.ones(3, 1),
            torch.ones(3, 1),
        ),
        batch_size=1,
        shuffle=False,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    _run_epoch(
        model,
        loader,
        torch.nn.MSELoss(),
        optimizer,
        NoOpScheduler(),
        torch.device("cpu"),
        torch,
        gradient_accumulation_steps=2,
        max_grad_norm=100.0,
    )

    assert model.weight.item() == pytest.approx(0.36, abs=1e-6)


def test_ai_x_bio_unsplit_duplicates_stay_in_one_split():
    rows = [
        {"sequence": f"ACGT{i}", "label": i % 2, "id": str(i)}
        for i in range(8)
    ]
    rows.append({"sequence": "ACGT0", "label": 0, "id": "duplicate"})
    frames, strategy = split_ai_x_bio_frame(pd.DataFrame(rows), seed=42)

    assert strategy == "seeded_stratified_group_70_15_15"
    locations = [
        split
        for split, frame in frames.items()
        if "ACGT0" in set(frame["sequence"])
    ]
    assert len(locations) == 1


def test_ipromp_fasta_ids_are_encoded_and_command_preserves_kmer_size(tmp_path):
    config = load_benchmark_config(CONFIG_PATH)
    config = replace(
        config,
        dataset=replace(config.dataset, id_field="id"),
        model=replace(
            config.model,
            params={**config.model.params, "kmer_size": 7},
        ),
    )
    frames = {
        split: pd.DataFrame(
            {"sequence": ["ACGT"], "label": [0], "id": ["ref|ABC"]}
        )
        for split in ("train", "validation", "test")
    }
    mapping = build_ipromp_mapping(config, frames)
    fasta_paths = write_ipromp_fastas(config, frames, tmp_path / "fasta", mapping=mapping)
    command_path = write_ipromp_run_commands(config, fasta_paths, tmp_path / "run")

    assert "sequence_id=url:ref%7CABC|label=0" in fasta_paths["train"].read_text()
    assert "--kmer-size 7" in command_path.read_text()
