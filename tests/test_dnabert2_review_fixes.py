from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from seqtrainer.adapters.ipromp import build_ipromp_mapping
from seqtrainer.benchmarks.config import load_benchmark_config
from seqtrainer.torch.dnabert2_benchmark import _normalize_binary_labels


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
