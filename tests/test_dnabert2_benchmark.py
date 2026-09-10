from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from seqtrainer.benchmarks.config import load_benchmark_config
from seqtrainer.torch.dnabert2_benchmark import _normalize_binary_labels, _select_dnabert2_threshold

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config-examples" / "benchmarks" / "dnabert2_finetune.toml"


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
