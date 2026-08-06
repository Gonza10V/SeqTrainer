"""Offline tests for the ProkBERT benchmark integration."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from seqtrainer.benchmarks import load_benchmark_config, run_benchmark
from seqtrainer.benchmarks.runner import BenchmarkRunResult, BenchmarkSkipped


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config-examples" / "benchmarks" / "prokbert_mini.toml"


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, sequences, **kwargs):
        rows = []
        masks = []
        for sequence in sequences:
            ids = ["ACGTN".index(base) + 1 for base in str(sequence)]
            rows.append(ids)
            masks.append([1] * len(ids))
        width = max(len(row) for row in rows)
        max_length = kwargs.get("max_length")
        if kwargs.get("truncation") and max_length:
            rows = [row[:max_length] for row in rows]
            masks = [mask[:max_length] for mask in masks]
            width = min(width, max_length)
        rows = [row + [0] * (width - len(row)) for row in rows]
        masks = [mask + [0] * (width - len(mask)) for mask in masks]
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("fake", encoding="utf-8")


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(6, 4)
        self.config = SimpleNamespace(hidden_size=4, use_cache=True)
        self.gradient_checkpointing_was_enabled = False

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_was_enabled = True

    def forward(self, *, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def _config_with_tmp_splits(tmp_path):
    config = load_benchmark_config(CONFIG_PATH)
    split_paths = {}
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.csv"
        pd.DataFrame(
            {
                "sequence": ["ACGT" * 5, "TGCA" * 5, "AAAA" * 5, "CCCC" * 5, "GGGG" * 5, "TTTT" * 5],
                "label": [0, 1, 0, 1, 0, 1],
            }
        ).to_csv(path, index=False)
        split_paths[split] = str(path)
    return replace(
        config,
        dataset=replace(config.dataset, split_files=split_paths),
        model=replace(config.model, params={**config.model.params, "allow_download": False}),
        training=replace(
            config.training,
            batch_size=2,
            max_epochs=1,
            learning_rate=1e-3,
            params={
                **config.training.params,
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": True,
            },
        ),
        environment=replace(config.environment, device="cpu", precision="float32"),
    )


def test_prokbert_family_is_accepted_and_invalid_family_remains_rejected():
    config = load_benchmark_config(CONFIG_PATH)
    assert config.model.family == "prokbert"
    raw = {
        "experiment": config.experiment.__dict__,
        "dataset": {**config.dataset.__dict__, "split_files": dict(config.dataset.split_files)},
        "label": config.label.__dict__,
        "split": config.split.__dict__,
        "preprocessing": {**config.preprocessing.__dict__, "params": dict(config.preprocessing.params)},
        "model": {**config.model.__dict__, "params": dict(config.model.params), "family": "not_a_model"},
        "training": {**config.training.__dict__, "params": dict(config.training.params)},
        "evaluation": {**config.evaluation.__dict__, "metrics": list(config.evaluation.metrics)},
        "outputs": config.outputs.__dict__,
        "environment": {**config.environment.__dict__, "params": dict(config.environment.params)},
    }
    from seqtrainer.benchmarks.config import ConfigValidationError, parse_benchmark_config

    with pytest.raises(ConfigValidationError):
        parse_benchmark_config(raw)


def test_mean_pooling_ignores_padding_and_cls_shape():
    from seqtrainer.torch.prokbert_benchmark import ProkBERTClassifier, mean_pool_hidden_states

    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(mean_pool_hidden_states(hidden, mask), torch.tensor([[2.0, 3.0]]))
    classifier = ProkBERTClassifier(FakeEncoder(), hidden_size=4, pooling="cls")
    logits = classifier(torch.tensor([[1, 2, 0]]), torch.tensor([[1, 1, 0]]))
    assert logits.shape == (1,)


def test_one_training_step_is_finite_and_artifacts_follow_standard_schema(tmp_path):
    from seqtrainer.torch.prokbert_benchmark import run_prokbert_csv_splits

    config = _config_with_tmp_splits(tmp_path)
    result = run_prokbert_csv_splits(
        config,
        output_dir=tmp_path / "run",
        tokenizer=FakeTokenizer(),
        encoder=FakeEncoder(),
    )
    assert result.status == "completed"
    assert np.isfinite(result.metrics["validation"]["loss"])
    assert (tmp_path / "run" / "best_checkpoint.pt").exists()
    assert (tmp_path / "run" / "tokenizer").exists()
    predictions = pd.read_csv(tmp_path / "run" / "predictions.csv")
    assert len(predictions) == 18
    assert {"split", "row_index", "label", "probability", "prediction", "threshold"}.issubset(predictions.columns)
    manifest = pd.read_json(tmp_path / "run" / "manifest.json", typ="series")
    assert manifest["model"]["metadata"]["model_revision"] == config.model.version
    assert set(manifest["dataset"]["split_summary"]["file_hashes"]) == {"train", "validation", "test"}


def test_runner_dispatches_to_prokbert(monkeypatch, tmp_path):
    import seqtrainer.benchmarks.runner as runner

    config = load_benchmark_config(CONFIG_PATH)
    expected = BenchmarkRunResult(tmp_path, "completed", {}, {})
    fake_module = types.ModuleType("seqtrainer.torch.prokbert_benchmark")
    fake_module.run_prokbert_csv_splits = lambda *args, **kwargs: expected
    monkeypatch.setitem(sys.modules, "seqtrainer.torch.prokbert_benchmark", fake_module)
    assert runner.run_benchmark(config, output_dir=tmp_path / "out") is expected


def test_optional_dependency_failure_only_skips_when_allowed(monkeypatch, tmp_path):
    import seqtrainer.benchmarks.runner as runner

    config = load_benchmark_config(CONFIG_PATH)
    fake_module = types.ModuleType("seqtrainer.torch.prokbert_benchmark")

    def unavailable(*args, **kwargs):
        raise BenchmarkSkipped("fake transformers unavailable")

    fake_module.run_prokbert_csv_splits = unavailable
    monkeypatch.setitem(sys.modules, "seqtrainer.torch.prokbert_benchmark", fake_module)
    skipped = runner.run_benchmark(config, base_dir=tmp_path, output_dir=tmp_path / "skip", allow_skip=True)
    assert skipped.status == "skipped"
    with pytest.raises(BenchmarkSkipped):
        runner.run_benchmark(config, base_dir=tmp_path, output_dir=tmp_path / "fail", allow_skip=False)


def test_validation_threshold_is_selected_without_test_data(monkeypatch, tmp_path):
    import seqtrainer.torch.prokbert_benchmark as module

    config = _config_with_tmp_splits(tmp_path)
    calls = []
    original = module.best_threshold_by_metric

    def wrapped(labels, scores, **kwargs):
        calls.append((len(labels), kwargs["metric"]))
        return original(labels, scores, **kwargs)

    monkeypatch.setattr(module, "best_threshold_by_metric", wrapped)
    module.run_prokbert_csv_splits(config, output_dir=tmp_path / "run", tokenizer=FakeTokenizer(), encoder=FakeEncoder())
    assert calls == [(6, "mcc")]


def test_oom_fallback_preserves_effective_batch_size():
    from seqtrainer.torch.prokbert_benchmark import resolve_batch_profile

    assert resolve_batch_profile(16, 2) == (16, 2, 32)
    assert resolve_batch_profile(16, 2, cuda_oom=True) == (8, 4, 32)
    assert resolve_batch_profile(8, 4, cuda_oom=True) == (8, 4, 32)


def test_non_oom_runtime_errors_are_not_swallowed(monkeypatch, tmp_path):
    import seqtrainer.benchmarks.runner as runner

    config = load_benchmark_config(CONFIG_PATH)
    fake_module = types.ModuleType("seqtrainer.torch.prokbert_benchmark")
    fake_module.run_prokbert_csv_splits = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("programming error"))
    monkeypatch.setitem(sys.modules, "seqtrainer.torch.prokbert_benchmark", fake_module)
    with pytest.raises(RuntimeError, match="programming error"):
        runner.run_benchmark(config, base_dir=tmp_path, output_dir=tmp_path / "run", allow_skip=True)


def test_resume_rejects_mismatched_configuration():
    from seqtrainer.torch.prokbert_benchmark import validate_resume_checkpoint

    expected = {
        "config_hash": "config-a",
        "split_hashes": {"train": "hash-a"},
        "model_revision": "revision-a",
        "seqtrainer_commit": "commit-a",
    }
    with pytest.raises(ValueError, match="config_hash"):
        validate_resume_checkpoint({**expected, "config_hash": "config-b"}, expected)


def test_smoke_mode_is_not_allowed_to_update_final_results():
    import json

    notebook = json.loads((ROOT / "notebooks" / "colab_benchmarks" / "prokbert_mini_t4_colab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(line for cell in notebook["cells"] for line in cell.get("source", []))
    assert 'RUN_MODE = "smoke"' in source
    assert "Results_Final.md" in source
    assert "if RUN_MODE == 'smoke'" in source
