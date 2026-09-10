from dataclasses import asdict
from pathlib import Path

import pytest

from seqtrainer.benchmarks import (
    ConfigValidationError,
    REQUIRED_CLASSIFICATION_METRICS,
    load_benchmark_config,
)
from seqtrainer.benchmarks.config import parse_benchmark_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config-examples" / "benchmarks"
def test_example_benchmark_configs_load():
    for name in (
        "cnn.toml",
        "cnn_v2.toml",
        "dnabert2_smoke.toml",
        "dnabert2_frozen.toml",
        "dnabert2_finetune.toml",
        "ipromp.toml",
        "ipromp_external.toml",
    ):
        config = load_benchmark_config(CONFIG_DIR / name)
        assert config.dataset.name == "ep_dnabert2_genomic_order"
        assert config.split.strategy == "predefined"
        assert set(REQUIRED_CLASSIFICATION_METRICS).issubset(config.evaluation.metrics)


def test_model_examples_share_dataset_and_split_contract():
    configs = [
        load_benchmark_config(CONFIG_DIR / name)
        for name in (
            "cnn.toml",
            "cnn_v2.toml",
            "dnabert2_frozen.toml",
            "dnabert2_finetune.toml",
            "ipromp_external.toml",
        )
    ]
    dataset_names = {config.dataset.name for config in configs}
    split_files = {tuple(sorted(config.dataset.split_files.items())) for config in configs}
    split_strategies = {config.split.strategy for config in configs}

    assert dataset_names == {"ep_dnabert2_genomic_order"}
    assert len(split_files) == 1
    assert split_strategies == {"predefined"}


def test_t4_profiles_preserve_the_shared_scientific_contract():
    dnabert2 = load_benchmark_config(
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "benchmarks"
        / "dnabert2"
        / "config.toml"
    )
    ipromp = load_benchmark_config(
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "benchmarks"
        / "ipromp"
        / "config.toml"
    )
    assert dnabert2.dataset.split_files == ipromp.dataset.split_files
    assert dnabert2.training.seed == ipromp.training.seed == 42
    assert dnabert2.evaluation.threshold_strategy == "validation_mcc"
    assert ipromp.evaluation.threshold_strategy == "validation_mcc"
    assert set(REQUIRED_CLASSIFICATION_METRICS).issubset(dnabert2.evaluation.metrics)
    assert set(REQUIRED_CLASSIFICATION_METRICS).issubset(ipromp.evaluation.metrics)

    assert dnabert2.model.name == "zhihan1996/DNABERT-2-117M"
    assert dnabert2.model.params["mode"] == "full_finetune"
    assert dnabert2.training.batch_size == 2
    assert dnabert2.training.params["gradient_accumulation_steps"] == 16
    assert dnabert2.environment.precision == "fp16"

    assert ipromp.model.params["folds"] == 5
    assert ipromp.model.params["species_id"] == 10
    assert ipromp.training.max_epochs == 0


@pytest.fixture
def raw_config():
    raw = asdict(load_benchmark_config(CONFIG_DIR / "cnn.toml"))
    raw["evaluation"]["metrics"] = list(raw["evaluation"]["metrics"])
    return raw


def test_missing_required_section_fails_clearly():
    with pytest.raises(ConfigValidationError, match="missing required section"):
        parse_benchmark_config({}, source="demo.toml")


def test_invalid_model_family_fails_clearly(raw_config):
    raw_config["model"]["family"] = "random_forest"
    with pytest.raises(ConfigValidationError, match="model.family"):
        parse_benchmark_config(raw_config)


@pytest.mark.parametrize("strategy", ["train_val_test", "k_fold", "stratified_group_k_fold"])
def test_unimplemented_split_strategy_fails_during_config_validation(raw_config, strategy):
    raw_config["split"]["strategy"] = strategy
    with pytest.raises(ConfigValidationError, match="not implemented by the benchmark runners"):
        parse_benchmark_config(raw_config)


def test_unimplemented_dataset_format_fails_during_config_validation(raw_config):
    raw_config["dataset"]["format"] = "fasta"
    with pytest.raises(ConfigValidationError, match="dataset.format"):
        parse_benchmark_config(raw_config)


def test_required_metric_suite_is_enforced(raw_config):
    raw_config["evaluation"]["metrics"] = ["accuracy", "mcc"]
    with pytest.raises(ConfigValidationError, match="evaluation.metrics"):
        parse_benchmark_config(raw_config)


def test_numeric_target_threshold_is_rejected_as_unimplemented(raw_config):
    raw_config["label"]["source"] = "numeric_target_threshold"
    with pytest.raises(ConfigValidationError, match="not implemented"):
        parse_benchmark_config(raw_config)


def test_cnn_rejects_disabled_pad_or_trim(raw_config):
    raw_config["preprocessing"]["pad_or_trim"] = False
    with pytest.raises(ConfigValidationError, match="pad_or_trim=true"):
        parse_benchmark_config(raw_config)


def test_cnn_rejects_unsupported_loss(raw_config):
    raw_config["training"]["params"]["loss"] = "bce_with_logits"
    with pytest.raises(ConfigValidationError, match="only implement"):
        parse_benchmark_config(raw_config)


def test_cnn_rejects_unimplemented_precision(raw_config):
    raw_config["environment"]["precision"] = "fp16"
    with pytest.raises(ConfigValidationError, match="only implement"):
        parse_benchmark_config(raw_config)
