from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


REQUIRED_CLASSIFICATION_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "auroc",
    "auprc",
    "f1",
    "mcc",
    "precision",
    "sensitivity",
    "specificity",
    "confusion_matrix",
}

_ALLOWED_MODEL_FAMILIES = {"cnn", "dnabert2", "ipromp"}
_ALLOWED_DATASET_FORMATS = {"csv", "materialized_csv"}
_ALLOWED_LABEL_SOURCES = {
    "provided_binary",
    "curated_binary",
}
_ALLOWED_THRESHOLD_STRATEGIES = {
    "validation_mcc",
    "validation_f1",
    "validation_balanced_accuracy",
    "fixed_0_5",
    "none",
}


class ConfigValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    task: str
    seed: int
    description: str = ""


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    format: str
    sequence_field: str
    label_field: str
    source_accession: str = ""
    source_url: str = ""
    version: str = ""
    id_field: Optional[str] = None
    split_files: Mapping[str, str] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelConfig:
    source: str
    positive_label: Any = 1
    negative_label: Any = 0


@dataclass(frozen=True)
class SplitConfig:
    strategy: str
    seed: int
    validation_name: str = "validation"


@dataclass(frozen=True)
class PreprocessingConfig:
    encoding: str
    sequence_length: Optional[int] = None
    pad_or_trim: bool = True
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    family: str
    name: str
    version: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    batch_size: Optional[int] = None
    max_epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationConfig:
    primary_metric: str
    threshold_strategy: str
    metrics: tuple[str, ...]


@dataclass(frozen=True)
class OutputsConfig:
    output_dir: str
    save_json: bool = True
    save_csv: bool = True
    save_predictions: bool = True


@dataclass(frozen=True)
class EnvironmentConfig:
    device: str = "auto"
    precision: str = "float32"
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkConfig:
    experiment: ExperimentConfig
    dataset: DatasetConfig
    label: LabelConfig
    split: SplitConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    outputs: OutputsConfig
    environment: EnvironmentConfig


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    return parse_benchmark_config(raw, source=str(config_path))


def parse_benchmark_config(raw: Mapping[str, Any], source: str = "<memory>") -> BenchmarkConfig:
    _require_sections(
        raw,
        (
            "experiment",
            "dataset",
            "label",
            "split",
            "preprocessing",
            "model",
            "training",
            "evaluation",
            "outputs",
        ),
        source,
    )

    experiment_raw = _section(raw, "experiment")
    dataset_raw = _section(raw, "dataset")
    label_raw = _section(raw, "label")
    split_raw = _section(raw, "split")
    preprocessing_raw = _section(raw, "preprocessing")
    model_raw = _section(raw, "model")
    training_raw = _section(raw, "training")
    evaluation_raw = _section(raw, "evaluation")
    outputs_raw = _section(raw, "outputs")
    environment_raw = dict(raw.get("environment", {}))

    config = BenchmarkConfig(
        experiment=ExperimentConfig(
            name=_required_str(experiment_raw, "experiment.name", source),
            task=_required_str(experiment_raw, "experiment.task", source),
            seed=_required_int(experiment_raw, "experiment.seed", source),
            description=str(experiment_raw.get("description", "")),
        ),
        dataset=DatasetConfig(
            name=_required_str(dataset_raw, "dataset.name", source),
            format=_required_str(dataset_raw, "dataset.format", source),
            sequence_field=_required_str(dataset_raw, "dataset.sequence_field", source),
            label_field=_required_str(dataset_raw, "dataset.label_field", source),
            source_accession=str(dataset_raw.get("source_accession", "")),
            source_url=str(dataset_raw.get("source_url", "")),
            version=str(dataset_raw.get("version", "")),
            id_field=_optional_str(dataset_raw.get("id_field")),
            split_files=_string_mapping(dataset_raw.get("split_files", {}), "dataset.split_files", source),
            params=_mapping(dataset_raw.get("params", {}), "dataset.params", source),
        ),
        label=LabelConfig(
            source=_required_str(label_raw, "label.source", source),
            positive_label=label_raw.get("positive_label", 1),
            negative_label=label_raw.get("negative_label", 0),
        ),
        split=SplitConfig(
            strategy=_required_str(split_raw, "split.strategy", source),
            seed=_required_int(split_raw, "split.seed", source),
            validation_name=str(split_raw.get("validation_name", "validation")),
        ),
        preprocessing=PreprocessingConfig(
            encoding=_required_str(preprocessing_raw, "preprocessing.encoding", source),
            sequence_length=_optional_int(
                preprocessing_raw.get("sequence_length"),
                "preprocessing.sequence_length",
                source,
            ),
            pad_or_trim=bool(preprocessing_raw.get("pad_or_trim", True)),
            params=_mapping(preprocessing_raw.get("params", {}), "preprocessing.params", source),
        ),
        model=ModelConfig(
            family=_required_str(model_raw, "model.family", source),
            name=_required_str(model_raw, "model.name", source),
            version=str(model_raw.get("version", "")),
            params=_mapping(model_raw.get("params", {}), "model.params", source),
        ),
        training=TrainingConfig(
            seed=_required_int(training_raw, "training.seed", source),
            batch_size=_optional_int(training_raw.get("batch_size"), "training.batch_size", source),
            max_epochs=_optional_int(training_raw.get("max_epochs"), "training.max_epochs", source),
            learning_rate=_optional_float(training_raw.get("learning_rate"), "training.learning_rate", source),
            params=_mapping(training_raw.get("params", {}), "training.params", source),
        ),
        evaluation=EvaluationConfig(
            primary_metric=_required_str(evaluation_raw, "evaluation.primary_metric", source),
            threshold_strategy=_required_str(evaluation_raw, "evaluation.threshold_strategy", source),
            metrics=tuple(_string_list(evaluation_raw.get("metrics"), "evaluation.metrics", source)),
        ),
        outputs=OutputsConfig(
            output_dir=_required_str(outputs_raw, "outputs.output_dir", source),
            save_json=bool(outputs_raw.get("save_json", True)),
            save_csv=bool(outputs_raw.get("save_csv", True)),
            save_predictions=bool(outputs_raw.get("save_predictions", True)),
        ),
        environment=EnvironmentConfig(
            device=str(environment_raw.get("device", "auto")),
            precision=str(environment_raw.get("precision", "float32")),
            params=_mapping(environment_raw.get("params", {}), "environment.params", source),
        ),
    )

    _validate_config(config, source)
    return config


def _validate_config(config: BenchmarkConfig, source: str) -> None:
    _validate_allowed(config.dataset.format, _ALLOWED_DATASET_FORMATS, "dataset.format", source)
    _validate_allowed(config.label.source, _ALLOWED_LABEL_SOURCES, "label.source", source)
    _validate_allowed(config.model.family, _ALLOWED_MODEL_FAMILIES, "model.family", source)
    _validate_allowed(
        config.evaluation.threshold_strategy,
        _ALLOWED_THRESHOLD_STRATEGIES,
        "evaluation.threshold_strategy",
        source,
    )

    if config.split.seed != config.training.seed or config.split.seed != config.experiment.seed:
        raise ConfigValidationError(
            f"{source}: experiment.seed, split.seed, and training.seed must match for reproducibility"
        )

    if config.split.strategy == "predefined":
        required = {"train", "validation", "test"}
        missing = required.difference(config.dataset.split_files)
        if missing:
            raise ConfigValidationError(
                f"{source}: split.strategy='predefined' requires dataset.split_files keys "
                f"{sorted(required)}; missing {sorted(missing)}"
            )

    if config.split.strategy != "predefined":
        raise ConfigValidationError(
            f"{source}: split.strategy={config.split.strategy!r} is not implemented by the benchmark runners; "
            "use split.strategy='predefined' with train/validation/test CSV files."
        )

    if config.model.family == "cnn" and not config.preprocessing.pad_or_trim:
        raise ConfigValidationError(
            f"{source}: CNN benchmarks require preprocessing.pad_or_trim=true"
        )

    if config.model.family == "cnn":
        loss = str(config.training.params.get("loss", "cross_entropy")).lower()
        if loss != "cross_entropy":
            raise ConfigValidationError(
                f"{source}: CNN benchmarks only implement training.params.loss='cross_entropy'; "
                f"got {loss!r}"
            )
        precision = str(config.environment.precision).lower()
        if precision not in {"float32", "fp32"}:
            raise ConfigValidationError(
                f"{source}: CNN benchmarks only implement environment.precision='float32'; "
                f"got {config.environment.precision!r}"
            )

    if config.model.family == "dnabert2":
        pooling = str(config.model.params.get("pooling", "mean")).lower()
        if pooling not in {"mean", "cls"}:
            raise ConfigValidationError(
                f"{source}: DNABERT2 only implements model.params.pooling='mean' or 'cls'; "
                f"got {pooling!r}"
            )
        mode = str(config.model.params.get("mode", "frozen_embedding_classifier")).lower()
        precision = str(config.environment.precision).lower()
        if mode == "frozen_embedding_classifier" and precision in {"fp16", "float16"}:
            raise ConfigValidationError(
                f"{source}: frozen DNABERT2 classifier does not implement fp16 gradient scaling; "
                "use environment.precision='float32' or 'bf16'."
            )

    missing_metrics = REQUIRED_CLASSIFICATION_METRICS.difference(config.evaluation.metrics)
    if missing_metrics:
        raise ConfigValidationError(
            f"{source}: evaluation.metrics is missing required metrics {sorted(missing_metrics)}"
        )
    if config.evaluation.primary_metric not in config.evaluation.metrics:
        raise ConfigValidationError(
            f"{source}: evaluation.primary_metric must also be listed in evaluation.metrics"
        )


def _require_sections(raw: Mapping[str, Any], sections: tuple[str, ...], source: str) -> None:
    missing = [section for section in sections if section not in raw]
    if missing:
        raise ConfigValidationError(f"{source}: missing required section(s): {', '.join(missing)}")


def _section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw[key]
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"{key} must be a table")
    return value


def _required_str(raw: Mapping[str, Any], path: str, source: str) -> str:
    key = path.rsplit(".", 1)[-1]
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{source}: {path} is required and must be a non-empty string")
    return value


def _required_int(raw: Mapping[str, Any], path: str, source: str) -> int:
    key = path.rsplit(".", 1)[-1]
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigValidationError(f"{source}: {path} is required and must be an integer")
    return value


def _optional_int(value: Any, path: str, source: str) -> Optional[int]:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigValidationError(f"{source}: {path} must be an integer")
    return value


def _optional_float(value: Any, path: str, source: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(f"{source}: {path} must be numeric")
    return float(value)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _mapping(value: Any, path: str, source: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"{source}: {path} must be a table")
    return dict(value)


def _string_mapping(value: Any, path: str, source: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"{source}: {path} must be a table")
    out = {}
    for key, item in value.items():
        if not isinstance(item, str) or not item.strip():
            raise ConfigValidationError(f"{source}: {path}.{key} must be a non-empty string")
        out[str(key)] = item
    return out


def _string_list(value: Any, path: str, source: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConfigValidationError(f"{source}: {path} must be a non-empty list")
    out = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigValidationError(f"{source}: {path} entries must be non-empty strings")
        out.append(item)
    return out


def _validate_allowed(value: str, allowed: set[str], path: str, source: str) -> None:
    if value not in allowed:
        raise ConfigValidationError(
            f"{source}: {path} must be one of {sorted(allowed)}; got {value!r}"
        )
