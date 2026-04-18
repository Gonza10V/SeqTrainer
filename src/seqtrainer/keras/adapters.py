"""Keras adapters that tensorize `MaterializedDataset` examples."""

from __future__ import annotations

from dataclasses import dataclass, field

from seqtrainer.data.materialized import MaterializedDataset
from seqtrainer.data.tensorization import SequenceTensorizationConfig, tensorize_materialized_dataset


@dataclass(slots=True)
class KerasAdapterConfig:
    """Config for converting MaterializedDataset into tf.data pipelines."""

    tensorization: SequenceTensorizationConfig = field(default_factory=SequenceTensorizationConfig)
    batch_size: int = 32
    shuffle: bool = False
    prefetch: bool = True


def to_tf_dataset(dataset: MaterializedDataset, config: KerasAdapterConfig | None = None):
    """Convert MaterializedDataset into a `tf.data.Dataset` pipeline."""
    cfg = config or KerasAdapterConfig()

    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover
        raise ImportError("Install seqtrainer[keras] to use Keras adapters") from exc

    features_np, labels_np = tensorize_materialized_dataset(dataset, cfg.tensorization)

    if labels_np is not None:
        ds = tf.data.Dataset.from_tensor_slices((features_np, labels_np))
    else:
        ds = tf.data.Dataset.from_tensor_slices(features_np)

    if cfg.shuffle:
        ds = ds.shuffle(buffer_size=max(len(dataset.examples), 1))
    ds = ds.batch(cfg.batch_size)
    if cfg.prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
