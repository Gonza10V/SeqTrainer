"""PyTorch adapters that tensorize `MaterializedDataset` examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from seqtrainer.data.materialized import MaterializedDataset
from seqtrainer.data.tensorization import SequenceTensorizationConfig, tensorize_materialized_dataset


@dataclass(slots=True)
class TorchAdapterConfig:
    """Config for converting MaterializedDataset into torch tensors."""

    tensorization: SequenceTensorizationConfig = field(default_factory=SequenceTensorizationConfig)


class SequenceDataset:
    """Torch-style dataset over tensorized sequence examples."""

    def __init__(self, features: dict[str, Any], labels: Any | None = None):
        self._features = features
        self._labels = labels
        first_feature = next(iter(features.values()))
        self._length = int(first_feature.shape[0]) if first_feature is not None else 0

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        item = {key: value[idx] for key, value in self._features.items()}
        if self._labels is None:
            return item
        return item, self._labels[idx]


def to_torch_dataset(dataset: MaterializedDataset, config: TorchAdapterConfig | None = None) -> SequenceDataset:
    """Convert MaterializedDataset to a tensorized torch Dataset-like object."""
    cfg = config or TorchAdapterConfig()
    features_np, labels_np = tensorize_materialized_dataset(dataset, cfg.tensorization)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install seqtrainer[torch] to use torch adapters") from exc

    features = {}
    for key, value in features_np.items():
        dtype = torch.float32 if key == "sequence" else torch.long
        features[key] = torch.tensor(value, dtype=dtype)

    labels = None
    if labels_np is not None:
        label_dtype = torch.float32
        if cfg.tensorization.label_dtype.startswith("int"):
            label_dtype = torch.long
        labels = torch.tensor(labels_np, dtype=label_dtype)

    return SequenceDataset(features=features, labels=labels)


def to_torch_dataloader(
    dataset: MaterializedDataset,
    *,
    config: TorchAdapterConfig | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
):
    """Convert MaterializedDataset to torch DataLoader."""
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install seqtrainer[torch] to use torch adapters") from exc

    torch_dataset = to_torch_dataset(dataset, config=config)
    return DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle)
