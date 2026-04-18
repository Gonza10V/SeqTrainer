import sys

import numpy as np
import pytest

from seqtrainer.data.materialized import MaterializedDataset
from seqtrainer.data.tensorization import SequenceTensorizationConfig, tensorize_materialized_dataset
from seqtrainer.keras.adapters import KerasAdapterConfig, to_tf_dataset
from seqtrainer.torch.adapters import TorchAdapterConfig, to_torch_dataset


def _example_dataset() -> MaterializedDataset:
    return MaterializedDataset(
        examples=[
            {"sequence": "ACGT", "target": 1.0},
            {"sequence": "NNAA", "target": 0.0},
        ]
    )


def test_tensorization_one_hot_shapes():
    features, labels = tensorize_materialized_dataset(
        _example_dataset(),
        SequenceTensorizationConfig(encoding="one_hot", max_length=6),
    )
    assert features["sequence"].shape == (2, 6, 5)
    assert labels is not None and labels.shape == (2,)


def test_tensorization_token_ids_contains_attention_mask():
    features, labels = tensorize_materialized_dataset(
        _example_dataset(),
        SequenceTensorizationConfig(encoding="token_ids", max_length=5),
    )
    assert features["input_ids"].shape == (2, 5)
    assert features["attention_mask"].shape == (2, 5)
    assert labels is not None and np.allclose(labels, [1.0, 0.0])


def test_torch_adapter_raises_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match="seqtrainer\\[torch\\]"):
        to_torch_dataset(_example_dataset(), TorchAdapterConfig())


def test_keras_adapter_raises_without_tensorflow(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    with pytest.raises(ImportError, match="seqtrainer\\[keras\\]"):
        to_tf_dataset(_example_dataset(), KerasAdapterConfig())
