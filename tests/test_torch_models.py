import importlib

import pytest

from seqtrainer.models import default_registry
from seqtrainer.torch.backbones import HFBackboneConfig, build_hf_backbone
from seqtrainer.torch.finetune import build_finetune_config, default_loss_for_task
from seqtrainer.torch.heads import ClassificationHeadConfig, RegressionHeadConfig, build_classification_head, build_regression_head


def test_default_registry_contains_torch_specs():
    registry = default_registry()
    assert "dnabert2" in registry.list_backbones()
    assert "hf-sequence" in registry.list_backbones()
    assert "regression-mlp" in registry.list_heads()
    assert "classification-mlp" in registry.list_heads()


def test_default_loss_for_task():
    assert default_loss_for_task("regression") == "mse"
    assert default_loss_for_task("classification") == "cross_entropy"


def test_build_finetune_config_includes_task_and_loss():
    cfg = build_finetune_config(backbone="dnabert2", head="regression-mlp", task="regression")
    assert cfg["loss"] == "mse"
    assert cfg["task"] == "regression"


def test_torch_builders_raise_helpful_import_error_without_torch(monkeypatch):
    original = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name.startswith("torch"):
            raise ImportError("missing torch")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(ImportError, match="seqtrainer\\[torch\\]"):
        build_hf_backbone(HFBackboneConfig(model_name="zhihan1996/DNABERT-2-117M"))

    with pytest.raises(ImportError, match="seqtrainer\\[torch\\]"):
        build_regression_head(RegressionHeadConfig(input_dim=768))

    with pytest.raises(ImportError, match="seqtrainer\\[torch\\]"):
        build_classification_head(ClassificationHeadConfig(input_dim=768, num_classes=2))
