import pytest

from seqtrainer.torch import backbones


def test_preset_lookup_is_pinned_and_offline():
    preset = backbones.get_hosted_model_preset("nucleotide-transformer-v2")

    assert preset.model_id.startswith("InstaDeepAI/")
    assert len(preset.revision) == 40


def test_preset_construction_uses_a_lazy_injectable_loader(monkeypatch):
    captured = {}

    def fake_builder(config):
        captured["config"] = config
        return "offline-model"

    monkeypatch.setattr(backbones, "build_hf_backbone", fake_builder)

    assert backbones.build_hosted_model_preset("hyenadna-tiny", trainable=False) == "offline-model"
    assert captured["config"].model_kwargs["revision"] == backbones.HOSTED_MODEL_PRESETS["hyenadna-tiny"].revision
    assert captured["config"].trainable is False


def test_unknown_preset_has_actionable_error():
    with pytest.raises(ValueError, match="Available presets"):
        backbones.get_hosted_model_preset("not-a-preset")
