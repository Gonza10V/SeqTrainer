from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class PromoterPredictor(Protocol):
    def predict_proba(self, sequences: list[str]) -> list[float]:
        ...

    def metadata(self) -> dict[str, Any]:
        ...


class DummyPromoterPredictor:
    """Deterministic smoke-test predictor.

    This is intentionally simple and should not be used for biological claims.
    It assigns high scores to windows containing a TATA-like motif so tests can
    exercise annotation writing deterministically.
    """

    def predict_proba(self, sequences: list[str]) -> list[float]:
        scores = []
        for sequence in sequences:
            seq = sequence.upper()
            if "TATA" in seq or "TTGACA" in seq:
                scores.append(0.95)
            else:
                scores.append(0.10)
        return scores

    def metadata(self) -> dict[str, Any]:
        return {
            "model_family": "dummy",
            "mode": "deterministic_smoke_test",
            "biological_claims": False,
        }


class DNABERT2PromoterPredictor:
    def __init__(self, checkpoint: str | Path | None = None, benchmark_manifest: str | Path | None = None):
        if checkpoint is None:
            raise ValueError("DNABERT2 annotation requires --checkpoint from a completed benchmark run.")
        if benchmark_manifest is None:
            raise ValueError(
                "DNABERT2 annotation requires --benchmark-manifest so the tokenizer, "
                "pooling, dropout, revision, and window settings match the benchmark run."
            )
        self.checkpoint = Path(checkpoint)
        self.benchmark_manifest = Path(benchmark_manifest)
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "DNABERT2 annotation requires torch/transformers. Install with "
                "`pip install -e \".[annotation,torch]\"`."
            ) from exc
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"DNABERT2 checkpoint not found: {self.checkpoint}")
        if not self.benchmark_manifest.exists():
            raise FileNotFoundError(f"DNABERT2 benchmark manifest not found: {self.benchmark_manifest}")

        try:
            from seqtrainer.torch.dnabert2_benchmark import (
                _DnaBert2Classifier,
                _load_huggingface_dnabert2,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "DNABERT2 annotation requires torch/transformers. Install with "
                "`pip install -e \".[annotation,torch]\"`."
            ) from exc

        self._torch = torch
        self._manifest = json.loads(self.benchmark_manifest.read_text(encoding="utf-8-sig"))
        self._model_name = str(_required_manifest_value(self._manifest, ("model", "name")))
        self._model_params = dict(_required_manifest_value(self._manifest, ("model", "params")))
        self._preprocessing = dict(_required_manifest_value(self._manifest, ("preprocessing", "params")))
        self._pooling = str(_required_mapping_value(self._model_params, "pooling", "model.params"))
        self._batch_size = int(_required_manifest_value(self._manifest, ("training", "batch_size")))
        self._max_length = int(_required_mapping_value(self._preprocessing, "model_max_length", "preprocessing.params"))
        self._padding = str(_required_mapping_value(self._preprocessing, "padding", "preprocessing.params"))
        self._revision = str(_required_mapping_value(self._model_params, "revision", "model.params"))
        self._classifier_dropout = float(_required_mapping_value(self._model_params, "classifier_dropout", "model.params"))
        self._allow_download = bool(_required_mapping_value(self._model_params, "allow_download", "model.params"))
        self._trust_remote_code = bool(_required_mapping_value(self._model_params, "trust_remote_code", "model.params"))
        self._device = _resolve_torch_device(torch)

        tokenizer, encoder = _load_huggingface_dnabert2(
            self._model_name,
            device=self._device,
            trust_remote_code=self._trust_remote_code,
            local_files_only=not self._allow_download,
            disable_flash_attention=bool(self._model_params.get("disable_flash_attention", True)),
            revision=self._revision,
        )
        self._tokenizer = tokenizer
        hidden_size = int(getattr(encoder.config, "hidden_size", 768))
        self._model = _DnaBert2Classifier(
            encoder=encoder,
            hidden_size=hidden_size,
            pooling=self._pooling,
            dropout=self._classifier_dropout,
        ).to(self._device)
        state_dict = _load_torch_state_dict(torch, self.checkpoint)
        self._model.load_state_dict(state_dict, strict=True)
        self._model.eval()

    def predict_proba(self, sequences: list[str]) -> list[float]:
        if not sequences:
            return []
        torch = self._torch
        probabilities: list[float] = []
        pad_to_multiple_of = self._preprocessing.get("pad_to_multiple_of")
        with torch.inference_mode():
            for start in range(0, len(sequences), self._batch_size):
                batch_sequences = sequences[start : start + self._batch_size]
                encoded = self._tokenizer(
                    batch_sequences,
                    padding=self._padding,
                    truncation=True,
                    max_length=self._max_length,
                    pad_to_multiple_of=int(pad_to_multiple_of) if pad_to_multiple_of is not None else None,
                    return_tensors="pt",
                )
                batch = {
                    key: value.to(self._device)
                    for key, value in encoded.items()
                    if hasattr(value, "to")
                }
                if "attention_mask" not in batch:
                    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
                logits = self._model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                probabilities.extend(torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist())
        return probabilities

    def metadata(self) -> dict[str, Any]:
        return {
            "model_family": "dnabert2",
            "checkpoint": str(self.checkpoint),
            "benchmark_manifest": str(self.benchmark_manifest),
            "model_name": self._model_name,
            "pooling": self._pooling,
            "revision": self._revision,
            "model_download_enabled": self._allow_download,
            "device": str(self._device),
            "batch_size": self._batch_size,
            "mode": self._model_params.get("mode"),
        }


def build_predictor(
    model_family: str,
    *,
    checkpoint: str | Path | None = None,
    benchmark_manifest: str | Path | None = None,
) -> PromoterPredictor:
    if model_family == "dummy":
        return DummyPromoterPredictor()
    if model_family == "dnabert2":
        return DNABERT2PromoterPredictor(checkpoint=checkpoint, benchmark_manifest=benchmark_manifest)
    raise ValueError(f"Unsupported model family: {model_family}")


def _manifest_get(manifest: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    current: Any = manifest
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _required_manifest_value(manifest: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value = _manifest_get(manifest, keys)
    if value is None or value == "":
        raise ValueError(f"DNABERT2 benchmark manifest is missing required field: {'.'.join(keys)}")
    return value


def _required_mapping_value(values: dict[str, Any], key: str, section: str) -> Any:
    value = values.get(key)
    if value is None or value == "":
        raise ValueError(f"DNABERT2 benchmark manifest is missing required field: {section}.{key}")
    return value


def _resolve_torch_device(torch: Any) -> Any:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_torch_state_dict(torch: Any, checkpoint: Path) -> dict[str, Any]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"DNABERT2 checkpoint did not contain a state dict: {checkpoint}")
    return state

