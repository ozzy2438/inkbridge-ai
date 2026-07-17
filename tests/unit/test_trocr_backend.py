"""Tests for immutable TrOCR model and runtime provenance."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

from src.evaluation import trocr_backend
from src.evaluation.trocr_backend import TrOCRPredictionBackend


class FakeEngine:
    """Capture backend configuration without loading model weights."""

    def __init__(self, **kwargs: Any) -> None:
        self.device = kwargs["device"]
        self.use_fp16 = kwargs["use_fp16"] and self.device == "cuda"


class FakeHfApi:
    """Resolve a known immutable model revision."""

    def model_info(self, *, repo_id: str, revision: str) -> SimpleNamespace:
        assert repo_id == "microsoft/trocr-base-handwritten"
        assert revision == "main"
        return SimpleNamespace(sha="a" * 40)


def test_backend_records_model_and_execution_environment(monkeypatch) -> None:
    """Latency evidence must identify both model bytes and runtime environment."""
    monkeypatch.setattr(trocr_backend, "TrOCREngine", FakeEngine)

    backend = TrOCRPredictionBackend(
        model_id="microsoft/trocr-base-handwritten",
        requested_revision="main",
        device="cpu",
        use_fp16=False,
        hf_api=FakeHfApi(),  # type: ignore[arg-type]
    )

    provenance = backend.provenance
    assert provenance["resolved_revision"] == "a" * 40
    assert provenance["model_version"].endswith(f"@{'a' * 40}")
    assert provenance["runtime"]["python"]
    assert provenance["runtime"]["os"]
    assert provenance["runtime"]["os_release"]
    assert provenance["runtime"]["machine"]

    provenance["runtime"]["python"] = "tampered"
    assert backend.provenance["runtime"]["python"] != "tampered"
