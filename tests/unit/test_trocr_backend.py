"""Tests for immutable TrOCR model and runtime provenance."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

from src.evaluation import protected_trocr_backend, trocr_backend
from src.evaluation.protected_trocr_backend import ProtectedTrOCRPredictionBackend
from src.evaluation.research_trocr_backend import ResearchTrOCRPredictionBackend
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


class FakeProtectedEngine:
    """Capture the local-only settings required by the protected backend."""

    kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).kwargs = kwargs
        self.device = kwargs["device"]
        self.use_fp16 = kwargs["use_fp16"] and self.device == "cuda"


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


def test_protected_backend_loads_only_the_materialized_local_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "sealed-model"
    model_dir.mkdir()
    monkeypatch.setattr(protected_trocr_backend, "TrOCREngine", FakeProtectedEngine)

    backend = ProtectedTrOCRPredictionBackend(
        model_dir=model_dir,
        model_version="trocr-protected-v1",
        model_artifact_sha256="b" * 64,
        device="cpu",
        batch_size=2,
        use_fp16=False,
    )

    assert FakeProtectedEngine.kwargs["model_path"] == str(model_dir.resolve())
    assert FakeProtectedEngine.kwargs["revision"] is None
    assert FakeProtectedEngine.kwargs["local_files_only"] is True
    assert backend.provenance["network_access_allowed"] is False
    assert backend.provenance["remote_code_allowed"] is False
    assert backend.provenance["processor_use_fast"] is False


def test_research_backend_does_not_claim_protected_pilot_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "sealed-model"
    model_dir.mkdir()
    monkeypatch.setattr(protected_trocr_backend, "TrOCREngine", FakeProtectedEngine)

    backend = ResearchTrOCRPredictionBackend(
        model_dir=model_dir,
        model_version="trocr-research-v1",
        model_artifact_sha256="c" * 64,
        device="cpu",
        batch_size=2,
        use_fp16=False,
    )

    assert backend.provenance["backend"] == "trocr_local_research"
    assert backend.provenance["evidence_scope"] == "noncommercial_research_rehearsal_only"
    assert backend.provenance["protected_pilot_evidence"] is False
