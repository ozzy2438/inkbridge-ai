"""Tests for sealed-model, reference-free protected inference production."""

from __future__ import annotations

import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from src.data.pilot_intake import validate_protected_pilot, write_pilot_audit
from src.data.protected_manifest import freeze_protected_evaluation
from src.evaluation.prediction_producer import PredictionResult
from src.evaluation.protected_inference import (
    inspect_local_model_artifact,
    produce_protected_prediction_artifact,
)
from src.evaluation.protected_runner import run_protected_evaluation
from tests.unit.test_pilot_intake import _fixture
from tests.unit.test_protected_evaluation import _clear_ci, _freeze

_CONFIG = Path(__file__).parents[2] / "configs" / "evaluation_config.yaml"
_MODEL_VERSION = "trocr-protected-v1"


class FakeProtectedBackend:
    """Deterministic backend with the required protected provenance."""

    def __init__(self, model_version: str, model_sha256: str, device: str) -> None:
        self.model_version = model_version
        self.model_sha256 = model_sha256
        self.device = device
        self.batch_size = 2
        self.images: list[bytes] = []

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "backend": "fake_protected",
            "model_version": self.model_version,
            "model_artifact_sha256": self.model_sha256,
            "device": self.device,
            "local_files_only": True,
            "network_access_allowed": False,
            "external_ai_service": False,
            "batch_size": self.batch_size,
            "beam_width": 4,
            "max_length": 128,
            "confidence_threshold": 0.7,
            "abstention_threshold": 0.4,
            "dtype": "float32",
        }

    def predict_batch(self, images: list[bytes]) -> list[PredictionResult]:
        self.images.extend(images)
        return [
            PredictionResult(
                raw_text=f"offline prediction {index}",
                confidence=0.8,
                latency_ms=12.5,
                is_uncertain=False,
                is_unreadable=False,
                model_version=self.model_version,
            )
            for index, _ in enumerate(images)
        ]


class NetworkAttemptBackend(FakeProtectedBackend):
    """Backend that proves the Python socket guard is active."""

    def predict_batch(self, images: list[bytes]) -> list[PredictionResult]:
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)
        return super().predict_batch(images)


class FailOnSecondBatchBackend(FakeProtectedBackend):
    """Simulate one interruption after the first batch has been persisted."""

    def __init__(self, model_version: str, model_sha256: str, device: str) -> None:
        super().__init__(model_version, model_sha256, device)
        self.batch_size = 1
        self.calls = 0

    def predict_batch(self, images: list[bytes]) -> list[PredictionResult]:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("synthetic protected interruption")
        return super().predict_batch(images)


def _sealed_model(tmp_path: Path, repository: Path) -> tuple[Path, dict[str, Any]]:
    model = tmp_path / "preloaded-model" / _MODEL_VERSION
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"vision-encoder-decoder"}')
    (model / "preprocessor_config.json").write_text('{"do_resize":true}')
    (model / "model.safetensors").write_bytes(b"synthetic-safe-weight-fixture")
    for path in model.iterdir():
        path.chmod(0o400)
    model.chmod(0o500)
    artifact = inspect_local_model_artifact(model, repository_root=repository)
    return model, artifact


def _authorization(
    dataset: Path,
    metadata: dict[str, Any],
    model_artifact: dict[str, Any],
    *,
    expired: bool = False,
) -> Path:
    now = datetime.now(timezone.utc)
    approved_at = now - timedelta(hours=2)
    expires_at = now - timedelta(hours=1) if expired else now + timedelta(days=1)
    value = {
        "schema_version": 1,
        "status": "approved",
        "approval_reference": "inference_approval_001",
        "approved_at": approved_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "accountable_owner_role": "pilot_privacy_owner",
        "operator_role": "ml_evaluation_operator",
        "execution_environment": "approved_self_hosted",
        "evaluation_set_id": metadata["evaluation_set_id"],
        "manifest_sha256": metadata["manifest"]["sha256"],
        "model_version": _MODEL_VERSION,
        "model_artifact_sha256": model_artifact["model_artifact_sha256"],
        "device": "cpu",
        "inference_config": {
            "batch_size": 2,
            "beam_width": 4,
            "max_length": 128,
            "confidence_threshold": 0.7,
            "abstention_threshold": 0.4,
            "dtype": "float32",
        },
        "prediction_schema": "inkbridge-reference-free-v1",
        "protected_storage_required": True,
        "network_access_during_inference": False,
        "external_ai_services_allowed": False,
        "github_actions_allowed": False,
        "model_artifacts_preloaded": True,
        "training_allowed": False,
        "automated_decisions_allowed": False,
    }
    path = dataset / "protected_inference_authorization.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o400)
    return path


def _inputs(
    tmp_path: Path, *, expired: bool = False
) -> tuple[Path, Path, Path, Path, Path, dict[str, Any]]:
    repository, dataset, contract, manifest_dir, metadata = _freeze(tmp_path)
    model, model_artifact = _sealed_model(tmp_path, repository)
    authorization = _authorization(
        dataset,
        metadata,
        model_artifact,
        expired=expired,
    )
    return repository, dataset, contract, manifest_dir, model, {
        "metadata": metadata,
        "model_artifact": model_artifact,
        "authorization": authorization,
    }


def _inputs_with_two_test_records(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, dict[str, Any]]:
    repository, dataset, contract, _ = _fixture(tmp_path)
    contract_value = json.loads(contract.read_text(encoding="utf-8"))
    contract_value["gold"]["validation_writer_ratio"] = 0.34
    contract_value["gold"]["test_writer_ratio"] = 0.66
    contract.write_text(json.dumps(contract_value), encoding="utf-8")
    audit = validate_protected_pilot(contract, dataset, repository_root=repository)
    write_pilot_audit(
        audit,
        dataset / "pilot_intake.audit.json",
        dataset,
        repository_root=repository,
    )
    manifest_dir = dataset / "frozen_evaluation"
    metadata = freeze_protected_evaluation(
        contract,
        dataset,
        manifest_dir,
        repository_root=repository,
    )
    assert metadata["split"]["sample_counts"]["test"] == 2
    model, model_artifact = _sealed_model(tmp_path, repository)
    authorization = _authorization(dataset, metadata, model_artifact)
    authorization.chmod(0o600)
    authorization_value = json.loads(authorization.read_text(encoding="utf-8"))
    authorization_value["inference_config"]["batch_size"] = 1
    authorization.write_text(json.dumps(authorization_value), encoding="utf-8")
    authorization.chmod(0o400)
    return repository, dataset, contract, manifest_dir, model, {
        "metadata": metadata,
        "model_artifact": model_artifact,
        "authorization": authorization,
    }


def _run(
    repository: Path,
    dataset: Path,
    contract: Path,
    manifest_dir: Path,
    model: Path,
    context: dict[str, Any],
    backend_class: type[FakeProtectedBackend] = FakeProtectedBackend,
) -> tuple[dict[str, Any], FakeProtectedBackend | None]:
    created: list[FakeProtectedBackend] = []

    def factory(model_version: str, model_sha256: str, device: str) -> FakeProtectedBackend:
        backend = backend_class(model_version, model_sha256, device)
        created.append(backend)
        return backend

    result = produce_protected_prediction_artifact(
        contract_path=contract,
        dataset_dir=dataset,
        manifest_dir=manifest_dir,
        authorization_path=context["authorization"],
        model_dir=model,
        output_dir=dataset / "runs" / _MODEL_VERSION,
        backend_factory=factory,
        batch_size=2,
        image_loader=lambda content: content,
        repository_root=repository,
    )
    return result, created[0] if created else None


def test_protected_inference_produces_reference_free_evaluator_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, dataset, contract, manifest_dir, model, context = _inputs(tmp_path)

    result, backend = _run(
        repository, dataset, contract, manifest_dir, model, context
    )

    run_dir = dataset / "runs" / _MODEL_VERSION
    predictions = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert result["status"] == "protected_predictions_ready"
    assert result["predictions"]["contains_references"] is False
    assert len(predictions) == context["metadata"]["split"]["sample_counts"]["test"]
    assert all(
        set(record)
        == {"sample_id", "source_sha256", "prediction", "confidence", "latency_ms"}
        for record in predictions
    )
    serialized = json.dumps(predictions)
    assert "First line" not in serialized
    assert "writer-" not in serialized
    assert "image_path" not in serialized
    assert backend is not None
    assert len(backend.images) == len(predictions)
    assert not (run_dir.stat().st_mode & 0o222)
    assert not ((run_dir / "predictions.jsonl").stat().st_mode & 0o222)

    evaluation, _ = run_protected_evaluation(
        contract_path=contract,
        dataset_dir=dataset,
        manifest_dir=manifest_dir,
        predictions_path=run_dir / "predictions.jsonl",
        attestation_path=run_dir / "execution_attestation.json",
        model_version=_MODEL_VERSION,
        config_path=_CONFIG,
        output_dir=dataset / "protected_evaluation_results",
        repository_root=repository,
    )
    assert evaluation["status"] == "protected_shadow_evaluated"
    assert evaluation["protected_evaluation"]["references_exported"] is False


def test_protected_inference_is_idempotent_after_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, dataset, contract, manifest_dir, model, context = _inputs(tmp_path)
    first, _ = _run(repository, dataset, contract, manifest_dir, model, context)
    second, backend = _run(repository, dataset, contract, manifest_dir, model, context)

    assert second == first
    assert backend is None


def test_protected_inference_resumes_after_a_persisted_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, dataset, contract, manifest_dir, model, context = (
        _inputs_with_two_test_records(tmp_path)
    )
    backend = FailOnSecondBatchBackend(
        _MODEL_VERSION,
        context["model_artifact"]["model_artifact_sha256"],
        "cpu",
    )

    def factory(model_version: str, model_sha256: str, device: str) -> FakeProtectedBackend:
        assert (model_version, model_sha256, device) == (
            backend.model_version,
            backend.model_sha256,
            backend.device,
        )
        return backend

    arguments = {
        "contract_path": contract,
        "dataset_dir": dataset,
        "manifest_dir": manifest_dir,
        "authorization_path": context["authorization"],
        "model_dir": model,
        "output_dir": dataset / "runs" / _MODEL_VERSION,
        "backend_factory": factory,
        "batch_size": 1,
        "image_loader": lambda content: content,
        "repository_root": repository,
    }
    with pytest.raises(RuntimeError, match="synthetic protected interruption"):
        produce_protected_prediction_artifact(**arguments)

    result = produce_protected_prediction_artifact(**arguments)

    assert result["predictions"]["num_records"] == 2
    assert backend.calls == 3


def test_protected_inference_rejects_writable_model_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, dataset, contract, manifest_dir, model, context = _inputs(tmp_path)
    (model / "config.json").chmod(0o600)

    with pytest.raises(ValueError, match="no write permission"):
        _run(repository, dataset, contract, manifest_dir, model, context)


def test_protected_inference_rejects_expired_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, dataset, contract, manifest_dir, model, context = _inputs(
        tmp_path, expired=True
    )

    with pytest.raises(ValueError, match="expired"):
        _run(repository, dataset, contract, manifest_dir, model, context)


def test_protected_inference_rejects_backend_setting_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, dataset, contract, manifest_dir, model, context = _inputs(tmp_path)
    authorization = context["authorization"]
    authorization.chmod(0o600)
    value = json.loads(authorization.read_text(encoding="utf-8"))
    value["inference_config"]["beam_width"] = 5
    authorization.write_text(json.dumps(value), encoding="utf-8")
    authorization.chmod(0o400)

    with pytest.raises(ValueError, match="provenance beam_width does not match"):
        _run(repository, dataset, contract, manifest_dir, model, context)


def test_protected_inference_blocks_python_socket_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, dataset, contract, manifest_dir, model, context = _inputs(tmp_path)

    with pytest.raises(RuntimeError, match="blocked an outbound socket"):
        _run(
            repository,
            dataset,
            contract,
            manifest_dir,
            model,
            context,
            backend_class=NetworkAttemptBackend,
        )


def test_protected_inference_refuses_ci_before_opening_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, dataset, contract, manifest_dir, model, context = _inputs(tmp_path)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    with pytest.raises(RuntimeError, match="refuses CI"):
        _run(repository, dataset, contract, manifest_dir, model, context)


def test_model_artifact_rejects_pickle_weights(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "preprocessor_config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"safe")
    (model / "pytorch_model.bin").write_bytes(b"unsafe")
    for path in model.iterdir():
        path.chmod(0o400)
    model.chmod(0o500)

    with pytest.raises(ValueError, match="unsafe executable/pickle"):
        inspect_local_model_artifact(model, repository_root=repository)
