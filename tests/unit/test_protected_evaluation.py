"""Tests for frozen protected manifests and aggregate-only self-hosted evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src.data.pilot_intake import validate_protected_pilot, write_pilot_audit
from src.data.protected_manifest import (
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    freeze_protected_evaluation,
    verify_frozen_protected_evaluation,
)
from src.evaluation.protected_runner import run_protected_evaluation
from tests.unit.test_pilot_intake import _fixture

_CONFIG = Path(__file__).parents[2] / "configs" / "evaluation_config.yaml"
_CI_MARKERS = (
    "BUILDKITE",
    "CIRCLECI",
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "JENKINS_URL",
    "TF_BUILD",
)


def _validated_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repository, dataset, contract, _ = _fixture(tmp_path)
    audit = validate_protected_pilot(contract, dataset, repository_root=repository)
    write_pilot_audit(
        audit,
        dataset / "pilot_intake.audit.json",
        dataset,
        repository_root=repository,
    )
    return repository, dataset, contract


def _freeze(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    repository, dataset, contract = _validated_fixture(tmp_path)
    manifest_dir = dataset / "frozen_evaluation"
    metadata = freeze_protected_evaluation(
        contract,
        dataset,
        manifest_dir,
        repository_root=repository,
    )
    return repository, dataset, contract, manifest_dir, metadata


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_run_inputs(
    dataset: Path,
    manifest_dir: Path,
    metadata: dict[str, Any],
    *,
    include_reference: bool = False,
    external_ai_services_used: bool = False,
) -> tuple[Path, Path]:
    records = _read_manifest(manifest_dir / MANIFEST_FILENAME)
    test_records = [record for record in records if record["split"] == "test"]
    run_dir = dataset / "runs" / "trocr-protected-v1"
    run_dir.mkdir(parents=True)
    predictions = run_dir / "predictions.jsonl"
    with predictions.open("w", encoding="utf-8") as handle:
        for record in test_records:
            prediction = {
                "sample_id": record["sample_id"],
                "source_sha256": record["source_sha256"],
                "prediction": record["reference"],
            }
            if include_reference:
                prediction["reference"] = record["reference"]
            handle.write(json.dumps(prediction) + "\n")
    predictions_sha256 = hashlib.sha256(predictions.read_bytes()).hexdigest()
    attestation = run_dir / "execution_attestation.json"
    attestation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "execution_environment": "approved_self_hosted",
                "executed_at": "2026-01-02T00:00:00+11:00",
                "operator_role": "ml_evaluation_operator",
                "model_version": "trocr-protected-v1",
                "model_artifact_sha256": "a" * 64,
                "evaluation_set_id": metadata["evaluation_set_id"],
                "manifest_sha256": metadata["manifest"]["sha256"],
                "predictions_sha256": predictions_sha256,
                "protected_storage_mounted": True,
                "network_access_during_inference": False,
                "external_ai_services_used": external_ai_services_used,
                "github_actions_used": False,
                "model_artifacts_preloaded": True,
                "predictions_contain_references": False,
                "automated_decisions_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    return predictions, attestation


def _clear_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _CI_MARKERS:
        monkeypatch.delenv(name, raising=False)


def _refresh_prediction_hash(predictions: Path, attestation: Path) -> None:
    value = json.loads(attestation.read_text(encoding="utf-8"))
    value["predictions_sha256"] = hashlib.sha256(predictions.read_bytes()).hexdigest()
    attestation.write_text(json.dumps(value), encoding="utf-8")


def test_freeze_creates_read_only_writer_isolated_manifest(tmp_path: Path) -> None:
    repository, dataset, contract, manifest_dir, metadata = _freeze(tmp_path)

    verified, records = verify_frozen_protected_evaluation(
        contract,
        dataset,
        manifest_dir,
        repository_root=repository,
    )

    assert verified == metadata
    assert metadata["status"] == "frozen_protected_evaluation_ready"
    assert metadata["gold_ready"] is False
    assert metadata["split"]["training_split_present"] is False
    assert set(metadata["split"]["sample_counts"]) == {"test", "validation"}
    validation_writers = {
        record["writer_id"] for record in records if record["split"] == "validation"
    }
    test_writers = {record["writer_id"] for record in records if record["split"] == "test"}
    assert validation_writers.isdisjoint(test_writers)
    assert not (manifest_dir.stat().st_mode & 0o222)
    assert not ((manifest_dir / MANIFEST_FILENAME).stat().st_mode & 0o222)
    metadata_text = (manifest_dir / METADATA_FILENAME).read_text(encoding="utf-8")
    assert "First line" not in metadata_text
    assert "writer-" not in metadata_text
    assert "annotator-" not in metadata_text


def test_freeze_requires_current_persisted_intake_audit(tmp_path: Path) -> None:
    repository, dataset, contract = _validated_fixture(tmp_path)
    audit_path = dataset / "pilot_intake.audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["counts"]["samples"] = 999
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError, match="stale or invalid"):
        freeze_protected_evaluation(
            contract,
            dataset,
            dataset / "frozen_evaluation",
            repository_root=repository,
        )


def test_freeze_never_overwrites_existing_manifest(tmp_path: Path) -> None:
    repository, dataset, contract, manifest_dir, _ = _freeze(tmp_path)

    with pytest.raises(FileExistsError, match="cannot be replaced"):
        freeze_protected_evaluation(
            contract,
            dataset,
            manifest_dir,
            repository_root=repository,
        )


def test_verify_detects_manifest_drift(tmp_path: Path) -> None:
    repository, dataset, contract, manifest_dir, _ = _freeze(tmp_path)
    manifest_path = manifest_dir / MANIFEST_FILENAME
    manifest_dir.chmod(0o700)
    manifest_path.chmod(0o600)
    content = manifest_path.read_text(encoding="utf-8").replace("First line", "Changed line")
    manifest_path.write_text(content, encoding="utf-8")
    manifest_path.chmod(0o400)
    manifest_dir.chmod(0o500)

    with pytest.raises(ValueError, match="content drift"):
        verify_frozen_protected_evaluation(
            contract,
            dataset,
            manifest_dir,
            repository_root=repository,
        )


def test_protected_runner_writes_only_aggregate_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, dataset, contract, manifest_dir, metadata = _freeze(tmp_path)
    predictions, attestation = _write_run_inputs(dataset, manifest_dir, metadata)

    artifact, result_path = run_protected_evaluation(
        contract_path=contract,
        dataset_dir=dataset,
        manifest_dir=manifest_dir,
        predictions_path=predictions,
        attestation_path=attestation,
        model_version="trocr-protected-v1",
        config_path=_CONFIG,
        output_dir=dataset / "protected_evaluation_results",
        repository_root=repository,
    )

    assert artifact["status"] == "protected_shadow_evaluated"
    assert artifact["metrics"]["cer"] == 0
    assert artifact["protected_evaluation"]["aggregate_only"] is True
    assert artifact["protected_evaluation"]["gold_ready"] is False
    assert not (result_path.stat().st_mode & 0o222)
    report = result_path.read_text(encoding="utf-8")
    assert "First line" not in report
    assert "writer-" not in report
    assert "sample-" not in report


def test_protected_runner_rejects_reference_in_prediction_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, dataset, contract, manifest_dir, metadata = _freeze(tmp_path)
    predictions, attestation = _write_run_inputs(
        dataset, manifest_dir, metadata, include_reference=True
    )

    with pytest.raises(ValueError, match="reference-free allowlist"):
        run_protected_evaluation(
            contract_path=contract,
            dataset_dir=dataset,
            manifest_dir=manifest_dir,
            predictions_path=predictions,
            attestation_path=attestation,
            model_version="trocr-protected-v1",
            config_path=_CONFIG,
            output_dir=dataset / "protected_evaluation_results",
            repository_root=repository,
        )


def test_protected_runner_rejects_external_ai_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, dataset, contract, manifest_dir, metadata = _freeze(tmp_path)
    predictions, attestation = _write_run_inputs(
        dataset, manifest_dir, metadata, external_ai_services_used=True
    )

    with pytest.raises(ValueError, match="external_ai_services_used must be False"):
        run_protected_evaluation(
            contract_path=contract,
            dataset_dir=dataset,
            manifest_dir=manifest_dir,
            predictions_path=predictions,
            attestation_path=attestation,
            model_version="trocr-protected-v1",
            config_path=_CONFIG,
            output_dir=dataset / "protected_evaluation_results",
            repository_root=repository,
        )


def test_protected_runner_rejects_prediction_drift_after_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, dataset, contract, manifest_dir, metadata = _freeze(tmp_path)
    predictions, attestation = _write_run_inputs(dataset, manifest_dir, metadata)
    predictions.write_text(predictions.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="predictions_sha256 does not match"):
        run_protected_evaluation(
            contract_path=contract,
            dataset_dir=dataset,
            manifest_dir=manifest_dir,
            predictions_path=predictions,
            attestation_path=attestation,
            model_version="trocr-protected-v1",
            config_path=_CONFIG,
            output_dir=dataset / "protected_evaluation_results",
            repository_root=repository,
        )


def test_protected_runner_refuses_ci_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, dataset, contract, manifest_dir, metadata = _freeze(tmp_path)
    predictions, attestation = _write_run_inputs(dataset, manifest_dir, metadata)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    with pytest.raises(RuntimeError, match="refuses CI"):
        run_protected_evaluation(
            contract_path=contract,
            dataset_dir=dataset,
            manifest_dir=manifest_dir,
            predictions_path=predictions,
            attestation_path=attestation,
            model_version="trocr-protected-v1",
            config_path=_CONFIG,
            output_dir=dataset / "protected_evaluation_results",
            repository_root=repository,
        )


def test_protected_runner_rejects_incomplete_test_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, dataset, contract, manifest_dir, metadata = _freeze(tmp_path)
    predictions, attestation = _write_run_inputs(dataset, manifest_dir, metadata)
    predictions.write_text("", encoding="utf-8")
    _refresh_prediction_hash(predictions, attestation)

    with pytest.raises(ValueError, match="incomplete"):
        run_protected_evaluation(
            contract_path=contract,
            dataset_dir=dataset,
            manifest_dir=manifest_dir,
            predictions_path=predictions,
            attestation_path=attestation,
            model_version="trocr-protected-v1",
            config_path=_CONFIG,
            output_dir=dataset / "protected_evaluation_results",
            repository_root=repository,
        )
