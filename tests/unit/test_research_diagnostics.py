"""Tests for private SMHD failure analysis and calibration diagnostics."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from src.data.manifest import sha256_file
from src.evaluation.research_diagnostics import build_offline_research_diagnostics

_CI_MARKERS = (
    "BUILDKITE",
    "CIRCLECI",
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "JENKINS_URL",
    "TF_BUILD",
)
_DATASET = {
    "name": "smhd-research-rehearsal",
    "version": "figshare-24419986-v1-selection-v1",
    "license_id": "CC-BY-NC-4.0",
    "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
    "sample_type": "line",
}
_MODEL = {
    "backend": "trocr_local_research",
    "evidence_scope": "noncommercial_research_rehearsal_only",
    "protected_pilot_evidence": False,
    "model_version": "trocr-base-smhd-research-v1",
    "model_artifact_sha256": "a" * 64,
    "device": "cpu",
    "dtype": "float32",
    "batch_size": 2,
    "beam_width": 4,
    "max_length": 128,
    "confidence_threshold": 0.7,
    "abstention_threshold": 0.4,
    "local_files_only": True,
    "network_access_allowed": False,
    "external_ai_service": False,
    "remote_code_allowed": False,
    "processor_use_fast": False,
}


def _clear_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _CI_MARKERS:
        monkeypatch.delenv(name, raising=False)


def _private_tree(root: Path) -> None:
    if os.name != "posix":
        return
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o700 if path.is_dir() else 0o600)
    root.chmod(0o700)


def _record(split: str, index: int, *, source_offset: int = 0) -> dict[str, Any]:
    exact = index % 4 == 0
    return {
        "schema_version": 1,
        "sample_id": f"{split}-fixture-{index:02d}",
        "source_sha256": f"{source_offset + index + 1:064x}",
        "reference": f"fixture reference {index}",
        "prediction": f"fixture reference {index}" if exact else f"fixture referenc {index}",
        "confidence": 0.96 - index * 0.03,
        "slices": [
            "cc_by_nc_research_only",
            "publisher_hash_marker" if index % 2 == 0 else "publisher_no_hash_marker",
        ],
        "latency_ms": 10.0 + index,
        "model_version": _MODEL["model_version"],
        "is_uncertain": index == 8,
        "is_unreadable": False,
        "split": split,
    }


def _run_directory(
    private_root: Path,
    split: str,
    count: int,
    writers: int,
    *,
    source_offset: int,
) -> Path:
    root = private_root / split
    root.mkdir()
    records = [_record(split, index, source_offset=source_offset) for index in range(count)]
    predictions = root / "predictions.jsonl"
    predictions.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    prediction_sha = sha256_file(predictions)
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "input": {
            "manifest_filename": "manifest.jsonl",
            "manifest_sha256": "b" * 64,
            "manifest_metadata_sha256": "c" * 64,
            "dataset": _DATASET,
            "split": split,
            "num_records": count,
        },
        "model": _MODEL,
        "output": {
            "filename": "predictions.jsonl",
            "sha256": prediction_sha,
            "num_records": count,
        },
        "privacy": {"writer_ids_exported": False, "source_images_exported": False},
    }
    attestation = {
        "schema_version": 1,
        "status": "complete",
        "evidence_scope": "noncommercial_research_rehearsal_only",
        "dataset": {
            **_DATASET,
            "split": split,
            "num_records": count,
            "num_writers": writers,
            "source_metadata_sha256": "d" * 64,
            "labels_sha256": "e" * 64,
            "manifest_sha256": "b" * 64,
            "manifest_metadata_sha256": "c" * 64,
        },
        "model": _MODEL,
        "execution": {
            "network_access_allowed": False,
            "python_socket_guard_enabled": True,
            "external_ai_service": False,
            "github_actions_allowed": False,
            "local_files_only": True,
        },
        "output": {
            "predictions_filename": "predictions.jsonl",
            "predictions_sha256": prediction_sha,
            "num_records": count,
            "sample_level_artifacts_committed": False,
        },
        "claims": {
            "gold_benchmark": False,
            "production_pilot_evidence": False,
            "commercial_use_evidence": False,
            "protected_pilot_substitute": False,
        },
    }
    (root / "predictions.meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "research_execution_attestation.json").write_text(
        json.dumps(attestation), encoding="utf-8"
    )
    _private_tree(root)
    return root


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".git").mkdir()
    private = tmp_path / "private"
    private.mkdir()
    validation = _run_directory(private, "validation", 9, 3, source_offset=0)
    test = _run_directory(private, "test", 6, 2, source_offset=100)
    analysis_parent = private / "analysis"
    analysis_parent.mkdir()
    _private_tree(private)
    return repository, validation, test, analysis_parent


def test_builds_private_failure_atlas_and_aggregate_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, validation, test, analysis_parent = _inputs(tmp_path)
    output = analysis_parent / "v1"

    result = build_offline_research_diagnostics(
        validation_run_dir=validation,
        test_run_dir=test,
        output_dir=output,
        repository_root=repository,
    )

    aggregate = result["aggregate"]
    rendered = json.dumps(aggregate, sort_keys=True)
    assert aggregate["inputs"]["validation"]["num_samples"] == 9
    assert aggregate["inputs"]["test"]["num_samples"] == 6
    assert aggregate["calibration"]["fit"]["status"] == "blocked"
    assert "validation_sample_count_below_calibration_minimum" in rendered
    assert aggregate["operational_decision"]["automatic_acceptance_allowed"] is False
    assert aggregate["claims"]["calibrated_confidence"] is False
    assert "sample_id" not in rendered
    assert "fixture reference" not in rendered

    atlas = [
        json.loads(line)
        for line in (output / "failure_atlas.private.jsonl").read_text().splitlines()
    ]
    assert atlas
    assert all(record["human_review_required"] is True for record in atlas)
    assert any("publisher_correction_marker" in record["categories"] for record in atlas)
    assert any("high_confidence_exact_error" in record["categories"] for record in atlas)

    if os.name == "posix":
        assert stat.S_IMODE(output.stat().st_mode) == 0o700
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in output.iterdir())


def test_rejects_validation_test_sample_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, validation, test, analysis_parent = _inputs(tmp_path)
    test_predictions = test / "predictions.jsonl"
    records = [json.loads(line) for line in test_predictions.read_text().splitlines()]
    validation_first = json.loads((validation / "predictions.jsonl").read_text().splitlines()[0])
    records[0]["source_sha256"] = validation_first["source_sha256"]
    if os.name == "posix":
        test_predictions.chmod(0o600)
    test_predictions.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    prediction_sha = sha256_file(test_predictions)
    for filename in ("predictions.meta.json", "research_execution_attestation.json"):
        path = test / filename
        value = json.loads(path.read_text())
        if filename == "predictions.meta.json":
            value["output"]["sha256"] = prediction_sha
        else:
            value["output"]["predictions_sha256"] = prediction_sha
        path.write_text(json.dumps(value), encoding="utf-8")
    _private_tree(test)

    with pytest.raises(ValueError, match="not sample-isolated"):
        build_offline_research_diagnostics(
            validation_run_dir=validation,
            test_run_dir=test,
            output_dir=analysis_parent / "overlap",
            repository_root=repository,
        )


def test_rejects_ci_before_opening_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", "1")

    with pytest.raises(RuntimeError, match="refuses CI"):
        build_offline_research_diagnostics(
            validation_run_dir=tmp_path / "missing-validation",
            test_run_dir=tmp_path / "missing-test",
            output_dir=tmp_path / "missing-output",
            repository_root=tmp_path,
        )


def test_rejects_output_inside_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_ci(monkeypatch)
    repository, validation, test, _ = _inputs(tmp_path)

    with pytest.raises(ValueError, match="must remain outside the repository"):
        build_offline_research_diagnostics(
            validation_run_dir=validation,
            test_run_dir=test,
            output_dir=repository / "private-analysis",
            repository_root=repository,
        )


def test_rejects_output_nested_in_prediction_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, validation, test, _ = _inputs(tmp_path)

    with pytest.raises(ValueError, match="separate from prediction runs"):
        build_offline_research_diagnostics(
            validation_run_dir=validation,
            test_run_dir=test,
            output_dir=validation / "private-analysis",
            repository_root=repository,
        )


def test_rejects_production_claim_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_ci(monkeypatch)
    repository, validation, test, analysis_parent = _inputs(tmp_path)
    attestation_path = validation / "research_execution_attestation.json"
    attestation = json.loads(attestation_path.read_text())
    attestation["claims"]["production_pilot_evidence"] = True
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    _private_tree(validation)

    with pytest.raises(ValueError, match="production_pilot_evidence"):
        build_offline_research_diagnostics(
            validation_run_dir=validation,
            test_run_dir=test,
            output_dir=analysis_parent / "claim-drift",
            repository_root=repository,
        )
