"""Tests for local-only SMHD research inference production."""

from __future__ import annotations

import csv
import json
import os
import socket
import stat
from pathlib import Path
from typing import Any

import pytest

from src.data.manifest import (
    DatasetIdentity,
    build_manifest_records,
    sha256_file,
    write_split_manifest,
)
from src.evaluation.prediction_producer import PredictionResult
from src.evaluation.protected_inference import inspect_local_model_artifact
from src.evaluation.research_inference import produce_offline_research_prediction_artifact

_CI_MARKERS = (
    "BUILDKITE",
    "CIRCLECI",
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "JENKINS_URL",
    "TF_BUILD",
)


class FakeResearchBackend:
    """Deterministic backend with the required local research provenance."""

    def __init__(self, model_version: str, model_sha256: str, device: str) -> None:
        self.model_version = model_version
        self.model_sha256 = model_sha256
        self.device = device
        self.calls = 0

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "backend": "trocr_local_research",
            "evidence_scope": "noncommercial_research_rehearsal_only",
            "protected_pilot_evidence": False,
            "model_version": self.model_version,
            "model_artifact_sha256": self.model_sha256,
            "device": self.device,
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

    def predict_batch(self, images: list[Any]) -> list[PredictionResult]:
        self.calls += 1
        return [
            PredictionResult(
                raw_text=f"offline fixture prediction {index}",
                confidence=0.8,
                latency_ms=12.5,
                is_uncertain=False,
                is_unreadable=False,
                model_version=self.model_version,
            )
            for index, _ in enumerate(images)
        ]


class NetworkAttemptBackend(FakeResearchBackend):
    """Attempt a connection to prove the offline socket guard is active."""

    def predict_batch(self, images: list[Any]) -> list[PredictionResult]:
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)
        return super().predict_batch(images)


def _clear_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _CI_MARKERS:
        monkeypatch.delenv(name, raising=False)


def _private_tree(root: Path) -> None:
    if os.name != "posix":
        return
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o700 if path.is_dir() else 0o600)
    root.chmod(0o700)


def _dataset(private_root: Path) -> Path:
    dataset = private_root / "smhd-normalized"
    images = dataset / "images"
    images.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    for writer_number in range(12):
        for sample_number in range(3):
            filename = f"sample-{writer_number:02d}-{sample_number}.pgm"
            content = b"P5\n8 4\n255\n" + bytes([200 + sample_number]) * 32
            (images / filename).write_bytes(content)
            rows.append(
                {
                    "filename": filename,
                    "text": f"synthetic fixture reference {writer_number} {sample_number}",
                    "writer_id": f"fixture-writer-{writer_number:02d}",
                    "slices": "cc_by_nc_research_only|smhd|student",
                }
            )
    labels = dataset / "labels.csv"
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("filename", "text", "writer_id", "slices"),
        )
        writer.writeheader()
        writer.writerows(rows)
    source_meta = {
        "schema_version": 1,
        "purpose": "offline_student_handwriting_research_shadow_rehearsal_only",
        "dataset": {
            "article_id": 24419986,
            "doi": "10.25439/rmt.24419986.v1",
            "license_id": "CC-BY-NC-4.0",
            "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "commercial_use_allowed": False,
        },
        "privacy": {
            "contains_student_data": True,
            "repository_artifacts_allowed": False,
            "offline_only": True,
        },
        "reference_quality": {
            "status": "publisher_transcription_unreviewed",
            "gold_ready": False,
        },
        "evidence_boundary": {
            "research_rehearsal_only": True,
            "production_pilot_evidence": False,
            "consented_protected_pilot_substitute": False,
        },
        "output": {"labels_sha256": sha256_file(labels)},
    }
    (dataset / "source.meta.json").write_text(json.dumps(source_meta), encoding="utf-8")
    identity = DatasetIdentity(
        name="smhd-research-rehearsal",
        version="figshare-24419986-v1-selection-v1",
        license_id="CC-BY-NC-4.0",
        license_url="https://creativecommons.org/licenses/by-nc/4.0/",
        sample_type="line",
    )
    records = build_manifest_records(dataset, identity)
    write_split_manifest(
        records,
        dataset_dir=dataset,
        output_dir=dataset,
        identity=identity,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        min_samples_per_writer=3,
    )
    _private_tree(dataset)
    return dataset


def _sealed_model(private_root: Path, repository: Path) -> tuple[Path, dict[str, Any]]:
    model = private_root / "sealed-model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"vision-encoder-decoder"}')
    (model / "preprocessor_config.json").write_text('{"do_resize":true}')
    (model / "model.safetensors").write_bytes(b"research-safe-weight-fixture")
    if os.name == "posix":
        for path in model.iterdir():
            path.chmod(0o400)
        model.chmod(0o500)
    artifact = inspect_local_model_artifact(model, repository_root=repository)
    return model, artifact


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".git").mkdir()
    private_root = tmp_path / "private"
    private_root.mkdir()
    runs = private_root / "runs"
    runs.mkdir()
    if os.name == "posix":
        private_root.chmod(0o700)
        runs.chmod(0o700)
    dataset = _dataset(private_root)
    model, artifact = _sealed_model(private_root, repository)
    return repository, dataset, model, runs, artifact


def _run(
    repository: Path,
    dataset: Path,
    model: Path,
    output: Path,
    artifact: dict[str, Any],
    backend_class: type[FakeResearchBackend] = FakeResearchBackend,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    def factory(model_version: str, model_sha256: str, device: str) -> FakeResearchBackend:
        return backend_class(model_version, model_sha256, device)

    return produce_offline_research_prediction_artifact(
        dataset_dir=dataset,
        output_dir=output,
        model_dir=model,
        model_version="trocr-base-smhd-research-v1",
        expected_model_artifact_sha256=artifact["model_artifact_sha256"],
        backend_factory=factory,
        device="cpu",
        batch_size=2,
        resume=resume,
        repository_root=repository,
    )


def test_produces_private_offline_research_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, dataset, model, runs, artifact = _inputs(tmp_path)
    output = runs / "trocr-base-smhd-research-v1"

    result = _run(repository, dataset, model, output, artifact)
    repeated = _run(repository, dataset, model, output, artifact)
    restarted = _run(repository, dataset, model, output, artifact, resume=False)

    predictions = [
        json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()
    ]
    assert len(predictions) == 6
    assert all("writer_id" not in record for record in predictions)
    assert result["prediction_metadata"]["input"]["num_records"] == 6
    assert result["attestation"]["dataset"]["test_writers"] == 2
    assert result["attestation"]["execution"]["network_access_allowed"] is False
    assert result["attestation"]["claims"]["gold_benchmark"] is False
    assert repeated["attestation"] == result["attestation"]
    assert restarted["prediction_metadata"]["output"]["num_records"] == 6
    assert restarted["attestation"]["output"] == result["attestation"]["output"]

    if os.name == "posix":
        assert stat.S_IMODE(output.stat().st_mode) == 0o700
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in output.iterdir())


def test_blocks_network_connections_during_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, dataset, model, runs, artifact = _inputs(tmp_path)

    with pytest.raises(RuntimeError, match="blocked an outbound socket"):
        _run(
            repository,
            dataset,
            model,
            runs / "network-attempt",
            artifact,
            NetworkAttemptBackend,
        )


def test_rejects_ci_before_opening_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", "1")

    with pytest.raises(RuntimeError, match="refuses CI"):
        produce_offline_research_prediction_artifact(
            dataset_dir=tmp_path / "missing-dataset",
            output_dir=tmp_path / "missing-output",
            model_dir=tmp_path / "missing-model",
            model_version="trocr-base-smhd-research-v1",
            expected_model_artifact_sha256="a" * 64,
            backend_factory=lambda _version, _sha, _device: FakeResearchBackend(
                _version, _sha, _device
            ),
        )


def test_rejects_gold_or_production_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_ci(monkeypatch)
    repository, dataset, model, runs, artifact = _inputs(tmp_path)
    source_meta_path = dataset / "source.meta.json"
    value = json.loads(source_meta_path.read_text(encoding="utf-8"))
    value["reference_quality"]["gold_ready"] = True
    if os.name == "posix":
        source_meta_path.chmod(0o600)
    source_meta_path.write_text(json.dumps(value), encoding="utf-8")
    if os.name == "posix":
        source_meta_path.chmod(0o600)

    with pytest.raises(ValueError, match="gold_ready"):
        _run(repository, dataset, model, runs / "invalid-claim", artifact)


def test_rejects_model_identity_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_ci(monkeypatch)
    repository, dataset, model, runs, _ = _inputs(tmp_path)

    with pytest.raises(ValueError, match="does not match"):
        produce_offline_research_prediction_artifact(
            dataset_dir=dataset,
            output_dir=runs / "wrong-model",
            model_dir=model,
            model_version="trocr-base-smhd-research-v1",
            expected_model_artifact_sha256="f" * 64,
            backend_factory=lambda version, sha, device: FakeResearchBackend(version, sha, device),
            repository_root=repository,
        )


def test_rejects_output_inside_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_ci(monkeypatch)
    repository, dataset, model, _, artifact = _inputs(tmp_path)

    with pytest.raises(ValueError, match="outside the repository"):
        _run(repository, dataset, model, repository / "output", artifact)
