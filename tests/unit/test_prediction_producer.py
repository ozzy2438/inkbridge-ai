"""Unit tests for resumable, provenance-locked OCR prediction artifacts."""

import json
from pathlib import Path
from typing import Any

import pytest

from src.data.manifest import DatasetIdentity, build_manifest_records, write_split_manifest
from src.evaluation.prediction_producer import (
    PredictionResult,
    produce_prediction_artifact,
)


class FakeBackend:
    """Deterministic backend that can simulate an interrupted batch."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.fail_on_call = fail_on_call
        self.calls = 0
        self.completed_inputs: list[bytes] = []

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "model_id": "fake/trocr",
            "requested_revision": "main",
            "resolved_revision": "a" * 40,
            "model_version": f"fake/trocr@{'a' * 40}",
        }

    def predict_batch(self, images: list[bytes]) -> list[PredictionResult]:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated inference interruption")
        self.completed_inputs.extend(images)
        return [
            PredictionResult(
                raw_text=f"prediction-{image.decode()}",
                confidence=0.8,
                latency_ms=12.5,
                is_uncertain=False,
                is_unreadable=False,
                model_version=f"fake/trocr@{'a' * 40}",
            )
            for image in images
        ]


def _dataset_package(root: Path, *, writers: int = 9) -> list[dict[str, Any]]:
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    rows = ["filename,text,writer_id,slices"]
    for index in range(writers):
        filename = f"line-{index}.png"
        (images_dir / filename).write_bytes(f"image-{index}".encode())
        rows.append(f"{filename},reference {index},writer-{index},age_8|faint_pencil")
    (root / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    identity = DatasetIdentity(
        name="synthetic-contract",
        version="v1",
        license_id="CC0-1.0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        sample_type="line",
    )
    records = build_manifest_records(root, identity)
    write_split_manifest(
        records,
        dataset_dir=root,
        output_dir=root,
        identity=identity,
        train_ratio=1 / 3,
        val_ratio=1 / 3,
        test_ratio=1 / 3,
        min_samples_per_writer=1,
        seed=7,
    )
    return [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_prediction_artifact_is_complete_traceable_and_deidentified(tmp_path: Path) -> None:
    """Only the selected split and evaluator-safe identity should be exported."""
    dataset_dir = tmp_path / "dataset"
    manifest_records = _dataset_package(dataset_dir)
    output_dir = tmp_path / "predictions"
    backend = FakeBackend()

    metadata = produce_prediction_artifact(
        dataset_dir,
        output_dir,
        backend,
        split="test",
        batch_size=2,
        image_loader=lambda path: path.read_bytes(),
    )

    predictions_path = output_dir / "predictions.jsonl"
    predictions = [
        json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    expected_ids = {
        record["sample_id"] for record in manifest_records if record["split"] == "test"
    }
    assert {record["sample_id"] for record in predictions} == expected_ids
    assert all("writer_id" not in record for record in predictions)
    assert all(record["prediction"].startswith("prediction-image-") for record in predictions)
    assert all(record["slices"] == ["age_8", "faint_pencil"] for record in predictions)
    assert metadata["model"]["resolved_revision"] == "a" * 40
    assert metadata["privacy"]["writer_ids_exported"] is False
    assert metadata["output"]["num_records"] == len(expected_ids)
    assert not (output_dir / "prediction_state.json").exists()
    assert not (output_dir / "predictions.partial.jsonl").exists()


def test_interrupted_run_resumes_without_reprocessing_completed_batches(tmp_path: Path) -> None:
    """A failed batch should leave durable progress and resume from the next sample."""
    dataset_dir = tmp_path / "dataset"
    _dataset_package(dataset_dir)
    output_dir = tmp_path / "predictions"
    backend = FakeBackend(fail_on_call=2)

    with pytest.raises(RuntimeError, match="simulated inference interruption"):
        produce_prediction_artifact(
            dataset_dir,
            output_dir,
            backend,
            split="test",
            batch_size=1,
            image_loader=lambda path: path.read_bytes(),
        )

    assert (output_dir / "prediction_state.json").is_file()
    partial_lines = (output_dir / "predictions.partial.jsonl").read_text().splitlines()
    assert len(partial_lines) == 1
    first_completed = backend.completed_inputs[0]

    backend.fail_on_call = None
    produce_prediction_artifact(
        dataset_dir,
        output_dir,
        backend,
        split="test",
        batch_size=1,
        image_loader=lambda path: path.read_bytes(),
    )

    assert backend.completed_inputs.count(first_completed) == 1
    assert len((output_dir / "predictions.jsonl").read_text().splitlines()) == 3


def test_source_hash_mismatch_aborts_before_inference(tmp_path: Path) -> None:
    """Changed source bytes must be detected before any model call occurs."""
    dataset_dir = tmp_path / "dataset"
    manifest_records = _dataset_package(dataset_dir)
    test_record = next(record for record in manifest_records if record["split"] == "test")
    (dataset_dir / test_record["image_path"]).write_bytes(b"tampered")
    backend = FakeBackend()

    with pytest.raises(ValueError, match="Source image SHA-256 mismatch"):
        produce_prediction_artifact(
            dataset_dir,
            tmp_path / "predictions",
            backend,
            split="test",
            image_loader=lambda path: path.read_bytes(),
        )

    assert backend.calls == 0
