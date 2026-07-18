"""Resumable, provenance-locked production of offline OCR prediction artifacts."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from src.data.manifest import sha256_file


@dataclass(frozen=True)
class PredictionResult:
    """Backend-neutral result for one line image."""

    raw_text: str
    confidence: float
    latency_ms: float
    is_uncertain: bool
    is_unreadable: bool
    model_version: str


class PredictionBackend(Protocol):
    """Minimal inference interface used by the artifact producer."""

    @property
    def provenance(self) -> dict[str, Any]:
        """Return immutable, JSON-serializable model and runtime provenance."""
        ...

    def predict_batch(self, images: list[Any]) -> list[PredictionResult]:
        """Predict a batch in input order."""
        ...


ImageLoader = Callable[[Path], Any]


def produce_prediction_artifact(
    dataset_dir: str | Path,
    output_dir: str | Path,
    backend: PredictionBackend,
    *,
    split: str = "test",
    batch_size: int = 8,
    resume: bool = True,
    image_loader: ImageLoader | None = None,
) -> dict[str, Any]:
    """Validate a line manifest and atomically produce evaluator-ready JSONL."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")

    dataset_root = Path(dataset_dir).resolve(strict=True)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_root / "manifest.jsonl"
    metadata_path = dataset_root / "manifest.meta.json"
    manifest_metadata, selected_records = _load_and_validate_manifest(
        dataset_root, manifest_path, metadata_path, split
    )

    provenance = backend.provenance
    _require_json_object(provenance, "backend provenance")
    expected_state = {
        "schema_version": 1,
        "manifest_sha256": sha256_file(manifest_path),
        "split": split,
        "backend": provenance,
    }

    partial_path = destination / "predictions.partial.jsonl"
    state_path = destination / "prediction_state.json"
    final_path = destination / "predictions.jsonl"
    final_metadata_path = destination / "predictions.meta.json"

    if not resume:
        for stale_path in (partial_path, state_path, final_path, final_metadata_path):
            stale_path.unlink(missing_ok=True)
    elif final_path.is_file():
        return _validate_completed_artifact(
            final_path, final_metadata_path, expected_state, len(selected_records)
        )

    if state_path.exists():
        prior_state = _load_json_object(state_path, "prediction state")
        if prior_state != expected_state:
            raise ValueError(
                "Existing prediction state belongs to a different manifest, split, or backend"
            )
    elif partial_path.exists():
        raise ValueError("Partial predictions exist without a matching prediction state")
    else:
        _write_json_atomic(state_path, expected_state)

    records_by_id = {str(record["sample_id"]): record for record in selected_records}
    completed = _load_partial_predictions(partial_path, records_by_id)
    remaining = [
        record for record in selected_records if str(record["sample_id"]) not in completed
    ]
    loader = image_loader or _load_rgb_image

    with partial_path.open("a", encoding="utf-8") as partial_handle:
        for offset in range(0, len(remaining), batch_size):
            batch_records = remaining[offset : offset + batch_size]
            images = [loader(Path(record["_absolute_image_path"])) for record in batch_records]
            try:
                results = backend.predict_batch(images)
            finally:
                for image in images:
                    close = getattr(image, "close", None)
                    if callable(close):
                        close()
            if len(results) != len(batch_records):
                raise RuntimeError("Prediction backend returned a different number of results")

            for record, result in zip(batch_records, results):
                output_record = _prediction_record(record, result)
                json.dump(output_record, partial_handle, ensure_ascii=False, sort_keys=True)
                partial_handle.write("\n")
            partial_handle.flush()
            os.fsync(partial_handle.fileno())

    complete = _load_partial_predictions(partial_path, records_by_id)
    if set(complete) != set(records_by_id):
        raise RuntimeError("Prediction artifact is incomplete after inference")

    partial_path.replace(final_path)
    artifact_metadata = {
        "schema_version": 1,
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "manifest_filename": manifest_path.name,
            "manifest_sha256": expected_state["manifest_sha256"],
            "manifest_metadata_sha256": sha256_file(metadata_path),
            "dataset": manifest_metadata["dataset"],
            "split": split,
            "num_records": len(selected_records),
        },
        "model": provenance,
        "output": {
            "filename": final_path.name,
            "sha256": sha256_file(final_path),
            "num_records": len(selected_records),
        },
        "privacy": {
            "writer_ids_exported": False,
            "source_images_exported": False,
        },
    }
    _write_json_atomic(final_metadata_path, artifact_metadata)
    state_path.unlink(missing_ok=True)
    return artifact_metadata


def _load_and_validate_manifest(
    dataset_root: Path, manifest_path: Path, metadata_path: Path, split: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Dataset package must contain manifest.jsonl and manifest.meta.json at its root"
        )
    metadata = _load_json_object(metadata_path, "manifest metadata")
    if metadata.get("schema_version") != 2:
        raise ValueError("Prediction production requires manifest schema_version 2")
    dataset = metadata.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("sample_type") != "line":
        raise ValueError("Pretrained TrOCR baseline accepts only line-level manifests")
    manifest_details = metadata.get("manifest")
    if not isinstance(manifest_details, dict):
        raise ValueError("Manifest metadata is missing its manifest identity")
    actual_manifest_hash = sha256_file(manifest_path)
    if manifest_details.get("sha256") != actual_manifest_hash:
        raise ValueError("manifest.jsonl SHA-256 does not match manifest.meta.json")
    split_details = metadata.get("split")
    if (
        not isinstance(split_details, dict)
        or split_details.get("writer_isolation_verified") is not True
    ):
        raise ValueError("Manifest metadata does not assert writer-isolated splits")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    writers_by_split: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"manifest line {line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"manifest line {line_number}: record must be an object")
            _validate_manifest_record(value, line_number)
            sample_id = str(value["sample_id"])
            if sample_id in seen_ids:
                raise ValueError(f"manifest line {line_number}: duplicate sample_id")
            seen_ids.add(sample_id)
            record_split = str(value["split"])
            writers_by_split[record_split].add(str(value["writer_id"]))
            image_path = _contained_image_path(dataset_root, str(value["image_path"]), line_number)
            value["_absolute_image_path"] = str(image_path)
            records.append(value)

    if not records:
        raise ValueError("Manifest contains no records")
    _verify_writer_isolation(writers_by_split)
    selected = sorted(
        (record for record in records if record["split"] == split),
        key=lambda record: str(record["sample_id"]),
    )
    if not selected:
        raise ValueError(f"Manifest contains no records for split '{split}'")

    for record in selected:
        image_path = Path(str(record["_absolute_image_path"]))
        if sha256_file(image_path) != record["source_sha256"]:
            raise ValueError(f"Source image SHA-256 mismatch for sample '{record['sample_id']}'")
    return metadata, selected


def _validate_manifest_record(value: dict[str, Any], line_number: int) -> None:
    prefix = f"manifest line {line_number}"
    if value.get("schema_version") != 2:
        raise ValueError(f"{prefix}: schema_version must be 2")
    if value.get("sample_type") != "line":
        raise ValueError(f"{prefix}: sample_type must be 'line'")
    for key in ("sample_id", "reference", "writer_id", "image_path"):
        item = value.get(key)
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{prefix}: '{key}' must be a non-empty string")
    source_hash = value.get("source_sha256")
    if not _is_sha256(source_hash):
        raise ValueError(f"{prefix}: source_sha256 must be a SHA-256 digest")
    if value.get("split") not in {"train", "validation", "test"}:
        raise ValueError(f"{prefix}: split is invalid")
    slices = value.get("slices")
    if not isinstance(slices, list) or any(
        not isinstance(item, str) or not item.strip() for item in slices
    ):
        raise ValueError(f"{prefix}: slices must be a list of non-empty strings")


def _verify_writer_isolation(writers_by_split: dict[str, set[str]]) -> None:
    split_names = tuple(writers_by_split)
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            overlap = writers_by_split[left_name] & writers_by_split[right_name]
            if overlap:
                raise ValueError(
                    f"Writer isolation violation between {left_name} and {right_name} splits"
                )


def _contained_image_path(dataset_root: Path, relative_value: str, line_number: int) -> Path:
    relative_path = Path(relative_value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"manifest line {line_number}: image_path escapes the dataset package")
    try:
        image_path = (dataset_root / relative_path).resolve(strict=True)
        image_path.relative_to(dataset_root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(
            f"manifest line {line_number}: image is missing or outside the dataset package"
        ) from exc
    if not image_path.is_file():
        raise ValueError(f"manifest line {line_number}: image_path is not a file")
    return image_path


def _prediction_record(
    manifest_record: dict[str, Any], result: PredictionResult
) -> dict[str, Any]:
    if not isinstance(result.raw_text, str):
        raise TypeError("Prediction raw_text must be a string")
    if not 0 <= result.confidence <= 1:
        raise ValueError("Prediction confidence must be between 0 and 1")
    if result.latency_ms < 0:
        raise ValueError("Prediction latency_ms must be non-negative")
    if not result.model_version.strip():
        raise ValueError("Prediction model_version must be non-empty")
    return {
        "schema_version": 1,
        "sample_id": manifest_record["sample_id"],
        "source_sha256": manifest_record["source_sha256"],
        "reference": manifest_record["reference"],
        "prediction": result.raw_text,
        "confidence": float(result.confidence),
        "slices": manifest_record["slices"],
        "latency_ms": float(result.latency_ms),
        "model_version": result.model_version,
        "is_uncertain": result.is_uncertain,
        "is_unreadable": result.is_unreadable,
        "split": manifest_record["split"],
    }


def _load_partial_predictions(
    path: Path, manifest_records: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"partial prediction line {line_number}: invalid JSON") from exc
            if not isinstance(value, dict) or not isinstance(value.get("sample_id"), str):
                raise ValueError(f"partial prediction line {line_number}: invalid record")
            sample_id = value["sample_id"]
            source = manifest_records.get(sample_id)
            if source is None:
                raise ValueError("Partial predictions contain a sample outside the selected split")
            if sample_id in records:
                raise ValueError("Partial predictions contain a duplicate sample_id")
            if (
                value.get("source_sha256") != source["source_sha256"]
                or value.get("reference") != source["reference"]
            ):
                raise ValueError("Partial prediction identity does not match the manifest")
            records[sample_id] = value
    return records


def _validate_completed_artifact(
    final_path: Path,
    metadata_path: Path,
    expected_state: dict[str, Any],
    expected_count: int,
) -> dict[str, Any]:
    metadata = _load_json_object(metadata_path, "prediction metadata")
    input_details = metadata.get("input")
    output_details = metadata.get("output")
    if not isinstance(input_details, dict) or not isinstance(output_details, dict):
        raise ValueError("Completed prediction metadata is incomplete")
    if (
        input_details.get("manifest_sha256") != expected_state["manifest_sha256"]
        or input_details.get("split") != expected_state["split"]
        or metadata.get("model") != expected_state["backend"]
        or output_details.get("sha256") != sha256_file(final_path)
        or output_details.get("num_records") != expected_count
    ):
        raise ValueError("Completed prediction artifact does not match this run")
    return metadata


def _load_rgb_image(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {description}: {path}") from exc
    return _require_json_object(value, description)


def _require_json_object(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    try:
        json.dumps(value, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be JSON-serializable") from exc
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary_path.replace(path)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )
