"""Reference-free OCR prediction production for protected, offline evaluation."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from src.data.protected_manifest import verify_frozen_protected_evaluation
from src.evaluation.prediction_producer import PredictionBackend, PredictionResult
from src.evaluation.protected_runtime import (
    offline_inference_environment,
    reject_ci_environment,
)

AUTHORIZATION_FILENAME = "protected_inference_authorization.json"
PREDICTIONS_FILENAME = "predictions.jsonl"
ATTESTATION_FILENAME = "execution_attestation.json"
RUNS_DIRECTORY = "runs"

_STATE_FILENAME = "protected_prediction_state.json"
_PARTIAL_FILENAME = "predictions.partial.jsonl"
_MODEL_REQUIRED_FILES = {"config.json", "preprocessor_config.json"}
_UNSAFE_MODEL_SUFFIXES = {".bin", ".ckpt", ".pkl", ".pickle", ".pt", ".pth", ".py"}
_AUTHORIZATION_FIELDS = {
    "schema_version",
    "status",
    "approval_reference",
    "approved_at",
    "expires_at",
    "accountable_owner_role",
    "operator_role",
    "execution_environment",
    "evaluation_set_id",
    "manifest_sha256",
    "model_version",
    "model_artifact_sha256",
    "device",
    "inference_config",
    "prediction_schema",
    "protected_storage_required",
    "network_access_during_inference",
    "external_ai_services_allowed",
    "github_actions_allowed",
    "model_artifacts_preloaded",
    "training_allowed",
    "automated_decisions_allowed",
}
_PREDICTION_FIELDS = {
    "sample_id",
    "source_sha256",
    "prediction",
    "confidence",
    "latency_ms",
}

ImageLoader = Callable[[bytes], Any]
BackendFactory = Callable[[str, str, str], PredictionBackend]


def inspect_local_model_artifact(
    model_dir: str | Path,
    *,
    repository_root: str | Path | None = None,
    require_read_only: bool = True,
) -> dict[str, Any]:
    """Hash a materialized, code-free local model directory without following links."""
    repository = _repository_root(repository_root)
    raw_root = Path(model_dir)
    if raw_root.is_symlink():
        raise ValueError("Local model directory must not be a symbolic link")
    root = raw_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Local model artifact must be a directory")
    _require_outside_repository(root, repository, "Local model artifact")

    entries = [root, *sorted(root.rglob("*"), key=lambda path: path.as_posix())]
    files: list[Path] = []
    read_only = True
    for entry in entries:
        if entry.is_symlink():
            raise ValueError("Local model artifact must not contain symbolic links")
        relative_parts = entry.relative_to(root).parts if entry != root else ()
        if any(part in {".cache", ".git"} for part in relative_parts):
            raise ValueError("Local model artifact must not contain cache or Git directories")
        mode = entry.stat().st_mode
        if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            read_only = False
        if entry.is_dir():
            continue
        if not entry.is_file():
            raise ValueError("Local model artifact must contain only regular files")
        if mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            raise ValueError("Local model artifact files must not be executable")
        if entry.suffix.casefold() in _UNSAFE_MODEL_SUFFIXES:
            raise ValueError("Local model artifact contains an unsafe executable/pickle format")
        files.append(entry)
    if require_read_only and not read_only:
        raise ValueError("Local model artifact must have no write permission bits")
    relative_files = {path.relative_to(root).as_posix() for path in files}
    if not _MODEL_REQUIRED_FILES.issubset(relative_files):
        raise ValueError("Local TrOCR artifact is missing required configuration files")
    if not any(path.suffix.casefold() == ".safetensors" for path in files):
        raise ValueError("Local TrOCR artifact must contain safetensors weights")

    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        file_sha256, size = _hash_file_and_size(path)
        total_bytes += size
        canonical = json.dumps(
            {"path": relative, "sha256": file_sha256, "size_bytes": size},
            separators=(",", ":"),
            sort_keys=True,
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return {
        "schema_version": 1,
        "model_artifact_sha256": digest.hexdigest(),
        "num_files": len(files),
        "total_bytes": total_bytes,
        "read_only": read_only,
        "symlinks_allowed": False,
        "pickle_weights_allowed": False,
        "remote_code_allowed": False,
    }


def produce_protected_prediction_artifact(
    *,
    contract_path: str | Path,
    dataset_dir: str | Path,
    manifest_dir: str | Path,
    authorization_path: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    backend_factory: BackendFactory,
    batch_size: int = 8,
    image_loader: ImageLoader | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Produce resumable reference-free predictions and an execution attestation."""
    reject_ci_environment("Protected inference")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    repository = _repository_root(repository_root)
    dataset_path = Path(dataset_dir)
    if dataset_path.is_symlink():
        raise ValueError("Protected dataset root must not be a symbolic link")
    root = dataset_path.resolve(strict=True)
    metadata, manifest_records = verify_frozen_protected_evaluation(
        contract_path,
        dataset_path,
        manifest_dir,
        repository_root=repository,
    )
    authorization_file = _required_root_file(
        authorization_path, root, AUTHORIZATION_FILENAME, "Inference authorization"
    )
    _require_read_only(authorization_file, "Inference authorization")
    authorization_content = authorization_file.read_bytes()
    authorization_sha256 = hashlib.sha256(authorization_content).hexdigest()
    authorization = _validate_authorization(
        authorization_content,
        evaluation_set_id=_required_string(metadata, "evaluation_set_id", "metadata"),
        manifest_sha256=_required_string(
            _required_mapping(metadata, "manifest", "metadata"), "sha256", "metadata.manifest"
        ),
    )
    model_artifact = inspect_local_model_artifact(
        model_dir,
        repository_root=repository,
        require_read_only=True,
    )
    if authorization["model_artifact_sha256"] != model_artifact["model_artifact_sha256"]:
        raise ValueError("Approved model_artifact_sha256 does not match the local model")
    inference_config = _required_mapping(authorization, "inference_config", "authorization")
    if inference_config["batch_size"] != batch_size:
        raise ValueError("Runtime batch_size does not match the inference authorization")

    destination, work, lock_path = _run_directories(
        output_dir, root, str(authorization["model_version"])
    )
    selected = [
        record
        for record in sorted(manifest_records, key=lambda item: str(item["sample_id"]))
        if record.get("split") == "test"
    ]
    if not selected:
        raise ValueError("Frozen manifest contains no test records")
    loader = image_loader or _load_rgb_image

    with _exclusive_lock(lock_path):
        if destination.exists():
            if work.exists():
                raise ValueError("Completed and incomplete protected runs both exist")
            return _validate_completed_run(
                destination,
                authorization,
                authorization_sha256,
                metadata,
                model_artifact,
                selected,
            )
        write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        if work.exists() and not (work.stat().st_mode & write_bits):
            recovered = _validate_completed_run(
                work,
                authorization,
                authorization_sha256,
                metadata,
                model_artifact,
                selected,
            )
            work.replace(destination)
            return {**recovered, "output_dir": str(destination)}
        work.mkdir(mode=0o700, exist_ok=True)
        if work.is_symlink() or not work.is_dir():
            raise ValueError("Incomplete protected run path is invalid")
        allowed_work_files = {
            _STATE_FILENAME,
            _PARTIAL_FILENAME,
            PREDICTIONS_FILENAME,
            ATTESTATION_FILENAME,
        }
        work_entries = list(work.iterdir())
        if any(path.name not in allowed_work_files for path in work_entries):
            raise ValueError("Incomplete protected run contains an unexpected file")
        if any(path.is_symlink() or not path.is_file() for path in work_entries):
            raise ValueError("Incomplete protected run must contain only regular files")

        with offline_inference_environment():
            backend = backend_factory(
                str(authorization["model_version"]),
                str(model_artifact["model_artifact_sha256"]),
                str(authorization["device"]),
            )
            provenance = backend.provenance
            _validate_backend_provenance(provenance, authorization, model_artifact)
            expected_state = {
                "schema_version": 1,
                "authorization_sha256": authorization_sha256,
                "evaluation_set_id": metadata["evaluation_set_id"],
                "manifest_sha256": metadata["manifest"]["sha256"],
                "model_artifact_sha256": model_artifact["model_artifact_sha256"],
                "model_version": authorization["model_version"],
                "device": authorization["device"],
                "batch_size": batch_size,
                "backend": provenance,
            }
            state_path = work / _STATE_FILENAME
            partial_path = work / _PARTIAL_FILENAME
            predictions_path = work / PREDICTIONS_FILENAME
            attestation_path = work / ATTESTATION_FILENAME
            _initialize_or_validate_state(state_path, expected_state)
            completed = _load_work_predictions(
                predictions_path if predictions_path.exists() else partial_path,
                selected,
            )
            remaining = [
                record for record in selected if str(record["sample_id"]) not in completed
            ]
            if predictions_path.exists() and remaining:
                raise ValueError("Finalized prediction file is incomplete")
            if predictions_path.exists() and partial_path.exists():
                raise ValueError("Protected run contains conflicting prediction files")

            if remaining:
                with _append_jsonl_no_follow(partial_path) as partial_handle:
                    for offset in range(0, len(remaining), batch_size):
                        batch_records = remaining[offset : offset + batch_size]
                        images = [
                            _load_verified_image(root, record, loader)
                            for record in batch_records
                        ]
                        try:
                            results = backend.predict_batch(images)
                        finally:
                            for image in images:
                                close = getattr(image, "close", None)
                                if callable(close):
                                    close()
                        if len(results) != len(batch_records):
                            raise RuntimeError(
                                "Prediction backend returned a different number of results"
                            )
                        for record, result in zip(batch_records, results):
                            output_record = _prediction_record(record, result, authorization)
                            partial_handle.write(
                                json.dumps(output_record, ensure_ascii=False, sort_keys=True) + "\n"
                            )
                        partial_handle.flush()
                        os.fsync(partial_handle.fileno())
                completed = _load_work_predictions(partial_path, selected)
            if set(completed) != {str(record["sample_id"]) for record in selected}:
                raise RuntimeError("Protected prediction artifact is incomplete after inference")

        _verify_inputs_unchanged(
            contract_path=contract_path,
            dataset_dir=root,
            manifest_dir=manifest_dir,
            repository=repository,
            expected_metadata=metadata,
            authorization_file=authorization_file,
            authorization_sha256=authorization_sha256,
            model_dir=model_dir,
            expected_model_artifact=model_artifact,
        )
        if not predictions_path.exists():
            partial_path.replace(predictions_path)
        predictions_sha256 = _sha256_file(predictions_path)
        attestation = _build_attestation(
            authorization,
            authorization_sha256=authorization_sha256,
            predictions_sha256=predictions_sha256,
        )
        if attestation_path.exists():
            existing_attestation = _load_object(attestation_path, "Execution attestation")
            attestation = _validate_existing_attestation(existing_attestation, attestation)
        else:
            _write_json_atomic(attestation_path, attestation)
        state_path.unlink(missing_ok=True)
        predictions_path.chmod(0o400)
        attestation_path.chmod(0o400)
        work.chmod(0o500)
        work.replace(destination)
        return _completion_summary(destination, attestation, len(selected))


def _validate_authorization(
    content: bytes, *, evaluation_set_id: str, manifest_sha256: str
) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Inference authorization is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or set(value) != _AUTHORIZATION_FIELDS:
        raise ValueError("Inference authorization fields must exactly match the allowlist")
    if value.get("schema_version") != 1 or value.get("status") != "approved":
        raise ValueError("Inference authorization must be approved schema_version 1")
    if value.get("execution_environment") != "approved_self_hosted":
        raise ValueError("Inference authorization environment must be approved_self_hosted")
    if value.get("evaluation_set_id") != evaluation_set_id:
        raise ValueError("Inference authorization evaluation_set_id does not match")
    if value.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Inference authorization manifest_sha256 does not match")
    _required_sha256(value, "model_artifact_sha256", "authorization")
    _safe_component(_required_string(value, "model_version", "authorization"), "model_version")
    if value.get("device") not in {"cpu", "cuda"}:
        raise ValueError("Inference authorization device must be cpu or cuda")
    inference_config = _required_mapping(value, "inference_config", "authorization")
    if set(inference_config) != {
        "batch_size",
        "beam_width",
        "max_length",
        "confidence_threshold",
        "abstention_threshold",
        "dtype",
    }:
        raise ValueError("Inference authorization inference_config fields are invalid")
    for key in ("batch_size", "beam_width", "max_length"):
        item = inference_config.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"Inference authorization inference_config.{key} is invalid")
    confidence_threshold = _finite_number(
        inference_config, "confidence_threshold", minimum=0.0, maximum=1.0
    )
    abstention_threshold = _finite_number(
        inference_config, "abstention_threshold", minimum=0.0, maximum=1.0
    )
    if abstention_threshold > confidence_threshold:
        raise ValueError("Inference authorization abstention threshold exceeds confidence")
    expected_dtype = "float16" if value.get("device") == "cuda" else "float32"
    if inference_config.get("dtype") != expected_dtype:
        raise ValueError(f"Inference authorization dtype must be {expected_dtype}")
    if value.get("prediction_schema") != "inkbridge-reference-free-v1":
        raise ValueError("Inference authorization prediction_schema is invalid")
    for key, expected in {
        "protected_storage_required": True,
        "network_access_during_inference": False,
        "external_ai_services_allowed": False,
        "github_actions_allowed": False,
        "model_artifacts_preloaded": True,
        "training_allowed": False,
        "automated_decisions_allowed": False,
    }.items():
        if not isinstance(value.get(key), bool) or value[key] is not expected:
            raise ValueError(f"Inference authorization {key} must be {expected}")
    _approval_reference(value, "approval_reference")
    _role(value, "accountable_owner_role")
    _role(value, "operator_role")
    approved_at = _timestamp(value, "approved_at")
    expires_at = _timestamp(value, "expires_at")
    now = datetime.now(timezone.utc)
    if approved_at > now:
        raise ValueError("Inference authorization approved_at must not be in the future")
    if expires_at <= now:
        raise ValueError("Inference authorization has expired")
    if expires_at <= approved_at or expires_at - approved_at > timedelta(days=30):
        raise ValueError(
            "Inference authorization validity must be greater than 0 and at most 30 days"
        )
    return value


def _validate_backend_provenance(
    provenance: Mapping[str, Any],
    authorization: Mapping[str, Any],
    model_artifact: Mapping[str, Any],
) -> None:
    try:
        json.dumps(provenance, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Protected backend provenance must be JSON-serializable") from exc
    required = {
        "model_version": authorization["model_version"],
        "model_artifact_sha256": model_artifact["model_artifact_sha256"],
        "device": authorization["device"],
        "local_files_only": True,
        "network_access_allowed": False,
        "external_ai_service": False,
    }
    inference_config = _required_mapping(authorization, "inference_config", "authorization")
    required.update({key: inference_config[key] for key in inference_config})
    for key, expected in required.items():
        if provenance.get(key) != expected:
            raise ValueError(f"Protected backend provenance {key} does not match")


def _prediction_record(
    manifest_record: Mapping[str, Any],
    result: PredictionResult,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(result.raw_text, str):
        raise TypeError("Prediction raw_text must be a string")
    if not math.isfinite(result.confidence) or not 0 <= result.confidence <= 1:
        raise ValueError("Prediction confidence must be finite and between 0 and 1")
    if not math.isfinite(result.latency_ms) or result.latency_ms < 0:
        raise ValueError("Prediction latency_ms must be finite and non-negative")
    if result.model_version != authorization["model_version"]:
        raise ValueError("Prediction model_version does not match the authorization")
    return {
        "sample_id": manifest_record["sample_id"],
        "source_sha256": manifest_record["source_sha256"],
        "prediction": result.raw_text,
        "confidence": float(result.confidence),
        "latency_ms": float(result.latency_ms),
    }


def _load_work_predictions(
    path: Path, selected_records: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    selected = {str(record["sample_id"]): record for record in selected_records}
    completed: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Protected prediction line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(value, dict) or set(value) != _PREDICTION_FIELDS:
                raise ValueError("Protected prediction fields violate the reference-free allowlist")
            sample_id = value.get("sample_id")
            if not isinstance(sample_id, str) or sample_id not in selected:
                raise ValueError("Protected predictions contain a sample outside the test split")
            if sample_id in completed:
                raise ValueError("Protected predictions contain a duplicate sample_id")
            source = selected[sample_id]
            if value.get("source_sha256") != source["source_sha256"]:
                raise ValueError("Protected prediction source hash does not match the manifest")
            if not isinstance(value.get("prediction"), str):
                raise ValueError("Protected prediction text must be a string")
            _finite_number(value, "confidence", minimum=0.0, maximum=1.0)
            _finite_number(value, "latency_ms", minimum=0.0)
            completed[sample_id] = value
    return completed


def _load_verified_image(root: Path, record: Mapping[str, Any], loader: ImageLoader) -> Any:
    relative = Path(str(record["image_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Protected image path escapes the dataset root")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError("Protected image must be a non-symlink file")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != record["source_sha256"]:
        raise ValueError("Protected source image SHA-256 changed before inference")
    return loader(content)


def _load_rgb_image(content: bytes) -> Image.Image:
    with Image.open(io.BytesIO(content)) as image:
        image.load()
        return image.convert("RGB")


def _initialize_or_validate_state(path: Path, expected: dict[str, Any]) -> None:
    if path.exists():
        if path.is_symlink() or _load_object(path, "Prediction state") != expected:
            raise ValueError("Existing protected prediction state belongs to another run")
        return
    _write_json_atomic(path, expected)


def _verify_inputs_unchanged(
    *,
    contract_path: str | Path,
    dataset_dir: Path,
    manifest_dir: str | Path,
    repository: Path,
    expected_metadata: Mapping[str, Any],
    authorization_file: Path,
    authorization_sha256: str,
    model_dir: str | Path,
    expected_model_artifact: Mapping[str, Any],
) -> None:
    metadata, _ = verify_frozen_protected_evaluation(
        contract_path,
        dataset_dir,
        manifest_dir,
        repository_root=repository,
    )
    if metadata != expected_metadata:
        raise ValueError("Frozen evaluation metadata changed during inference")
    if _sha256_file(authorization_file) != authorization_sha256:
        raise ValueError("Inference authorization changed during inference")
    model_artifact = inspect_local_model_artifact(
        model_dir,
        repository_root=repository,
        require_read_only=True,
    )
    if model_artifact != expected_model_artifact:
        raise ValueError("Local model artifact changed during inference")


def _build_attestation(
    authorization: Mapping[str, Any], *, authorization_sha256: str, predictions_sha256: str
) -> dict[str, Any]:
    inference_config_sha256 = _canonical_sha256(authorization["inference_config"])
    return {
        "schema_version": 1,
        "execution_environment": "approved_self_hosted",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "operator_role": authorization["operator_role"],
        "approval_reference": authorization["approval_reference"],
        "inference_authorization_sha256": authorization_sha256,
        "model_version": authorization["model_version"],
        "model_artifact_sha256": authorization["model_artifact_sha256"],
        "inference_config_sha256": inference_config_sha256,
        "evaluation_set_id": authorization["evaluation_set_id"],
        "manifest_sha256": authorization["manifest_sha256"],
        "predictions_sha256": predictions_sha256,
        "protected_storage_mounted": True,
        "network_access_during_inference": False,
        "external_ai_services_used": False,
        "github_actions_used": False,
        "model_artifacts_preloaded": True,
        "predictions_contain_references": False,
        "automated_decisions_enabled": False,
    }


def _validate_existing_attestation(
    existing: Mapping[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    executed_at = _timestamp(existing, "executed_at", context="attestation")
    if executed_at > datetime.now(timezone.utc):
        raise ValueError("Existing execution attestation timestamp is in the future")
    expected = {**current, "executed_at": existing.get("executed_at")}
    if existing != expected:
        raise ValueError("Existing execution attestation does not match this run")
    return dict(existing)


def _validate_completed_run(
    destination: Path,
    authorization: Mapping[str, Any],
    authorization_sha256: str,
    metadata: Mapping[str, Any],
    model_artifact: Mapping[str, Any],
    selected_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("Completed protected run directory is invalid")
    _require_read_only(destination, "Completed protected run directory")
    if {path.name for path in destination.iterdir()} != {
        PREDICTIONS_FILENAME,
        ATTESTATION_FILENAME,
    }:
        raise ValueError("Completed protected run must contain exactly two files")
    predictions = _required_run_file(destination / PREDICTIONS_FILENAME, "Predictions")
    attestation_path = _required_run_file(destination / ATTESTATION_FILENAME, "Attestation")
    attestation = _load_object(attestation_path, "Execution attestation")
    executed_at = _timestamp(attestation, "executed_at", context="attestation")
    if executed_at > datetime.now(timezone.utc):
        raise ValueError("Completed execution attestation timestamp is in the future")
    expected_attestation = {
        **_build_attestation(
            authorization,
            authorization_sha256=authorization_sha256,
            predictions_sha256=_sha256_file(predictions),
        ),
        "executed_at": attestation.get("executed_at"),
    }
    if attestation != expected_attestation:
        raise ValueError("Completed execution attestation does not match current inputs")
    if attestation.get("manifest_sha256") != metadata["manifest"]["sha256"]:
        raise ValueError("Completed run manifest identity does not match")
    if attestation.get("model_artifact_sha256") != model_artifact["model_artifact_sha256"]:
        raise ValueError("Completed run model identity does not match")
    records = _load_work_predictions(predictions, selected_records)
    if len(records) != len(selected_records):
        raise ValueError("Completed protected prediction count does not match")
    return _completion_summary(destination, attestation, len(selected_records))


def _completion_summary(
    destination: Path, attestation: Mapping[str, Any], count: int
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "protected_predictions_ready",
        "model_version": attestation["model_version"],
        "evaluation_set_id": attestation["evaluation_set_id"],
        "predictions": {
            "filename": PREDICTIONS_FILENAME,
            "sha256": attestation["predictions_sha256"],
            "num_records": count,
            "contains_references": False,
        },
        "attestation": {
            "filename": ATTESTATION_FILENAME,
            "sha256": _sha256_file(destination / ATTESTATION_FILENAME),
        },
        "output_dir": str(destination),
    }


def _run_directories(
    output_dir: str | Path, root: Path, model_version: str
) -> tuple[Path, Path, Path]:
    safe_version = _safe_component(model_version, "model_version")
    destination = Path(output_dir)
    runs = root / RUNS_DIRECTORY
    if destination.name != safe_version or destination.parent.resolve(strict=False) != runs:
        raise ValueError("Protected output must be <dataset-dir>/runs/<model-version>")
    if destination.is_symlink():
        raise ValueError("Protected output must not be a symbolic link")
    if runs.is_symlink():
        raise ValueError("Protected runs directory must not be a symbolic link")
    runs.mkdir(mode=0o700, exist_ok=True)
    if not runs.is_dir():
        raise ValueError("Protected runs path must be a directory")
    return destination, runs / f".{safe_version}.incomplete", runs / f".{safe_version}.lock"


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    if path.is_symlink():
        raise ValueError("Protected inference lock must not be a symbolic link")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another protected inference process holds this run lock") from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _append_jsonl_no_follow(path: Path) -> Iterator[Any]:
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Protected prediction progress must be a regular file")
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            descriptor = -1
            yield handle
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _required_root_file(path: str | Path, root: Path, name: str, label: str) -> Path:
    value = Path(path)
    if value.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    resolved = value.resolve(strict=True)
    if resolved.parent != root or resolved.name != name or not resolved.is_file():
        raise ValueError(f"{label} must be <dataset-dir>/{name}")
    return resolved


def _required_run_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a non-symlink file")
    _require_read_only(path, label)
    return path


def _require_read_only(path: Path, label: str) -> None:
    if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError(f"{label} must have no write permission bits")


def _repository_root(value: str | Path | None) -> Path:
    root = (
        Path(value).resolve(strict=True)
        if value is not None
        else Path(__file__).parents[2].resolve(strict=True)
    )
    if not (root / ".git").exists():
        raise ValueError("repository_root must identify the InkBridge Git checkout")
    return root


def _require_outside_repository(path: Path, repository: Path, label: str) -> None:
    try:
        path.resolve().relative_to(repository)
    except ValueError:
        return
    raise ValueError(f"{label} must be outside the Git repository")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _hash_file_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _sha256_file(path: Path) -> str:
    return _hash_file_and_size(path)[0]


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(value, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _required_mapping(value: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"{context}.{key} must be an object")
    return item


def _required_string(value: Mapping[str, Any], key: str, context: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return item.strip()


def _required_sha256(value: Mapping[str, Any], key: str, context: str) -> str:
    item = _required_string(value, key, context)
    if len(item) != 64 or any(character not in "0123456789abcdef" for character in item):
        raise ValueError(f"{context}.{key} must be a lowercase SHA-256 digest")
    return item


def _safe_component(value: str, label: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,127}", value) is None:
        raise ValueError(f"{label} must contain only safe version characters")
    return value


def _approval_reference(value: Mapping[str, Any], key: str) -> str:
    item = _required_string(value, key, "authorization")
    if re.fullmatch(r"[a-z][a-z0-9_-]{7,127}", item) is None:
        raise ValueError(f"authorization.{key} must be an opaque internal reference")
    return item


def _role(value: Mapping[str, Any], key: str) -> str:
    item = _required_string(value, key, "authorization")
    if re.fullmatch(r"[a-z][a-z0-9_]{7,63}", item) is None or item in {
        "pending",
        "replace_me",
    }:
        raise ValueError(f"authorization.{key} must be an approved role")
    return item


def _timestamp(
    value: Mapping[str, Any], key: str, *, context: str = "authorization"
) -> datetime:
    raw = _required_string(value, key, context)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{context}.{key} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{context}.{key} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _finite_number(
    value: Mapping[str, Any], key: str, *, minimum: float, maximum: float | None = None
) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise ValueError(f"Protected prediction {key} must be numeric")
    number = float(item)
    if not math.isfinite(number) or number < minimum or (
        maximum is not None and number > maximum
    ):
        raise ValueError(f"Protected prediction {key} is outside the allowed range")
    return number
