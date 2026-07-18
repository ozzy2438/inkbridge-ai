"""Fail-closed, local-model prediction production for the SMHD research rehearsal."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.manifest import sha256_file
from src.evaluation.prediction_producer import PredictionBackend, produce_prediction_artifact
from src.evaluation.protected_inference import inspect_local_model_artifact
from src.evaluation.protected_runtime import offline_inference_environment, reject_ci_environment

ATTESTATION_FILENAME = "research_execution_attestation.json"

BackendFactory = Callable[[str, str, str], PredictionBackend]

_EXPECTED_DATASET = {
    "name": "smhd-research-rehearsal",
    "version": "figshare-24419986-v1-selection-v1",
    "license_id": "CC-BY-NC-4.0",
    "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
    "sample_type": "line",
}
_EXPECTED_PURPOSE = "offline_student_handwriting_research_shadow_rehearsal_only"
_ALLOWED_OUTPUT_FILES = {
    "prediction_state.json",
    "predictions.partial.jsonl",
    "predictions.jsonl",
    "predictions.meta.json",
    ATTESTATION_FILENAME,
}


def produce_offline_research_prediction_artifact(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    model_dir: str | Path,
    model_version: str,
    expected_model_artifact_sha256: str,
    backend_factory: BackendFactory,
    device: str = "cpu",
    batch_size: int = 2,
    beam_width: int = 4,
    max_length: int = 128,
    confidence_threshold: float = 0.7,
    abstention_threshold: float = 0.4,
    resume: bool = True,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run test-split inference with a sealed model and no Python network access."""
    reject_ci_environment("Offline research inference")
    _validate_runtime_settings(
        model_version=model_version,
        expected_model_artifact_sha256=expected_model_artifact_sha256,
        device=device,
        batch_size=batch_size,
        beam_width=beam_width,
        max_length=max_length,
        confidence_threshold=confidence_threshold,
        abstention_threshold=abstention_threshold,
    )
    repository = Path(repository_root or Path.cwd()).resolve(strict=True)
    dataset = _validated_external_directory(dataset_dir, repository, "SMHD dataset")
    _require_owner_only_tree(dataset, "SMHD dataset")
    input_binding = _validate_smhd_package(dataset)

    model = Path(model_dir)
    model_artifact = inspect_local_model_artifact(
        model,
        repository_root=repository,
        require_read_only=True,
    )
    if model_artifact["model_artifact_sha256"] != expected_model_artifact_sha256:
        raise ValueError("Expected model artifact SHA-256 does not match the sealed local model")

    destination = _prepare_output_directory(output_dir, dataset=dataset, repository=repository)
    if not resume:
        (destination / ATTESTATION_FILENAME).unlink(missing_ok=True)
    expected_provenance = {
        "backend": "trocr_local_research",
        "evidence_scope": "noncommercial_research_rehearsal_only",
        "protected_pilot_evidence": False,
        "model_version": model_version,
        "model_artifact_sha256": expected_model_artifact_sha256,
        "device": device,
        "dtype": "float16" if device == "cuda" else "float32",
        "batch_size": batch_size,
        "beam_width": beam_width,
        "max_length": max_length,
        "confidence_threshold": confidence_threshold,
        "abstention_threshold": abstention_threshold,
        "local_files_only": True,
        "network_access_allowed": False,
        "external_ai_service": False,
        "remote_code_allowed": False,
        "processor_use_fast": False,
    }

    with _private_umask(), offline_inference_environment():
        backend = backend_factory(
            model_version,
            expected_model_artifact_sha256,
            device,
        )
        provenance = backend.provenance
        _validate_backend_provenance(provenance, expected_provenance)
        prediction_metadata = produce_prediction_artifact(
            dataset,
            destination,
            backend,
            split="test",
            batch_size=batch_size,
            resume=resume,
        )

    if _input_binding(dataset) != input_binding:
        raise RuntimeError("SMHD package changed during offline inference")
    final_model_artifact = inspect_local_model_artifact(
        model,
        repository_root=repository,
        require_read_only=True,
    )
    if final_model_artifact != model_artifact:
        raise RuntimeError("Sealed local model changed during offline inference")

    attestation = _build_attestation(
        input_binding=input_binding,
        model_artifact=model_artifact,
        prediction_metadata=prediction_metadata,
        expected_provenance=expected_provenance,
    )
    attestation_path = destination / ATTESTATION_FILENAME
    if attestation_path.exists():
        existing = _load_json_object(attestation_path, "research execution attestation")
        _validate_existing_attestation(existing, attestation)
        attestation = existing
    else:
        _write_json_private_atomic(attestation_path, attestation)
    _seal_private_output(destination)
    return {
        "schema_version": 1,
        "status": "complete",
        "output_dir": str(destination),
        "prediction_metadata": prediction_metadata,
        "attestation": attestation,
    }


def _validate_runtime_settings(
    *,
    model_version: str,
    expected_model_artifact_sha256: str,
    device: str,
    batch_size: int,
    beam_width: int,
    max_length: int,
    confidence_threshold: float,
    abstention_threshold: float,
) -> None:
    if (
        not model_version
        or len(model_version) > 128
        or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
            for character in model_version
        )
    ):
        raise ValueError("model_version must contain only safe version characters")
    if not _is_sha256(expected_model_artifact_sha256):
        raise ValueError("expected_model_artifact_sha256 must be a lowercase SHA-256 digest")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    integer_settings = (
        ("batch_size", batch_size),
        ("beam_width", beam_width),
        ("max_length", max_length),
    )
    for label, value in integer_settings:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if not 0 <= abstention_threshold <= confidence_threshold <= 1:
        raise ValueError(
            "thresholds must satisfy 0 <= abstention_threshold <= confidence_threshold <= 1"
        )


def _validate_smhd_package(dataset: Path) -> dict[str, str]:
    source_meta = _load_json_object(dataset / "source.meta.json", "SMHD source metadata")
    manifest_meta = _load_json_object(dataset / "manifest.meta.json", "SMHD manifest metadata")
    if source_meta.get("schema_version") != 1:
        raise ValueError("SMHD source metadata schema_version must be 1")
    if source_meta.get("purpose") != _EXPECTED_PURPOSE:
        raise ValueError("SMHD source metadata purpose is not the research rehearsal")
    _require_mapping_values(
        _required_mapping(source_meta, "privacy", "source metadata"),
        {
            "contains_student_data": True,
            "repository_artifacts_allowed": False,
            "offline_only": True,
        },
        "source metadata privacy",
    )
    _require_mapping_values(
        _required_mapping(source_meta, "reference_quality", "source metadata"),
        {"status": "publisher_transcription_unreviewed", "gold_ready": False},
        "source metadata reference quality",
    )
    _require_mapping_values(
        _required_mapping(source_meta, "evidence_boundary", "source metadata"),
        {
            "research_rehearsal_only": True,
            "production_pilot_evidence": False,
            "consented_protected_pilot_substitute": False,
        },
        "source metadata evidence boundary",
    )
    source_dataset = _required_mapping(source_meta, "dataset", "source metadata")
    _require_mapping_values(
        source_dataset,
        {
            "article_id": 24419986,
            "doi": "10.25439/rmt.24419986.v1",
            "license_id": "CC-BY-NC-4.0",
            "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "commercial_use_allowed": False,
        },
        "source metadata dataset",
    )
    if manifest_meta.get("schema_version") != 2:
        raise ValueError("SMHD manifest metadata schema_version must be 2")
    _require_mapping_values(
        _required_mapping(manifest_meta, "dataset", "manifest metadata"),
        _EXPECTED_DATASET,
        "manifest metadata dataset",
    )
    manifest_details = _required_mapping(manifest_meta, "manifest", "manifest metadata")
    if manifest_details.get("num_samples") != 36:
        raise ValueError("SMHD research manifest must contain 36 samples")
    split = _required_mapping(manifest_meta, "split", "manifest metadata")
    if split.get("writer_isolation_verified") is not True:
        raise ValueError("SMHD manifest must assert writer-isolated splits")
    sample_counts = _required_mapping(split, "sample_counts", "manifest metadata split")
    writer_counts = _required_mapping(split, "writer_counts", "manifest metadata split")
    if sample_counts.get("test") != 6 or writer_counts.get("test") != 2:
        raise ValueError("SMHD research test split must contain 6 samples from 2 writers")

    binding = _input_binding(dataset)
    if manifest_details.get("sha256") != binding["manifest_sha256"]:
        raise ValueError("SMHD manifest SHA-256 does not match its metadata")
    source_details = _required_mapping(manifest_meta, "source", "manifest metadata")
    source_output = _required_mapping(source_meta, "output", "source metadata")
    if (
        source_details.get("labels_sha256") != binding["labels_sha256"]
        or source_output.get("labels_sha256") != binding["labels_sha256"]
    ):
        raise ValueError("SMHD labels SHA-256 does not match its provenance")
    return binding


def _input_binding(dataset: Path) -> dict[str, str]:
    return {
        "source_metadata_sha256": sha256_file(dataset / "source.meta.json"),
        "labels_sha256": sha256_file(dataset / "labels.csv"),
        "manifest_sha256": sha256_file(dataset / "manifest.jsonl"),
        "manifest_metadata_sha256": sha256_file(dataset / "manifest.meta.json"),
    }


def _validate_backend_provenance(
    provenance: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    for key, expected_value in expected.items():
        if provenance.get(key) != expected_value:
            raise ValueError(
                f"Research backend provenance {key} does not match the runtime contract"
            )


def _build_attestation(
    *,
    input_binding: Mapping[str, str],
    model_artifact: Mapping[str, Any],
    prediction_metadata: Mapping[str, Any],
    expected_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    output = _required_mapping(prediction_metadata, "output", "prediction metadata")
    input_details = _required_mapping(prediction_metadata, "input", "prediction metadata")
    return {
        "schema_version": 1,
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "noncommercial_research_rehearsal_only",
        "dataset": {
            **_EXPECTED_DATASET,
            "split": "test",
            "num_records": input_details["num_records"],
            "test_writers": 2,
            **dict(input_binding),
        },
        "model": {
            **dict(expected_provenance),
            "num_files": model_artifact["num_files"],
            "total_bytes": model_artifact["total_bytes"],
            "read_only": model_artifact["read_only"],
        },
        "execution": {
            "network_access_allowed": False,
            "python_socket_guard_enabled": True,
            "external_ai_service": False,
            "github_actions_allowed": False,
            "local_files_only": True,
        },
        "output": {
            "predictions_filename": output["filename"],
            "predictions_sha256": output["sha256"],
            "num_records": output["num_records"],
            "sample_level_artifacts_committed": False,
        },
        "claims": {
            "gold_benchmark": False,
            "production_pilot_evidence": False,
            "commercial_use_evidence": False,
            "protected_pilot_substitute": False,
        },
    }


def _validate_existing_attestation(
    existing: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    bound_fields = (
        "schema_version",
        "status",
        "evidence_scope",
        "dataset",
        "model",
        "execution",
        "output",
        "claims",
    )
    for key in bound_fields:
        if existing.get(key) != expected.get(key):
            raise ValueError("Existing research attestation does not match this inference run")


def _prepare_output_directory(output_dir: str | Path, *, dataset: Path, repository: Path) -> Path:
    raw = Path(output_dir)
    if raw.is_symlink():
        raise ValueError("Research inference output cannot be a symbolic link")
    resolved = raw.resolve(strict=False)
    _require_outside_repository(resolved, repository, "Research inference output")
    if _is_within(resolved, dataset):
        raise ValueError("Research inference output must not modify the source dataset package")
    parent = resolved.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("Research inference output parent must be an existing regular directory")
    _require_owner_only_path(parent, "Research inference output parent")
    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError("Research inference output must be a directory")
        entries = list(resolved.iterdir())
        if any(path.name not in _ALLOWED_OUTPUT_FILES for path in entries):
            raise ValueError("Research inference output contains an unexpected file")
        _require_owner_only_tree(resolved, "Research inference output")
    else:
        resolved.mkdir(mode=0o700)
    if os.name == "posix":
        resolved.chmod(0o700)
    return resolved


def _validated_external_directory(value: str | Path, repository: Path, label: str) -> Path:
    raw = Path(value)
    if raw.is_symlink():
        raise ValueError(f"{label} cannot be a symbolic link")
    resolved = raw.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a directory")
    _require_outside_repository(resolved, repository, label)
    return resolved


def _require_outside_repository(path: Path, repository: Path, label: str) -> None:
    if _is_within(path, repository):
        raise ValueError(f"{label} must remain outside the repository")


def _require_owner_only_tree(root: Path, label: str) -> None:
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"{label} must not contain symbolic links")
        if not path.is_dir() and not path.is_file():
            raise ValueError(f"{label} must contain only regular files and directories")
        _require_owner_only_path(path, label)


def _require_owner_only_path(path: Path, label: str) -> None:
    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError(f"{label} must not grant group or world permissions: {path}")


def _seal_private_output(destination: Path) -> None:
    for path in destination.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("Research inference output must contain only regular files")
        if os.name == "posix":
            path.chmod(0o600)
    if os.name == "posix":
        destination.chmod(0o700)


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or invalid {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {description}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _write_json_private_atomic(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "posix":
            temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _required_mapping(value: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"{context}.{key} must be an object")
    return result


def _require_mapping_values(
    value: Mapping[str, Any], expected: Mapping[str, Any], context: str
) -> None:
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"{context}.{key} must be {expected_value!r}")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


@contextmanager
def _private_umask() -> Iterator[None]:
    if os.name != "posix":
        yield
        return
    prior = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(prior)
