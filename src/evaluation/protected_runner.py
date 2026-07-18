"""Aggregate-only evaluation for protected handwriting on an approved self-hosted machine."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.protected_manifest import (
    METADATA_FILENAME,
    verify_frozen_protected_evaluation,
)
from src.evaluation.runner import (
    EvaluationRecord,
    evaluate_records,
    evaluate_release_gate,
    evaluation_set_sha256,
    load_evaluation_artifact,
    load_evaluation_config,
    sha256_file,
)

RESULTS_DIRECTORY = "protected_evaluation_results"
ATTESTATION_FILENAME = "execution_attestation.json"
PREDICTIONS_FILENAME = "predictions.jsonl"

_PREDICTION_REQUIRED_FIELDS = {"sample_id", "source_sha256", "prediction"}
_PREDICTION_OPTIONAL_FIELDS = {"confidence", "latency_ms", "cost_per_page_usd"}
_ATTESTATION_FIELDS = {
    "schema_version",
    "execution_environment",
    "executed_at",
    "operator_role",
    "model_version",
    "model_artifact_sha256",
    "evaluation_set_id",
    "manifest_sha256",
    "predictions_sha256",
    "protected_storage_mounted",
    "network_access_during_inference",
    "external_ai_services_used",
    "github_actions_used",
    "model_artifacts_preloaded",
    "predictions_contain_references",
    "automated_decisions_enabled",
}
_CI_ENVIRONMENT_MARKERS = (
    "BUILDKITE",
    "CIRCLECI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "JENKINS_URL",
    "TF_BUILD",
)


def run_protected_evaluation(
    *,
    contract_path: str | Path,
    dataset_dir: str | Path,
    manifest_dir: str | Path,
    predictions_path: str | Path,
    attestation_path: str | Path,
    model_version: str,
    config_path: str | Path,
    output_dir: str | Path,
    baseline_results: str | Path | None = None,
    enforce_release_gate: bool = False,
    repository_root: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Join reference-free predictions in memory and persist only aggregate metrics."""
    _reject_ci_environment()
    safe_model_version = _safe_component(model_version, "model_version")
    metadata, manifest_records = verify_frozen_protected_evaluation(
        contract_path,
        dataset_dir,
        manifest_dir,
        repository_root=repository_root,
    )
    root = Path(dataset_dir).resolve(strict=True)
    predictions = _required_protected_file(
        predictions_path, root, PREDICTIONS_FILENAME, "Predictions"
    )
    attestation = _required_protected_file(
        attestation_path, root, ATTESTATION_FILENAME, "Execution attestation"
    )
    predictions_content = predictions.read_bytes()
    predictions_sha256 = hashlib.sha256(predictions_content).hexdigest()
    attestation_content = attestation.read_bytes()
    manifest_sha256 = str(_required_mapping(metadata, "manifest", "metadata")["sha256"])
    evaluation_set_id = _required_string(metadata, "evaluation_set_id", "metadata")
    attestation_value = _validate_attestation(
        model_version=model_version,
        evaluation_set_id=evaluation_set_id,
        manifest_sha256=manifest_sha256,
        predictions_sha256=predictions_sha256,
        content=attestation_content,
    )

    test_records = {
        str(record["sample_id"]): record
        for record in manifest_records
        if record.get("split") == "test"
    }
    records = _load_reference_free_predictions(predictions_content, test_records)
    evaluation_config = load_evaluation_config(config_path)
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "status": "protected_shadow_evaluated",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "test_set": evaluation_set_id,
        "data_classification": "restricted_student_handwriting_aggregate",
        "input": {
            "filename": predictions.name,
            "sha256": predictions_sha256,
            "evaluation_set_sha256": evaluation_set_sha256(records),
            "num_records": len(records),
        },
        "config": {
            "filename": Path(config_path).name,
            "sha256": sha256_file(config_path),
        },
        **evaluate_records(records),
    }
    artifact["protected_evaluation"] = {
        "execution_mode": "approved_self_hosted_offline",
        "evaluation_set_id": evaluation_set_id,
        "manifest_sha256": manifest_sha256,
        "manifest_metadata_sha256": sha256_file(Path(manifest_dir) / METADATA_FILENAME),
        "execution_attestation_sha256": hashlib.sha256(attestation_content).hexdigest(),
        "model_artifact_sha256": attestation_value["model_artifact_sha256"],
        "aggregate_only": True,
        "sample_level_output_persisted": False,
        "references_exported": False,
        "publication_allowed": False,
        "github_actions_allowed": False,
        "gold_ready": False,
    }

    if enforce_release_gate and baseline_results is None:
        raise ValueError("enforce_release_gate requires baseline_results")
    if baseline_results is not None:
        baseline_path = _required_contained_file(
            baseline_results, root, "Baseline evaluation result"
        )
        baseline = load_evaluation_artifact(baseline_path)
        _validate_protected_baseline(baseline, evaluation_set_id, manifest_sha256)
        thresholds = evaluation_config.get("release_gate")
        if not isinstance(thresholds, dict):
            raise ValueError("Evaluation config must contain a 'release_gate' mapping")
        artifact["baseline"] = {
            "filename": baseline_path.name,
            "model_version": baseline.get("model_version"),
        }
        artifact["release_gate"] = evaluate_release_gate(artifact, baseline, thresholds)

    destination = _required_results_directory(output_dir, root)
    destination.mkdir(mode=0o700, exist_ok=True)
    result_path = destination / f"eval_{safe_model_version}_{evaluation_set_id}.json"
    if result_path.exists() or result_path.is_symlink():
        raise FileExistsError("Protected evaluation result already exists and cannot be replaced")
    serialized = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    _assert_aggregate_only(serialized, test_records)
    _write_new_read_only(result_path, serialized)
    return artifact, result_path


def _load_reference_free_predictions(
    content: bytes, expected: Mapping[str, Mapping[str, Any]]
) -> list[EvaluationRecord]:
    records_by_id: dict[str, EvaluationRecord] = {}
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Predictions must be UTF-8 JSONL") from exc
    for line_number, raw_line in enumerate(decoded.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"predictions line {line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"predictions line {line_number}: value must be an object")
        fields = set(value)
        if not _PREDICTION_REQUIRED_FIELDS.issubset(fields) or not fields.issubset(
            _PREDICTION_REQUIRED_FIELDS | _PREDICTION_OPTIONAL_FIELDS
        ):
            raise ValueError(
                f"predictions line {line_number}: fields violate the reference-free allowlist"
            )
        sample_id = value.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in expected:
            raise ValueError(
                f"predictions line {line_number}: sample_id is not in the frozen test split"
            )
        if sample_id in records_by_id:
            raise ValueError(f"predictions line {line_number}: duplicate sample_id")
        gold = expected[sample_id]
        if value.get("source_sha256") != gold["source_sha256"]:
            raise ValueError(
                f"predictions line {line_number}: source_sha256 does not match the manifest"
            )
        joined = {
            **value,
            "reference": gold["reference"],
            "slices": gold["slices"],
        }
        records_by_id[sample_id] = EvaluationRecord.from_mapping(joined, line_number)
    missing = sorted(set(expected) - set(records_by_id))
    if missing:
        raise ValueError(
            f"Predictions are incomplete for the frozen test split: {len(missing)} missing"
        )
    for field_name in _PREDICTION_OPTIONAL_FIELDS:
        populated = [getattr(record, field_name) is not None for record in records_by_id.values()]
        if any(populated) and not all(populated):
            raise ValueError(
                f"Optional prediction field '{field_name}' must be present for every record or none"
            )
    return [records_by_id[sample_id] for sample_id in sorted(records_by_id)]


def _validate_attestation(
    *,
    model_version: str,
    evaluation_set_id: str,
    manifest_sha256: str,
    predictions_sha256: str,
    content: bytes,
) -> dict[str, Any]:
    value = _load_object_bytes(content, "Execution attestation")
    if set(value) != _ATTESTATION_FIELDS:
        raise ValueError("Execution attestation fields must exactly match the allowlist")
    if value.get("schema_version") != 1:
        raise ValueError("Execution attestation schema_version must be 1")
    if value.get("execution_environment") != "approved_self_hosted":
        raise ValueError("Execution attestation must declare approved_self_hosted")
    if value.get("model_version") != model_version:
        raise ValueError("Execution attestation model_version does not match")
    if value.get("evaluation_set_id") != evaluation_set_id:
        raise ValueError("Execution attestation evaluation_set_id does not match")
    if value.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Execution attestation manifest_sha256 does not match")
    if value.get("predictions_sha256") != predictions_sha256:
        raise ValueError("Execution attestation predictions_sha256 does not match")
    _required_sha256(value, "model_artifact_sha256", "attestation")
    _verified_timestamp(value, "executed_at")
    operator_role = _required_string(value, "operator_role", "attestation")
    if (
        re.fullmatch(r"[a-z][a-z0-9_]{7,63}", operator_role) is None
        or operator_role in {"pending", "replace_me"}
    ):
        raise ValueError("Execution attestation operator_role must be an approved role")
    expected_booleans = {
        "protected_storage_mounted": True,
        "network_access_during_inference": False,
        "external_ai_services_used": False,
        "github_actions_used": False,
        "model_artifacts_preloaded": True,
        "predictions_contain_references": False,
        "automated_decisions_enabled": False,
    }
    for key, expected in expected_booleans.items():
        if not isinstance(value.get(key), bool) or value[key] is not expected:
            raise ValueError(f"Execution attestation {key} must be {expected}")
    return value


def _validate_protected_baseline(
    baseline: Mapping[str, Any], evaluation_set_id: str, manifest_sha256: str
) -> None:
    protected = _required_mapping(baseline, "protected_evaluation", "baseline")
    required = {
        "execution_mode": "approved_self_hosted_offline",
        "evaluation_set_id": evaluation_set_id,
        "manifest_sha256": manifest_sha256,
        "aggregate_only": True,
        "sample_level_output_persisted": False,
        "references_exported": False,
        "publication_allowed": False,
        "github_actions_allowed": False,
        "gold_ready": False,
    }
    for key, expected in required.items():
        if protected.get(key) != expected:
            raise ValueError(f"Baseline protected_evaluation.{key} does not match")


def _reject_ci_environment() -> None:
    active = [name for name in _CI_ENVIRONMENT_MARKERS if _environment_truthy(name)]
    if _environment_truthy("CI"):
        active.append("CI")
    if active:
        raise RuntimeError(
            "Protected evaluation refuses CI environments; active markers=" + ",".join(active)
        )


def _environment_truthy(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip().casefold() not in {"", "0", "false", "no"}


def _required_protected_file(path: str | Path, root: Path, name: str, label: str) -> Path:
    value = _required_contained_file(path, root, label)
    if value.name != name:
        raise ValueError(f"{label} filename must be '{name}'")
    return value


def _required_contained_file(path: str | Path, root: Path, label: str) -> Path:
    value = Path(path)
    if value.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    try:
        resolved = value.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"{label} must be a file under the protected root") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} must be a file")
    return resolved


def _required_results_directory(path: str | Path, root: Path) -> Path:
    value = Path(path)
    if value.name != RESULTS_DIRECTORY or value.parent.resolve(strict=True) != root:
        raise ValueError(
            f"Protected results directory must be <dataset-dir>/{RESULTS_DIRECTORY}"
        )
    if value.is_symlink():
        raise ValueError("Protected results directory must not be a symbolic link")
    if value.exists() and not value.is_dir():
        raise ValueError("Protected results path must be a directory")
    return value


def _assert_aggregate_only(
    serialized: str, protected_records: Mapping[str, Mapping[str, Any]]
) -> None:
    decoded = json.loads(serialized)
    report_strings = _string_values(decoded)
    for sample_id, record in protected_records.items():
        sensitive_values = (sample_id, str(record["writer_id"]), str(record["reference"]))
        if any(value and value in report_strings for value in sensitive_values):
            raise RuntimeError("Aggregate report unexpectedly contains sample-level content")


def _string_values(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return {item for child in value.values() for item in _string_values(child)}
    if isinstance(value, list):
        return {item for child in value for item in _string_values(child)}
    return set()


def _write_new_read_only(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o400)


def _safe_component(value: str, label: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,127}", value) is None:
        raise ValueError(f"{label} must contain only safe version characters")
    return value


def _verified_timestamp(value: Mapping[str, Any], key: str) -> None:
    raw = _required_string(value, key, "attestation")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Execution attestation {key} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed > datetime.now(timezone.utc):
        raise ValueError(f"Execution attestation {key} must be a non-future timestamp")


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


def _load_object_bytes(content: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value
