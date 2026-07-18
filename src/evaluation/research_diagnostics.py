"""Private failure analysis and fail-closed calibration diagnostics for SMHD research runs."""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from src.data.manifest import sha256_file
from src.evaluation.metrics import calibration_error, character_error_rate
from src.evaluation.protected_runtime import reject_ci_environment
from src.evaluation.runner import EvaluationRecord, load_evaluation_records

_EXPECTED_DATASET = {
    "name": "smhd-research-rehearsal",
    "version": "figshare-24419986-v1-selection-v1",
    "license_id": "CC-BY-NC-4.0",
    "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
    "sample_type": "line",
}
_EXPECTED_SPLITS = {
    "validation": {"samples": 9, "writers": 3},
    "test": {"samples": 6, "writers": 2},
}
_EXPECTED_MODEL_FLAGS = {
    "backend": "trocr_local_research",
    "evidence_scope": "noncommercial_research_rehearsal_only",
    "protected_pilot_evidence": False,
    "local_files_only": True,
    "network_access_allowed": False,
    "external_ai_service": False,
    "remote_code_allowed": False,
    "processor_use_fast": False,
}
_EXPECTED_EXECUTION_FLAGS = {
    "network_access_allowed": False,
    "python_socket_guard_enabled": True,
    "external_ai_service": False,
    "github_actions_allowed": False,
    "local_files_only": True,
}
_EXPECTED_FALSE_CLAIMS = {
    "gold_benchmark": False,
    "production_pilot_evidence": False,
    "commercial_use_evidence": False,
    "protected_pilot_substitute": False,
}
_REQUIRED_RUN_FILES = {
    "predictions.jsonl",
    "predictions.meta.json",
    "research_execution_attestation.json",
}
_ALLOWED_OUTPUT_FILES = {
    "failure_atlas.private.jsonl",
    "research_diagnostics.aggregate.json",
}
_DATASET_BINDING_KEYS = (
    "source_metadata_sha256",
    "labels_sha256",
    "manifest_sha256",
    "manifest_metadata_sha256",
)


@dataclass(frozen=True)
class ResearchRun:
    """A provenance-validated private research prediction run."""

    split: str
    root: Path
    records: tuple[EvaluationRecord, ...]
    flags: Mapping[str, Mapping[str, Any]]
    metadata: Mapping[str, Any]
    dataset_binding: Mapping[str, str]
    predictions_sha256: str


def build_offline_research_diagnostics(
    *,
    validation_run_dir: str | Path,
    test_run_dir: str | Path,
    output_dir: str | Path,
    repository_root: str | Path | None = None,
    minimum_calibration_samples: int = 30,
    minimum_policy_coverage: float = 0.5,
    maximum_selective_cer: float = 0.1,
    high_confidence_threshold: float = 0.9,
) -> dict[str, Any]:
    """Build a private failure atlas and aggregate-only calibration diagnostic."""
    reject_ci_environment("Offline research diagnostics")
    _validate_settings(
        minimum_calibration_samples=minimum_calibration_samples,
        minimum_policy_coverage=minimum_policy_coverage,
        maximum_selective_cer=maximum_selective_cer,
        high_confidence_threshold=high_confidence_threshold,
    )
    repository = Path(repository_root or Path.cwd()).resolve(strict=True)
    validation = _load_run(validation_run_dir, "validation", repository)
    test = _load_run(test_run_dir, "test", repository)
    _validate_run_pair(validation, test)
    destination = _prepare_output_directory(
        output_dir,
        repository,
        input_roots=(validation.root, test.root),
    )

    atlas_records = _build_failure_records(
        (*validation.records, *test.records),
        flags={**validation.flags, **test.flags},
        high_confidence_threshold=high_confidence_threshold,
    )
    atlas_path = destination / "failure_atlas.private.jsonl"
    with _private_umask():
        _write_jsonl_private_atomic(atlas_path, atlas_records)

    validation_diagnostic = _calibration_diagnostic(validation.records)
    test_diagnostic = _calibration_diagnostic(test.records)
    policy = _select_validation_policy(
        validation.records,
        minimum_coverage=minimum_policy_coverage,
        maximum_selective_cer=maximum_selective_cer,
    )
    policy_application = _apply_policy_to_holdout(test.records, policy.get("threshold"))
    blockers = _calibration_blockers(
        validation_diagnostic,
        policy,
        minimum_calibration_samples=minimum_calibration_samples,
    )
    aggregate = {
        "schema_version": 1,
        "status": "diagnosed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "noncommercial_research_rehearsal_only",
        "inputs": {
            "validation": {
                "predictions_sha256": validation.predictions_sha256,
                "num_samples": len(validation.records),
                "num_writers": _EXPECTED_SPLITS["validation"]["writers"],
            },
            "test": {
                "predictions_sha256": test.predictions_sha256,
                "num_samples": len(test.records),
                "num_writers": _EXPECTED_SPLITS["test"]["writers"],
            },
            "model_version": _model_value(validation, "model_version"),
            "model_artifact_sha256": _model_value(validation, "model_artifact_sha256"),
        },
        "failure_atlas": {
            "private_filename": atlas_path.name,
            "private_sha256": sha256_file(atlas_path),
            "num_failure_records": len(atlas_records),
            "categories": _aggregate_categories(atlas_records),
            "sample_level_artifacts_committed": False,
        },
        "calibration": {
            "target": "exact_transcription_match",
            "minimum_fit_samples": minimum_calibration_samples,
            "validation": validation_diagnostic,
            "untouched_test_holdout": test_diagnostic,
            "fit": {
                "status": "blocked" if blockers else "eligible_not_fitted",
                "method_reserved": "temperature_scaling",
                "blockers": blockers,
                "parameters": None,
            },
            "validation_policy_search": policy,
            "test_policy_application": policy_application,
        },
        "operational_decision": {
            "status": "human_review_required",
            "automatic_acceptance_allowed": False,
            "release_gate_passed": False,
            "reasons": sorted(
                set(
                    blockers
                    + [
                        "publisher_references_are_not_independently_reviewed",
                        "noncommercial_research_evidence_only",
                    ]
                )
            ),
        },
        "claims": {
            **_EXPECTED_FALSE_CLAIMS,
            "calibrated_confidence": False,
            "automatic_acceptance_supported": False,
            "primary_school_accuracy": False,
        },
    }
    aggregate_path = destination / "research_diagnostics.aggregate.json"
    with _private_umask():
        _write_json_private_atomic(aggregate_path, aggregate)
    _seal_private_output(destination)
    return {
        "schema_version": 1,
        "status": "complete",
        "output_dir": str(destination),
        "aggregate": aggregate,
    }


def _validate_settings(
    *,
    minimum_calibration_samples: int,
    minimum_policy_coverage: float,
    maximum_selective_cer: float,
    high_confidence_threshold: float,
) -> None:
    if (
        isinstance(minimum_calibration_samples, bool)
        or not isinstance(minimum_calibration_samples, int)
        or minimum_calibration_samples < 2
    ):
        raise ValueError("minimum_calibration_samples must be an integer of at least 2")
    for name, value in (
        ("minimum_policy_coverage", minimum_policy_coverage),
        ("maximum_selective_cer", maximum_selective_cer),
        ("high_confidence_threshold", high_confidence_threshold),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")


def _load_run(value: str | Path, expected_split: str, repository: Path) -> ResearchRun:
    root = _validated_external_directory(value, repository, f"{expected_split} run")
    _require_owner_only_tree(root, f"{expected_split} run")
    filenames = {path.name for path in root.iterdir()}
    if filenames != _REQUIRED_RUN_FILES:
        raise ValueError(
            f"{expected_split} run must contain exactly the completed research artifacts"
        )
    predictions_path = root / "predictions.jsonl"
    metadata = _load_json_object(root / "predictions.meta.json", "prediction metadata")
    attestation = _load_json_object(
        root / "research_execution_attestation.json", "research attestation"
    )
    predictions_sha256 = sha256_file(predictions_path)
    _validate_run_provenance(
        metadata,
        attestation,
        expected_split=expected_split,
        predictions_sha256=predictions_sha256,
    )
    records = tuple(load_evaluation_records(predictions_path))
    expected_count = _EXPECTED_SPLITS[expected_split]["samples"]
    if len(records) != expected_count:
        raise ValueError(f"{expected_split} predictions must contain {expected_count} records")
    flags = _load_prediction_flags(
        predictions_path,
        expected_split=expected_split,
        expected_model_version=str(_required_mapping(metadata, "model")["model_version"]),
    )
    if set(flags) != {record.sample_id for record in records}:
        raise ValueError(f"{expected_split} prediction envelope does not match its records")
    attested_dataset = _required_mapping(attestation, "dataset")
    return ResearchRun(
        split=expected_split,
        root=root,
        records=records,
        flags=flags,
        metadata=metadata,
        dataset_binding={key: str(attested_dataset[key]) for key in _DATASET_BINDING_KEYS},
        predictions_sha256=predictions_sha256,
    )


def _validate_run_provenance(
    metadata: Mapping[str, Any],
    attestation: Mapping[str, Any],
    *,
    expected_split: str,
    predictions_sha256: str,
) -> None:
    expected = _EXPECTED_SPLITS[expected_split]
    if metadata.get("schema_version") != 1 or metadata.get("status") != "complete":
        raise ValueError(f"{expected_split} prediction metadata is not complete schema v1")
    input_details = _required_mapping(metadata, "input")
    _require_mapping_values(
        _required_mapping(input_details, "dataset"), _EXPECTED_DATASET, "prediction dataset"
    )
    if (
        input_details.get("split") != expected_split
        or input_details.get("num_records") != expected["samples"]
    ):
        raise ValueError(f"{expected_split} prediction metadata has the wrong split identity")
    model = _required_mapping(metadata, "model")
    _require_mapping_values(model, _EXPECTED_MODEL_FLAGS, "prediction model")
    if not _is_sha256(model.get("model_artifact_sha256")):
        raise ValueError("prediction model artifact SHA-256 is invalid")
    output = _required_mapping(metadata, "output")
    if (
        output.get("filename") != "predictions.jsonl"
        or output.get("sha256") != predictions_sha256
        or output.get("num_records") != expected["samples"]
    ):
        raise ValueError(f"{expected_split} prediction output identity does not match")
    _require_mapping_values(
        _required_mapping(metadata, "privacy"),
        {"writer_ids_exported": False, "source_images_exported": False},
        "prediction privacy",
    )

    if (
        attestation.get("schema_version") != 1
        or attestation.get("status") != "complete"
        or attestation.get("evidence_scope") != "noncommercial_research_rehearsal_only"
    ):
        raise ValueError(f"{expected_split} research attestation is invalid")
    dataset = _required_mapping(attestation, "dataset")
    _require_mapping_values(dataset, _EXPECTED_DATASET, "attested dataset")
    if (
        dataset.get("split") != expected_split
        or dataset.get("num_records") != expected["samples"]
        or dataset.get("num_writers") != expected["writers"]
    ):
        raise ValueError(f"{expected_split} attestation has the wrong split identity")
    for key in _DATASET_BINDING_KEYS:
        if not _is_sha256(dataset.get(key)):
            raise ValueError(f"{expected_split} attested dataset {key} is invalid")
    if dataset.get("manifest_sha256") != input_details.get("manifest_sha256") or dataset.get(
        "manifest_metadata_sha256"
    ) != input_details.get("manifest_metadata_sha256"):
        raise ValueError(f"{expected_split} attested dataset does not match prediction input")
    attested_model = _required_mapping(attestation, "model")
    _require_mapping_values(attested_model, _EXPECTED_MODEL_FLAGS, "attested model")
    for key in ("model_version", "model_artifact_sha256"):
        if attested_model.get(key) != model.get(key):
            raise ValueError(f"attested model {key} does not match prediction metadata")
    _require_mapping_values(
        _required_mapping(attestation, "execution"),
        _EXPECTED_EXECUTION_FLAGS,
        "attested execution",
    )
    attested_output = _required_mapping(attestation, "output")
    if (
        attested_output.get("predictions_filename") != "predictions.jsonl"
        or attested_output.get("predictions_sha256") != predictions_sha256
        or attested_output.get("num_records") != expected["samples"]
        or attested_output.get("sample_level_artifacts_committed") is not False
    ):
        raise ValueError(f"{expected_split} attested output identity does not match")
    _require_mapping_values(
        _required_mapping(attestation, "claims"),
        _EXPECTED_FALSE_CLAIMS,
        "attested claims",
    )


def _load_prediction_flags(
    path: Path, *, expected_split: str, expected_model_version: str
) -> dict[str, Mapping[str, Any]]:
    flags: dict[str, Mapping[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"prediction line {line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"prediction line {line_number}: record must be an object")
            sample_id = value.get("sample_id")
            if not isinstance(sample_id, str) or sample_id in flags:
                raise ValueError("prediction envelope has a missing or duplicate sample_id")
            if (
                value.get("schema_version") != 1
                or value.get("split") != expected_split
                or value.get("model_version") != expected_model_version
                or not isinstance(value.get("is_uncertain"), bool)
                or not isinstance(value.get("is_unreadable"), bool)
            ):
                raise ValueError(f"prediction line {line_number}: envelope identity is invalid")
            flags[sample_id] = {
                "is_uncertain": value["is_uncertain"],
                "is_unreadable": value["is_unreadable"],
                "split": expected_split,
            }
    return flags


def _validate_run_pair(validation: ResearchRun, test: ResearchRun) -> None:
    validation_ids = {record.sample_id for record in validation.records}
    test_ids = {record.sample_id for record in test.records}
    validation_sources = {record.source_sha256 for record in validation.records}
    test_sources = {record.source_sha256 for record in test.records}
    if validation_ids & test_ids or validation_sources & test_sources:
        raise ValueError("Validation and test research runs are not sample-isolated")
    if validation.dataset_binding != test.dataset_binding:
        raise ValueError("Validation and test research runs use different dataset bindings")
    for key in (
        "model_version",
        "model_artifact_sha256",
        "batch_size",
        "beam_width",
        "max_length",
        "confidence_threshold",
        "abstention_threshold",
        "device",
        "dtype",
    ):
        if _model_value(validation, key) != _model_value(test, key):
            raise ValueError(f"Validation and test model provenance differ for {key}")


def _build_failure_records(
    records: tuple[EvaluationRecord, ...],
    *,
    flags: Mapping[str, Mapping[str, Any]],
    high_confidence_threshold: float,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for record in records:
        exact_match = record.prediction.strip() == record.reference.strip()
        if exact_match:
            continue
        confidence = _required_confidence(record)
        line_cer = character_error_rate([record.prediction], [record.reference])
        categories = [_severity_category(line_cer)]
        if "publisher_hash_marker" in record.slices:
            categories.append("publisher_correction_marker")
        if confidence >= high_confidence_threshold:
            categories.append("high_confidence_exact_error")
        record_flags = flags[record.sample_id]
        if record_flags["is_uncertain"] is True:
            categories.append("model_marked_uncertain")
        if record_flags["is_unreadable"] is True:
            categories.append("model_marked_unreadable")
        failures.append(
            {
                "schema_version": 1,
                "sample_id": record.sample_id,
                "source_sha256": record.source_sha256,
                "split": record_flags["split"],
                "reference": record.reference,
                "prediction": record.prediction,
                "confidence": confidence,
                "line_cer": line_cer,
                "exact_match": False,
                "categories": sorted(categories),
                "slices": list(record.slices),
                "is_uncertain": record_flags["is_uncertain"],
                "is_unreadable": record_flags["is_unreadable"],
                "human_review_required": True,
            }
        )
    return sorted(failures, key=lambda item: (str(item["split"]), str(item["sample_id"])))


def _severity_category(line_cer: float) -> str:
    if line_cer >= 0.25:
        return "character_error_ge_0_25"
    if line_cer >= 0.1:
        return "character_error_0_10_to_0_25"
    return "character_error_below_0_10"


def _aggregate_categories(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for category in record["categories"]:
            grouped[str(category)].append(record)
    return {
        category: {
            "count": len(items),
            "validation_count": sum(item["split"] == "validation" for item in items),
            "test_count": sum(item["split"] == "test" for item in items),
            "mean_cer": fmean(float(item["line_cer"]) for item in items),
            "mean_confidence": fmean(float(item["confidence"]) for item in items),
        }
        for category, items in sorted(grouped.items())
    }


def _calibration_diagnostic(records: tuple[EvaluationRecord, ...]) -> dict[str, Any]:
    confidences = [_required_confidence(record) for record in records]
    exact = [record.prediction.strip() == record.reference.strip() for record in records]
    line_cers = [
        character_error_rate([record.prediction], [record.reference]) for record in records
    ]
    character_accuracies = [max(0.0, 1.0 - line_cer) for line_cer in line_cers]
    exact_targets = [float(value) for value in exact]
    return {
        "num_samples": len(records),
        "exact_matches": sum(exact),
        "exact_errors": len(records) - sum(exact),
        "exact_match_rate": fmean(exact_targets),
        "mean_confidence": fmean(confidences),
        "exact_match_ece": calibration_error(confidences, exact),
        "exact_match_brier_score": fmean(
            (confidence - target) ** 2 for confidence, target in zip(confidences, exact_targets)
        ),
        "mean_line_cer": fmean(line_cers),
        "mean_character_accuracy": fmean(character_accuracies),
        "confidence_character_accuracy_mae": fmean(
            abs(confidence - accuracy)
            for confidence, accuracy in zip(confidences, character_accuracies)
        ),
        "target_has_both_classes": 0 < sum(exact) < len(exact),
    }


def _select_validation_policy(
    records: tuple[EvaluationRecord, ...],
    *,
    minimum_coverage: float,
    maximum_selective_cer: float,
) -> dict[str, Any]:
    minimum_accepted = math.ceil(len(records) * minimum_coverage)
    eligible: list[dict[str, Any]] = []
    for step in range(21):
        threshold = step / 20
        accepted = [record for record in records if _required_confidence(record) >= threshold]
        if not accepted:
            continue
        selective_cer = character_error_rate(
            [record.prediction for record in accepted],
            [record.reference for record in accepted],
        )
        exact_error_rate = fmean(
            record.prediction.strip() != record.reference.strip() for record in accepted
        )
        candidate = {
            "threshold": threshold,
            "accepted_samples": len(accepted),
            "coverage": len(accepted) / len(records),
            "selective_cer": selective_cer,
            "exact_error_rate": exact_error_rate,
        }
        if len(accepted) >= minimum_accepted and selective_cer <= maximum_selective_cer:
            eligible.append(candidate)
    if not eligible:
        return {
            "status": "no_eligible_threshold",
            "threshold": None,
            "minimum_coverage": minimum_coverage,
            "minimum_accepted_samples": minimum_accepted,
            "maximum_selective_cer": maximum_selective_cer,
            "searched_on": "validation_only",
        }
    selected = min(
        eligible,
        key=lambda item: (
            -float(item["coverage"]),
            float(item["selective_cer"]),
            item["threshold"],
        ),
    )
    return {
        "status": "diagnostic_candidate_found",
        **selected,
        "minimum_coverage": minimum_coverage,
        "minimum_accepted_samples": minimum_accepted,
        "maximum_selective_cer": maximum_selective_cer,
        "searched_on": "validation_only",
    }


def _apply_policy_to_holdout(
    records: tuple[EvaluationRecord, ...], threshold: Any
) -> dict[str, Any]:
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        return {
            "status": "not_applied",
            "threshold": None,
            "reason": "validation_produced_no_eligible_threshold",
        }
    accepted = [record for record in records if _required_confidence(record) >= threshold]
    if not accepted:
        return {
            "status": "applied",
            "threshold": float(threshold),
            "accepted_samples": 0,
            "coverage": 0.0,
            "selective_cer": None,
            "exact_error_rate": None,
        }
    return {
        "status": "applied",
        "threshold": float(threshold),
        "accepted_samples": len(accepted),
        "coverage": len(accepted) / len(records),
        "selective_cer": character_error_rate(
            [record.prediction for record in accepted],
            [record.reference for record in accepted],
        ),
        "exact_error_rate": fmean(
            record.prediction.strip() != record.reference.strip() for record in accepted
        ),
    }


def _calibration_blockers(
    validation_diagnostic: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    minimum_calibration_samples: int,
) -> list[str]:
    blockers = []
    if validation_diagnostic["num_samples"] < minimum_calibration_samples:
        blockers.append("validation_sample_count_below_calibration_minimum")
    if validation_diagnostic["target_has_both_classes"] is not True:
        blockers.append("exact_match_target_has_only_one_class")
    if policy["status"] != "diagnostic_candidate_found":
        blockers.append("no_validation_threshold_met_selective_risk_policy")
    return blockers


def _model_value(run: ResearchRun, key: str) -> Any:
    return _required_mapping(run.metadata, "model").get(key)


def _required_confidence(record: EvaluationRecord) -> float:
    if record.confidence is None:
        raise ValueError("Research diagnostics require confidence on every prediction")
    return record.confidence


def _prepare_output_directory(
    output_dir: str | Path,
    repository: Path,
    *,
    input_roots: tuple[Path, ...],
) -> Path:
    raw = Path(output_dir)
    if raw.is_symlink():
        raise ValueError("Research diagnostics output cannot be a symbolic link")
    resolved = raw.resolve(strict=False)
    _require_outside_repository(resolved, repository, "Research diagnostics output")
    if any(
        _is_within(resolved, input_root) or _is_within(input_root, resolved)
        for input_root in input_roots
    ):
        raise ValueError("Research diagnostics output must be separate from prediction runs")
    if resolved.parent.is_symlink() or not resolved.parent.is_dir():
        raise ValueError("Research diagnostics output parent must be an existing directory")
    _require_owner_only_path(resolved.parent, "Research diagnostics output parent")
    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError("Research diagnostics output must be a directory")
        if any(path.name not in _ALLOWED_OUTPUT_FILES for path in resolved.iterdir()):
            raise ValueError("Research diagnostics output contains an unexpected file")
        _require_owner_only_tree(resolved, "Research diagnostics output")
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
            raise ValueError("Research diagnostics output must contain only regular files")
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


def _write_jsonl_private_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                json.dump(record, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "posix":
            temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"{key} must be an object")
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
