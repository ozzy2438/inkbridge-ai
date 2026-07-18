"""Cryptographically freeze a writer-isolated protected evaluation manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.pilot_intake import validate_protected_pilot
from src.data.writer_split import create_writer_split

MANIFEST_DIRECTORY = "frozen_evaluation"
MANIFEST_FILENAME = "protected_evaluation.manifest.jsonl"
METADATA_FILENAME = "protected_evaluation.manifest.meta.json"

_MANIFEST_FIELDS = {
    "schema_version",
    "sample_id",
    "source_sha256",
    "image_path",
    "reference",
    "writer_id",
    "slices",
    "split",
}
_AUDIT_COMPARISON_FIELDS = {
    "schema_version",
    "status",
    "gold_ready",
    "pilot_id",
    "data_classification",
    "policy_boundary",
    "data_use_scope",
    "input",
    "counts",
    "annotation_quality",
    "storage_boundary",
    "next_gate",
}
_METADATA_FIELDS = {
    "schema_version",
    "status",
    "gold_ready",
    "generated_at",
    "pilot_id",
    "evaluation_set_id",
    "data_classification",
    "policy_boundary",
    "source",
    "manifest",
    "split",
    "intake_audit_snapshot",
    "next_gate",
}


def freeze_protected_evaluation(
    contract_path: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Create a new read-only manifest after revalidating the protected pilot audit."""
    root, repository = _protected_root(dataset_dir, repository_root)
    contract_file = _required_root_file(contract_path, root, "Pilot contract")
    destination = _required_manifest_destination(output_dir, root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Frozen evaluation directory already exists and cannot be replaced")

    audit_path, persisted_audit, fresh_audit = _validated_intake_state(
        contract_file, root, repository
    )
    contract = _load_object(contract_file, "Pilot contract")
    records, split_config = _build_split_records(root, contract)
    manifest_content = _manifest_content(records)
    manifest_sha256 = _sha256_bytes(manifest_content.encode("utf-8"))
    evaluation_set_id = _evaluation_set_id(
        str(fresh_audit["pilot_id"]), str(split_config["evaluation_set_version"])
    )
    split_sample_counts, split_writer_counts = _split_counts(records)

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "frozen_protected_evaluation_ready",
        "gold_ready": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pilot_id": fresh_audit["pilot_id"],
        "evaluation_set_id": evaluation_set_id,
        "data_classification": "restricted_student_handwriting",
        "policy_boundary": {
            "public_repository_allowed": False,
            "github_actions_allowed": False,
            "third_party_ai_upload_allowed": False,
            "automated_education_decisions_allowed": False,
            "model_training_allowed": False,
        },
        "source": {
            "contract_filename": contract_file.name,
            "contract_sha256": _sha256_file(contract_file),
            "pilot_audit_filename": audit_path.name,
            "pilot_audit_sha256": _sha256_file(audit_path),
            "labels_sha256": fresh_audit["input"]["labels_sha256"],
            "annotation_review_sha256": fresh_audit["input"][
                "annotation_review_sha256"
            ],
            "storage_control_sha256": fresh_audit["storage_boundary"][
                "storage_control_sha256"
            ],
            "lifecycle_ledger_head_sha256": fresh_audit["storage_boundary"]["lifecycle"][
                "head_sha256"
            ],
        },
        "manifest": {
            "filename": MANIFEST_FILENAME,
            "sha256": manifest_sha256,
            "num_samples": len(records),
            "contains_references": True,
            "repository_export_allowed": False,
        },
        "split": {
            **split_config,
            "sample_counts": split_sample_counts,
            "writer_counts": split_writer_counts,
            "writer_isolation_verified": True,
            "training_split_present": False,
        },
        "intake_audit_snapshot": {
            key: persisted_audit[key] for key in sorted(_AUDIT_COMPARISON_FIELDS)
        },
        "next_gate": "run_self_hosted_shadow_evaluation_without_exporting_sample_content",
    }

    try:
        destination.mkdir(mode=0o700)
        manifest_path = destination / MANIFEST_FILENAME
        metadata_path = destination / METADATA_FILENAME
        _write_new_file(manifest_path, manifest_content)
        _write_new_file(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        manifest_path.chmod(0o400)
        metadata_path.chmod(0o400)
        destination.chmod(0o500)
    except Exception:
        _remove_incomplete_directory(destination)
        raise
    return metadata


def verify_frozen_protected_evaluation(
    contract_path: str | Path,
    dataset_dir: str | Path,
    manifest_dir: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Recompute every binding and reject drift in a frozen protected manifest."""
    root, repository = _protected_root(dataset_dir, repository_root)
    contract_file = _required_root_file(contract_path, root, "Pilot contract")
    destination = _required_existing_manifest_directory(manifest_dir, root)
    manifest_path = _required_frozen_file(destination / MANIFEST_FILENAME, "Manifest")
    metadata_path = _required_frozen_file(destination / METADATA_FILENAME, "Manifest metadata")
    _require_read_only(destination, "Frozen evaluation directory")

    metadata = _load_object(metadata_path, "Manifest metadata")
    _validate_metadata_boundary(metadata)
    audit_path, persisted_audit, fresh_audit = _validated_intake_state(
        contract_file, root, repository
    )
    source = _required_mapping(metadata, "source", "metadata")
    expected_source = {
        "contract_filename": contract_file.name,
        "contract_sha256": _sha256_file(contract_file),
        "pilot_audit_filename": audit_path.name,
        "pilot_audit_sha256": _sha256_file(audit_path),
        "labels_sha256": fresh_audit["input"]["labels_sha256"],
        "annotation_review_sha256": fresh_audit["input"]["annotation_review_sha256"],
        "storage_control_sha256": fresh_audit["storage_boundary"][
            "storage_control_sha256"
        ],
        "lifecycle_ledger_head_sha256": fresh_audit["storage_boundary"]["lifecycle"][
            "head_sha256"
        ],
    }
    if source != expected_source:
        raise ValueError("Frozen manifest source bindings no longer match the pilot package")
    expected_snapshot = {
        key: persisted_audit[key] for key in sorted(_AUDIT_COMPARISON_FIELDS)
    }
    if metadata.get("intake_audit_snapshot") != expected_snapshot:
        raise ValueError("Frozen manifest intake-audit snapshot does not match")

    contract = _load_object(contract_file, "Pilot contract")
    expected_records, split_config = _build_split_records(root, contract)
    actual_content = manifest_path.read_text(encoding="utf-8")
    expected_content = _manifest_content(expected_records)
    if actual_content != expected_content:
        raise ValueError("Frozen manifest content drift detected")
    manifest_metadata = _required_mapping(metadata, "manifest", "metadata")
    if manifest_metadata.get("sha256") != _sha256_file(manifest_path):
        raise ValueError("Frozen manifest SHA-256 does not match metadata")
    if manifest_metadata.get("num_samples") != len(expected_records):
        raise ValueError("Frozen manifest sample count does not match metadata")

    expected_id = _evaluation_set_id(
        str(fresh_audit["pilot_id"]), str(split_config["evaluation_set_version"])
    )
    if metadata.get("pilot_id") != fresh_audit["pilot_id"]:
        raise ValueError("Frozen manifest pilot identifier does not match")
    if metadata.get("evaluation_set_id") != expected_id:
        raise ValueError("Frozen manifest evaluation-set identifier does not match")
    sample_counts, writer_counts = _split_counts(expected_records)
    expected_split = {
        **split_config,
        "sample_counts": sample_counts,
        "writer_counts": writer_counts,
        "writer_isolation_verified": True,
        "training_split_present": False,
    }
    if metadata.get("split") != expected_split:
        raise ValueError("Frozen manifest split metadata does not match")
    return metadata, expected_records


def _validated_intake_state(
    contract_file: Path, root: Path, repository: Path
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    audit_path = _required_root_file(
        root / "pilot_intake.audit.json", root, "Pilot intake audit"
    )
    persisted = _load_object(audit_path, "Pilot intake audit")
    fresh = validate_protected_pilot(contract_file, root, repository_root=repository)
    for field in _AUDIT_COMPARISON_FIELDS:
        if persisted.get(field) != fresh.get(field):
            raise ValueError(f"Pilot intake audit is stale or invalid at field '{field}'")
    return audit_path, persisted, fresh


def _build_split_records(
    root: Path, contract: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold = _required_mapping(contract, "gold", "contract")
    evaluation_set_version = _required_string(
        gold, "evaluation_set_version", "contract.gold"
    )
    seed = _required_int(gold, "split_seed", "contract.gold")
    validation_ratio = _required_number(gold, "validation_writer_ratio", "contract.gold")
    test_ratio = _required_number(gold, "test_writer_ratio", "contract.gold")
    minimum_samples = _required_int(gold, "minimum_samples_per_writer", "contract.gold")
    split_config = {
        "evaluation_set_version": evaluation_set_version,
        "seed": seed,
        "validation_writer_ratio": validation_ratio,
        "test_writer_ratio": test_ratio,
        "minimum_samples_per_writer": minimum_samples,
    }
    base_records = _load_label_records(root)
    _, validation, test = create_writer_split(
        base_records,
        train_ratio=0.0,
        val_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
        min_samples_per_writer=minimum_samples,
    )
    split_by_sample = {
        str(record["sample_id"]): split_name
        for split_name, split in (("validation", validation), ("test", test))
        for record in split
    }
    if len(split_by_sample) != len(base_records):
        raise RuntimeError("Protected split did not assign every sample exactly once")
    return [
        {**record, "split": split_by_sample[str(record["sample_id"])]}
        for record in sorted(base_records, key=lambda item: str(item["sample_id"]))
    ], split_config


def _load_label_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with (root / "labels.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            filename = _csv_value(row, "filename", row_number)
            sample_id = Path(filename).stem
            writer_id = _csv_value(row, "writer_id", row_number)
            age_band = _csv_value(row, "age_band", row_number).replace("-", "_")
            slices = {
                *_csv_value(row, "slices", row_number).split("|"),
                f"age_band_{age_band}",
                _csv_value(row, "document_type", row_number),
                _csv_value(row, "capture_type", row_number),
            }
            records.append(
                {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "source_sha256": _sha256_file(root / "images" / filename),
                    "image_path": f"images/{filename}",
                    "reference": _csv_value(row, "text", row_number, preserve=True),
                    "writer_id": writer_id,
                    "slices": sorted(slices),
                }
            )
    return records


def _manifest_content(records: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        for record in records
    )


def _split_counts(records: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    sample_counts: dict[str, int] = {}
    writer_sets: dict[str, set[str]] = {}
    for record in records:
        if set(record) != _MANIFEST_FIELDS:
            raise ValueError("Protected manifest record fields are invalid")
        split = str(record["split"])
        sample_counts[split] = sample_counts.get(split, 0) + 1
        writer_sets.setdefault(split, set()).add(str(record["writer_id"]))
    if set(sample_counts) != {"validation", "test"}:
        raise ValueError("Protected manifest requires non-empty validation and test splits")
    if writer_sets["validation"] & writer_sets["test"]:
        raise ValueError("Writer leakage detected in protected manifest")
    return (
        {key: sample_counts[key] for key in sorted(sample_counts)},
        {key: len(writer_sets[key]) for key in sorted(writer_sets)},
    )


def _protected_root(
    dataset_dir: str | Path, repository_root: str | Path | None
) -> tuple[Path, Path]:
    root_path = Path(dataset_dir)
    if root_path.is_symlink():
        raise ValueError("Protected pilot root must not be a symbolic link")
    root = root_path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Protected pilot root must be a directory")
    repository = (
        Path(repository_root).resolve(strict=True)
        if repository_root is not None
        else Path(__file__).parents[2].resolve(strict=True)
    )
    if not (repository / ".git").exists():
        raise ValueError("repository_root must identify the InkBridge Git checkout")
    _require_outside_repository(root, repository, "Protected pilot root")
    return root, repository


def _required_manifest_destination(output_dir: str | Path, root: Path) -> Path:
    output = Path(output_dir)
    if output.name != MANIFEST_DIRECTORY:
        raise ValueError(f"Frozen evaluation directory must be named '{MANIFEST_DIRECTORY}'")
    if output.parent.resolve(strict=True) != root:
        raise ValueError("Frozen evaluation directory must be directly under the protected root")
    return output


def _required_existing_manifest_directory(manifest_dir: str | Path, root: Path) -> Path:
    path = Path(manifest_dir)
    if path.is_symlink():
        raise ValueError("Frozen evaluation directory must not be a symbolic link")
    path = path.resolve(strict=True)
    if path.name != MANIFEST_DIRECTORY or path.parent != root or not path.is_dir():
        raise ValueError("Frozen evaluation directory location is invalid")
    return path


def _required_root_file(path: str | Path, root: Path, label: str) -> Path:
    value = Path(path)
    if value.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    try:
        resolved = value.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {value}") from exc
    if resolved.parent != root or not resolved.is_file():
        raise ValueError(f"{label} must be a file at the protected root")
    return resolved


def _required_frozen_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a non-symlink file")
    _require_read_only(path, label)
    return path


def _require_read_only(path: Path, label: str) -> None:
    if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError(f"{label} must have no write permission bits")


def _validate_metadata_boundary(metadata: Mapping[str, Any]) -> None:
    if set(metadata) != _METADATA_FIELDS:
        raise ValueError("Manifest metadata fields must exactly match the allowlist")
    if metadata.get("schema_version") != 1:
        raise ValueError("Manifest metadata schema_version must be 1")
    if metadata.get("status") != "frozen_protected_evaluation_ready":
        raise ValueError("Manifest metadata status is invalid")
    if metadata.get("gold_ready") is not False:
        raise ValueError("Protected pilot must not claim final gold readiness")
    if metadata.get("data_classification") != "restricted_student_handwriting":
        raise ValueError("Manifest metadata data classification is invalid")
    if (
        metadata.get("next_gate")
        != "run_self_hosted_shadow_evaluation_without_exporting_sample_content"
    ):
        raise ValueError("Manifest metadata next gate is invalid")
    _verified_timestamp(metadata, "generated_at", "metadata")
    expected = {
        "public_repository_allowed": False,
        "github_actions_allowed": False,
        "third_party_ai_upload_allowed": False,
        "automated_education_decisions_allowed": False,
        "model_training_allowed": False,
    }
    if metadata.get("policy_boundary") != expected:
        raise ValueError("Manifest metadata policy boundary is invalid")
    manifest = _required_mapping(metadata, "manifest", "metadata")
    if set(manifest) != {
        "filename",
        "sha256",
        "num_samples",
        "contains_references",
        "repository_export_allowed",
    }:
        raise ValueError("Manifest metadata manifest fields are invalid")
    if manifest.get("filename") != MANIFEST_FILENAME:
        raise ValueError("Manifest metadata filename is invalid")
    if manifest.get("contains_references") is not True:
        raise ValueError("Protected manifest must declare protected references")
    if manifest.get("repository_export_allowed") is not False:
        raise ValueError("Protected manifest cannot allow repository export")


def _evaluation_set_id(pilot_id: str, version: str) -> str:
    value = f"{pilot_id}-{version}"
    if re.fullmatch(r"pilot-[a-z0-9-]{8,64}-v[1-9][0-9]{0,5}", value) is None:
        raise ValueError("Protected evaluation-set identifier is invalid")
    return value


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _verified_timestamp(value: Mapping[str, Any], key: str, context: str) -> None:
    raw = _required_string(value, key, context)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{context}.{key} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed > datetime.now(timezone.utc):
        raise ValueError(f"{context}.{key} must be a non-future timestamp")


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


def _required_number(value: Mapping[str, Any], key: str, context: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise ValueError(f"{context}.{key} must be numeric")
    return float(item)


def _required_int(value: Mapping[str, Any], key: str, context: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"{context}.{key} must be an integer")
    return item


def _csv_value(
    row: Mapping[str, str | None], key: str, row_number: int, *, preserve: bool = False
) -> str:
    item = row.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"labels.csv row {row_number}: {key} must be non-empty")
    if preserve and item != item.strip():
        raise ValueError(f"labels.csv row {row_number}: {key} has boundary whitespace")
    return item if preserve else item.strip()


def _write_new_file(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _remove_incomplete_directory(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        return
    path.chmod(0o700)
    for child in path.iterdir():
        child.chmod(0o600)
        child.unlink()
    path.rmdir()


def _require_outside_repository(path: Path, repository: Path, label: str) -> None:
    try:
        path.resolve().relative_to(repository)
    except ValueError:
        return
    raise ValueError(f"{label} must be outside the Git repository")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
