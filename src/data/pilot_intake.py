"""Fail-closed governance and double-annotation gate for protected student pilots."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

_SAMPLE_FILENAME = re.compile(r"sample-[0-9a-f]{16,64}\.(?:jpeg|jpg|pgm|png)")
_WRITER_ID = re.compile(r"writer-[0-9a-f]{16,64}")
_ANNOTATOR_ID = re.compile(r"annotator-[0-9a-f]{12,64}")
_PIL_SAFE_INFO = {"dpi", "jfif", "jfif_density", "jfif_unit", "jfif_version"}
_LABEL_COLUMNS = [
    "filename",
    "text",
    "writer_id",
    "slices",
    "age_band",
    "document_type",
    "capture_type",
]
_REVIEW_COLUMNS = [
    "filename",
    "annotator_1_id",
    "annotator_2_id",
    "pass_1_sha256",
    "pass_2_sha256",
    "final_sha256",
    "status",
    "adjudicator_id",
    "adjudication_reason",
    "crop_reviewed",
    "pass_1_seconds",
    "pass_2_seconds",
    "adjudication_seconds",
]
_AGE_BANDS = {"5-7", "8-10", "11-13", "14-17"}
_DOCUMENT_TYPES = {"essay", "short_answer", "worksheet"}
_CAPTURE_TYPES = {"camera", "scan"}
_ALLOWED_SLICES = {
    "age_band_5_7",
    "age_band_8_10",
    "age_band_11_13",
    "age_band_14_17",
    "camera",
    "crossed_out",
    "essay",
    "faint_pencil",
    "heavy_blur",
    "joined_cursive",
    "margin_insertion",
    "missing_page_edge",
    "mixed_content",
    "perspective",
    "scan",
    "short_answer",
    "student_handwriting",
    "teacher_annotation",
    "touching_lines",
    "worksheet",
}
_ADJUDICATION_REASONS = {
    "crop_boundary_disagreement",
    "reading_order_disagreement",
    "region_type_disagreement",
    "transcription_disagreement",
    "uncertain_character",
}


def validate_protected_pilot(
    contract_path: str | Path,
    dataset_dir: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a protected normalized pilot and return a PII-free readiness audit."""
    root = Path(dataset_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Protected pilot dataset path must be a directory")
    repo = _resolve_repository_root(repository_root)
    _require_outside_repository(root, repo, "Protected pilot dataset")

    contract_file = Path(contract_path)
    if contract_file.is_symlink():
        raise ValueError("Pilot contract must not be a symbolic link")
    contract_file = contract_file.resolve(strict=True)
    if contract_file.parent != root:
        raise ValueError("Pilot contract must be stored at the protected dataset root")
    _require_outside_repository(contract_file, repo, "Pilot contract")
    contract = _load_json(contract_file)
    _validate_contract(contract)

    labels_path = root / "labels.csv"
    images_dir = root / "images"
    if labels_path.is_symlink() or not labels_path.is_file():
        raise ValueError("Protected pilot must contain a non-symlink labels.csv")
    if images_dir.is_symlink() or not images_dir.is_dir():
        raise ValueError("Protected pilot must contain a non-symlink images directory")

    labels, image_hashes, writer_count = _validate_labels_and_images(labels_path, images_dir)
    minimum_writers = _required_positive_int(
        _required_mapping(contract, "gold", "contract"), "minimum_distinct_writers", "gold"
    )
    if writer_count < minimum_writers:
        raise ValueError(
            f"Pilot has {writer_count} writers but gold.minimum_distinct_writers is "
            f"{minimum_writers}"
        )

    annotation = _required_mapping(contract, "annotation", "contract")
    review_filename = _required_string(annotation, "review_filename", "annotation")
    if Path(review_filename).name != review_filename or review_filename != "annotation_review.csv":
        raise ValueError("annotation.review_filename must be 'annotation_review.csv'")
    review_path = root / review_filename
    if review_path.is_symlink() or not review_path.is_file():
        raise ValueError("Protected pilot must contain a non-symlink annotation_review.csv")
    review_metrics = _validate_annotation_review(review_path, labels)

    pilot_id = _required_string(contract, "pilot_id", "contract")
    return {
        "schema_version": 1,
        "status": "gold_candidate_ready",
        "gold_ready": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pilot_id": pilot_id,
        "data_classification": "restricted_student_handwriting",
        "policy_boundary": {
            "public_repository_allowed": False,
            "github_actions_allowed": False,
            "third_party_ai_upload_allowed": False,
            "automated_education_decisions_allowed": False,
        },
        "data_use_scope": {
            "evaluation_allowed": True,
            "model_training_allowed": False,
            "publication_allowed": False,
        },
        "input": {
            "contract_filename": contract_file.name,
            "contract_sha256": _sha256_file(contract_file),
            "labels_filename": labels_path.name,
            "labels_sha256": _sha256_file(labels_path),
            "annotation_review_filename": review_path.name,
            "annotation_review_sha256": _sha256_file(review_path),
            "image_content_sha256": sorted(image_hashes),
        },
        "counts": {
            "samples": len(labels),
            "writers": writer_count,
        },
        "annotation_quality": review_metrics,
        "next_gate": "create_and_freeze_writer_isolated_manifest_after_owner_approval",
    }


def write_pilot_audit(
    audit: Mapping[str, Any],
    output_path: str | Path,
    dataset_dir: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> None:
    """Atomically persist a validated audit beside, and only beside, protected data."""
    root = Path(dataset_dir).resolve(strict=True)
    repo = _resolve_repository_root(repository_root)
    output = Path(output_path)
    if output.parent.resolve(strict=True) != root:
        raise ValueError("Pilot audit must be written at the protected dataset root")
    if output.name != "pilot_intake.audit.json":
        raise ValueError("Pilot audit filename must be 'pilot_intake.audit.json'")
    _require_outside_repository(output, repo, "Pilot audit")
    _atomic_write_text(
        output,
        json.dumps(dict(audit), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != 1:
        raise ValueError("Pilot contract schema_version must be 1")
    pilot_id = _required_string(contract, "pilot_id", "contract")
    if re.fullmatch(r"pilot-[a-z0-9][a-z0-9-]{7,63}", pilot_id) is None:
        raise ValueError("contract.pilot_id must be an opaque pilot identifier")
    _require_exact(contract, "data_stage", "normalized_line_gold_candidate", "contract")
    _require_exact(contract, "jurisdiction", "AU-VIC", "contract")

    approval = _required_mapping(contract, "approval", "contract")
    _require_exact(approval, "privacy_review_status", "approved", "approval")
    _approval_reference(approval, "privacy_review_reference", "approval")
    _require_exact(approval, "educator_approval_status", "approved", "approval")
    _approval_reference(approval, "educator_approval_reference", "approval")
    _non_placeholder(approval, "accountable_owner_role", "approval")

    consent = _required_mapping(approval, "consent", "approval")
    _require_exact(consent, "status", "active", "approval.consent")
    authority = _required_string(consent, "authority", "approval.consent")
    if authority not in {"participant", "parent_guardian", "participant_and_parent_guardian"}:
        raise ValueError("approval.consent.authority is invalid")
    _require_bool(consent, "capacity_assessment_complete", True, "approval.consent")
    _require_exact(
        consent,
        "collection_purpose",
        "handwriting_ocr_shadow_evaluation",
        "approval.consent",
    )
    _require_bool(consent, "evaluation_allowed", True, "approval.consent")
    _require_bool(consent, "publication_allowed", False, "approval.consent")
    _require_bool(consent, "withdrawal_process_documented", True, "approval.consent")
    _require_bool(consent, "consent_records_separate_from_dataset", True, "approval.consent")
    _verified_timestamp(consent, "verified_at", "approval.consent")
    _require_bool(consent, "training_allowed", False, "approval.consent")

    privacy = _required_mapping(contract, "privacy", "contract")
    for key in (
        "direct_identifiers_removed",
        "visual_identifier_review_complete",
        "image_metadata_removed",
    ):
        _require_bool(privacy, key, True, "privacy")
    _require_exact(privacy, "filename_policy", "opaque_sample_id", "privacy")
    _require_exact(privacy, "writer_id_policy", "opaque_pseudonym", "privacy")
    demographics = privacy.get("allowed_demographics")
    if demographics != ["age_band"]:
        raise ValueError("privacy.allowed_demographics must contain only 'age_band'")
    for key in (
        "public_repository_allowed",
        "github_actions_allowed",
        "third_party_ai_upload_allowed",
        "automated_education_decisions_allowed",
    ):
        _require_bool(privacy, key, False, "privacy")

    storage = _required_mapping(contract, "storage", "contract")
    _require_exact(storage, "location_class", "approved_private", "storage")
    for key in (
        "encryption_at_rest",
        "encryption_in_transit",
        "audit_logging",
        "backup_copies_in_retention_scope",
    ):
        _require_bool(storage, key, True, "storage")
    _require_exact(storage, "access_control", "named_least_privilege", "storage")
    _require_exact(storage, "cross_border_disclosure", "none", "storage")
    _bounded_days(storage, "raw_source_retention_days", maximum=30)
    _bounded_days(storage, "normalized_candidate_retention_days", maximum=180)
    _bounded_days(storage, "deletion_on_withdrawal_days", maximum=30)
    _require_exact(
        storage,
        "deletion_method",
        "verified_secure_deletion_or_deidentification",
        "storage",
    )
    _non_placeholder(storage, "deletion_verification_owner_role", "storage")

    annotation = _required_mapping(contract, "annotation", "contract")
    _require_exact(annotation, "review_filename", "annotation_review.csv", "annotation")
    _require_exact(annotation, "guideline_version", "inkbridge-annotation-v2", "annotation")
    _require_exact(annotation, "independent_annotators_per_sample", 2, "annotation")
    for key in (
        "blind_first_passes",
        "adjudication_required_on_disagreement",
        "adjudicator_must_be_independent",
        "crop_boundary_review_required",
    ):
        _require_bool(annotation, key, True, "annotation")
    _require_bool(annotation, "model_preannotations_allowed", False, "annotation")

    gold = _required_mapping(contract, "gold", "contract")
    minimum = _required_positive_int(gold, "minimum_distinct_writers", "gold")
    if minimum < 3:
        raise ValueError("gold.minimum_distinct_writers must be at least 3")
    for key in ("writer_isolation_required", "split_locked_before_model_selection"):
        _require_bool(gold, key, True, "gold")
    for key in ("test_set_training_access_allowed", "gold_ready"):
        _require_bool(gold, key, False, "gold")


def _validate_labels_and_images(
    labels_path: Path, images_dir: Path
) -> tuple[dict[str, str], list[str], int]:
    labels: dict[str, str] = {}
    writers: set[str] = set()
    image_hashes: list[str] = []
    with labels_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != _LABEL_COLUMNS:
            raise ValueError("labels.csv columns must exactly match the protected-pilot allowlist")
        for row_number, row in enumerate(reader, start=2):
            filename = _csv_value(row, "filename", row_number, "labels.csv")
            if _SAMPLE_FILENAME.fullmatch(filename) is None:
                raise ValueError(f"labels.csv row {row_number}: filename is not opaque")
            if filename in labels:
                raise ValueError(f"labels.csv row {row_number}: duplicate filename")
            text = _csv_value(row, "text", row_number, "labels.csv", preserve=True)
            writer_id = _csv_value(row, "writer_id", row_number, "labels.csv")
            if _WRITER_ID.fullmatch(writer_id) is None:
                raise ValueError(f"labels.csv row {row_number}: writer_id is not pseudonymous")
            if _csv_value(row, "age_band", row_number, "labels.csv") not in _AGE_BANDS:
                raise ValueError(f"labels.csv row {row_number}: age_band is invalid")
            if _csv_value(row, "document_type", row_number, "labels.csv") not in _DOCUMENT_TYPES:
                raise ValueError(f"labels.csv row {row_number}: document_type is invalid")
            if _csv_value(row, "capture_type", row_number, "labels.csv") not in _CAPTURE_TYPES:
                raise ValueError(f"labels.csv row {row_number}: capture_type is invalid")
            _validate_slices(_csv_value(row, "slices", row_number, "labels.csv"), row_number)

            image_path = images_dir / filename
            _validate_protected_image(image_path, images_dir, row_number)
            labels[filename] = text
            writers.add(writer_id)
            image_hashes.append(_sha256_file(image_path))
    if not labels:
        raise ValueError("labels.csv contains no samples")

    indexed = set(labels)
    actual: set[str] = set()
    for path in images_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("images/ must contain only non-symlink image files")
        actual.add(path.name)
    if actual != indexed:
        raise ValueError("images/ contents must exactly match labels.csv")
    return labels, image_hashes, len(writers)


def _validate_protected_image(image_path: Path, images_dir: Path, row_number: int) -> None:
    if image_path.is_symlink() or not image_path.is_file():
        raise ValueError(f"labels.csv row {row_number}: image is missing or is a symlink")
    try:
        image_path.resolve(strict=True).relative_to(images_dir.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"labels.csv row {row_number}: image escapes images/") from exc
    try:
        with Image.open(image_path) as image:
            image.load()
            if image.format not in {"JPEG", "PNG", "PPM"}:
                raise ValueError(f"labels.csv row {row_number}: image format is not allowed")
            if len(image.getexif()) > 0 or set(image.info) - _PIL_SAFE_INFO:
                raise ValueError(f"labels.csv row {row_number}: image metadata was not removed")
    except OSError as exc:
        raise ValueError(f"labels.csv row {row_number}: image is not decodable") from exc


def _validate_slices(raw: str, row_number: int) -> None:
    slices = raw.split("|")
    if slices != sorted(set(slices)) or any(value not in _ALLOWED_SLICES for value in slices):
        raise ValueError(f"labels.csv row {row_number}: slices violate the controlled vocabulary")


def _validate_annotation_review(review_path: Path, labels: Mapping[str, str]) -> dict[str, Any]:
    reviewed: set[str] = set()
    agreed = 0
    adjudicated = 0
    total_seconds = 0.0
    with review_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != _REVIEW_COLUMNS:
            raise ValueError("annotation_review.csv columns must exactly match the review contract")
        for row_number, row in enumerate(reader, start=2):
            filename = _csv_value(row, "filename", row_number, "annotation_review.csv")
            if filename not in labels:
                raise ValueError(
                    f"annotation_review.csv row {row_number}: filename is not in labels.csv"
                )
            if filename in reviewed:
                raise ValueError(f"annotation_review.csv row {row_number}: duplicate filename")
            reviewed.add(filename)
            first = _annotator(row, "annotator_1_id", row_number)
            second = _annotator(row, "annotator_2_id", row_number)
            if first == second:
                raise ValueError(
                    f"annotation_review.csv row {row_number}: annotators must be independent"
                )
            pass_1 = _csv_sha(row, "pass_1_sha256", row_number)
            pass_2 = _csv_sha(row, "pass_2_sha256", row_number)
            final = _csv_sha(row, "final_sha256", row_number)
            expected_final = _sha256_bytes(labels[filename].encode("utf-8"))
            if final != expected_final:
                raise ValueError(
                    f"annotation_review.csv row {row_number}: final transcript hash mismatch"
                )
            if _csv_value(row, "crop_reviewed", row_number, "annotation_review.csv") != "true":
                raise ValueError(
                    f"annotation_review.csv row {row_number}: crop_reviewed must be true"
                )
            first_seconds = _csv_positive_float(row, "pass_1_seconds", row_number)
            second_seconds = _csv_positive_float(row, "pass_2_seconds", row_number)
            adjudication_seconds = _csv_nonnegative_float(row, "adjudication_seconds", row_number)
            total_seconds += first_seconds + second_seconds + adjudication_seconds

            status = _csv_value(row, "status", row_number, "annotation_review.csv")
            adjudicator = (row.get("adjudicator_id") or "").strip()
            reason = (row.get("adjudication_reason") or "").strip()
            if status == "agreed":
                if pass_1 != pass_2 or pass_1 != final:
                    raise ValueError(
                        f"annotation_review.csv row {row_number}: agreed hashes must match"
                    )
                if adjudicator or reason or adjudication_seconds != 0:
                    raise ValueError(
                        f"annotation_review.csv row {row_number}: agreed row has adjudication data"
                    )
                agreed += 1
            elif status == "adjudicated":
                if pass_1 == pass_2:
                    raise ValueError(
                        f"annotation_review.csv row {row_number}: adjudicated passes must differ"
                    )
                if _ANNOTATOR_ID.fullmatch(adjudicator) is None:
                    raise ValueError(
                        f"annotation_review.csv row {row_number}: adjudicator_id is invalid"
                    )
                if adjudicator in {first, second}:
                    raise ValueError(
                        f"annotation_review.csv row {row_number}: adjudicator must be independent"
                    )
                if reason not in _ADJUDICATION_REASONS or adjudication_seconds <= 0:
                    raise ValueError(
                        f"annotation_review.csv row {row_number}: adjudication evidence is invalid"
                    )
                adjudicated += 1
            else:
                raise ValueError(
                    f"annotation_review.csv row {row_number}: status must be agreed or adjudicated"
                )
    if reviewed != set(labels):
        raise ValueError("Every labels.csv sample must have exactly one annotation review")
    sample_count = len(labels)
    return {
        "samples": sample_count,
        "agreed_samples": agreed,
        "adjudicated_samples": adjudicated,
        "agreement_rate": agreed / sample_count,
        "adjudication_rate": adjudicated / sample_count,
        "total_review_hours": total_seconds / 3600,
        "samples_per_review_hour": sample_count / (total_seconds / 3600),
        "two_independent_passes_verified": True,
        "crop_review_verified": True,
    }


def _resolve_repository_root(value: str | Path | None) -> Path:
    if value is not None:
        root = Path(value).resolve(strict=True)
    else:
        root = Path(__file__).parents[2].resolve(strict=True)
    if not (root / ".git").exists():
        raise ValueError("repository_root must identify the InkBridge Git checkout")
    return root


def _require_outside_repository(path: Path, repository_root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(repository_root)
    except ValueError:
        return
    raise ValueError(f"{label} must be outside the Git repository")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Pilot contract is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Pilot contract must be a JSON object")
    return raw


def _verified_timestamp(value: Mapping[str, Any], key: str, context: str) -> None:
    raw = _required_string(value, key, context)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{context}.{key} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{context}.{key} must include a timezone")
    if parsed > datetime.now(timezone.utc):
        raise ValueError(f"{context}.{key} must not be in the future")


def _approval_reference(value: Mapping[str, Any], key: str, context: str) -> str:
    result = _required_string(value, key, context)
    if re.fullmatch(r"[a-z][a-z0-9_-]{7,127}", result) is None:
        raise ValueError(f"{context}.{key} must be an opaque internal approval reference")
    return result


def _non_placeholder(value: Mapping[str, Any], key: str, context: str) -> str:
    result = _required_string(value, key, context)
    if result.casefold() in {"pending", "replace_me", "tbd"}:
        raise ValueError(f"{context}.{key} must not be a placeholder")
    return result


def _bounded_days(value: Mapping[str, Any], key: str, *, maximum: int) -> int:
    result = _required_positive_int(value, key, "storage")
    if result > maximum:
        raise ValueError(f"storage.{key} exceeds the InkBridge pilot maximum of {maximum} days")
    return result


def _required_mapping(value: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"{context}.{key} must be an object")
    return result


def _required_string(value: Mapping[str, Any], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return result.strip()


def _required_positive_int(value: Mapping[str, Any], key: str, context: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result <= 0:
        raise ValueError(f"{context}.{key} must be a positive integer")
    return result


def _require_exact(value: Mapping[str, Any], key: str, expected: Any, context: str) -> None:
    if value.get(key) != expected:
        raise ValueError(f"{context}.{key} must be {expected!r}")


def _require_bool(value: Mapping[str, Any], key: str, expected: bool, context: str) -> None:
    result = value.get(key)
    if not isinstance(result, bool) or result is not expected:
        raise ValueError(f"{context}.{key} must be {expected!r}")


def _csv_value(
    row: Mapping[str, str | None],
    key: str,
    row_number: int,
    filename: str,
    *,
    preserve: bool = False,
) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{filename} row {row_number}: {key} must be non-empty")
    if preserve and value != value.strip():
        raise ValueError(f"{filename} row {row_number}: {key} has boundary whitespace")
    return value if preserve else value.strip()


def _annotator(row: Mapping[str, str | None], key: str, row_number: int) -> str:
    value = _csv_value(row, key, row_number, "annotation_review.csv")
    if _ANNOTATOR_ID.fullmatch(value) is None:
        raise ValueError(f"annotation_review.csv row {row_number}: {key} is invalid")
    return value


def _csv_sha(row: Mapping[str, str | None], key: str, row_number: int) -> str:
    value = _csv_value(row, key, row_number, "annotation_review.csv")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"annotation_review.csv row {row_number}: {key} is not SHA-256")
    return value


def _csv_positive_float(row: Mapping[str, str | None], key: str, row_number: int) -> float:
    value = _csv_nonnegative_float(row, key, row_number)
    if value <= 0:
        raise ValueError(f"annotation_review.csv row {row_number}: {key} must be positive")
    return value


def _csv_nonnegative_float(row: Mapping[str, str | None], key: str, row_number: int) -> float:
    raw = _csv_value(row, key, row_number, "annotation_review.csv")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"annotation_review.csv row {row_number}: {key} must be numeric") from exc
    if not 0 <= value < float("inf"):
        raise ValueError(f"annotation_review.csv row {row_number}: {key} must be finite")
    return value


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
