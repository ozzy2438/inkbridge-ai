"""Tests for the protected student-pilot governance and annotation gate."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, PngImagePlugin

from src.data.pilot_intake import validate_protected_pilot, write_pilot_audit

LABEL_COLUMNS = [
    "filename",
    "text",
    "writer_id",
    "slices",
    "age_band",
    "document_type",
    "capture_type",
]
REVIEW_COLUMNS = [
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


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _approved_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pilot_id": "pilot-shadow-001",
        "data_stage": "normalized_line_gold_candidate",
        "jurisdiction": "AU-VIC",
        "approval": {
            "privacy_review_status": "approved",
            "privacy_review_reference": "privacy_review_001",
            "educator_approval_status": "approved",
            "educator_approval_reference": "educator_review_001",
            "accountable_owner_role": "pilot_privacy_owner",
            "consent": {
                "status": "active",
                "authority": "participant_and_parent_guardian",
                "capacity_assessment_complete": True,
                "collection_purpose": "handwriting_ocr_shadow_evaluation",
                "evaluation_allowed": True,
                "training_allowed": False,
                "publication_allowed": False,
                "withdrawal_process_documented": True,
                "consent_records_separate_from_dataset": True,
                "verified_at": "2026-01-01T00:00:00+11:00",
            },
        },
        "privacy": {
            "direct_identifiers_removed": True,
            "visual_identifier_review_complete": True,
            "image_metadata_removed": True,
            "filename_policy": "opaque_sample_id",
            "writer_id_policy": "opaque_pseudonym",
            "allowed_demographics": ["age_band"],
            "public_repository_allowed": False,
            "github_actions_allowed": False,
            "third_party_ai_upload_allowed": False,
            "automated_education_decisions_allowed": False,
        },
        "storage": {
            "location_class": "approved_private",
            "encryption_at_rest": True,
            "encryption_in_transit": True,
            "access_control": "named_least_privilege",
            "audit_logging": True,
            "backup_copies_in_retention_scope": True,
            "cross_border_disclosure": "none",
            "raw_source_retention_days": 30,
            "normalized_candidate_retention_days": 180,
            "deletion_on_withdrawal_days": 30,
            "deletion_method": "verified_secure_deletion_or_deidentification",
            "deletion_verification_owner_role": "pilot_privacy_owner",
        },
        "annotation": {
            "review_filename": "annotation_review.csv",
            "guideline_version": "inkbridge-annotation-v2",
            "independent_annotators_per_sample": 2,
            "blind_first_passes": True,
            "model_preannotations_allowed": False,
            "adjudication_required_on_disagreement": True,
            "adjudicator_must_be_independent": True,
            "crop_boundary_review_required": True,
        },
        "gold": {
            "minimum_distinct_writers": 3,
            "evaluation_set_version": "v1",
            "split_seed": 42,
            "validation_writer_ratio": 0.5,
            "test_writer_ratio": 0.5,
            "minimum_samples_per_writer": 1,
            "writer_isolation_required": True,
            "split_locked_before_model_selection": True,
            "test_set_training_access_allowed": False,
            "gold_ready": False,
        },
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[dict[str, str]]]:
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    dataset = tmp_path / "protected" / "pilot"
    images = dataset / "images"
    images.mkdir(parents=True)

    labels: list[dict[str, str]] = []
    for index, text in enumerate(("First line", "Second line", "Third line"), start=1):
        token = str(index) * 16
        filename = f"sample-{token}.pgm"
        (images / filename).write_bytes(b"P5\n2 2\n255\n" + bytes([index] * 4))
        labels.append(
            {
                "filename": filename,
                "text": text,
                "writer_id": f"writer-{token}",
                "slices": "age_band_8_10|camera|student_handwriting",
                "age_band": "8-10",
                "document_type": "essay",
                "capture_type": "camera",
            }
        )
    _write_csv(dataset / "labels.csv", LABEL_COLUMNS, labels)

    reviews: list[dict[str, str]] = []
    for index, label in enumerate(labels):
        final_hash = _sha(label["text"])
        adjudicated = index == 2
        reviews.append(
            {
                "filename": label["filename"],
                "annotator_1_id": "annotator-aaaaaaaaaaaa",
                "annotator_2_id": "annotator-bbbbbbbbbbbb",
                "pass_1_sha256": "d" * 64 if adjudicated else final_hash,
                "pass_2_sha256": "e" * 64 if adjudicated else final_hash,
                "final_sha256": final_hash,
                "status": "adjudicated" if adjudicated else "agreed",
                "adjudicator_id": "annotator-cccccccccccc" if adjudicated else "",
                "adjudication_reason": ("transcription_disagreement" if adjudicated else ""),
                "crop_reviewed": "true",
                "pass_1_seconds": "30",
                "pass_2_seconds": "35",
                "adjudication_seconds": "20" if adjudicated else "0",
            }
        )
    _write_csv(dataset / "annotation_review.csv", REVIEW_COLUMNS, reviews)
    contract_path = dataset / "pilot_intake.json"
    contract_path.write_text(json.dumps(_approved_contract()), encoding="utf-8")
    return repository, dataset, contract_path, labels


def test_protected_pilot_gate_writes_pii_free_candidate_audit(tmp_path: Path) -> None:
    repository, dataset, contract, _ = _fixture(tmp_path)

    audit = validate_protected_pilot(contract, dataset, repository_root=repository)
    output = dataset / "pilot_intake.audit.json"
    write_pilot_audit(audit, output, dataset, repository_root=repository)

    assert audit["status"] == "gold_candidate_ready"
    assert audit["gold_ready"] is False
    assert audit["counts"] == {"samples": 3, "writers": 3}
    assert audit["data_use_scope"] == {
        "evaluation_allowed": True,
        "model_training_allowed": False,
        "publication_allowed": False,
    }
    assert audit["annotation_quality"]["agreement_rate"] == pytest.approx(2 / 3)
    assert audit["annotation_quality"]["adjudication_rate"] == pytest.approx(1 / 3)
    persisted = output.read_text(encoding="utf-8")
    assert "First line" not in persisted
    assert "writer-" not in persisted
    assert "annotator-" not in persisted


def test_protected_pilot_gate_rejects_data_inside_repository(tmp_path: Path) -> None:
    repository, dataset, _, _ = _fixture(tmp_path)
    unsafe = repository / "student-data"
    dataset.rename(unsafe)

    with pytest.raises(ValueError, match="outside the Git repository"):
        validate_protected_pilot(unsafe / "pilot_intake.json", unsafe, repository_root=repository)


def test_protected_pilot_gate_rejects_pending_approval(tmp_path: Path) -> None:
    repository, dataset, contract_path, _ = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["approval"]["privacy_review_status"] = "pending"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="privacy_review_status"):
        validate_protected_pilot(contract_path, dataset, repository_root=repository)


def test_protected_pilot_gate_rejects_invalid_writer_split_contract(tmp_path: Path) -> None:
    repository, dataset, contract_path, _ = _fixture(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["gold"]["test_writer_ratio"] = 0.6
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="ratios must sum to 1"):
        validate_protected_pilot(contract_path, dataset, repository_root=repository)


def test_protected_pilot_gate_rejects_identifier_columns(tmp_path: Path) -> None:
    repository, dataset, contract_path, _ = _fixture(tmp_path)
    labels_path = dataset / "labels.csv"
    fields, rows = _read_csv(labels_path)
    fields.append("student_name")
    for row in rows:
        row["student_name"] = "Not allowed"
    _write_csv(labels_path, fields, rows)

    with pytest.raises(ValueError, match="allowlist"):
        validate_protected_pilot(contract_path, dataset, repository_root=repository)


def test_protected_pilot_gate_rejects_uncontrolled_slice_metadata(tmp_path: Path) -> None:
    repository, dataset, contract_path, _ = _fixture(tmp_path)
    labels_path = dataset / "labels.csv"
    fields, rows = _read_csv(labels_path)
    rows[0]["slices"] = "age_band_8_10|camera|student_name_alice"
    _write_csv(labels_path, fields, rows)

    with pytest.raises(ValueError, match="controlled vocabulary"):
        validate_protected_pilot(contract_path, dataset, repository_root=repository)


def test_protected_pilot_gate_rejects_non_independent_adjudicator(tmp_path: Path) -> None:
    repository, dataset, contract_path, _ = _fixture(tmp_path)
    review_path = dataset / "annotation_review.csv"
    fields, rows = _read_csv(review_path)
    rows[2]["adjudicator_id"] = rows[2]["annotator_1_id"]
    _write_csv(review_path, fields, rows)

    with pytest.raises(ValueError, match="adjudicator must be independent"):
        validate_protected_pilot(contract_path, dataset, repository_root=repository)


def test_protected_pilot_gate_rejects_final_transcript_hash_drift(tmp_path: Path) -> None:
    repository, dataset, contract_path, _ = _fixture(tmp_path)
    review_path = dataset / "annotation_review.csv"
    fields, rows = _read_csv(review_path)
    rows[0]["final_sha256"] = "f" * 64
    _write_csv(review_path, fields, rows)

    with pytest.raises(ValueError, match="final transcript hash mismatch"):
        validate_protected_pilot(contract_path, dataset, repository_root=repository)


def test_protected_pilot_gate_rejects_embedded_image_metadata(tmp_path: Path) -> None:
    repository, dataset, contract_path, labels = _fixture(tmp_path)
    images = dataset / "images"
    old_name = labels[0]["filename"]
    new_name = old_name.removesuffix(".pgm") + ".png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Author", "Sensitive scanner metadata")
    Image.new("L", (2, 2), color=255).save(images / new_name, pnginfo=metadata)
    (images / old_name).unlink()

    labels_path = dataset / "labels.csv"
    label_fields, label_rows = _read_csv(labels_path)
    label_rows[0]["filename"] = new_name
    _write_csv(labels_path, label_fields, label_rows)
    review_path = dataset / "annotation_review.csv"
    review_fields, review_rows = _read_csv(review_path)
    review_rows[0]["filename"] = new_name
    _write_csv(review_path, review_fields, review_rows)

    with pytest.raises(ValueError, match="image metadata was not removed"):
        validate_protected_pilot(contract_path, dataset, repository_root=repository)
