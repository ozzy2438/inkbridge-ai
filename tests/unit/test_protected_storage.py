"""Tests for protected POSIX storage and lifecycle evidence."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from src.data.protected_storage import (
    LIFECYCLE_LEDGER_FILENAME,
    STORAGE_CONTROL_FILENAME,
    append_lifecycle_event,
    prepare_protected_storage,
    validate_protected_storage_boundary,
    verify_lifecycle_ledger,
)
from tests.unit.test_protected_evaluation import _clear_ci

_PILOT_ID = "pilot-shadow-001"
_OWNER_ROLE = "pilot_privacy_owner"
_OPERATOR_ROLE = "ml_evaluation_operator"
_DELETION_OWNER_ROLE = "deletion_evidence_owner"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _contract_storage() -> dict[str, Any]:
    return {
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
        "deletion_verification_owner_role": _DELETION_OWNER_ROLE,
    }


def _approved_control(root: Path, *, approved_at: datetime, expires_at: datetime) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "approved",
        "storage_id": "storage-shadow-001",
        "pilot_id": _PILOT_ID,
        "approved_at": approved_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "accountable_owner_role": _OWNER_ROLE,
        "operator_role": _OPERATOR_ROLE,
        "deletion_verification_owner_role": _DELETION_OWNER_ROLE,
        "platform": "self_hosted_posix",
        "root_identity_sha256": _sha(str(root.resolve())),
        "operator_uid": os.getuid(),
        "operator_gid": os.getgid(),
        "controls": {
            "encryption_at_rest_attested": True,
            "encryption_in_transit_attested": True,
            "named_least_privilege_attested": True,
            "audit_logging_attested": True,
            "backup_copies_in_retention_scope": True,
            "network_isolation_attested": True,
            "public_access_allowed": False,
            "cross_border_disclosure": "none",
        },
        "retention": {
            "raw_source_retention_days": 30,
            "normalized_candidate_retention_days": 180,
            "deletion_on_withdrawal_days": 30,
        },
        "lifecycle_ledger_filename": LIFECYCLE_LEDGER_FILENAME,
    }


def _storage_fixture(
    tmp_path: Path, *, approval_age_days: int = 1
) -> tuple[Path, Path, dict[str, Any], datetime]:
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    root = tmp_path / "protected" / "pilot"
    root.mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    now = datetime.now(timezone.utc)
    control = _approved_control(
        root,
        approved_at=now - timedelta(days=approval_age_days),
        expires_at=now + timedelta(days=1),
    )
    control_path = root / STORAGE_CONTROL_FILENAME
    control_path.write_text(json.dumps(control), encoding="utf-8")
    control_path.chmod(0o400)
    ledger = root / LIFECYCLE_LEDGER_FILENAME
    ledger.write_text("", encoding="utf-8")
    ledger.chmod(0o600)
    return repository, root, _contract_storage(), now


def _append(
    repository: Path,
    root: Path,
    contract_storage: dict[str, Any],
    *,
    event_type: str,
    actor_role: str,
    occurred_at: datetime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return append_lifecycle_event(
        root,
        pilot_id=_PILOT_ID,
        event_type=event_type,
        subject_token_sha256=_sha("subject-keyed-token"),
        scope_sha256=_sha("object-scope"),
        actor_role=actor_role,
        evidence_reference=f"evidence_{event_type}",
        evidence_sha256=_sha(f"evidence-bytes-{event_type}"),
        contract_storage=contract_storage,
        accountable_owner_role=_OWNER_ROLE,
        occurred_at=occurred_at,
        repository_root=repository,
    )


def test_prepare_storage_is_private_and_deliberately_pending(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    root = tmp_path / "protected" / "pilot"
    root.mkdir(parents=True)
    sample = root / "sample.txt"
    sample.write_text("synthetic")

    result = prepare_protected_storage(
        root,
        pilot_id=_PILOT_ID,
        storage_id="storage-shadow-001",
        repository_root=repository,
    )

    control = json.loads((root / STORAGE_CONTROL_FILENAME).read_text(encoding="utf-8"))
    assert result["status"] == "protected_storage_prepared_pending_approval"
    assert control["status"] == "pending"
    assert control["controls"]["encryption_at_rest_attested"] is False
    assert not (root.stat().st_mode & 0o077)
    assert not (sample.stat().st_mode & 0o077)
    assert not ((root / STORAGE_CONTROL_FILENAME).stat().st_mode & 0o222)


def test_storage_boundary_validates_local_iam_and_empty_ledger(tmp_path: Path) -> None:
    repository, root, contract_storage, _ = _storage_fixture(tmp_path)

    result = validate_protected_storage_boundary(
        root,
        pilot_id=_PILOT_ID,
        contract_storage=contract_storage,
        accountable_owner_role=_OWNER_ROLE,
        repository_root=repository,
    )

    assert result["status"] == "protected_storage_boundary_ready"
    assert result["local_iam"]["group_access_allowed"] is False
    assert result["lifecycle"]["head_sha256"] == "0" * 64
    assert result["lifecycle"]["num_events"] == 0


def test_storage_boundary_rejects_group_access(tmp_path: Path) -> None:
    repository, root, contract_storage, _ = _storage_fixture(tmp_path)
    root.chmod(0o750)

    with pytest.raises(ValueError, match="deny all group and other access"):
        validate_protected_storage_boundary(
            root,
            pilot_id=_PILOT_ID,
            contract_storage=contract_storage,
            accountable_owner_role=_OWNER_ROLE,
            repository_root=repository,
        )


def test_storage_boundary_rejects_accountable_owner_role_drift(tmp_path: Path) -> None:
    repository, root, contract_storage, _ = _storage_fixture(tmp_path)

    with pytest.raises(ValueError, match="accountable owner role does not match"):
        validate_protected_storage_boundary(
            root,
            pilot_id=_PILOT_ID,
            contract_storage=contract_storage,
            accountable_owner_role="different_privacy_owner",
            repository_root=repository,
        )


def test_lifecycle_ledger_enforces_complete_withdrawal_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, root, contract_storage, now = _storage_fixture(tmp_path)
    sequence = (
        ("consent_withdrawal_requested", _OWNER_ROLE),
        ("primary_deletion_verified", _OPERATOR_ROLE),
        ("backup_deletion_verified", _OPERATOR_ROLE),
        ("withdrawal_closed", _DELETION_OWNER_ROLE),
    )
    prior_hash = "0" * 64
    started_at = now - timedelta(seconds=10)
    for offset, (event_type, role) in enumerate(sequence, start=1):
        event, summary = _append(
            repository,
            root,
            contract_storage,
            event_type=event_type,
            actor_role=role,
            occurred_at=started_at + timedelta(seconds=offset),
        )
        assert event["previous_event_sha256"] == prior_hash
        prior_hash = event["event_sha256"]

    assert summary["status"] == "lifecycle_ready"
    assert summary["num_events"] == 4
    assert summary["open_withdrawals"] == 0
    assert summary["closed_withdrawals"] == 1
    assert summary["head_sha256"] == prior_hash


def test_open_withdrawal_blocks_protected_storage_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, root, contract_storage, now = _storage_fixture(tmp_path)
    _append(
        repository,
        root,
        contract_storage,
        event_type="consent_withdrawal_requested",
        actor_role=_OWNER_ROLE,
        occurred_at=now - timedelta(seconds=1),
    )

    with pytest.raises(ValueError, match="open consent withdrawal"):
        validate_protected_storage_boundary(
            root,
            pilot_id=_PILOT_ID,
            contract_storage=contract_storage,
            accountable_owner_role=_OWNER_ROLE,
            repository_root=repository,
        )


def test_withdrawal_cannot_close_before_backup_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, root, contract_storage, now = _storage_fixture(tmp_path)
    _append(
        repository,
        root,
        contract_storage,
        event_type="consent_withdrawal_requested",
        actor_role=_OWNER_ROLE,
        occurred_at=now - timedelta(seconds=3),
    )
    _append(
        repository,
        root,
        contract_storage,
        event_type="primary_deletion_verified",
        actor_role=_OPERATOR_ROLE,
        occurred_at=now - timedelta(seconds=2),
    )

    with pytest.raises(ValueError, match="primary and backup deletion evidence"):
        _append(
            repository,
            root,
            contract_storage,
            event_type="withdrawal_closed",
            actor_role=_DELETION_OWNER_ROLE,
            occurred_at=now - timedelta(seconds=1),
        )

    assert len((root / LIFECYCLE_LEDGER_FILENAME).read_text().splitlines()) == 2


def test_lifecycle_ledger_detects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, root, contract_storage, now = _storage_fixture(tmp_path)
    _append(
        repository,
        root,
        contract_storage,
        event_type="consent_withdrawal_requested",
        actor_role=_OWNER_ROLE,
        occurred_at=now - timedelta(seconds=1),
    )
    ledger = root / LIFECYCLE_LEDGER_FILENAME
    ledger.write_text(
        ledger.read_text().replace("evidence_consent", "tampered_consent"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="content hash does not match"):
        verify_lifecycle_ledger(
            ledger,
            pilot_id=_PILOT_ID,
            accountable_owner_role=_OWNER_ROLE,
            operator_role=_OPERATOR_ROLE,
            deletion_verification_owner_role=_DELETION_OWNER_ROLE,
            deletion_on_withdrawal_days=30,
        )


def test_lifecycle_recording_refuses_ci_before_opening_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, root, contract_storage, now = _storage_fixture(tmp_path)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    with pytest.raises(RuntimeError, match="refuses CI"):
        _append(
            repository,
            root,
            contract_storage,
            event_type="consent_withdrawal_requested",
            actor_role=_OWNER_ROLE,
            occurred_at=now - timedelta(seconds=1),
        )


def test_late_deletion_can_be_recorded_but_deadline_breach_remains_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_ci(monkeypatch)
    repository, root, contract_storage, now = _storage_fixture(
        tmp_path, approval_age_days=40
    )
    events = (
        ("consent_withdrawal_requested", _OWNER_ROLE, now - timedelta(days=31)),
        ("primary_deletion_verified", _OPERATOR_ROLE, now - timedelta(minutes=3)),
        ("backup_deletion_verified", _OPERATOR_ROLE, now - timedelta(minutes=2)),
        ("withdrawal_closed", _DELETION_OWNER_ROLE, now - timedelta(minutes=1)),
    )
    summary: dict[str, Any] = {}
    for event_type, role, occurred_at in events:
        _, summary = _append(
            repository,
            root,
            contract_storage,
            event_type=event_type,
            actor_role=role,
            occurred_at=occurred_at,
        )

    assert summary["open_withdrawals"] == 0
    assert summary["closed_withdrawals"] == 1
    assert summary["deadline_breaches"] == 1
    with pytest.raises(ValueError, match="deletion deadline breach"):
        validate_protected_storage_boundary(
            root,
            pilot_id=_PILOT_ID,
            contract_storage=contract_storage,
            accountable_owner_role=_OWNER_ROLE,
            repository_root=repository,
        )
