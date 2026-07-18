"""Protected POSIX storage boundary and tamper-evident lifecycle evidence."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.evaluation.protected_runtime import reject_ci_environment

STORAGE_CONTROL_FILENAME = "protected_storage_control.json"
LIFECYCLE_LEDGER_FILENAME = "protected_lifecycle_events.jsonl"

_ZERO_SHA256 = "0" * 64
_CONTROL_FIELDS = {
    "schema_version",
    "status",
    "storage_id",
    "pilot_id",
    "approved_at",
    "expires_at",
    "accountable_owner_role",
    "operator_role",
    "deletion_verification_owner_role",
    "platform",
    "root_identity_sha256",
    "operator_uid",
    "operator_gid",
    "controls",
    "retention",
    "lifecycle_ledger_filename",
}
_CONTROL_ASSERTIONS = {
    "encryption_at_rest_attested": True,
    "encryption_in_transit_attested": True,
    "named_least_privilege_attested": True,
    "audit_logging_attested": True,
    "backup_copies_in_retention_scope": True,
    "network_isolation_attested": True,
    "public_access_allowed": False,
    "cross_border_disclosure": "none",
}
_RETENTION_FIELDS = {
    "raw_source_retention_days",
    "normalized_candidate_retention_days",
    "deletion_on_withdrawal_days",
}
_EVENT_FIELDS = {
    "schema_version",
    "sequence",
    "event_id",
    "event_type",
    "occurred_at",
    "recorded_at",
    "pilot_id",
    "subject_token_sha256",
    "scope_sha256",
    "actor_role",
    "evidence_reference",
    "evidence_sha256",
    "previous_event_sha256",
    "event_sha256",
}
_EVENT_TYPES = {
    "consent_withdrawal_requested",
    "primary_deletion_verified",
    "backup_deletion_verified",
    "withdrawal_closed",
}


def prepare_protected_storage(
    dataset_dir: str | Path,
    *,
    pilot_id: str,
    storage_id: str,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Harden a repository-external tree and create deliberately pending control files."""
    root, repository = _protected_root(dataset_dir, repository_root)
    _require_safe_id(pilot_id, "pilot", "pilot_id")
    _require_safe_id(storage_id, "storage", "storage_id")
    _require_outside_repository(root, repository, "Protected storage root")
    control_path = root / STORAGE_CONTROL_FILENAME
    ledger_path = root / LIFECYCLE_LEDGER_FILENAME
    if control_path.exists() or control_path.is_symlink():
        raise FileExistsError("Protected storage control already exists")
    if ledger_path.exists() or ledger_path.is_symlink():
        raise FileExistsError("Protected lifecycle ledger already exists")

    _harden_private_tree(root)
    _write_new_file(ledger_path, b"", mode=0o600)
    pending = {
        "schema_version": 1,
        "status": "pending",
        "storage_id": storage_id,
        "pilot_id": pilot_id,
        "approved_at": "replace_me",
        "expires_at": "replace_me",
        "accountable_owner_role": "replace_me",
        "operator_role": "replace_me",
        "deletion_verification_owner_role": "replace_me",
        "platform": "self_hosted_posix",
        "root_identity_sha256": _root_identity(root),
        "operator_uid": os.getuid(),
        "operator_gid": os.getgid(),
        "controls": {
            **_CONTROL_ASSERTIONS,
            "encryption_at_rest_attested": False,
            "encryption_in_transit_attested": False,
            "named_least_privilege_attested": False,
            "audit_logging_attested": False,
            "backup_copies_in_retention_scope": False,
            "network_isolation_attested": False,
        },
        "retention": {
            "raw_source_retention_days": 30,
            "normalized_candidate_retention_days": 180,
            "deletion_on_withdrawal_days": 30,
        },
        "lifecycle_ledger_filename": LIFECYCLE_LEDGER_FILENAME,
    }
    _write_new_file(
        control_path,
        (json.dumps(pending, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        mode=0o600,
    )
    control_path.chmod(0o400)
    return {
        "schema_version": 1,
        "status": "protected_storage_prepared_pending_approval",
        "storage_id": storage_id,
        "pilot_id": pilot_id,
        "control_path": str(control_path),
        "ledger_path": str(ledger_path),
    }


def validate_protected_storage_boundary(
    dataset_dir: str | Path,
    *,
    pilot_id: str,
    contract_storage: Mapping[str, Any],
    accountable_owner_role: str,
    repository_root: str | Path | None = None,
    require_no_open_withdrawals: bool = True,
    allow_deadline_breaches: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate local IAM facts, asserted controls, retention, and ledger integrity."""
    root, repository = _protected_root(dataset_dir, repository_root)
    _require_outside_repository(root, repository, "Protected storage root")
    _validate_private_tree(root)
    control_path = _required_root_file(root / STORAGE_CONTROL_FILENAME, root, "Storage control")
    _require_read_only(control_path, "Storage control")
    control = _load_object_no_follow(control_path, "Storage control")
    _validate_storage_control(
        control,
        root,
        pilot_id,
        contract_storage,
        accountable_owner_role=accountable_owner_role,
        now=now,
    )
    ledger_path = _required_root_file(
        root / LIFECYCLE_LEDGER_FILENAME, root, "Lifecycle ledger"
    )
    _require_owner_read_write(ledger_path, "Lifecycle ledger")
    ledger = verify_lifecycle_ledger(
        ledger_path,
        pilot_id=pilot_id,
        accountable_owner_role=str(control["accountable_owner_role"]),
        operator_role=str(control["operator_role"]),
        deletion_verification_owner_role=str(
            control["deletion_verification_owner_role"]
        ),
        deletion_on_withdrawal_days=int(
            _required_mapping(control, "retention", "storage control")[
                "deletion_on_withdrawal_days"
            ]
        ),
        now=now,
    )
    if require_no_open_withdrawals and ledger["open_withdrawals"]:
        raise ValueError("Protected storage has an open consent withdrawal")
    if ledger["deadline_breaches"] and not allow_deadline_breaches:
        raise ValueError("Protected storage lifecycle ledger contains a deletion deadline breach")

    principal = json.dumps(
        {"uid": os.getuid(), "gid": os.getgid()}, separators=(",", ":"), sort_keys=True
    )
    return {
        "schema_version": 1,
        "status": "protected_storage_boundary_ready",
        "storage_id": control["storage_id"],
        "storage_control_sha256": _sha256_file(control_path),
        "asserted_controls_sha256": _canonical_sha256(control["controls"]),
        "root_identity_sha256": control["root_identity_sha256"],
        "execution_principal_sha256": hashlib.sha256(principal.encode("utf-8")).hexdigest(),
        "accountable_owner_role": control["accountable_owner_role"],
        "operator_role": control["operator_role"],
        "deletion_verification_owner_role": control[
            "deletion_verification_owner_role"
        ],
        "local_iam": {
            "owner_uid_matches": True,
            "owner_gid_matches": True,
            "group_access_allowed": False,
            "other_access_allowed": False,
            "symbolic_links_allowed": False,
        },
        "lifecycle": ledger,
    }


def verify_lifecycle_ledger(
    ledger_path: str | Path,
    *,
    pilot_id: str,
    accountable_owner_role: str,
    operator_role: str,
    deletion_verification_owner_role: str,
    deletion_on_withdrawal_days: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify every event hash, transition, evidence binding, and withdrawal deadline."""
    path = Path(ledger_path)
    if path.is_symlink():
        raise ValueError("Lifecycle ledger must not be a symbolic link")
    content = _read_file_no_follow(path)
    summary, _ = _verify_ledger_content(
        content,
        pilot_id=pilot_id,
        accountable_owner_role=accountable_owner_role,
        operator_role=operator_role,
        deletion_verification_owner_role=deletion_verification_owner_role,
        deletion_on_withdrawal_days=deletion_on_withdrawal_days,
        now=now,
    )
    return summary


def append_lifecycle_event(
    dataset_dir: str | Path,
    *,
    pilot_id: str,
    event_type: str,
    subject_token_sha256: str,
    scope_sha256: str,
    actor_role: str,
    evidence_reference: str,
    evidence_sha256: str,
    contract_storage: Mapping[str, Any],
    accountable_owner_role: str,
    event_id: str | None = None,
    occurred_at: datetime | None = None,
    repository_root: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append one fsynced lifecycle event after validating the complete candidate chain."""
    reject_ci_environment("Protected lifecycle recording")
    root, repository = _protected_root(dataset_dir, repository_root)
    _require_outside_repository(root, repository, "Protected storage root")
    boundary = validate_protected_storage_boundary(
        root,
        pilot_id=pilot_id,
        contract_storage=contract_storage,
        accountable_owner_role=accountable_owner_role,
        repository_root=repository,
        require_no_open_withdrawals=False,
        allow_deadline_breaches=True,
    )
    control = _load_object_no_follow(
        root / STORAGE_CONTROL_FILENAME, "Storage control"
    )
    _validate_event_request(
        event_type=event_type,
        actor_role=actor_role,
        control=control,
    )
    _required_sha256(subject_token_sha256, "subject_token_sha256")
    _required_sha256(scope_sha256, "scope_sha256")
    _required_sha256(evidence_sha256, "evidence_sha256")
    _require_role(actor_role, "actor_role")
    _require_reference(evidence_reference, "evidence_reference")
    identifier = event_id or f"evt-{uuid.uuid4().hex}"
    if re.fullmatch(r"evt-[0-9a-f]{32}", identifier) is None:
        raise ValueError("event_id must be evt- followed by 32 lowercase hex characters")
    recorded_at = datetime.now(timezone.utc)
    raw_timestamp = occurred_at or recorded_at
    if raw_timestamp.tzinfo is None:
        raise ValueError("occurred_at must include a timezone")
    timestamp = raw_timestamp.astimezone(timezone.utc)
    if timestamp > recorded_at:
        raise ValueError("occurred_at must not be in the future")
    if timestamp < _timestamp(control["approved_at"], "approved_at"):
        raise ValueError("Lifecycle event must not predate the storage approval")
    ledger_path = root / LIFECYCLE_LEDGER_FILENAME
    descriptor = _open_no_follow(ledger_path, os.O_RDWR | os.O_APPEND)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
            descriptor = -1
            handle.seek(0)
            content = handle.read().encode("utf-8")
            prior_summary, _ = _verify_ledger_content(
                content,
                pilot_id=pilot_id,
                accountable_owner_role=str(control["accountable_owner_role"]),
                operator_role=str(control["operator_role"]),
                deletion_verification_owner_role=str(
                    control["deletion_verification_owner_role"]
                ),
                deletion_on_withdrawal_days=int(
                    _required_mapping(control, "retention", "storage control")[
                        "deletion_on_withdrawal_days"
                    ]
                ),
                now=recorded_at,
            )
            base_event = {
                "schema_version": 1,
                "sequence": int(prior_summary["num_events"]) + 1,
                "event_id": identifier,
                "event_type": event_type,
                "occurred_at": timestamp.isoformat(),
                "recorded_at": recorded_at.isoformat(),
                "pilot_id": pilot_id,
                "subject_token_sha256": subject_token_sha256,
                "scope_sha256": scope_sha256,
                "actor_role": actor_role,
                "evidence_reference": evidence_reference,
                "evidence_sha256": evidence_sha256,
                "previous_event_sha256": prior_summary["head_sha256"],
            }
            event = {**base_event, "event_sha256": _canonical_sha256(base_event)}
            serialized = json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n"
            candidate = content + serialized.encode("utf-8")
            summary, _ = _verify_ledger_content(
                candidate,
                pilot_id=pilot_id,
                accountable_owner_role=str(control["accountable_owner_role"]),
                operator_role=str(control["operator_role"]),
                deletion_verification_owner_role=str(
                    control["deletion_verification_owner_role"]
                ),
                deletion_on_withdrawal_days=int(
                    _required_mapping(control, "retention", "storage control")[
                        "deletion_on_withdrawal_days"
                    ]
                ),
                now=recorded_at,
            )
            handle.seek(0, os.SEEK_END)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
    if boundary["storage_id"] != control["storage_id"]:
        raise RuntimeError("Protected storage identity changed while recording lifecycle event")
    return event, summary


def _verify_ledger_content(
    content: bytes,
    *,
    pilot_id: str,
    accountable_owner_role: str,
    operator_role: str,
    deletion_verification_owner_role: str,
    deletion_on_withdrawal_days: int,
    now: datetime | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not 1 <= deletion_on_withdrawal_days <= 30:
        raise ValueError("deletion_on_withdrawal_days must be between 1 and 30")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Lifecycle ledger must be UTF-8") from exc
    if decoded and not decoded.endswith("\n"):
        raise ValueError("Lifecycle ledger must end with a newline")
    previous_sha256 = _ZERO_SHA256
    previous_occurred_at: datetime | None = None
    previous_recorded_at: datetime | None = None
    states: dict[str, dict[str, Any]] = {}
    event_ids: set[str] = set()
    event_count = 0
    for line_number, raw in enumerate(decoded.splitlines(), start=1):
        if not raw:
            raise ValueError("Lifecycle ledger must not contain blank records")
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Lifecycle ledger line {line_number} is invalid JSON") from exc
        if not isinstance(event, dict) or set(event) != _EVENT_FIELDS:
            raise ValueError("Lifecycle event fields must exactly match the allowlist")
        if event.get("schema_version") != 1 or event.get("sequence") != line_number:
            raise ValueError("Lifecycle event schema or sequence is invalid")
        if event.get("pilot_id") != pilot_id:
            raise ValueError("Lifecycle event pilot_id does not match")
        if event.get("event_type") not in _EVENT_TYPES:
            raise ValueError("Lifecycle event_type is invalid")
        if re.fullmatch(r"evt-[0-9a-f]{32}", str(event.get("event_id"))) is None:
            raise ValueError("Lifecycle event_id is invalid")
        if event["event_id"] in event_ids:
            raise ValueError("Lifecycle event_id must be unique")
        event_ids.add(str(event["event_id"]))
        for key in (
            "subject_token_sha256",
            "scope_sha256",
            "evidence_sha256",
            "previous_event_sha256",
            "event_sha256",
        ):
            _required_sha256(event.get(key), key)
        _require_role(event.get("actor_role"), "actor_role")
        expected_actor = (
            accountable_owner_role
            if event["event_type"] == "consent_withdrawal_requested"
            else (
                deletion_verification_owner_role
                if event["event_type"] == "withdrawal_closed"
                else operator_role
            )
        )
        if event["actor_role"] != expected_actor:
            raise ValueError("Lifecycle event actor role does not match the event type")
        _require_reference(event.get("evidence_reference"), "evidence_reference")
        if event["previous_event_sha256"] != previous_sha256:
            raise ValueError("Lifecycle event hash chain is broken")
        expected_hash = _canonical_sha256(
            {key: event[key] for key in event if key != "event_sha256"}
        )
        if event["event_sha256"] != expected_hash:
            raise ValueError("Lifecycle event content hash does not match")
        occurred_at = _timestamp(event.get("occurred_at"), "occurred_at")
        recorded_at = _timestamp(event.get("recorded_at"), "recorded_at")
        if occurred_at > recorded_at or recorded_at > current:
            raise ValueError("Lifecycle event timestamps must not be in the future")
        if previous_occurred_at is not None and occurred_at < previous_occurred_at:
            raise ValueError("Lifecycle event timestamps must be non-decreasing")
        if previous_recorded_at is not None and recorded_at < previous_recorded_at:
            raise ValueError("Lifecycle recording timestamps must be non-decreasing")
        _apply_transition(states, event, occurred_at, recorded_at)
        previous_sha256 = event["event_sha256"]
        previous_occurred_at = occurred_at
        previous_recorded_at = recorded_at
        event_count += 1

    open_withdrawals = 0
    closed_withdrawals = 0
    deadline_breaches = 0
    deadline = timedelta(days=deletion_on_withdrawal_days)
    for state in states.values():
        requested_at = state["requested_at"]
        closed_at = state.get("closed_at")
        if closed_at is None:
            open_withdrawals += 1
            if current > requested_at + deadline:
                deadline_breaches += 1
        else:
            closed_withdrawals += 1
            if (
                closed_at > requested_at + deadline
                or state["closed_recorded_at"] > requested_at + deadline
            ):
                deadline_breaches += 1
    status = (
        "lifecycle_ready"
        if open_withdrawals == 0 and deadline_breaches == 0
        else "lifecycle_attention_required"
    )
    return (
        {
            "schema_version": 1,
            "status": status,
            "ledger_sha256": hashlib.sha256(content).hexdigest(),
            "head_sha256": previous_sha256,
            "num_events": event_count,
            "open_withdrawals": open_withdrawals,
            "closed_withdrawals": closed_withdrawals,
            "deadline_breaches": deadline_breaches,
        },
        states,
    )


def _apply_transition(
    states: dict[str, dict[str, Any]],
    event: Mapping[str, Any],
    occurred_at: datetime,
    recorded_at: datetime,
) -> None:
    subject = str(event["subject_token_sha256"])
    event_type = str(event["event_type"])
    state = states.get(subject)
    if event_type == "consent_withdrawal_requested":
        if state is not None:
            raise ValueError("Lifecycle ledger contains a duplicate withdrawal request")
        states[subject] = {
            "scope_sha256": event["scope_sha256"],
            "requested_at": occurred_at,
            "requested_recorded_at": recorded_at,
        }
        return
    if state is None:
        raise ValueError("Lifecycle deletion evidence appears before a withdrawal request")
    if event["scope_sha256"] != state["scope_sha256"]:
        raise ValueError("Lifecycle event scope changed within a withdrawal")
    if state.get("closed_at") is not None:
        raise ValueError("Lifecycle event appears after withdrawal closure")
    if event_type == "primary_deletion_verified":
        if state.get("primary_at") is not None:
            raise ValueError("Primary deletion was verified more than once")
        state["primary_at"] = occurred_at
    elif event_type == "backup_deletion_verified":
        if state.get("backup_at") is not None:
            raise ValueError("Backup deletion was verified more than once")
        state["backup_at"] = occurred_at
    elif event_type == "withdrawal_closed":
        if state.get("primary_at") is None or state.get("backup_at") is None:
            raise ValueError("Withdrawal cannot close before primary and backup deletion evidence")
        state["closed_at"] = occurred_at
        state["closed_recorded_at"] = recorded_at


def _validate_storage_control(
    control: Mapping[str, Any],
    root: Path,
    pilot_id: str,
    contract_storage: Mapping[str, Any],
    *,
    accountable_owner_role: str,
    now: datetime | None,
) -> None:
    if set(control) != _CONTROL_FIELDS:
        raise ValueError("Storage control fields must exactly match the allowlist")
    if control.get("schema_version") != 1 or control.get("status") != "approved":
        raise ValueError("Storage control must be approved schema_version 1")
    _require_safe_id(control.get("storage_id"), "storage", "storage_id")
    if control.get("pilot_id") != pilot_id:
        raise ValueError("Storage control pilot_id does not match")
    _require_safe_id(pilot_id, "pilot", "pilot_id")
    if control.get("platform") != "self_hosted_posix":
        raise ValueError("Storage control platform must be self_hosted_posix")
    if control.get("root_identity_sha256") != _root_identity(root):
        raise ValueError("Storage control root identity does not match")
    if control.get("operator_uid") != os.getuid() or control.get("operator_gid") != os.getgid():
        raise ValueError("Storage control execution principal does not match the current process")
    if control.get("accountable_owner_role") != accountable_owner_role:
        raise ValueError("Storage control accountable owner role does not match the pilot contract")
    _require_role(control.get("accountable_owner_role"), "accountable_owner_role")
    _require_role(control.get("operator_role"), "operator_role")
    expected_deletion_owner = contract_storage.get("deletion_verification_owner_role")
    if control.get("deletion_verification_owner_role") != expected_deletion_owner:
        raise ValueError("Storage control deletion verification role does not match")
    _require_role(
        control.get("deletion_verification_owner_role"),
        "deletion_verification_owner_role",
    )
    approved_at = _timestamp(control.get("approved_at"), "approved_at")
    expires_at = _timestamp(control.get("expires_at"), "expires_at")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if approved_at > current or expires_at <= current:
        raise ValueError("Storage control approval is not currently valid")
    if expires_at <= approved_at or expires_at - approved_at > timedelta(days=90):
        raise ValueError("Storage control validity must be greater than 0 and at most 90 days")
    controls = _required_mapping(control, "controls", "storage control")
    if controls != _CONTROL_ASSERTIONS:
        raise ValueError("Storage control assertions do not match the protected boundary")
    retention = _required_mapping(control, "retention", "storage control")
    if set(retention) != _RETENTION_FIELDS:
        raise ValueError("Storage control retention fields are invalid")
    expected_retention = {
        key: contract_storage.get(key) for key in sorted(_RETENTION_FIELDS)
    }
    if dict(retention) != expected_retention:
        raise ValueError("Storage control retention does not match the pilot contract")
    for key, maximum in {
        "raw_source_retention_days": 30,
        "normalized_candidate_retention_days": 180,
        "deletion_on_withdrawal_days": 30,
    }.items():
        value = retention.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f"Storage control retention.{key} is invalid")
    if control.get("lifecycle_ledger_filename") != LIFECYCLE_LEDGER_FILENAME:
        raise ValueError("Storage control lifecycle ledger filename is invalid")


def _validate_event_request(
    *, event_type: str, actor_role: str, control: Mapping[str, Any]
) -> None:
    if event_type not in _EVENT_TYPES:
        raise ValueError("event_type is invalid")
    expected_role = (
        control["accountable_owner_role"]
        if event_type == "consent_withdrawal_requested"
        else (
            control["deletion_verification_owner_role"]
            if event_type == "withdrawal_closed"
            else control["operator_role"]
        )
    )
    if actor_role != expected_role:
        raise ValueError(f"{event_type} must be recorded by role {expected_role}")


def _protected_root(
    dataset_dir: str | Path, repository_root: str | Path | None
) -> tuple[Path, Path]:
    raw = Path(dataset_dir)
    if raw.is_symlink():
        raise ValueError("Protected storage root must not be a symbolic link")
    root = raw.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Protected storage root must be a directory")
    repository = (
        Path(repository_root).resolve(strict=True)
        if repository_root is not None
        else Path(__file__).parents[2].resolve(strict=True)
    )
    if not (repository / ".git").exists():
        raise ValueError("repository_root must identify the InkBridge Git checkout")
    return root, repository


def _harden_private_tree(root: Path) -> None:
    entries = [root, *sorted(root.rglob("*"), key=lambda path: path.as_posix())]
    for path in entries:
        if path.is_symlink():
            raise ValueError("Protected storage must not contain symbolic links")
        if path.is_dir():
            path.chmod(0o700)
        elif path.is_file():
            path.chmod(0o600)
        else:
            raise ValueError("Protected storage must contain only files and directories")


def _validate_private_tree(root: Path) -> None:
    entries = [root, *sorted(root.rglob("*"), key=lambda path: path.as_posix())]
    for path in entries:
        if path.is_symlink():
            raise ValueError("Protected storage must not contain symbolic links")
        details = path.stat()
        if details.st_uid != os.getuid() or details.st_gid != os.getgid():
            raise ValueError("Protected storage must be owned by the approved execution principal")
        if details.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise ValueError("Protected storage must deny all group and other access")
        if not path.is_dir() and not path.is_file():
            raise ValueError("Protected storage must contain only files and directories")


def _required_root_file(path: Path, root: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    resolved = path.resolve(strict=True)
    if resolved.parent != root or not resolved.is_file():
        raise ValueError(f"{label} must be a file at the protected root")
    return resolved


def _require_read_only(path: Path, label: str) -> None:
    if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError(f"{label} must have no write permission bits")


def _require_owner_read_write(path: Path, label: str) -> None:
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO) or not mode & stat.S_IRUSR or not mode & stat.S_IWUSR:
        raise ValueError(f"{label} must be owner-readable/writable and private")


def _load_object_no_follow(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_read_file_no_follow(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_file_no_follow(path: Path) -> bytes:
    descriptor = _open_no_follow(path, os.O_RDONLY)
    with os.fdopen(descriptor, "rb") as handle:
        return handle.read()


def _open_no_follow(path: Path, flags: int) -> int:
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("Protected storage evidence must be a regular file")
    return descriptor


def _write_new_file(path: Path, content: bytes, *, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _root_identity(root: Path) -> str:
    return hashlib.sha256(str(root).encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(value, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(_read_file_no_follow(path)).hexdigest()


def _required_mapping(value: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"{context}.{key} must be an object")
    return item


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _required_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_safe_id(value: Any, prefix: str, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(
        rf"{prefix}-[a-z0-9][a-z0-9-]{{7,63}}", value
    ) is None:
        raise ValueError(f"{label} must be an opaque {prefix} identifier")
    return value


def _require_role(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9_]{7,63}", value) is None:
        raise ValueError(f"{label} must be an approved role")
    if value in {"pending", "replace_me"}:
        raise ValueError(f"{label} must not be a placeholder")
    return value


def _require_reference(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9_-]{7,127}", value) is None:
        raise ValueError(f"{label} must be an opaque internal reference")
    return value


def _require_outside_repository(path: Path, repository: Path, label: str) -> None:
    try:
        path.resolve().relative_to(repository)
    except ValueError:
        return
    raise ValueError(f"{label} must be outside the Git repository")
