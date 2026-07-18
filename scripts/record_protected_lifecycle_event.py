"""Append a hash-chained consent-withdrawal or deletion-evidence event."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src.data.protected_storage import append_lifecycle_event
from src.evaluation.protected_runtime import reject_ci_environment

logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    """Parse lifecycle event arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--event-type",
        required=True,
        choices=(
            "consent_withdrawal_requested",
            "primary_deletion_verified",
            "backup_deletion_verified",
            "withdrawal_closed",
        ),
    )
    parser.add_argument("--subject-token-sha256", required=True)
    parser.add_argument("--scope-sha256", required=True)
    parser.add_argument("--actor-role", required=True)
    parser.add_argument("--evidence-reference", required=True)
    parser.add_argument("--evidence-sha256", required=True)
    parser.add_argument("--event-id")
    parser.add_argument("--occurred-at", type=datetime.fromisoformat)
    return parser.parse_args()


def _contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("storage"), dict)
        or not isinstance(value.get("approval"), dict)
    ):
        raise ValueError("Pilot contract must contain storage and approval objects")
    return value


def main() -> int:
    """Record one evidence event and log only aggregate lifecycle state."""
    args = parse_args()
    reject_ci_environment("Protected lifecycle recording")
    contract = _contract(args.contract)
    event, summary = append_lifecycle_event(
        args.dataset_dir,
        pilot_id=str(contract.get("pilot_id", "")),
        event_type=args.event_type,
        subject_token_sha256=args.subject_token_sha256,
        scope_sha256=args.scope_sha256,
        actor_role=args.actor_role,
        evidence_reference=args.evidence_reference,
        evidence_sha256=args.evidence_sha256,
        contract_storage=contract["storage"],
        accountable_owner_role=str(
            contract["approval"].get("accountable_owner_role", "")
        ),
        event_id=args.event_id,
        occurred_at=args.occurred_at,
    )
    logger.info(
        "protected_lifecycle.recorded",
        event_id=event["event_id"],
        event_type=event["event_type"],
        sequence=event["sequence"],
        head_sha256=summary["head_sha256"],
        open_withdrawals=summary["open_withdrawals"],
        deadline_breaches=summary["deadline_breaches"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
