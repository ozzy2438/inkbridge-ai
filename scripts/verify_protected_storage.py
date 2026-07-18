"""Verify protected storage/IAM facts and lifecycle evidence without exposing records."""

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.protected_storage import validate_protected_storage_boundary
from src.evaluation.protected_runtime import reject_ci_environment


def parse_args() -> argparse.Namespace:
    """Parse protected storage validation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-open-withdrawals",
        action="store_true",
        help="Audit an attention state; normal pilot processing refuses open withdrawals",
    )
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
    """Print only aggregate storage and lifecycle evidence."""
    args = parse_args()
    reject_ci_environment("Protected storage verification")
    contract = _contract(args.contract)
    result = validate_protected_storage_boundary(
        args.dataset_dir,
        pilot_id=str(contract.get("pilot_id", "")),
        contract_storage=contract["storage"],
        accountable_owner_role=str(
            contract["approval"].get("accountable_owner_role", "")
        ),
        require_no_open_withdrawals=not args.allow_open_withdrawals,
        allow_deadline_breaches=args.allow_open_withdrawals,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
