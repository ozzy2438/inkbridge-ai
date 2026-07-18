"""Prepare a private POSIX pilot tree without fabricating storage approval."""

import argparse
from pathlib import Path

import structlog

from src.data.protected_storage import prepare_protected_storage
from src.evaluation.protected_runtime import reject_ci_environment

logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    """Parse protected storage preparation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--pilot-id", required=True)
    parser.add_argument("--storage-id", required=True)
    return parser.parse_args()


def main() -> int:
    """Harden permissions and create pending control/lifecycle files."""
    args = parse_args()
    reject_ci_environment("Protected storage preparation")
    result = prepare_protected_storage(
        args.dataset_dir,
        pilot_id=args.pilot_id,
        storage_id=args.storage_id,
    )
    logger.info(
        "protected_storage.prepared",
        status=result["status"],
        pilot_id=result["pilot_id"],
        storage_id=result["storage_id"],
        control_path=result["control_path"],
        ledger_path=result["ledger_path"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
