"""Validate a protected student-handwriting pilot package and write its local audit."""

import argparse
from pathlib import Path

import structlog

from src.data.pilot_intake import validate_protected_pilot, write_pilot_audit

logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    """Parse protected-pilot validation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--audit-output",
        type=Path,
        help="Defaults to <dataset-dir>/pilot_intake.audit.json",
    )
    return parser.parse_args()


def main() -> int:
    """Run the fail-closed gate without exporting student content."""
    args = parse_args()
    output = args.audit_output or args.dataset_dir / "pilot_intake.audit.json"
    audit = validate_protected_pilot(args.contract, args.dataset_dir)
    write_pilot_audit(audit, output, args.dataset_dir)
    logger.info(
        "protected_pilot.validated",
        pilot_id=audit["pilot_id"],
        samples=audit["counts"]["samples"],
        writers=audit["counts"]["writers"],
        status=audit["status"],
        gold_ready=audit["gold_ready"],
        output=str(output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
