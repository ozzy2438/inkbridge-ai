"""Inspect and hash a materialized local model artifact without modifying it."""

import argparse
import json
from pathlib import Path

from src.evaluation.protected_inference import inspect_local_model_artifact


def parse_args() -> argparse.Namespace:
    """Parse local model inspection arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-writable",
        action="store_true",
        help=(
            "Report a staging artifact before read-only sealing; "
            "protected inference still refuses it"
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Print the deterministic model artifact identity."""
    args = parse_args()
    result = inspect_local_model_artifact(
        args.model_dir,
        require_read_only=not args.allow_writable,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
