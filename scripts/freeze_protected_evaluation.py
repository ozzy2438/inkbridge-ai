"""Freeze a writer-isolated evaluation manifest inside an approved protected pilot."""

import argparse
from pathlib import Path

import structlog

from src.data.protected_manifest import MANIFEST_DIRECTORY, freeze_protected_evaluation

logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    """Parse protected-manifest arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=f"Defaults to <dataset-dir>/{MANIFEST_DIRECTORY}",
    )
    return parser.parse_args()


def main() -> int:
    """Run the promotion gate without exporting protected content."""
    args = parse_args()
    output = args.output_dir or args.dataset_dir / MANIFEST_DIRECTORY
    metadata = freeze_protected_evaluation(args.contract, args.dataset_dir, output)
    logger.info(
        "protected_evaluation.frozen",
        evaluation_set_id=metadata["evaluation_set_id"],
        samples=metadata["manifest"]["num_samples"],
        validation_writers=metadata["split"]["writer_counts"]["validation"],
        test_writers=metadata["split"]["writer_counts"]["test"],
        gold_ready=metadata["gold_ready"],
        output=str(output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
