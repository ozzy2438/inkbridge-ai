"""Prepare the pinned CSAFE real-handwriting engineering smoke subset."""

import argparse
from pathlib import Path

import structlog

from src.data.csafe_smoke_subset import prepare_csafe_smoke_subset

logger = structlog.get_logger()
DEFAULT_SPEC = Path("configs/datasets/csafe_real_handwriting_smoke.json")


def parse_args() -> argparse.Namespace:
    """Parse CSAFE smoke subset preparation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Prepare the subset and emit its immutable provenance summary."""
    args = parse_args()
    metadata = prepare_csafe_smoke_subset(args.spec, args.output_dir)
    logger.info(
        "csafe_smoke_subset.complete",
        output=str(args.output_dir),
        samples=metadata["output"]["num_samples"],
        writers=metadata["output"]["num_writers"],
        doi=metadata["dataset"]["doi"],
        labels_sha256=metadata["output"]["labels_sha256"],
        gold_ready=metadata["reference_quality"]["gold_ready"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
