"""Prepare the pinned SMHD student-handwriting offline research subset."""

import argparse
from pathlib import Path

import structlog

from src.data.smhd_research_subset import prepare_smhd_research_subset

logger = structlog.get_logger()
DEFAULT_SPEC = Path("configs/datasets/smhd_research_rehearsal.json")


def parse_args() -> argparse.Namespace:
    """Parse local SMHD subset preparation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> int:
    """Prepare the private subset and emit only aggregate provenance."""
    args = parse_args()
    metadata = prepare_smhd_research_subset(
        args.spec,
        args.archive,
        args.output_dir,
        repository_root=args.repository_root,
    )
    logger.info(
        "smhd_research_subset.complete",
        output=str(args.output_dir),
        samples=metadata["output"]["num_samples"],
        writers=metadata["output"]["num_writers"],
        doi=metadata["dataset"]["doi"],
        license_id=metadata["dataset"]["license_id"],
        labels_sha256=metadata["output"]["labels_sha256"],
        gold_ready=metadata["reference_quality"]["gold_ready"],
        production_pilot_evidence=metadata["evidence_boundary"]["production_pilot_evidence"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
