"""Build a licensed, writer-isolated dataset manifest from normalized labels."""

import argparse
from pathlib import Path

import structlog

from src.data.manifest import DatasetIdentity, build_manifest_records, write_split_manifest

logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    """Parse manifest builder arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--license-id", required=True)
    parser.add_argument("--license-url", required=True)
    parser.add_argument("--sample-type", choices=("line", "page"), required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-samples-per-writer", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    """Validate normalized data and write its reproducibility artifacts."""
    args = parse_args()
    identity = DatasetIdentity(
        name=args.dataset_name,
        version=args.dataset_version,
        license_id=args.license_id,
        license_url=args.license_url,
        sample_type=args.sample_type,
    )
    records = build_manifest_records(args.dataset_dir, identity)
    metadata = write_split_manifest(
        records,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        identity=identity,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_samples_per_writer=args.min_samples_per_writer,
    )
    logger.info(
        "dataset_manifest.complete",
        output=str(args.output_dir),
        samples=metadata["manifest"]["num_samples"],
        manifest_sha256=metadata["manifest"]["sha256"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
