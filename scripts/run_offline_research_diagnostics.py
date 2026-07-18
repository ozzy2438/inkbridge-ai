"""Build a private SMHD failure atlas and aggregate calibration diagnostic."""

import argparse
from pathlib import Path

import structlog

from src.evaluation.research_diagnostics import build_offline_research_diagnostics

logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    """Parse local-only diagnostic arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-run-dir", type=Path, required=True)
    parser.add_argument("--test-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--minimum-calibration-samples", type=int, default=30)
    parser.add_argument("--minimum-policy-coverage", type=float, default=0.5)
    parser.add_argument("--maximum-selective-cer", type=float, default=0.1)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.9)
    return parser.parse_args()


def main() -> int:
    """Run the fail-closed private diagnostic."""
    args = parse_args()
    result = build_offline_research_diagnostics(
        validation_run_dir=args.validation_run_dir,
        test_run_dir=args.test_run_dir,
        output_dir=args.output_dir,
        repository_root=args.repository_root,
        minimum_calibration_samples=args.minimum_calibration_samples,
        minimum_policy_coverage=args.minimum_policy_coverage,
        maximum_selective_cer=args.maximum_selective_cer,
        high_confidence_threshold=args.high_confidence_threshold,
    )
    aggregate = result["aggregate"]
    logger.info(
        "offline_research_diagnostics.complete",
        output=result["output_dir"],
        failure_records=aggregate["failure_atlas"]["num_failure_records"],
        calibration_status=aggregate["calibration"]["fit"]["status"],
        operational_status=aggregate["operational_decision"]["status"],
        sample_level_artifacts_committed=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
