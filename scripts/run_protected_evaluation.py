"""Run aggregate-only OCR evaluation on an approved self-hosted machine."""

import argparse
from pathlib import Path

import structlog

from src.evaluation.protected_runner import RESULTS_DIRECTORY, run_protected_evaluation

logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    """Parse protected evaluation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation_config.yaml"))
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--enforce-release-gate", action="store_true")
    args = parser.parse_args()
    if args.enforce_release_gate and args.baseline_results is None:
        parser.error("--enforce-release-gate requires --baseline-results")
    return args


def main() -> int:
    """Evaluate reference-free predictions and write an aggregate-only result."""
    args = parse_args()
    output = args.output_dir or args.dataset_dir / RESULTS_DIRECTORY
    artifact, result_path = run_protected_evaluation(
        contract_path=args.contract,
        dataset_dir=args.dataset_dir,
        manifest_dir=args.manifest_dir,
        predictions_path=args.predictions,
        attestation_path=args.attestation,
        model_version=args.model_version,
        config_path=args.config,
        output_dir=output,
        baseline_results=args.baseline_results,
        enforce_release_gate=args.enforce_release_gate,
    )
    logger.info(
        "protected_evaluation.complete",
        evaluation_set_id=artifact["test_set"],
        model_version=artifact["model_version"],
        samples=artifact["input"]["num_records"],
        release_gate=artifact.get("release_gate", {}).get("passed"),
        output=str(result_path),
    )
    if args.enforce_release_gate and not artifact["release_gate"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
