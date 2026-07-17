"""Evaluate schema-validated, offline model predictions."""

import argparse
import json
import re
from pathlib import Path

import structlog

from src.evaluation.runner import (
    build_evaluation_artifact,
    evaluate_release_gate,
    load_evaluation_artifact,
    load_evaluation_config,
    load_evaluation_records,
)

logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate an offline JSONL prediction artifact against gold references"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--test-set", default="gold")
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation_config.yaml"))
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results"))
    parser.add_argument(
        "--enforce-release-gate",
        action="store_true",
        help="Exit non-zero when any release-gate check fails",
    )
    args = parser.parse_args()
    if args.enforce_release_gate and args.baseline_results is None:
        parser.error("--enforce-release-gate requires --baseline-results")
    return args


def main() -> int:
    """Run evaluation and persist a traceable JSON result."""
    args = parse_args()
    evaluation_config = load_evaluation_config(args.config)
    records = load_evaluation_records(args.predictions)
    artifact = build_evaluation_artifact(
        records,
        model_version=args.model_version,
        test_set=args.test_set,
        input_path=args.predictions,
        config_path=args.config,
    )

    if args.baseline_results is not None:
        baseline = load_evaluation_artifact(args.baseline_results)
        thresholds = evaluation_config.get("release_gate")
        if not isinstance(thresholds, dict):
            raise ValueError("Evaluation config must contain a 'release_gate' mapping")
        artifact["baseline"] = {
            "filename": args.baseline_results.name,
            "model_version": baseline.get("model_version"),
        }
        artifact["release_gate"] = evaluate_release_gate(artifact, baseline, thresholds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_model_version = _safe_filename_component(args.model_version)
    safe_test_set = _safe_filename_component(args.test_set)
    results_path = args.output_dir / f"eval_{safe_model_version}_{safe_test_set}.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")

    logger.info(
        "evaluation.complete",
        output=str(results_path),
        model_version=args.model_version,
        samples=len(records),
        release_gate=artifact.get("release_gate", {}).get("passed"),
    )
    if args.enforce_release_gate and not artifact["release_gate"]["passed"]:
        return 2
    return 0


def _safe_filename_component(value: str) -> str:
    """Prevent model or dataset labels from escaping the output directory."""
    component = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not component:
        raise ValueError("Model version and test-set names must contain a safe filename character")
    return component


if __name__ == "__main__":
    raise SystemExit(main())
