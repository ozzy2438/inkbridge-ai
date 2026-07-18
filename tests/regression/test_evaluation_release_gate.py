"""Regression coverage for evidence-backed model promotion gates."""

from copy import deepcopy
from pathlib import Path

from src.evaluation.runner import (
    build_evaluation_artifact,
    evaluate_release_gate,
    load_evaluation_config,
    load_evaluation_records,
)

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "evaluation"
CONFIG = ROOT / "configs" / "evaluation_config.yaml"


def _artifact(filename: str, model_version: str) -> dict:
    input_path = FIXTURES / filename
    return build_evaluation_artifact(
        load_evaluation_records(input_path),
        model_version=model_version,
        test_set="synthetic-regression",
        input_path=input_path,
        config_path=CONFIG,
    )


def test_improved_candidate_passes_release_gate() -> None:
    """A better, faster, cheaper candidate should pass every configured check."""
    baseline = _artifact("baseline.jsonl", "baseline-v1")
    candidate = _artifact("candidate.jsonl", "candidate-v1")
    thresholds = load_evaluation_config(CONFIG)["release_gate"]

    gate = evaluate_release_gate(candidate, baseline, thresholds)

    assert gate["passed"] is True
    assert all(check["passed"] for check in gate["checks"])


def test_slice_regression_blocks_release() -> None:
    """Aggregate gains must not hide a regression on a required slice."""
    baseline = _artifact("baseline.jsonl", "baseline-v1")
    candidate = deepcopy(_artifact("candidate.jsonl", "candidate-v1"))
    thresholds = load_evaluation_config(CONFIG)["release_gate"]
    candidate["slices"]["faint_pencil"]["cer"] = (
        baseline["slices"]["faint_pencil"]["cer"] + thresholds["max_slice_regression"] + 0.001
    )

    gate = evaluate_release_gate(candidate, baseline, thresholds)
    slice_check = next(check for check in gate["checks"] if check["name"] == "slice_regression")

    assert gate["passed"] is False
    assert slice_check["passed"] is False


def test_different_evaluation_set_blocks_release() -> None:
    """Candidate and baseline metrics are incomparable when gold references differ."""
    baseline = _artifact("baseline.jsonl", "baseline-v1")
    candidate = deepcopy(_artifact("candidate.jsonl", "candidate-v1"))
    thresholds = load_evaluation_config(CONFIG)["release_gate"]
    candidate["input"]["evaluation_set_sha256"] = "different"

    gate = evaluate_release_gate(candidate, baseline, thresholds)
    set_check = next(check for check in gate["checks"] if check["name"] == "evaluation_set_match")

    assert gate["passed"] is False
    assert set_check["passed"] is False


def test_missing_baseline_slices_blocks_release() -> None:
    """A slice gate must not pass vacuously when no required slices were defined."""
    baseline = deepcopy(_artifact("baseline.jsonl", "baseline-v1"))
    candidate = deepcopy(_artifact("candidate.jsonl", "candidate-v1"))
    thresholds = load_evaluation_config(CONFIG)["release_gate"]
    baseline["slices"] = {}
    candidate["slices"] = {}

    gate = evaluate_release_gate(candidate, baseline, thresholds)
    slice_check = next(check for check in gate["checks"] if check["name"] == "slice_regression")

    assert gate["passed"] is False
    assert slice_check["required_slices_present"] is False
