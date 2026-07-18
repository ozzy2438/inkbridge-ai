"""Reproducible evaluation of offline handwriting-model predictions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.evaluation.metrics import compute_full_metrics


@dataclass(frozen=True)
class EvaluationRecord:
    """One schema-validated model prediction and its gold reference."""

    sample_id: str
    source_sha256: str
    reference: str
    prediction: str
    confidence: float | None = None
    slices: tuple[str, ...] = ()
    latency_ms: float | None = None
    cost_per_page_usd: float | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any], line_number: int) -> EvaluationRecord:
        """Validate and construct a record from a decoded JSON object."""
        prefix = f"line {line_number}"
        sample_id = _required_string(value, "sample_id", prefix)
        source_sha256 = _required_sha256(value, "source_sha256", prefix)
        reference = _required_string(value, "reference", prefix, allow_empty=True)
        prediction = _required_string(value, "prediction", prefix, allow_empty=True)
        confidence = _optional_number(value, "confidence", prefix, minimum=0.0, maximum=1.0)
        latency_ms = _optional_number(value, "latency_ms", prefix, minimum=0.0)
        cost_per_page_usd = _optional_number(value, "cost_per_page_usd", prefix, minimum=0.0)

        raw_slices = value.get("slices", [])
        if not isinstance(raw_slices, list) or any(
            not isinstance(item, str) or not item.strip() for item in raw_slices
        ):
            raise ValueError(f"{prefix}: 'slices' must be a list of non-empty strings")

        return cls(
            sample_id=sample_id,
            source_sha256=source_sha256,
            reference=reference,
            prediction=prediction,
            confidence=confidence,
            slices=tuple(sorted({item.strip() for item in raw_slices})),
            latency_ms=latency_ms,
            cost_per_page_usd=cost_per_page_usd,
        )


def load_evaluation_records(path: str | Path) -> list[EvaluationRecord]:
    """Load strict JSONL records and reject incomplete or ambiguous inputs."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Evaluation input does not exist: {input_path}")

    records: list[EvaluationRecord] = []
    seen_ids: set[str] = set()
    with input_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}: each JSONL value must be an object")

            record = EvaluationRecord.from_mapping(value, line_number)
            if record.sample_id in seen_ids:
                raise ValueError(f"line {line_number}: duplicate sample_id '{record.sample_id}'")
            seen_ids.add(record.sample_id)
            records.append(record)

    if not records:
        raise ValueError("Evaluation input contains no records")

    for field_name in ("confidence", "latency_ms", "cost_per_page_usd"):
        populated = [getattr(record, field_name) is not None for record in records]
        if any(populated) and not all(populated):
            raise ValueError(
                f"Optional field '{field_name}' must be present for every record or none"
            )

    return records


def evaluate_records(records: list[EvaluationRecord]) -> dict[str, Any]:
    """Compute overall, operational, and per-slice metrics."""
    if not records:
        raise ValueError("At least one evaluation record is required")

    predictions = [record.prediction for record in records]
    references = [record.reference for record in records]
    confidences = _complete_optional_values(records, "confidence")
    metrics = compute_full_metrics(predictions, references, confidences)
    metrics["valid_output_rate"] = 1.0

    latencies = _complete_optional_values(records, "latency_ms")
    if latencies is not None:
        metrics.update(
            {
                "p50_latency_ms": float(np.percentile(latencies, 50)),
                "p95_latency_ms": float(np.percentile(latencies, 95)),
                "samples_per_minute": float(60_000 / np.mean(latencies))
                if np.mean(latencies) > 0
                else None,
            }
        )

    costs = _complete_optional_values(records, "cost_per_page_usd")
    if costs is not None:
        metrics["cost_per_page_usd"] = float(np.mean(costs))

    slice_names = sorted({slice_name for record in records for slice_name in record.slices})
    slice_metrics: dict[str, dict[str, Any]] = {}
    for slice_name in slice_names:
        subset = [record for record in records if slice_name in record.slices]
        subset_confidences = _complete_optional_values(subset, "confidence")
        slice_metrics[slice_name] = compute_full_metrics(
            [record.prediction for record in subset],
            [record.reference for record in subset],
            subset_confidences,
        )

    return {"metrics": metrics, "slices": slice_metrics}


def build_evaluation_artifact(
    records: list[EvaluationRecord],
    *,
    model_version: str,
    test_set: str,
    input_path: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Build a traceable evaluation result without inventing missing measurements."""
    evaluation = evaluate_records(records)
    return {
        "schema_version": 1,
        "status": "evaluated",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "test_set": test_set,
        "input": {
            "filename": Path(input_path).name,
            "sha256": sha256_file(input_path),
            "evaluation_set_sha256": evaluation_set_sha256(records),
            "num_records": len(records),
        },
        "config": {
            "filename": Path(config_path).name,
            "sha256": sha256_file(config_path),
        },
        **evaluation,
    }


def load_evaluation_config(path: str | Path) -> dict[str, Any]:
    """Load the evaluation YAML and return its evaluation section."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("evaluation"), dict):
        raise ValueError("Evaluation config must contain an 'evaluation' mapping")
    return dict(config["evaluation"])


def load_evaluation_artifact(path: str | Path) -> dict[str, Any]:
    """Load a previously generated baseline evaluation artifact."""
    with Path(path).open(encoding="utf-8") as handle:
        artifact = json.load(handle)
    if not isinstance(artifact, dict):
        raise ValueError("Baseline result must be a JSON object")
    return artifact


def evaluate_release_gate(
    candidate: dict[str, Any], baseline: dict[str, Any], thresholds: dict[str, Any]
) -> dict[str, Any]:
    """Compare a candidate artifact with a baseline using explicit quality gates."""
    candidate_metrics = _required_mapping(candidate, "metrics")
    baseline_metrics = _required_mapping(baseline, "metrics")
    checks: list[dict[str, Any]] = []

    candidate_input = _required_mapping(candidate, "input")
    baseline_input = _required_mapping(baseline, "input")
    candidate_set_hash = candidate_input.get("evaluation_set_sha256")
    baseline_set_hash = baseline_input.get("evaluation_set_sha256")
    same_evaluation_set = (
        isinstance(candidate_set_hash, str)
        and candidate_set_hash == baseline_set_hash
        and candidate.get("test_set") == baseline.get("test_set")
    )
    checks.append(
        {
            "name": "evaluation_set_match",
            "actual": {
                "test_set": candidate.get("test_set"),
                "evaluation_set_sha256": candidate_set_hash,
            },
            "expected": {
                "test_set": baseline.get("test_set"),
                "evaluation_set_sha256": baseline_set_hash,
            },
            "passed": same_evaluation_set,
        }
    )

    candidate_cer = _required_metric(candidate_metrics, "cer")
    baseline_cer = _required_metric(baseline_metrics, "cer")
    min_improvement = _required_threshold(thresholds, "min_cer_improvement")
    improvement = baseline_cer - candidate_cer
    checks.append(
        _gate_check(
            "cer_improvement",
            improvement,
            f">= {min_improvement}",
            improvement >= min_improvement,
        )
    )

    candidate_slices = _required_mapping(candidate, "slices")
    baseline_slices = _required_mapping(baseline, "slices")
    max_slice_regression = _required_threshold(thresholds, "max_slice_regression")
    missing_slices = sorted(set(baseline_slices) - set(candidate_slices))
    regressions: dict[str, float] = {}
    for slice_name in sorted(set(candidate_slices) & set(baseline_slices)):
        candidate_slice = _required_mapping(candidate_slices, slice_name)
        baseline_slice = _required_mapping(baseline_slices, slice_name)
        regressions[slice_name] = _required_metric(candidate_slice, "cer") - _required_metric(
            baseline_slice, "cer"
        )
    worst_regression = max(regressions.values(), default=0.0)
    has_required_slices = bool(baseline_slices)
    checks.append(
        {
            **_gate_check(
                "slice_regression",
                worst_regression,
                f"<= {max_slice_regression}",
                has_required_slices
                and not missing_slices
                and worst_regression <= max_slice_regression,
            ),
            "required_slices_present": has_required_slices,
            "missing_slices": missing_slices,
            "regressions": regressions,
        }
    )

    _append_upper_bound_check(
        checks,
        candidate_metrics,
        thresholds,
        metric="false_confidence_rate",
        threshold="max_false_confidence_rate",
    )
    _append_upper_bound_check(
        checks,
        candidate_metrics,
        thresholds,
        metric="p95_latency_ms",
        threshold="max_p95_latency_ms",
    )
    _append_upper_bound_check(
        checks,
        candidate_metrics,
        thresholds,
        metric="cost_per_page_usd",
        threshold="max_cost_per_page_usd",
    )

    valid_output_rate = _required_metric(candidate_metrics, "valid_output_rate")
    minimum_valid_rate = _required_threshold(thresholds, "min_valid_output_rate")
    checks.append(
        _gate_check(
            "valid_output_rate",
            valid_output_rate,
            f">= {minimum_valid_rate}",
            valid_output_rate >= minimum_valid_rate,
        )
    )

    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 hash for an input artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluation_set_sha256(records: list[EvaluationRecord]) -> str:
    """Hash sample identity, references, and slices independently of predictions."""
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.sample_id):
        canonical = json.dumps(
            {
                "sample_id": record.sample_id,
                "source_sha256": record.source_sha256,
                "reference": record.reference,
                "slices": record.slices,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _required_string(
    value: dict[str, Any], key: str, prefix: str, *, allow_empty: bool = False
) -> str:
    item = value.get(key)
    if not isinstance(item, str) or (not allow_empty and not item.strip()):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{prefix}: '{key}' must be {qualifier}")
    return item


def _optional_number(
    value: dict[str, Any],
    key: str,
    prefix: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    item = value.get(key)
    if item is None:
        return None
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise ValueError(f"{prefix}: '{key}' must be numeric")
    number = float(item)
    if not np.isfinite(number):
        raise ValueError(f"{prefix}: '{key}' must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{prefix}: '{key}' must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{prefix}: '{key}' must be <= {maximum}")
    return number


def _required_sha256(value: dict[str, Any], key: str, prefix: str) -> str:
    item = _required_string(value, key, prefix).lower()
    if len(item) != 64 or any(character not in "0123456789abcdef" for character in item):
        raise ValueError(f"{prefix}: '{key}' must be a 64-character SHA-256 hex digest")
    return item


def _complete_optional_values(
    records: list[EvaluationRecord], field_name: str
) -> list[float] | None:
    values = [getattr(record, field_name) for record in records]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"Optional field '{field_name}' is incomplete")
    return [float(value) for value in values if value is not None]


def _required_mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"Evaluation artifact is missing mapping '{key}'")
    return item


def _required_metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Evaluation artifact is missing numeric metric '{key}'")
    return float(value)


def _required_threshold(thresholds: dict[str, Any], key: str) -> float:
    value = thresholds.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Release gate is missing numeric threshold '{key}'")
    return float(value)


def _gate_check(name: str, actual: float, expected: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "actual": actual, "expected": expected, "passed": passed}


def _append_upper_bound_check(
    checks: list[dict[str, Any]],
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
    *,
    metric: str,
    threshold: str,
) -> None:
    actual = _required_metric(metrics, metric)
    maximum = _required_threshold(thresholds, threshold)
    checks.append(_gate_check(metric, actual, f"<= {maximum}", actual <= maximum))
