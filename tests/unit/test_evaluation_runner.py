"""Unit tests for schema-validated evaluation artifacts."""

from pathlib import Path

import pytest

from src.evaluation.runner import (
    build_evaluation_artifact,
    load_evaluation_records,
    sha256_file,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "evaluation"
CONFIG = Path(__file__).parents[2] / "configs" / "evaluation_config.yaml"


def test_evaluation_artifact_is_traceable_and_sliced() -> None:
    """Metrics should retain input identity and edge-case slices."""
    input_path = FIXTURES / "candidate.jsonl"
    records = load_evaluation_records(input_path)
    artifact = build_evaluation_artifact(
        records,
        model_version="candidate-v1",
        test_set="synthetic-regression",
        input_path=input_path,
        config_path=CONFIG,
    )

    assert artifact["status"] == "evaluated"
    assert artifact["input"]["num_records"] == 3
    assert artifact["input"]["sha256"] == sha256_file(input_path)
    assert len(artifact["input"]["evaluation_set_sha256"]) == 64
    assert artifact["metrics"]["cer"] > 0
    assert artifact["metrics"]["false_confidence_rate"] == 0
    assert set(artifact["slices"]) == {"clean", "cursive", "faint_pencil", "print"}


def test_partial_optional_metric_is_rejected(tmp_path: Path) -> None:
    """Partial latency coverage must not produce a biased operational metric."""
    input_path = tmp_path / "partial.jsonl"
    input_path.write_text(
        '{"sample_id":"one","source_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","reference":"a","prediction":"a","latency_ms":10}\n'
        '{"sample_id":"two","source_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","reference":"b","prediction":"b"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="latency_ms.*every record"):
        load_evaluation_records(input_path)


def test_duplicate_sample_id_is_rejected(tmp_path: Path) -> None:
    """Duplicate IDs would invalidate paired baseline comparisons."""
    input_path = tmp_path / "duplicate.jsonl"
    input_path.write_text(
        '{"sample_id":"same","source_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","reference":"a","prediction":"a"}\n'
        '{"sample_id":"same","source_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","reference":"b","prediction":"b"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_evaluation_records(input_path)
