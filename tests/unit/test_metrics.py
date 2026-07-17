"""Unit tests for evaluation metrics."""

import pytest

from src.evaluation.metrics import (
    calibration_error,
    character_error_rate,
    compute_full_metrics,
    false_confidence_rate,
    word_error_rate,
)


def test_cer_perfect():
    """Perfect predictions should have 0 CER."""
    preds = ["hello world", "test string"]
    refs = ["hello world", "test string"]
    assert character_error_rate(preds, refs) == 0.0


def test_cer_complete_mismatch():
    """Completely wrong predictions should have high CER."""
    preds = ["xxxxx"]
    refs = ["hello"]
    cer = character_error_rate(preds, refs)
    assert cer > 0.5


def test_wer_perfect():
    """Perfect predictions should have 0 WER."""
    preds = ["hello world"]
    refs = ["hello world"]
    assert word_error_rate(preds, refs) == 0.0


def test_false_confidence_rate_no_errors():
    """No errors in high-confidence predictions = 0 FCR."""
    preds = ["correct", "also correct"]
    refs = ["correct", "also correct"]
    confs = [0.95, 0.92]
    assert false_confidence_rate(preds, refs, confs) == 0.0


def test_false_confidence_rate_with_errors():
    """Confident wrong predictions should increase FCR."""
    preds = ["wrong", "correct"]
    refs = ["right", "correct"]
    confs = [0.95, 0.92]
    fcr = false_confidence_rate(preds, refs, confs)
    assert fcr == 0.5  # 1 out of 2 high-conf predictions is wrong


def test_compute_full_metrics():
    """Full metrics computation should return all expected keys."""
    preds = ["hello world"]
    refs = ["hello world"]
    metrics = compute_full_metrics(preds, refs)

    assert "cer" in metrics
    assert "wer" in metrics
    assert "normalized_edit_distance" in metrics
    assert metrics["cer"] == 0.0


def test_metric_length_mismatch_is_rejected():
    """Parallel metric inputs must never be silently truncated by zip."""
    with pytest.raises(ValueError, match="equal lengths"):
        character_error_rate(["one", "two"], ["one"])


def test_zero_confidence_is_included_in_calibration():
    """The first ECE bin must include confidence exactly equal to zero."""
    assert calibration_error([0.0], [True]) == 1.0
