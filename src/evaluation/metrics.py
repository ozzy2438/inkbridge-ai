"""Evaluation Metrics for Handwriting Recognition.

Computes:
- Character Error Rate (CER)
- Word Error Rate (WER)
- Normalized Edit Distance
- Calibration Error
- False Confidence Rate
- Per-slice metrics
"""

import numpy as np
import structlog

logger = structlog.get_logger()


def character_error_rate(predictions: list[str], references: list[str]) -> float:
    """Compute Character Error Rate.

    CER = (substitutions + insertions + deletions) / reference_length
    """
    import jiwer

    # Handle empty cases
    if not predictions or not references:
        return 1.0

    # jiwer computes CER using edit distance
    cer = jiwer.cer(references, predictions)
    return float(cer)


def word_error_rate(predictions: list[str], references: list[str]) -> float:
    """Compute Word Error Rate."""
    import jiwer

    if not predictions or not references:
        return 1.0

    wer = jiwer.wer(references, predictions)
    return float(wer)


def normalized_edit_distance(predictions: list[str], references: list[str]) -> float:
    """Compute normalized edit distance (average over samples)."""
    if not predictions or not references:
        return 1.0

    distances = []
    for pred, ref in zip(predictions, references):
        dist = _edit_distance(pred, ref)
        max_len = max(len(pred), len(ref), 1)
        distances.append(dist / max_len)

    return float(np.mean(distances))


def _edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calibration_error(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Measures how well confidence scores correlate with actual accuracy.
    A well-calibrated model: when it says 80% confidence, it's correct 80% of the time.
    """
    if not confidences:
        return 0.0

    confs = np.array(confidences)
    accs = np.array(correct, dtype=float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue

        bin_conf = confs[mask].mean()
        bin_acc = accs[mask].mean()
        bin_weight = mask.sum() / len(confs)

        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def false_confidence_rate(
    predictions: list[str],
    references: list[str],
    confidences: list[float],
    threshold: float = 0.9,
) -> float:
    """Rate of high-confidence predictions that are actually wrong.

    This is critical for production: the system should not be confidently wrong.
    """
    high_conf_mask = np.array(confidences) >= threshold

    if high_conf_mask.sum() == 0:
        return 0.0

    # Check which high-confidence predictions are wrong
    wrong_count = 0
    total_high_conf = 0

    for pred, ref, is_high_conf in zip(predictions, references, high_conf_mask):
        if is_high_conf:
            total_high_conf += 1
            if pred.strip() != ref.strip():
                wrong_count += 1

    return wrong_count / total_high_conf if total_high_conf > 0 else 0.0


def abstention_precision(
    predictions: list[str],
    references: list[str],
    confidences: list[float],
    abstention_threshold: float = 0.5,
) -> dict:
    """Evaluate the quality of abstention decisions.

    Returns:
    - abstention_rate: fraction of items abstained
    - correct_abstentions: abstained items that were actually wrong
    - unnecessary_abstentions: abstained items that were actually correct
    """
    abstained = np.array(confidences) < abstention_threshold

    if abstained.sum() == 0:
        return {
            "abstention_rate": 0.0,
            "correct_abstentions": 0.0,
            "unnecessary_abstentions": 0.0,
        }

    actually_wrong = []
    for pred, ref in zip(predictions, references):
        actually_wrong.append(pred.strip() != ref.strip())

    actually_wrong = np.array(actually_wrong)

    abstention_rate = abstained.mean()
    correct_abstentions = (abstained & actually_wrong).sum() / abstained.sum()
    unnecessary_abstentions = (abstained & ~actually_wrong).sum() / abstained.sum()

    return {
        "abstention_rate": float(abstention_rate),
        "correct_abstentions": float(correct_abstentions),
        "unnecessary_abstentions": float(unnecessary_abstentions),
    }


def compute_full_metrics(
    predictions: list[str],
    references: list[str],
    confidences: list[float] | None = None,
) -> dict:
    """Compute all evaluation metrics."""
    metrics: dict[str, object] = {
        "cer": character_error_rate(predictions, references),
        "wer": word_error_rate(predictions, references),
        "normalized_edit_distance": normalized_edit_distance(predictions, references),
        "num_samples": len(predictions),
    }

    if confidences:
        # Check correctness for calibration metrics
        correct = [pred.strip() == ref.strip() for pred, ref in zip(predictions, references)]

        metrics["calibration_error"] = calibration_error(confidences, correct)
        metrics["false_confidence_rate"] = false_confidence_rate(
            predictions, references, confidences
        )
        metrics["abstention_metrics"] = abstention_precision(predictions, references, confidences)
        metrics["mean_confidence"] = float(np.mean(confidences))

    return metrics
