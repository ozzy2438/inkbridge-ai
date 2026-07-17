"""Deterministic writer-independent dataset splitting."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any

import structlog

logger = structlog.get_logger()


def create_writer_split(
    samples: list[dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    min_samples_per_writer: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by writer while rejecting leakage-prone or silently dropped data."""
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("train_ratio, val_ratio, and test_ratio must be non-negative and sum to 1")
    if min_samples_per_writer < 1:
        raise ValueError("min_samples_per_writer must be at least 1")
    if not samples:
        raise ValueError("At least one sample is required for writer splitting")

    writer_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        writer_id = sample.get("writer_id")
        if not isinstance(writer_id, str) or not writer_id.strip():
            raise ValueError(f"Sample {index} is missing a non-empty writer_id")
        writer_samples[writer_id].append(sample)

    undersized = {
        writer_id: len(writer_records)
        for writer_id, writer_records in writer_samples.items()
        if len(writer_records) < min_samples_per_writer
    }
    if undersized:
        distribution = Counter(undersized.values())
        raise ValueError(
            "Writers below min_samples_per_writer would be dropped; "
            f"count_by_sample_size={dict(sorted(distribution.items()))}"
        )

    writers = sorted(writer_samples)
    nonzero_split_count = sum(ratio > 0 for ratio in ratios)
    if len(writers) < nonzero_split_count:
        raise ValueError(
            f"Need at least {nonzero_split_count} writers for non-empty requested splits"
        )

    random.Random(seed).shuffle(writers)
    train_count, val_count, _ = _allocate_counts(len(writers), ratios)
    train_writers = set(writers[:train_count])
    val_writers = set(writers[train_count : train_count + val_count])
    test_writers = set(writers[train_count + val_count :])

    train_samples = _records_for_writers(writer_samples, train_writers)
    val_samples = _records_for_writers(writer_samples, val_writers)
    test_samples = _records_for_writers(writer_samples, test_writers)
    _assert_writer_isolation(train_samples, val_samples, test_samples)

    logger.info(
        "writer_split.created",
        total_writers=len(writers),
        train_writers=len(train_writers),
        val_writers=len(val_writers),
        test_writers=len(test_writers),
        train_samples=len(train_samples),
        val_samples=len(val_samples),
        test_samples=len(test_samples),
        seed=seed,
    )
    return train_samples, val_samples, test_samples


def _allocate_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Allocate writers with non-empty positive-ratio splits and deterministic remainders."""
    raw_counts = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw_counts]
    unallocated = total - sum(counts)
    remainder_order = sorted(
        range(len(ratios)),
        key=lambda index: (raw_counts[index] - counts[index], ratios[index], -index),
        reverse=True,
    )
    for index in remainder_order[:unallocated]:
        counts[index] += 1

    for index, ratio in enumerate(ratios):
        if ratio == 0 or counts[index] > 0:
            continue
        donors = [candidate for candidate, count in enumerate(counts) if count > 1]
        if not donors:
            raise RuntimeError("Unable to allocate a writer to every positive-ratio split")
        donor = max(
            donors,
            key=lambda candidate: (
                counts[candidate] - raw_counts[candidate],
                counts[candidate],
                ratios[candidate],
                -candidate,
            ),
        )
        counts[donor] -= 1
        counts[index] += 1

    return counts[0], counts[1], counts[2]


def _records_for_writers(
    writer_samples: dict[str, list[dict[str, Any]]], writer_ids: set[str]
) -> list[dict[str, Any]]:
    return [
        sample
        for writer_id in sorted(writer_ids)
        for sample in sorted(
            writer_samples[writer_id], key=lambda value: str(value.get("sample_id", ""))
        )
    ]


def _assert_writer_isolation(
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    test_samples: list[dict[str, Any]],
) -> None:
    writer_sets = [
        {str(sample["writer_id"]) for sample in split}
        for split in (train_samples, val_samples, test_samples)
    ]
    if writer_sets[0] & writer_sets[1]:
        raise RuntimeError("Writer leakage detected between train and validation")
    if writer_sets[0] & writer_sets[2]:
        raise RuntimeError("Writer leakage detected between train and test")
    if writer_sets[1] & writer_sets[2]:
        raise RuntimeError("Writer leakage detected between validation and test")
