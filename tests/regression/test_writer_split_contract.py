"""Regression tests for deterministic writer-independent splitting."""

import pytest

from src.data.writer_split import create_writer_split


def _samples(writer_count: int = 9, samples_per_writer: int = 3) -> list[dict[str, str]]:
    return [
        {"sample_id": f"writer-{writer}-sample-{sample}", "writer_id": f"writer-{writer}"}
        for writer in range(writer_count)
        for sample in range(samples_per_writer)
    ]


def test_writer_split_is_deterministic_and_leakage_free() -> None:
    """The same seed must reproduce assignments with no writer overlap."""
    first = create_writer_split(_samples(), seed=123)
    second = create_writer_split(_samples(), seed=123)

    assert [[record["sample_id"] for record in split] for split in first] == [
        [record["sample_id"] for record in split] for split in second
    ]
    writer_sets = [{record["writer_id"] for record in split} for split in first]
    assert writer_sets[0].isdisjoint(writer_sets[1])
    assert writer_sets[0].isdisjoint(writer_sets[2])
    assert writer_sets[1].isdisjoint(writer_sets[2])
    assert all(writer_sets)
    assert [len(writer_set) for writer_set in writer_sets] == [6, 2, 1]


def test_writer_split_rejects_silent_sample_loss() -> None:
    """Undersized writers should stop the build instead of disappearing from metrics."""
    with pytest.raises(ValueError, match="would be dropped"):
        create_writer_split(_samples(samples_per_writer=2), min_samples_per_writer=3)
