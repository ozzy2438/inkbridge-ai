"""Writer-Independent Data Splitting.

CRITICAL: Train/test split MUST be writer-based, not image-based.

If the same writer's pages appear in both train and test:
- The model memorizes handwriting style
- Test metrics are artificially inflated
- Real-world performance will be worse

This module ensures strict writer isolation.
"""

import random
from collections import defaultdict

import structlog

logger = structlog.get_logger()


def create_writer_split(
    samples: list[dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    min_samples_per_writer: int = 3,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split dataset by writer ID, ensuring no writer leakage.

    Args:
        samples: List of {image_path, text, writer_id}
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility
        min_samples_per_writer: Minimum samples to include a writer

    Returns:
        (train_samples, val_samples, test_samples)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.seed(seed)

    # Group by writer
    writer_samples = defaultdict(list)
    for sample in samples:
        writer_id = sample.get("writer_id", "unknown")
        writer_samples[writer_id].append(sample)

    # Filter writers with too few samples
    valid_writers = [w for w, s in writer_samples.items() if len(s) >= min_samples_per_writer]

    # Shuffle writers
    random.shuffle(valid_writers)

    # Split writers
    n_writers = len(valid_writers)
    n_train = int(n_writers * train_ratio)
    n_val = int(n_writers * val_ratio)

    train_writers = set(valid_writers[:n_train])
    val_writers = set(valid_writers[n_train : n_train + n_val])
    test_writers = set(valid_writers[n_train + n_val :])

    # Assign samples
    train_samples = []
    val_samples = []
    test_samples = []

    for writer_id, writer_samps in writer_samples.items():
        if writer_id in train_writers:
            train_samples.extend(writer_samps)
        elif writer_id in val_writers:
            val_samples.extend(writer_samps)
        elif writer_id in test_writers:
            test_samples.extend(writer_samps)

    # Verify no leakage
    train_writer_set = set(s["writer_id"] for s in train_samples)
    val_writer_set = set(s["writer_id"] for s in val_samples)
    test_writer_set = set(s["writer_id"] for s in test_samples)

    assert len(train_writer_set & val_writer_set) == 0, "Writer leakage: train/val"
    assert len(train_writer_set & test_writer_set) == 0, "Writer leakage: train/test"
    assert len(val_writer_set & test_writer_set) == 0, "Writer leakage: val/test"

    logger.info(
        "writer_split.created",
        total_writers=n_writers,
        train_writers=len(train_writers),
        val_writers=len(val_writers),
        test_writers=len(test_writers),
        train_samples=len(train_samples),
        val_samples=len(val_samples),
        test_samples=len(test_samples),
    )

    return train_samples, val_samples, test_samples
