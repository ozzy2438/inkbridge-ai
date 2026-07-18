"""Active Learning Loop.

Implements priority-based active learning for efficient annotation:

Priority = error_risk × frequency × operational_impact ÷ label_time

The loop:
1. Model makes predictions on unlabeled data
2. Predictions scored by uncertainty
3. High-uncertainty samples prioritized for annotation
4. Annotations feed back into training
5. Model retrained on expanded dataset
6. Repeat
"""

from dataclasses import dataclass

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class ActiveLearningItem:
    """An item in the active learning queue."""

    item_id: str
    image_path: str
    predicted_text: str
    confidence: float
    priority_score: float
    failure_category: str | None = None
    estimated_label_time_seconds: float = 10.0
    page_type: str = "unknown"


class ActiveLearningManager:
    """Manages the active learning loop for continuous model improvement.

    Priority scoring considers:
    - Model uncertainty (low confidence = high priority)
    - Error frequency (common failure types)
    - Operational impact (correction time impact)
    - Label cost (time to annotate)
    """

    def __init__(
        self,
        uncertainty_weight: float = 0.4,
        frequency_weight: float = 0.2,
        impact_weight: float = 0.3,
        cost_weight: float = 0.1,
        min_confidence_for_auto: float = 0.95,
        max_queue_size: int = 1000,
    ):
        self.uncertainty_weight = uncertainty_weight
        self.frequency_weight = frequency_weight
        self.impact_weight = impact_weight
        self.cost_weight = cost_weight
        self.min_confidence_for_auto = min_confidence_for_auto
        self.max_queue_size = max_queue_size

        self.queue: list[ActiveLearningItem] = []
        self.failure_frequencies: dict[str, float] = {}

    def score_item(
        self,
        item_id: str,
        image_path: str,
        predicted_text: str,
        confidence: float,
        failure_category: str | None = None,
        estimated_correction_impact: float = 1.0,
        estimated_label_time: float = 10.0,
    ) -> ActiveLearningItem:
        """Score an item for active learning priority.

        Args:
            item_id: Unique identifier
            image_path: Path to the image
            predicted_text: Model's prediction
            confidence: Model confidence (0-1)
            failure_category: Type of potential failure
            estimated_correction_impact: How much correction time this saves
            estimated_label_time: Expected annotation time in seconds

        Returns:
            ActiveLearningItem with computed priority score
        """
        # Uncertainty score (inverted confidence)
        uncertainty = 1.0 - confidence

        # Frequency score (how common is this failure type)
        frequency = self.failure_frequencies.get(failure_category or "unknown", 0.5)

        # Impact score (normalized)
        impact = min(1.0, estimated_correction_impact / 60.0)  # Normalize to 1 min

        # Cost efficiency (inverse of label time, normalized)
        cost_efficiency = 1.0 / max(1.0, estimated_label_time / 10.0)

        # Weighted priority
        priority = (
            self.uncertainty_weight * uncertainty
            + self.frequency_weight * frequency
            + self.impact_weight * impact
            + self.cost_weight * cost_efficiency
        )

        item = ActiveLearningItem(
            item_id=item_id,
            image_path=image_path,
            predicted_text=predicted_text,
            confidence=confidence,
            priority_score=priority,
            failure_category=failure_category,
            estimated_label_time_seconds=estimated_label_time,
        )

        return item

    def add_to_queue(self, item: ActiveLearningItem):
        """Add an item to the active learning queue."""
        # Don't queue high-confidence items
        if item.confidence >= self.min_confidence_for_auto:
            return

        self.queue.append(item)

        # Sort by priority (highest first)
        self.queue.sort(key=lambda x: x.priority_score, reverse=True)

        # Trim queue to max size
        if len(self.queue) > self.max_queue_size:
            self.queue = self.queue[: self.max_queue_size]

    def get_next_batch(self, batch_size: int = 20) -> list[ActiveLearningItem]:
        """Get next batch of items for annotation."""
        batch = self.queue[:batch_size]
        self.queue = self.queue[batch_size:]
        return batch

    def update_failure_frequencies(self, corrections: list[dict]):
        """Update failure type frequencies from corrections."""
        category_counts: dict[str, int] = {}
        total = len(corrections)

        for correction in corrections:
            cat = correction.get("failure_category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        if total > 0:
            for cat, count in category_counts.items():
                self.failure_frequencies[cat] = count / total

    def get_queue_stats(self) -> dict:
        """Get statistics about the current queue."""
        if not self.queue:
            return {"queue_size": 0}

        priorities = [item.priority_score for item in self.queue]
        confidences = [item.confidence for item in self.queue]

        return {
            "queue_size": len(self.queue),
            "avg_priority": float(np.mean(priorities)),
            "max_priority": float(np.max(priorities)),
            "avg_confidence": float(np.mean(confidences)),
            "failure_categories": dict(
                sorted(
                    self.failure_frequencies.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),
        }
