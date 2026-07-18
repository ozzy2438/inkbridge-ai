"""Human review endpoints for uncertain predictions."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ReviewItem(BaseModel):
    """A single item requiring human review."""

    item_id: str
    page_id: str
    job_id: str
    image_url: str
    predicted_text: str
    confidence: float
    bbox: list[int]
    context_before: str | None = None
    context_after: str | None = None
    model_version: str
    failure_category: str | None = None


class ReviewSubmission(BaseModel):
    """Human reviewer's correction."""

    item_id: str
    corrected_text: str
    is_unreadable: bool = False
    is_crossed_out: bool = False
    reviewer_notes: str | None = None


@router.get("/review/queue")
async def get_review_queue(
    job_id: str | None = None,
    limit: int = 20,
    priority: str = "uncertainty",
):
    """Get items needing human review, prioritized by uncertainty.

    Priority modes:
    - uncertainty: Lowest confidence first
    - impact: Highest operational impact first
    - active_learning: Best for model improvement
    """
    # TODO: Implement review queue logic
    return {"items": [], "total_pending": 0, "priority_mode": priority}


@router.post("/review/submit")
async def submit_review(submission: ReviewSubmission):
    """Submit a human review correction.

    The correction is:
    1. Stored in the audit trail
    2. Applied to the transcript
    3. Queued for active learning (model retraining)
    """
    # TODO: Implement review submission
    return {
        "status": "accepted",
        "item_id": submission.item_id,
        "queued_for_training": True,
    }


@router.get("/review/stats")
async def get_review_stats(job_id: str | None = None):
    """Get review operation statistics."""
    return {
        "total_reviewed": 0,
        "avg_review_time_seconds": 0,
        "corrections_rate": 0,
        "agreement_rate": 0,
        "adjudication_rate": 0,
        "pages_per_hour": 0,
    }
