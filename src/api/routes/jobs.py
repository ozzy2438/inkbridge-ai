"""Job management endpoints for batch processing."""

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued, processing, completed, failed
    pages_total: int
    pages_completed: int
    pages_needs_review: int
    pages_failed: int
    created_at: datetime
    completed_at: datetime | None = None
    avg_confidence: float | None = None
    processing_time_seconds: float | None = None


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a batch processing job."""
    from src.pipeline.batch_processor import BatchProcessor

    processor = BatchProcessor()
    status = await processor.get_status(job_id)

    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return status


@router.get("/jobs/{job_id}/results")
async def get_job_results(job_id: str, page: int = 1, per_page: int = 20):
    """Get transcription results for a completed job."""
    from src.pipeline.batch_processor import BatchProcessor

    processor = BatchProcessor()
    results = await processor.get_results(job_id, page=page, per_page=per_page)

    if results is None:
        raise HTTPException(status_code=404, detail="Job not found or not completed")

    return results


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a queued or in-progress job."""
    from src.pipeline.batch_processor import BatchProcessor

    processor = BatchProcessor()
    cancelled = await processor.cancel(job_id)

    if not cancelled:
        raise HTTPException(status_code=404, detail="Job not found or already completed")

    return {"status": "cancelled", "job_id": job_id}
