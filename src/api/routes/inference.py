"""Inference endpoints — single page and batch processing."""

import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

router = APIRouter()


class TranscriptionResult(BaseModel):
    """Result for a single transcribed page."""

    page_id: str
    text: str
    blocks: list[dict]
    confidence_score: float
    model_version: str
    processing_time_ms: float
    needs_review: bool
    uncertain_spans: list[dict]
    metadata: dict


class BatchJobResponse(BaseModel):
    """Response for batch job submission."""

    job_id: str
    status: str
    pages_queued: int
    estimated_completion_seconds: float
    webhook_url: str | None = None


@router.post("/transcribe/single", response_model=TranscriptionResult)
async def transcribe_single_page(
    request: Request,
    file: UploadFile = File(...),
    force_vlm: bool = Form(default=False),
):
    """Transcribe a single handwritten page.

    The pipeline:
    1. Quality gate (blur, skew, exposure check)
    2. Layout analysis (segment lines, blocks, reading order)
    3. Confidence router (normal → TrOCR, hard → VLM)
    4. Return structured transcript with confidence scores
    """
    from src.pipeline.orchestrator import PipelineOrchestrator

    # Validate file type
    if file.content_type not in ["image/png", "image/jpeg", "image/tiff", "application/pdf"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Read file
    contents = await file.read()

    # Run pipeline
    orchestrator = PipelineOrchestrator(request.app.state.model_registry)
    result = await orchestrator.process_single_page(
        image_bytes=contents,
        filename=file.filename,
        force_vlm=force_vlm,
    )

    return result


@router.post("/transcribe/batch", response_model=BatchJobResponse)
async def transcribe_batch(
    request: Request,
    files: list[UploadFile] = File(...),
    webhook_url: str | None = Form(default=None),
    priority: str = Form(default="normal"),
):
    """Submit a batch of pages (class-set) for async processing.

    Pages are queued for asynchronous processing. Results can be:
    - Polled via GET /api/v1/jobs/{job_id}
    - Delivered via webhook when complete
    """
    if len(files) > 200:
        raise HTTPException(status_code=400, detail="Maximum 200 pages per batch")

    job_id = str(uuid.uuid4())

    # Queue the batch job
    from src.pipeline.batch_processor import BatchProcessor

    processor = BatchProcessor()

    pages_queued = await processor.queue_batch(
        job_id=job_id,
        files=files,
        webhook_url=webhook_url,
        priority=priority,
    )

    # Estimate completion time (based on historical throughput)
    estimated_seconds = pages_queued * 2.5  # ~2.5s per page average

    return BatchJobResponse(
        job_id=job_id,
        status="queued",
        pages_queued=pages_queued,
        estimated_completion_seconds=estimated_seconds,
        webhook_url=webhook_url,
    )
