"""Export endpoints — TXT, JSON, DOCX, LMS formats."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
import io

router = APIRouter()


@router.get("/export/{job_id}")
async def export_transcript(
    job_id: str,
    format: str = "json",  # json, txt, docx, lms
    include_metadata: bool = True,
    include_confidence: bool = True,
):
    """Export transcription results in various formats.
    
    Formats:
    - json: Structured JSON with blocks, confidence, and metadata
    - txt: Plain text preserving reading order
    - docx: Microsoft Word document with layout preservation
    - lms: LMS-compatible API format (Moodle, Canvas, etc.)
    """
    from src.export.transcript import TranscriptExporter
    
    exporter = TranscriptExporter()
    
    try:
        content, content_type, filename = await exporter.export(
            job_id=job_id,
            format=format,
            include_metadata=include_metadata,
            include_confidence=include_confidence,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return StreamingResponse(
        io.BytesIO(content),
        media_type=content_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
