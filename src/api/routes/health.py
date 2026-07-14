"""Health check and readiness probe endpoints."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    model_version: str | None = None


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Liveness probe — service is running."""
    registry = request.app.state.model_registry
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model_loaded=registry.is_loaded,
        model_version=registry.champion_version,
    )


@router.get("/ready")
async def readiness_check(request: Request):
    """Readiness probe — models loaded and ready to serve."""
    registry = request.app.state.model_registry
    if not registry.is_loaded:
        return {"status": "not_ready", "reason": "Models still loading"}
    return {"status": "ready"}
