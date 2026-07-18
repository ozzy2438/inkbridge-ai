"""InkBridge AI — FastAPI Application Entry Point.

Production-grade API for handwriting intelligence with:
- Async job processing for class-set batch uploads
- Presigned secure upload to object storage
- Versioned model inference
- Audit logging and monitoring
- Health checks and readiness probes
"""

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app

from src.api.middleware.logging import LoggingMiddleware
from src.api.routes import exports, health, inference, jobs, review
from src.models.model_registry import ModelRegistry

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter(
    "inkbridge_requests_total", "Total API requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "inkbridge_request_latency_seconds", "Request latency in seconds", ["method", "endpoint"]
)
INFERENCE_LATENCY = Histogram(
    "inkbridge_inference_latency_seconds", "Model inference latency", ["model_version", "page_type"]
)
PAGES_PROCESSED = Counter(
    "inkbridge_pages_processed_total", "Total pages processed", ["model", "route"]
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler — load models at startup."""
    logger.info("inkbridge.startup", msg="Loading model registry...")

    # Initialize model registry
    app.state.model_registry = ModelRegistry()
    await app.state.model_registry.load_champion_model()

    logger.info("inkbridge.startup.complete", msg="Models loaded successfully")
    yield

    # Cleanup
    logger.info("inkbridge.shutdown", msg="Shutting down InkBridge AI")
    await app.state.model_registry.unload_models()


app = FastAPI(
    title="InkBridge AI",
    description="Production Handwriting Intelligence & ModelOps Platform",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom logging middleware
app.add_middleware(LoggingMiddleware)

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.middleware("http")
async def track_metrics(request: Request, call_next):
    """Track request metrics for monitoring."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method, endpoint=request.url.path, status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(duration)

    return response


# Include route modules
app.include_router(health.router, tags=["Health"])
app.include_router(inference.router, prefix="/api/v1", tags=["Inference"])
app.include_router(jobs.router, prefix="/api/v1", tags=["Jobs"])
app.include_router(review.router, prefix="/api/v1", tags=["Review"])
app.include_router(exports.router, prefix="/api/v1", tags=["Export"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with structured logging."""
    logger.error(
        "inkbridge.unhandled_error",
        error=str(exc),
        path=request.url.path,
        method=request.method,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Request ID logged for investigation."},
    )
