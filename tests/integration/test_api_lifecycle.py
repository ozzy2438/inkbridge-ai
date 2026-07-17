"""Integration checks for the API application lifecycle."""

from fastapi.testclient import TestClient

from src.api.main import app


def test_health_and_readiness_follow_startup_lifecycle() -> None:
    """The service should expose a loaded baseline registry after startup."""
    with TestClient(app) as client:
        health_response = client.get("/health")
        readiness_response = client.get("/ready")

    assert health_response.status_code == 200
    assert health_response.json() == {
        "status": "healthy",
        "version": "0.1.0",
        "model_loaded": True,
        "model_version": "v0.1.0-baseline",
    }
    assert readiness_response.status_code == 200
    assert readiness_response.json() == {"status": "ready"}
