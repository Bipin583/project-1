"""
Unit and integration tests for FastAPI health check and diagnostic endpoints.
"""

from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """Verify that GET / returns a 200 OK with valid service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "ConfTest"
    assert "version" in data
    assert data["docs"] == "/docs"
    assert data["health"] == "/health"


def test_health_check_endpoint(client: TestClient):
    """Verify that GET /health returns 200 OK with healthy status and database connectivity."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ConfTest"
    assert data["database"] == "connected"
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0


def test_api_v1_health_alias(client: TestClient):
    """Verify that GET /api/v1/health functions identically as an alias."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
