"""
Comprehensive Integration and Unit tests for FastAPI REST Endpoints.
"""

from fastapi.testclient import TestClient
import pytest

from conftest.features.pipeline import FEATURE_NAMES


def test_root_endpoint(client: TestClient):
    """Verify GET / returns documentation links and metadata."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "app" in data
    assert "endpoints" in data
    assert data["endpoints"]["select"] == "/api/v1/select"


def test_health_endpoint(client: TestClient):
    """Verify GET /api/v1/health probe."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"


def test_repositories_crud_endpoints(client: TestClient):
    """Verify POST and GET /api/v1/repositories."""
    # 1. Create repository
    payload = {
        "full_name": "owner/api-test-repo",
        "url": "https://github.com/owner/api-test-repo",
        "local_path": "./tests/sample_suite",
    }
    resp = client.post("/api/v1/repositories", json=payload)
    assert resp.status_code == 201
    created = resp.json()
    repo_id = created["id"]
    assert created["full_name"] == "owner/api-test-repo"

    # 2. List repositories
    list_resp = client.get("/api/v1/repositories")
    assert list_resp.status_code == 200
    repos = list_resp.json()
    assert len(repos) >= 1

    # 3. Get repository details
    detail_resp = client.get(f"/api/v1/repositories/{repo_id}")
    assert detail_resp.status_code == 200
    det = detail_resp.json()
    assert det["id"] == repo_id
    assert "total_test_cases" in det


def test_select_endpoint_fast_and_fallback(client: TestClient):
    """Verify POST /api/v1/select executes end-to-end RTS prediction."""
    payload = {
        "repository_name": "owner/api-test-repo",
        "commit_sha": "abc123456789",
        "changed_files": [
            {"file_path": "src_app/auth.py", "change_type": "M", "lines_added": 15, "lines_deleted": 3}
        ],
        "commit_message": "fix: update authentication token expiration",
        "budget_ratio": 0.25,
        "repo_path": "./tests/sample_suite",
        "execute": False,
    }
    resp = client.post("/api/v1/select", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "decision_mode" in data
    assert data["decision_mode"] in ("FAST_SELECTED", "SAFE_FULL_SUITE")
    assert "selected_test_ids" in data
    assert "ranked_tests" in data
    assert len(data["ranked_tests"]) >= 5
    assert "markdown_summary" in data
    assert "🛡️ ConfTest" in data["markdown_summary"]


def test_explain_endpoint_shap_and_rules(client: TestClient):
    """Verify POST /api/v1/explain returns SHAP drivers and developer cards."""
    # Build 32-feature vector dict
    dummy_feats = {name: 0.0 for name in FEATURE_NAMES}
    dummy_feats["dep_is_direct_import"] = 1.0
    dummy_feats["hist_recent_10_failure_rate"] = 0.20
    dummy_feats["diff_total_churn"] = 45.0

    payload = {
        "test_id": "tests/sample_suite/tests/test_auth.py::test_password_hashing",
        "features": dummy_feats,
        "top_k": 3,
    }
    resp = client.post("/api/v1/explain", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["test_id"] == payload["test_id"]
    assert "predicted_probability" in data
    assert "primary_reasons" in data
    assert len(data["top_risk_increasing_features"]) <= 3


def test_calibration_endpoint(client: TestClient):
    """Verify GET /api/v1/calibration returns ECE and Brier diagnostics."""
    resp = client.get("/api/v1/calibration")
    assert resp.status_code == 200
    data = resp.json()

    assert "best_method" in data
    assert "uncalibrated" in data
    assert "calibrated" in data
    assert "ece" in data["calibrated"]


def test_analytics_endpoint(client: TestClient):
    """Verify GET /api/v1/analytics aggregates database metrics."""
    resp = client.get("/api/v1/analytics")
    assert resp.status_code == 200
    data = resp.json()

    assert "total_repositories" in data
    assert "total_decisions" in data
    assert "average_test_reduction_pct" in data
    assert "recent_decisions" in data
