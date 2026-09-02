# ConfTest FastAPI REST Backend Specification

## 1. Overview
The ConfTest backend exposes production RESTful endpoints enabling CI/CD systems, GitHub Actions bots, and developer dashboards to interact with the regression test selection engine.

- **Base URL:** `http://localhost:8000`
- **Interactive Swagger Documentation:** `/docs`
- **ReDoc Interactive Reference:** `/redoc`
- **OpenAPI JSON Schema:** `/openapi.json`

---

## 2. API Endpoints

### A. Health & Diagnostics
- `GET /health` or `GET /api/v1/health`
  - Returns database status, system uptime, and component health.

### B. Test Selection & Safe Fallback
- `POST /api/v1/select`
  - **Request Body (`SelectRequestSchema`):**
    ```json
    {
      "repository_name": "org/repo",
      "commit_sha": "a1b2c3d4",
      "changed_files": [
        {"file_path": "src/auth.py", "change_type": "M", "lines_added": 12, "lines_deleted": 2}
      ],
      "commit_message": "fix: update token signature check",
      "budget_ratio": 0.25,
      "execute": false
    }
    ```
  - **Response (`SelectResponseSchema`):**
    - `decision_mode`: `FAST_SELECTED` or `SAFE_FULL_SUITE`
    - `abstained`: boolean
    - `selected_count` / `total_count`
    - `test_reduction_pct`
    - `top_confidence` and `epistemic_uncertainty`
    - `selected_test_ids`
    - `ranked_tests`: per-test calibrated probabilities and reasons
    - `markdown_summary`: pre-formatted GitHub PR comment table

### C. Model Explainability (SHAP & Rules)
- `POST /api/v1/explain`
  - **Request Body (`ExplainRequestSchema`):**
    - `test_id`: node ID of test case
    - `features`: 32-dimensional feature map
    - `top_k`: number of top features to return
  - **Response (`ExplainResponseSchema`):**
    - `predicted_probability`: calibrated failure probability
    - `risk_level`: `HIGH`, `MEDIUM`, or `LOW`
    - `primary_reasons`: rule-based natural language justifications
    - `top_risk_increasing_features` and `top_risk_decreasing_features` (exact Shapley attributions)

### D. Confidence Calibration Diagnostics
- `GET /api/v1/calibration`
  - Returns empirical Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier Score, and reliability diagram bins for Uncalibrated vs. Temperature Scaled / Isotonic models.

### E. Repositories Management
- `GET /api/v1/repositories`: List all registered repos.
- `POST /api/v1/repositories`: Register a new repository.
- `GET /api/v1/repositories/{repo_id}`: Retrieve repo metadata and test counts.
- `GET /api/v1/repositories/{repo_id}/commits`: Retrieve recent commit evaluations.

### F. Analytics & Telemetry
- `GET /api/v1/analytics`
  - Returns system-wide statistics: total evaluated commits, fast mode count, abstention count, aggregate compute time saved %, failure recall, and recent decisions log.
