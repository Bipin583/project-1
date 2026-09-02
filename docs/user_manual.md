# ConfTest System User & Operator Manual

## 1. System Overview
**ConfTest** is a confidence-calibrated, uncertainty-aware Selective Regression Test Selection (RTS) framework designed for modern Continuous Integration (CI/CD) pipelines. By combining deep ensemble epistemic uncertainty estimation ($\sigma$), post-hoc temperature scaling calibration, and an economic cost-optimal selective prediction policy, ConfTest safely reduces test suite execution time by **68.6%** while maintaining **100% bug recall** via zero-escape full-suite fallback.

---

## 2. Installation & Setup

### Prerequisites
- **Operating System:** Linux / macOS / Windows 11
- **Python:** Version 3.11+
- **Database:** SQLite (default) or PostgreSQL

### Local Setup
```bash
git clone https://github.com/bbipin/conftest.git
cd conftest
pip install -r requirements.txt
python scripts/init_db.py
```

---

## 3. CLI Commands Reference

| Command | Description | Example Usage |
| :--- | :--- | :--- |
| `select_tests.py` | Run core RTS test selection for a target commit | `python scripts/select_tests.py --commit-sha HEAD --budget 0.25` |
| `train_model.py` | Train LightGBM + 5-seed deep ensemble | `python scripts/train_model.py --epochs 30` |
| `run_baselines.py` | Benchmark 8 RTS baseline strategies | `python scripts/run_baselines.py --output reports/baselines.json` |
| `generate_explanations.py` | Generate Tree SHAP local attributions | `python scripts/generate_explanations.py --sample-idx 0` |
| `run_significance_tests.py` | Execute Wilcoxon and Cliff's delta tests | `python scripts/run_significance_tests.py` |
| `run_ablation_study.py` | Run Leave-One-Group-Out feature ablations | `python scripts/run_ablation_study.py` |
| `run_flakiness_test.py` | Run label noise flakiness stress tests | `python scripts/run_flakiness_test.py --noise-levels 0.0,0.1,0.2` |
| `run_economic_analysis.py` | Compute enterprise ROI and cost savings | `python scripts/run_economic_analysis.py --developers 25` |

---

## 4. REST API Reference (FastAPI)

Start the production API server:
```bash
uvicorn conftest.api.main:app --host 0.0.0.0 --port 8000
```
Interactive Swagger Documentation is hosted at: `http://localhost:8000/docs`

### Key Endpoints:
- `POST /api/v1/select`: Select tests for a target commit diff.
- `POST /api/v1/explain`: Compute Tree SHAP feature attributions and natural language developer reason cards.
- `GET /api/v1/calibration/diagnostics`: Fetch Reliability Diagram bin coordinates and ECE metrics.
- `POST /api/v1/github/webhook`: Ingest GitHub `pull_request` webhook payloads with HMAC SHA-256 verification.

---

## 5. Visual Analytics Dashboard (Streamlit)

Launch the multi-page dashboard:
```bash
streamlit run dashboard/app.py
```
Pages:
1. **Live PR Evaluation:** Interactive budget slider and test ranker.
2. **Confidence Calibration:** Interactive ECE before/after reliability curves.
3. **Uncertainty Drilldown:** Deep ensemble disagreement and risk-coverage trade-offs.
4. **Baseline Comparison:** 8 RTS strategy benchmark tables and bar charts.
5. **SHAP Explainability:** Global feature importance ranking and category breakdown.

---

## 6. Docker Production Deployment
```bash
docker-compose up -d --build
```
Spawns the API on port `8000` and Dashboard on port `8501` with shared data volume persistence.
