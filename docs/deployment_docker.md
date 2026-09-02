# ConfTest Docker Production Deployment Guide

## 1. Quickstart with Docker Compose

Deploy the complete ConfTest stack (FastAPI Backend + Streamlit Analytics Dashboard + SQLite Database) in one command:

```bash
docker-compose up -d --build
```

### Endpoints Available:
- **FastAPI REST API:** [http://localhost:8000](http://localhost:8000)
- **FastAPI Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit Analytics Dashboard:** [http://localhost:8501](http://localhost:8501)
- **Health Check:** `curl http://localhost:8000/health`

---

## 2. Container Architecture
- **Multi-Stage Build:** Python 3.11-slim builder stage compiles wheel dependencies; minimal runtime stage delivers lean image footprint (~280MB).
- **Persistent Volume:** SQLite database and serialized model weights are mounted to named volume `conftest-data` at `/app/data`.
- **Microservices Network:** Internal bridge network enables Streamlit dashboard to communicate with FastAPI backend at `http://conftest-api:8000`.

---

## 3. Environment Variables Reference
| Variable | Description | Default |
| :--- | :--- | :--- |
| `CONFTEST_ENV` | Runtime environment mode | `production` |
| `CONFTEST_DB_URL` | SQLAlchemy database connection URI | `sqlite:////app/data/conftest.db` |
| `CONFTEST_API_PORT` | FastAPI listening port | `8000` |
| `CONFTEST_LOG_LEVEL` | Logging verbosity | `INFO` |
