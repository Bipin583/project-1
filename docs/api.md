# ConfTest REST API Reference

## Base URL
`http://127.0.0.1:8000`

## Endpoints

### 1. Root & Diagnostics
- `GET /` - Service metadata and links.
- `GET /health` - Liveness, uptime, environment, and database connectivity.
- `GET /docs` - Swagger UI interactive documentation.
- `GET /redoc` - ReDoc API specification.

### 2. Repositories (Milestone 2+)
- `POST /api/v1/repositories` - Register a repository.
- `GET /api/v1/repositories` - List tracked repositories.

### 3. Selection & Execution (Milestone 11+)
- `POST /api/v1/commits/{sha}/select` - Run selective prediction policy.
- `POST /api/v1/commits/{sha}/execute` - Execute selected or full test suite.
