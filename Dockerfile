# Multi-stage Dockerfile for ConfTest API and Dashboard
FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CONFTEST_ENV=production

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code and configuration
COPY pyproject.toml README.md .env.example ./
COPY configs/ ./configs/
COPY src/ ./src/
COPY dashboard/ ./dashboard/

# Install conftest package in editable mode
RUN pip install -e .

# Create data and models directory
RUN mkdir -p data models

EXPOSE 8000 8501

# Default startup command launches FastAPI
CMD ["uvicorn", "conftest.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
