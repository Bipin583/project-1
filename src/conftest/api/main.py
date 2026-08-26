"""
ConfTest FastAPI Main Application Entry Point.

Provides RESTful endpoints for repository registration, AST analysis,
test-selection prediction, calibrated confidence reporting, and CI/CD webhook processing.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from conftest.config import settings
from conftest.logging_config import get_logger
from conftest.db.init_db import init_db
from conftest.api.routes.health import router as health_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager: runs startup and shutdown tasks."""
    logger.info(f"Starting {settings.app_name} v{settings.version} [{settings.env}]...")
    # Initialize database schema automatically on startup
    init_db()
    yield
    logger.info(f"Shutting down {settings.app_name}...")


# Create main FastAPI application
app = FastAPI(
    title="ConfTest API",
    description=(
        "Confidence-Calibrated Regression Test Selection API with Selective Prediction "
        "and Safe Full-Suite Fallback for CI/CD Optimization."
    ),
    version=settings.version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health_router, prefix="", tags=["System & Diagnostics"])
app.include_router(health_router, prefix="/api/v1", tags=["System & Diagnostics"])


@app.get(
    "/",
    status_code=status.HTTP_200_OK,
    tags=["Root"],
    summary="ConfTest API Root",
    description="Returns welcome information and link to API documentation.",
)
def root():
    """Root endpoint welcoming clients and directing to interactive documentation."""
    return JSONResponse(
        content={
            "app": settings.app_name,
            "version": settings.version,
            "description": "Confidence-Calibrated Regression Test Selection API",
            "docs": "/docs",
            "health": "/health",
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "conftest.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=settings.api_workers,
    )
