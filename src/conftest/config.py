"""
ConfTest Global Configuration Module.

Provides robust, type-safe settings management using Pydantic Settings.
Environment variables can override defaults, using the 'CONFTEST_' prefix.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application configuration model with default fallbacks."""

    model_config = SettingsConfigDict(
        env_prefix="CONFTEST_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application & Environment
    app_name: str = Field(default="ConfTest", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    env: str = Field(default="development", description="Runtime environment")
    debug: bool = Field(default=False, description="Debug mode flag")
    log_level: str = Field(default="INFO", description="Logging level")

    # API Server Settings
    api_host: str = Field(default="127.0.0.1", description="FastAPI host")
    api_port: int = Field(default=8000, description="FastAPI port")
    api_workers: int = Field(default=1, description="FastAPI worker count")

    # Database Configuration (Defaults to SQLite with WAL mode)
    database_url: str = Field(
        default="sqlite:///./data/conftest.db",
        description="Database connection URI",
    )
    db_echo: bool = Field(default=False, description="SQLAlchemy query echoing")

    # ML & Decision Policy
    model_path: str = Field(
        default="./models/ensembles/lgbm_ensemble.pkl",
        description="Path to serialized ML model or ensemble artifact",
    )
    default_risk_tolerance: float = Field(
        default=0.18,
        description="Default risk tolerance threshold for selective execution",
    )
    default_budget_ratio: float = Field(
        default=0.25,
        description="Default test budget fraction (e.g. 0.25 = top 25% tests)",
    )
    abstention_threshold: float = Field(
        default=0.15,
        description="Epistemic uncertainty threshold triggering full-suite fallback",
    )

    # Security & Integration
    github_webhook_secret: str = Field(
        default="development_secret_only_change_in_ci",
        description="HMAC SHA-256 secret for GitHub webhook payload verification",
    )
    github_token: Optional[str] = Field(
        default=None,
        description="Optional GitHub personal access token for higher API rate limits",
    )

    # Directories
    data_dir: Path = Field(
        default=PROJECT_ROOT / "data",
        description="Base directory for raw, processed, and split datasets",
    )
    models_dir: Path = Field(
        default=PROJECT_ROOT / "models",
        description="Base directory for serialized ML models and calibration artifacts",
    )

    @property
    def is_production(self) -> bool:
        """Return True if running in production mode."""
        return self.env.lower() in ("production", "prod")

    @property
    def is_testing(self) -> bool:
        """Return True if running under a test harness."""
        return self.env.lower() in ("test", "testing")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Retrieve cached global application settings singleton.
    Cached with LRU cache to avoid re-reading disk on every access.
    """
    settings = Settings()
    # Ensure standard directories exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    return settings


# Global settings singleton instance
settings: Settings = get_settings()
