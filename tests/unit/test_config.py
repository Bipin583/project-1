"""
Unit tests for the ConfTest configuration loader and settings management.
"""

from pathlib import Path
from conftest.config import Settings, get_settings


def test_default_settings_instantiation():
    """Verify that default settings instantiate with expected values."""
    settings = Settings()
    assert settings.app_name == "ConfTest"
    assert settings.version == "0.1.0"
    assert settings.api_port == 8000
    assert settings.default_risk_tolerance == 0.18
    assert settings.default_budget_ratio == 0.25
    assert settings.abstention_threshold == 0.15
    assert isinstance(settings.data_dir, Path)
    assert isinstance(settings.models_dir, Path)


def test_settings_environment_flags():
    """Verify environment detection helpers."""
    dev_settings = Settings(env="development")
    assert dev_settings.is_production is False
    assert dev_settings.is_testing is False

    prod_settings = Settings(env="production")
    assert prod_settings.is_production is True
    assert prod_settings.is_testing is False

    test_settings = Settings(env="testing")
    assert test_settings.is_production is False
    assert test_settings.is_testing is True


def test_get_settings_singleton():
    """Verify that get_settings() returns a cached singleton instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
