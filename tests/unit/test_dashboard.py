import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import load_baseline_data, load_calibration_data, load_shap_report, get_cached_engine


def test_load_baseline_data_schema():
    """Verify load_baseline_data returns valid benchmark DataFrame."""
    df = load_baseline_data()
    assert not df.empty
    assert "strategy" in df.columns
    assert "failure_recall_pct" in df.columns
    assert "time_reduction_pct" in df.columns
    assert any("ConfTest" in s for s in df["strategy"])


def test_load_calibration_data_structure():
    """Verify load_calibration_data returns valid metrics dictionary."""
    data = load_calibration_data()
    assert "best_method" in data
    assert "test_metrics" in data
    assert "uncalibrated" in data["test_metrics"]
    assert "ece" in data["test_metrics"]["uncalibrated"]


def test_load_shap_report_structure():
    """Verify load_shap_report returns global SHAP feature list."""
    data = load_shap_report()
    assert "global_shap_importance" in data
    assert len(data["global_shap_importance"]) >= 5
    assert "feature" in data["global_shap_importance"][0]
    assert "mean_abs_shap" in data["global_shap_importance"][0]


def test_get_cached_engine_initialization():
    """Verify get_cached_engine returns functional ConfTestEngine."""
    engine = get_cached_engine()
    assert engine is not None
    assert engine.calibrator is not None
