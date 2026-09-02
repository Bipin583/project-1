"""
Unit tests for SHAP Attributions and Rule-Based Explainability Engine.
"""

import numpy as np
import pytest

from conftest.features.pipeline import FEATURE_NAMES
from conftest.models.lightgbm_model import LightGBMTestPredictor
from conftest.explainability.shap_explainer import ShapExplainer
from conftest.explainability.rules import RuleBasedExplainer


@pytest.fixture
def trained_lightgbm():
    rng = np.random.RandomState(42)
    X = rng.randn(80, len(FEATURE_NAMES)).astype(np.float32)
    y = (rng.rand(80) < 0.20).astype(int)
    y[0] = 1
    y[1] = 1

    predictor = LightGBMTestPredictor(random_seed=42, n_estimators=15)
    predictor.train(X_train=X, y_train=y)
    return predictor, X, y


def test_shap_explainer_single_instance(trained_lightgbm):
    """Verify SHAP TreeExplainer computes feature attributions for a single test prediction."""
    predictor, X, _ = trained_lightgbm
    explainer = ShapExplainer(predictor)

    x_sample = X[0]
    res = explainer.explain_instance(x_sample, top_k=3)

    assert "predicted_probability" in res
    assert "base_expected_value" in res
    assert "top_risk_increasing_features" in res
    assert "top_risk_decreasing_features" in res
    assert len(res["all_attributions"]) == 32


def test_shap_explainer_global_dataset(trained_lightgbm):
    """Verify global mean absolute SHAP importance rankings across dataset."""
    predictor, X, _ = trained_lightgbm
    explainer = ShapExplainer(predictor)

    res = explainer.explain_dataset(X[:20])

    assert "global_feature_importance_shap" in res
    rankings = res["global_feature_importance_shap"]
    assert len(rankings) == 32
    assert rankings[0]["mean_abs_shap"] >= rankings[-1]["mean_abs_shap"]


def test_rule_based_explainer_reason_cards():
    """Verify RuleBasedExplainer translates feature signals into natural language reason strings."""
    explainer = RuleBasedExplainer()

    # Case 1: Direct dependency with historical failures
    feats_1 = {
        "dep_is_direct_import": 1.0,
        "hist_recent_10_failure_rate": 0.30,
        "diff_total_churn": 150.0,
        "diff_is_fix_commit": 1.0,
    }
    card_1 = explainer.generate_test_reason_card(
        test_id="tests/test_auth.py::test_login",
        feature_dict=feats_1,
        is_selected=True,
        confidence=0.85,
    )
    assert card_1["risk_level"] == "HIGH"
    assert any("Direct Dependency" in r for r in card_1["primary_reasons"])
    assert any("Recent Regression History" in r for r in card_1["primary_reasons"])
    assert any("High Code Churn" in r for r in card_1["primary_reasons"])

    # Case 2: Clean baseline test without coupling
    feats_2 = {
        "dep_is_direct_import": 0.0,
        "dep_shortest_path_depth": 10.0,
        "hist_recent_10_failure_rate": 0.0,
        "hist_lifetime_failure_rate": 0.0,
        "diff_total_churn": 10.0,
    }
    card_2 = explainer.generate_test_reason_card(
        test_id="tests/test_unrelated.py::test_fn",
        feature_dict=feats_2,
        is_selected=False,
        confidence=0.02,
    )
    assert card_2["risk_level"] == "LOW"
    assert any("Low Regression Risk" in r for r in card_2["primary_reasons"])


def test_rule_based_explainer_markdown_summary():
    """Verify generated Markdown report conforms to GitHub PR format."""
    explainer = RuleBasedExplainer()

    decision_dict = {
        "decision_mode": "FAST_SELECTED",
        "abstained": False,
        "test_reduction_pct": 75.0,
        "selected_count": 5,
        "total_count": 20,
        "top_confidence": 0.92,
        "epistemic_uncertainty": 0.008,
    }
    top_tests = [
        {"test_id": "tests/test_auth.py::test_login", "risk_level": "HIGH", "confidence": 0.92, "primary_reasons": ["Direct Dependency"]}
    ]

    md = explainer.generate_commit_markdown_summary(
        commit_sha="a1b2c3d4e5f6",
        decision_dict=decision_dict,
        top_tests=top_tests,
    )

    assert "## 🛡️ ConfTest CI Regression Test Selection Report" in md
    assert "75.0% test execution reduction" in md
    assert "tests/test_auth.py::test_login" in md
