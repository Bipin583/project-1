"""
Unit tests for Statistical Significance and Non-Parametric Hypothesis Testing.
"""

import numpy as np
import pytest

from conftest.evaluation.statistics import (
    compute_cliffs_delta,
    compute_wilcoxon_test,
    bootstrap_confidence_interval,
    StatisticalSignificanceTester,
)


def test_cliffs_delta_extremes():
    """Verify Cliff's delta calculations for boundary distributions."""
    # 1. Perfectly superior distribution
    x_high = np.array([10, 11, 12, 13, 14])
    y_low = np.array([1, 2, 3, 4, 5])
    delta_1, mag_1 = compute_cliffs_delta(x_high, y_low)
    assert delta_1 == 1.0
    assert mag_1 == "Large"

    # 2. Identical distributions
    same_x = np.array([5, 5, 5, 5])
    same_y = np.array([5, 5, 5, 5])
    delta_2, mag_2 = compute_cliffs_delta(same_x, same_y)
    assert delta_2 == 0.0
    assert mag_2 == "Negligible"

    # 3. Inverted distribution
    delta_3, mag_3 = compute_cliffs_delta(y_low, x_high)
    assert delta_3 == -1.0
    assert mag_3 == "Large"


def test_wilcoxon_signed_rank_paired():
    """Verify Wilcoxon test detects significant difference on paired distributions."""
    rng = np.random.RandomState(42)
    # Distinct superior performance
    conftest_scores = rng.normal(0.98, 0.02, 30)
    baseline_scores = rng.normal(0.70, 0.05, 30)

    w_stat, p_val, is_sig = compute_wilcoxon_test(conftest_scores, baseline_scores)
    assert p_val < 0.01
    assert is_sig is True


def test_bootstrap_confidence_interval_bounds():
    """Verify Bootstrap 95% CI bounds consistency."""
    rng = np.random.RandomState(42)
    sample_data = rng.normal(50.0, 5.0, 100)

    ci_dict = bootstrap_confidence_interval(sample_data, num_bootstraps=500, ci=0.95)

    assert "mean" in ci_dict
    assert "ci_lower" in ci_dict
    assert "ci_upper" in ci_dict
    assert ci_dict["ci_lower"] <= ci_dict["mean"] <= ci_dict["ci_upper"]
    assert 48.0 < ci_dict["mean"] < 52.0


def test_statistical_significance_tester_pairwise():
    """Verify StatisticalSignificanceTester end-to-end report generation."""
    rng = np.random.RandomState(42)
    c_metrics = {
        "failure_recall": rng.uniform(0.95, 1.0, 20),
        "time_reduction": rng.uniform(0.60, 0.75, 20),
    }
    b_metrics = {
        "failure_recall": rng.uniform(0.20, 0.40, 20),
        "time_reduction": rng.uniform(0.70, 0.80, 20),
    }

    tester = StatisticalSignificanceTester()
    report = tester.evaluate_pairwise(c_metrics, b_metrics, "Random-K")

    assert report["baseline_name"] == "Random-K"
    assert "failure_recall" in report
    assert "time_reduction" in report
    assert report["failure_recall"]["statistically_significant_p05"] is True
    assert report["failure_recall"]["effect_size"] == "Large"
