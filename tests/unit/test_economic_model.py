"""
Unit tests for Economic Cost-Benefit and CI/CD Financial Modeling module.
"""

import pytest
from conftest.evaluation.economic_model import EnterpriseEconomicConfig, EconomicCostBenefitModel


def test_economic_model_baseline_costs():
    """Verify full suite annual cost computations."""
    config = EnterpriseEconomicConfig(
        num_developers=10,
        commits_per_dev_per_day=2.0,
        working_days_per_year=200,
        full_suite_duration_minutes=30.0,
        ci_runner_cost_per_minute_usd=0.01,
        developer_hourly_rate_usd=50.0,
        cost_per_escaped_bug_usd=2000.0,
        baseline_annual_bugs_escaped=2,
    )

    model = EconomicCostBenefitModel(config)
    assert model.total_annual_commits == 4000  # 10 * 2 * 200

    baseline = model.calculate_full_suite_annual_cost()
    assert baseline["total_ci_minutes"] == 120000.0  # 4000 * 30
    assert baseline["annual_ci_compute_cost_usd"] == 1200.0  # 120000 * 0.01
    # 30% wait time = 36000 mins = 600 hours -> 600 * $50 = $30,000
    assert baseline["annual_developer_wait_cost_usd"] == 30000.0
    assert baseline["annual_escape_cost_usd"] == 4000.0


def test_economic_model_conftest_strategy_savings():
    """Verify strategy evaluation and net financial benefits."""
    config = EnterpriseEconomicConfig(
        num_developers=20,
        commits_per_dev_per_day=2.0,
        working_days_per_year=250,
        full_suite_duration_minutes=60.0,
        ci_runner_cost_per_minute_usd=0.02,
        developer_hourly_rate_usd=60.0,
    )
    model = EconomicCostBenefitModel(config)

    res = model.evaluate_rts_strategy(
        strategy_name="ConfTest Selective",
        test_reduction_rate=0.70,
        failure_recall_rate=1.0,
        annual_escapes_with_strategy=0,
    )

    assert res["test_reduction_rate_pct"] == 70.0
    assert res["annual_ci_compute_savings_usd"] > 0.0
    assert res["annual_developer_time_savings_usd"] > 0.0
    assert res["net_annual_benefit_usd"] > 0.0
    assert res["breakeven_escaped_bugs_threshold"] > 0
