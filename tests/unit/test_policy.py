"""
Unit tests for Selective Prediction Policy, Abstention Engine, and CI Utility Model.
"""

from pathlib import Path
import numpy as np
import pytest

from conftest.models.policy import SelectivePredictionPolicy, CostBenefitModel, PolicyDecision


def test_selective_policy_fast_mode_trigger():
    """Verify Fast Selective Mode is triggered under high confidence and low uncertainty."""
    policy = SelectivePredictionPolicy(tau_abstain=0.02, tau_conf=0.60, budget_ratio=0.25)

    test_ids = [f"tests/test_{i}.py::test_fn" for i in range(10)]
    confidences = np.array([0.95, 0.80, 0.10, 0.05, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01])
    uncertainties = np.array([0.005] * 10)  # Low uncertainty (< 0.02)

    decision = policy.evaluate_commit(
        commit_sha="abc1234",
        candidate_test_ids=test_ids,
        calibrated_confidences=confidences,
        epistemic_uncertainties=uncertainties,
        num_changed_files=2,
        total_churn_lines=30,
    )

    assert decision.decision_mode == "FAST_SELECTED"
    assert decision.abstained is False
    assert len(decision.selected_test_ids) == 2  # 25% of 10 tests = 2 tests
    assert decision.selected_test_ids[0] == "tests/test_0.py::test_fn"
    assert decision.selected_test_ids[1] == "tests/test_1.py::test_fn"
    assert decision.estimated_time_saved_pct == 80.0


def test_selective_policy_abstention_on_high_uncertainty():
    """Verify Safe Full-Suite Fallback is triggered when epistemic uncertainty spikes."""
    policy = SelectivePredictionPolicy(tau_abstain=0.02, tau_conf=0.60, budget_ratio=0.25)

    test_ids = [f"tests/test_{i}.py::test_fn" for i in range(8)]
    confidences = np.array([0.90] + [0.05] * 7)
    uncertainties = np.array([0.05] + [0.005] * 7)  # Test 0 has high uncertainty 0.05 > 0.02

    decision = policy.evaluate_commit(
        commit_sha="sha_uncertain",
        candidate_test_ids=test_ids,
        calibrated_confidences=confidences,
        epistemic_uncertainties=uncertainties,
    )

    assert decision.decision_mode == "SAFE_FULL_SUITE"
    assert decision.abstained is True
    assert len(decision.selected_test_ids) == 8  # Full suite execution
    assert any("High epistemic uncertainty" in r for r in decision.reasons)


def test_selective_policy_abstention_on_low_confidence():
    """Verify Safe Full-Suite Fallback is triggered when model is not confident."""
    policy = SelectivePredictionPolicy(tau_abstain=0.02, tau_conf=0.70, budget_ratio=0.25)

    test_ids = [f"tests/test_{i}.py::test_fn" for i in range(6)]
    confidences = np.array([0.45] + [0.10] * 5)  # Top confidence 0.45 < 0.70
    uncertainties = np.array([0.005] * 6)

    decision = policy.evaluate_commit(
        commit_sha="sha_low_conf",
        candidate_test_ids=test_ids,
        calibrated_confidences=confidences,
        epistemic_uncertainties=uncertainties,
    )

    assert decision.decision_mode == "SAFE_FULL_SUITE"
    assert decision.abstained is True
    assert len(decision.selected_test_ids) == 6


def test_selective_policy_abstention_on_ood_refactoring():
    """Verify Safe Full-Suite Fallback is triggered on large architectural diffs."""
    policy = SelectivePredictionPolicy(tau_abstain=0.05, tau_conf=0.50, ood_file_limit=10, ood_churn_limit=300)

    test_ids = [f"tests/test_{i}.py::test_fn" for i in range(10)]
    confidences = np.array([0.95] * 10)
    uncertainties = np.array([0.001] * 10)

    decision = policy.evaluate_commit(
        commit_sha="sha_huge_refactor",
        candidate_test_ids=test_ids,
        calibrated_confidences=confidences,
        epistemic_uncertainties=uncertainties,
        num_changed_files=25,  # > 10 files
        total_churn_lines=1200,  # > 300 lines
    )

    assert decision.decision_mode == "SAFE_FULL_SUITE"
    assert decision.abstained is True
    assert any("OOD architectural refactoring" in r for r in decision.reasons)


def test_cost_benefit_model_calculations():
    """Verify CostBenefitModel calculates time savings and penalty costs."""
    cost_model = CostBenefitModel(cost_per_test_second=0.02, penalty_per_escaped_failure=100.0)

    # 100s full suite -> 25s selected (75s saved, 0 escapes)
    util_clean = cost_model.compute_utility(
        full_suite_duration_sec=100.0,
        selected_duration_sec=25.0,
        escaped_failures_count=0,
    )
    assert util_clean["time_saved_sec"] == 75.0
    assert util_clean["gross_savings_dollars"] == 1.50  # 75 * 0.02
    assert util_clean["penalty_cost_dollars"] == 0.0
    assert util_clean["net_ci_utility_dollars"] == 1.50

    # 1 escaped bug penalty ($100)
    util_escaped = cost_model.compute_utility(
        full_suite_duration_sec=100.0,
        selected_duration_sec=25.0,
        escaped_failures_count=1,
    )
    assert util_escaped["penalty_cost_dollars"] == 100.0
    assert util_escaped["net_ci_utility_dollars"] == -98.50


def test_policy_save_and_load(tmp_path: Path):
    """Verify JSON configuration serialization of policy thresholds."""
    policy = SelectivePredictionPolicy(tau_abstain=0.0123, tau_conf=0.75, budget_ratio=0.30)
    config_path = tmp_path / "policy_config.json"

    policy.save(str(config_path))
    assert config_path.exists()

    loaded = SelectivePredictionPolicy.load(str(config_path))
    assert loaded.tau_abstain == 0.0123
    assert loaded.tau_conf == 0.75
    assert loaded.budget_ratio == 0.30
