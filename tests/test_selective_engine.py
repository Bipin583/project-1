"""
Unit tests for the Selective Prediction & Abstention Decision Engine.
"""
import numpy as np
from src.engine.selective_engine import SelectiveDecisionEngine

def test_selective_decision_high_confidence():
    engine = SelectiveDecisionEngine(tau_abstain=0.18, theta_select=0.10)
    tests = ["test_auth.py", "test_db.py", "test_payment.py", "test_ui.py"]
    probs = np.array([0.45, 0.02, 0.01, 0.03])
    uncertainties = np.array([0.05, 0.04, 0.03, 0.02])

    decision = engine.decide(tests, probs, uncertainties)
    assert decision.action == "SELECTIVE_RUN"
    assert "test_auth.py" in decision.selected_tests
    assert decision.test_reduction_ratio > 0.0
    assert decision.max_uncertainty < 0.18

def test_selective_decision_abstention_on_high_uncertainty():
    engine = SelectiveDecisionEngine(tau_abstain=0.18, theta_select=0.10)
    tests = ["test_auth.py", "test_db.py", "test_payment.py", "test_ui.py"]
    probs = np.array([0.25, 0.15, 0.12, 0.09])
    # High uncertainty triggers abstention
    uncertainties = np.array([0.24, 0.08, 0.06, 0.05])

    decision = engine.decide(tests, probs, uncertainties)
    assert decision.action == "ABSTAIN_SAFE_FALLBACK"
    assert len(decision.selected_tests) == 4 # Executes full suite for safety
    assert decision.test_reduction_ratio == 0.0
    assert "safety budget" in decision.reason

def test_selective_decision_abstention_on_ood_refactoring():
    engine = SelectiveDecisionEngine(tau_abstain=0.18, theta_select=0.10)
    tests = ["test_1.py", "test_2.py", "test_3.py"]
    probs = np.array([0.05, 0.04, 0.02])
    uncertainties = np.array([0.05, 0.04, 0.03])

    decision = engine.decide(tests, probs, uncertainties, is_ood_refactoring=True)
    assert decision.action == "ABSTAIN_SAFE_FALLBACK"
    assert len(decision.selected_tests) == 3
