"""
Unit tests for Flakiness Stress Testing and Robustness Evaluation module.
"""

import numpy as np
import pytest

from conftest.features.pipeline import FEATURE_NAMES
from conftest.evaluation.flakiness import inject_flakiness_noise, FlakinessStressTester


def test_flakiness_noise_injection():
    """Verify flakiness label flipping mechanics and rates."""
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)

    # 1. Zero noise
    clean_y, scores_0 = inject_flakiness_noise(y, noise_rate=0.0)
    assert np.array_equal(clean_y, y)
    assert np.all(scores_0 == 0.0)

    # 2. 20% noise (flips 2 items)
    noisy_y, scores = inject_flakiness_noise(y, noise_rate=0.20, random_seed=42)
    flips = np.sum(noisy_y != y)
    assert flips == 2
    assert len(scores) == len(y)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_flakiness_stress_tester_synthetic():
    """Verify FlakinessStressTester compares standard vs robust models."""
    rng = np.random.RandomState(42)
    n_feats = len(FEATURE_NAMES)

    X_train = rng.randn(60, n_feats).astype(np.float32)
    y_train = (rng.rand(60) < 0.20).astype(int)
    y_train[0] = 1
    y_train[1] = 1

    X_val = rng.randn(20, n_feats).astype(np.float32)
    y_val = (rng.rand(20) < 0.20).astype(int)
    y_val[0] = 1

    X_test = rng.randn(20, n_feats).astype(np.float32)
    y_test = (rng.rand(20) < 0.20).astype(int)
    y_test[0] = 1

    tester = FlakinessStressTester(random_seed=42)
    res = tester.evaluate_noise_level(X_train, y_train, X_val, y_val, X_test, y_test, noise_rate=0.10)

    assert "noise_rate_pct" in res
    assert res["noise_rate_pct"] == 10.0
    assert "standard_unweighted" in res
    assert "conftest_robust" in res
    assert "robustness_advantage" in res
    assert "pr_auc" in res["conftest_robust"]
    assert "failure_recall" in res["conftest_robust"]


def test_flakiness_stress_grid_execution():
    """Verify stress grid across multiple noise rates."""
    rng = np.random.RandomState(42)
    n_feats = len(FEATURE_NAMES)

    X_train = rng.randn(50, n_feats).astype(np.float32)
    y_train = (rng.rand(50) < 0.20).astype(int)
    y_train[0] = 1

    X_val = rng.randn(20, n_feats).astype(np.float32)
    y_val = (rng.rand(20) < 0.20).astype(int)
    y_val[0] = 1

    X_test = rng.randn(20, n_feats).astype(np.float32)
    y_test = (rng.rand(20) < 0.20).astype(int)
    y_test[0] = 1

    tester = FlakinessStressTester(random_seed=42)
    grid = tester.run_stress_grid(X_train, y_train, X_val, y_val, X_test, y_test, noise_levels=[0.0, 0.20])

    assert len(grid) == 2
    assert grid[0]["noise_rate_pct"] == 0.0
    assert grid[1]["noise_rate_pct"] == 20.0
