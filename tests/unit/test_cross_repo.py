"""
Unit tests for Multi-Repository Cross-Project Generalization module.
"""

import numpy as np
import pytest

from conftest.features.pipeline import FEATURE_NAMES
from conftest.evaluation.cross_repo import CrossRepoEvaluator


def test_cross_repo_evaluator_validation():
    """Verify input validation rules."""
    evaluator = CrossRepoEvaluator()
    with pytest.raises(ValueError, match="at least 2 repositories"):
        evaluator.evaluate_lopo_transfer({"single_repo": {"X": np.zeros((10, 32)), "y": np.zeros(10)}})


def test_cross_repo_evaluator_lopo_execution():
    """Verify Leave-One-Project-Out transfer across 3 synthetic repos."""
    rng = np.random.RandomState(42)
    n_feats = len(FEATURE_NAMES)

    def make_repo(n):
        X = rng.randn(n, n_feats).astype(np.float32)
        y = (rng.rand(n) < 0.10).astype(int)
        y[0] = 1
        y[1] = 1
        return {"X": X, "y": y}

    repos = {
        "repo_a": make_repo(60),
        "repo_b": make_repo(50),
        "repo_c": make_repo(40),
    }

    evaluator = CrossRepoEvaluator(random_seed=42)
    results = evaluator.evaluate_lopo_transfer(repos)

    assert "per_repository" in results
    assert "macro_average" in results
    assert len(results["per_repository"]) == 3
    assert "repo_a" in results["per_repository"]
    assert results["macro_average"]["total_repositories_evaluated"] == 3
    assert 0.0 <= results["macro_average"]["mean_pr_auc"] <= 1.0
