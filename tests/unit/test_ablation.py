"""
Unit tests for Feature Ablation and Contribution Study module.
"""

import numpy as np
import pytest

from conftest.features.pipeline import FEATURE_NAMES
from conftest.evaluation.ablation import (
    FEATURE_GROUPS,
    get_feature_indices,
    FeatureAblationStudy,
)


def test_feature_indices_mapping():
    """Verify feature group index resolution matches schema."""
    diff_idx = get_feature_indices(["diff_churn"])
    assert len(diff_idx) == 12

    ast_idx = get_feature_indices(["ast_complexity"])
    assert len(ast_idx) == 6

    dep_idx = get_feature_indices(["dependency_graph"])
    assert len(dep_idx) == 6

    hist_idx = get_feature_indices(["history_telemetry"])
    assert len(hist_idx) == 8

    all_idx = get_feature_indices(list(FEATURE_GROUPS.keys()))
    assert len(all_idx) == 32
    assert sorted(all_idx) == list(range(32))


def test_ablation_study_synthetic_dataset():
    """Verify FeatureAblationStudy executes on synthetic data and computes metric deltas."""
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

    study = FeatureAblationStudy(random_seed=42)
    results = study.run_study(X_train, y_train, X_val, y_val, X_test, y_test)

    assert "full_model" in results
    assert "leave_one_group_out" in results
    assert "single_group_only" in results

    assert "pr_auc" in results["full_model"]
    assert "without_history_telemetry" in results["leave_one_group_out"]
    assert "delta_pr_auc" in results["leave_one_group_out"]["without_history_telemetry"]

    assert "dependency_graph_only" in results["single_group_only"]
    assert "delta_recall" in results["single_group_only"]["dependency_graph_only"]
