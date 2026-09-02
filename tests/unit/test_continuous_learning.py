"""
Unit tests for Online Continuous Learning and Concept Drift Adaptation.
"""

import numpy as np
import pytest

from conftest.features.pipeline import FEATURE_NAMES
from conftest.models.continuous_learning import PageHinkleyDriftDetector, OnlineContinualLearner


def test_page_hinkley_drift_detection():
    """Verify Page-Hinkley triggers on persistent error surge."""
    detector = PageHinkleyDriftDetector(delta=0.01, threshold=2.0)

    # Low error baseline
    for _ in range(20):
        drift = detector.update(0.05)
        assert drift is False

    # High error surge
    drift_detected = False
    for _ in range(30):
        if detector.update(0.85):
            drift_detected = True
            break

    assert drift_detected is True


def test_online_continual_learner_lifecycle():
    """Verify OnlineContinualLearner buffer management and model adaptation."""
    rng = np.random.RandomState(42)
    n_feats = len(FEATURE_NAMES)

    X_init = rng.randn(40, n_feats).astype(np.float32)
    y_init = (rng.rand(40) < 0.20).astype(int)
    y_init[0] = 1

    learner = OnlineContinualLearner(buffer_capacity=50, drift_threshold=1.5, random_seed=42)
    learner.initialize_base_model(X_init, y_init)

    assert len(learner.buffer_X) == 40
    assert learner.adaptation_count == 0

    # Stream clean commit
    res1 = learner.process_streaming_commit(rng.randn(5, n_feats).astype(np.float32), np.zeros(5, dtype=int))
    assert res1["buffer_size"] == 45

    # Stream out-of-distribution high-error commit to induce drift
    drift_seen = False
    for _ in range(15):
        X_drift = (rng.randn(5, n_feats) + 5.0).astype(np.float32)
        y_drift = np.ones(5, dtype=int)
        res = learner.process_streaming_commit(X_drift, y_drift)
        if res["drift_detected"]:
            drift_seen = True
            break

    assert drift_seen is True
    assert learner.adaptation_count >= 1
