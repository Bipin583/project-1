"""
Unit tests for Temperature Scaling, Calibration, and ECE calculation.
"""
import numpy as np
from src.models.calibration import TemperatureCalibrator, UncertaintyEstimator

def test_temperature_calibration():
    # Generate uncalibrated overconfident logits
    np.random.seed(42)
    logits = np.random.normal(loc=2.0, scale=1.5, size=200)
    labels = (logits > 1.5).astype(int)
    # Add some label noise
    labels[np.random.rand(200) < 0.2] = 0

    calibrator = TemperatureCalibrator()
    optimal_temp = calibrator.fit(logits, labels)

    assert optimal_temp > 0.0
    calibrated_probs = calibrator.predict_proba(logits)
    assert np.all(calibrated_probs >= 0.0)
    assert np.all(calibrated_probs <= 1.0)

def test_ece_computation():
    probs = np.array([0.9, 0.8, 0.2, 0.1, 0.95])
    labels = np.array([1, 1, 0, 0, 1])

    ece = UncertaintyEstimator.compute_ece(probs, labels, n_bins=5)
    assert ece >= 0.0
    assert ece <= 1.0
