"""
ConfTest Post-Hoc Confidence Calibration Module.

Implements Isotonic Regression, Temperature Scaling (Platt Scaling),
Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
and Reliability Diagram binning for Regression Test Selection.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from conftest.logging_config import get_logger

logger = get_logger(__name__)


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        y_true: Binary ground-truth labels {0, 1}.
        y_prob: Predicted confidence probabilities in range [0, 1].
        n_bins: Number of equal-width probability bins.

    Returns:
        Tuple of (ece, mce, reliability_diagram_bins).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    n_samples = len(y_true)
    bins_data = []

    for bin_idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Calculate sample indices in this probability bin
        if bin_idx == 0:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_accuracy = float(np.mean(y_true[in_bin]))
            bin_confidence = float(np.mean(y_prob[in_bin]))
            bin_error = abs(bin_accuracy - bin_confidence)

            ece += (bin_count / n_samples) * bin_error
            mce = max(mce, bin_error)

            bins_data.append({
                "bin_idx": bin_idx,
                "bin_range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "sample_count": bin_count,
                "confidence": round(bin_confidence, 4),
                "accuracy": round(bin_accuracy, 4),
                "calibration_gap": round(bin_error, 4),
            })
        else:
            bins_data.append({
                "bin_idx": bin_idx,
                "bin_range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "sample_count": 0,
                "confidence": round((bin_lower + bin_upper) / 2.0, 4),
                "accuracy": 0.0,
                "calibration_gap": 0.0,
            })

    return float(ece), float(mce), bins_data


class TemperatureScalingCalibrator:
    """Parametric Temperature Scaling calibrator optimizing scalar T > 0 on validation logits."""

    def __init__(self):
        self.temperature: float = 1.0

    def _logit(self, p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        p_c = np.clip(p, eps, 1.0 - eps)
        return np.log(p_c / (1.0 - p_c))

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> "TemperatureScalingCalibrator":
        """
        Fit optimal temperature T by minimizing negative log-likelihood on validation split.

        Args:
            val_probs: Raw uncalibrated probabilities in [0, 1].
            y_val: Binary ground-truth labels {0, 1}.
        """
        logits = self._logit(val_probs)
        y = np.asarray(y_val).astype(float)

        def nll_objective(t: np.ndarray) -> float:
            temp = t[0]
            scaled_logits = logits / max(1e-3, temp)
            probs = self._sigmoid(scaled_logits)
            eps = 1e-9
            probs = np.clip(probs, eps, 1.0 - eps)
            loss = -np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))
            return float(loss)

        res = minimize(nll_objective, x0=[1.0], bounds=[(0.01, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        logger.info(f"Fitted Temperature Scaling calibrator: T = {self.temperature:.4f}")
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling transformation."""
        logits = self._logit(probs)
        scaled = logits / max(1e-3, self.temperature)
        return self._sigmoid(scaled).astype(np.float32)


class IsotonicCalibrator:
    """Non-parametric piecewise monotonic calibration using Isotonic Regression."""

    def __init__(self):
        self.regressor = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> "IsotonicCalibrator":
        """Fit isotonic step regression on validation predictions."""
        self.regressor.fit(val_probs, y_val)
        logger.info("Fitted Isotonic Regression calibrator.")
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Transform raw probabilities to calibrated empirical probabilities."""
        calibrated = self.regressor.predict(probs)
        return np.clip(calibrated, 0.0, 1.0).astype(np.float32)


class ConfidenceCalibrator:
    """Unified Confidence Calibration Manager supporting Isotonic and Temperature Scaling."""

    def __init__(self, method: str = "isotonic"):
        """
        Initialize calibrator.

        Args:
            method: 'isotonic' or 'temperature_scaling'.
        """
        self.method = method.lower()
        if self.method == "isotonic":
            self.calibrator = IsotonicCalibrator()
        elif self.method in ("temperature", "temperature_scaling", "platt"):
            self.calibrator = TemperatureScalingCalibrator()
        else:
            raise ValueError(f"Unknown calibration method: {method}. Choose 'isotonic' or 'temperature_scaling'.")

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> "ConfidenceCalibrator":
        """Fit calibration model on validation predictions."""
        self.calibrator.fit(val_probs, y_val)
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Calibrate input probability array."""
        return self.calibrator.calibrate(probs)

    def save(self, filepath: str) -> str:
        """Serialize calibrator to disk."""
        out_path = Path(filepath).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(out_path))
        logger.info(f"Calibrator saved to {out_path}")
        return str(out_path)

    @classmethod
    def load(cls, filepath: str) -> "ConfidenceCalibrator":
        """Load serialized calibrator from disk."""
        in_path = Path(filepath).resolve()
        if not in_path.exists():
            raise FileNotFoundError(f"Calibrator file not found: {in_path}")
        instance = joblib.load(str(in_path))
        logger.info(f"Loaded {instance.method} calibrator from {in_path}")
        return instance
