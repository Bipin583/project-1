"""
ConfTest Online Continuous Learning & Concept Drift Adaptation Subsystem.

Implements Page-Hinkley drift detection and sliding-window experience replay
to adapt RTS models to evolving codebases without catastrophic forgetting.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import lightgbm as lgb

from conftest.features.pipeline import FEATURE_NAMES
from conftest.models.lightgbm_model import LightGBMTestPredictor
from conftest.models.task_definition import compute_class_weights
from conftest.logging_config import get_logger

logger = get_logger(__name__)


class PageHinkleyDriftDetector:
    """
    Page-Hinkley test for online statistical concept drift detection on streaming error signals.
    """

    def __init__(self, delta: float = 0.005, threshold: float = 15.0, alpha: float = 0.99):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.mean_error = 0.0
        self.cumulative_sum = 0.0
        self.min_cumulative_sum = 0.0
        self.num_samples = 0

    def update(self, error: float) -> bool:
        """
        Update detector with instantaneous test error (e.g., |y_true - y_prob|).
        Returns True if statistical concept drift is detected.
        """
        self.num_samples += 1
        # Exponentially decaying historical error mean
        if self.num_samples == 1:
            self.mean_error = error
        else:
            self.mean_error = self.alpha * self.mean_error + (1.0 - self.alpha) * error

        # Update cumulative sum
        self.cumulative_sum += (error - self.mean_error - self.delta)
        if self.cumulative_sum < self.min_cumulative_sum:
            self.min_cumulative_sum = self.cumulative_sum

        ph_statistic = self.cumulative_sum - self.min_cumulative_sum
        if ph_statistic > self.threshold:
            # Drift detected, reset detector
            self.reset()
            return True
        return False

    def reset(self) -> None:
        """Reset internal accumulator states."""
        self.cumulative_sum = 0.0
        self.min_cumulative_sum = 0.0
        self.num_samples = 0


class OnlineContinualLearner:
    """
    Manages online experience replay and incremental model retraining upon concept drift.
    """

    def __init__(
        self,
        buffer_capacity: int = 1000,
        drift_threshold: float = 10.0,
        random_seed: int = 42,
    ):
        self.buffer_capacity = buffer_capacity
        self.random_seed = random_seed
        self.detector = PageHinkleyDriftDetector(threshold=drift_threshold)

        self.buffer_X: List[np.ndarray] = []
        self.buffer_y: List[int] = []
        self.current_model: Optional[LightGBMTestPredictor] = None
        self.adaptation_count: int = 0

    def initialize_base_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit initial base model on historical training data."""
        self.current_model = LightGBMTestPredictor(random_seed=self.random_seed, n_estimators=30)
        self.current_model.train(X_train=X_train, y_train=y_train)

        # Seed replay buffer
        n_seed = min(len(X_train), self.buffer_capacity)
        self.buffer_X = list(X_train[-n_seed:])
        self.buffer_y = list(y_train[-n_seed:])

    def process_streaming_commit(
        self,
        X_commit: np.ndarray,
        y_commit: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Process a newly executed commit:
        1. Predict with current model.
        2. Compute error and update drift detector.
        3. Add samples to experience replay buffer.
        4. If drift detected, trigger incremental model adaptation.
        """
        if self.current_model is None:
            raise RuntimeError("Model not initialized. Call initialize_base_model first.")

        probs = self.current_model.predict_proba(X_commit)
        abs_errors = np.abs(y_commit - probs)
        mean_commit_error = float(np.mean(abs_errors))

        # Check drift
        drift_detected = self.detector.update(mean_commit_error)

        # Append to circular replay buffer
        for x_row, y_val in zip(X_commit, y_commit):
            self.buffer_X.append(x_row)
            self.buffer_y.append(int(y_val))
            if len(self.buffer_X) > self.buffer_capacity:
                self.buffer_X.pop(0)
                self.buffer_y.pop(0)

        adapted = False
        if drift_detected:
            self._adapt_model()
            adapted = True

        return {
            "mean_commit_error": round(mean_commit_error, 4),
            "drift_detected": drift_detected,
            "model_adapted": adapted,
            "buffer_size": len(self.buffer_X),
            "total_adaptations": self.adaptation_count,
        }

    def _adapt_model(self) -> None:
        """Incrementally retrain model on refreshed replay buffer."""
        self.adaptation_count += 1
        X_buf = np.array(self.buffer_X, dtype=np.float32)
        y_buf = np.array(self.buffer_y, dtype=int)

        logger.info(f"Concept drift detected! Triggering adaptation #{self.adaptation_count} on {len(X_buf)} buffer samples...")
        new_model = LightGBMTestPredictor(random_seed=self.random_seed, n_estimators=30)
        new_model.train(X_train=X_buf, y_train=y_buf)
        self.current_model = new_model
