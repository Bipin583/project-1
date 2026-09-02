"""
ConfTest SHAP Model Explainability Engine.

Uses TreeExplainer on gradient-boosted tree ensembles to compute exact Shapley
feature attribution values (phi_i) explaining why test failure risk increased or decreased.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import shap

from conftest.features.pipeline import FEATURE_NAMES
from conftest.models.lightgbm_model import LightGBMTestPredictor
from conftest.logging_config import get_logger

logger = get_logger(__name__)


class ShapExplainer:
    """Computes exact Shapley feature attributions using SHAP TreeExplainer."""

    def __init__(self, model_predictor: LightGBMTestPredictor):
        """
        Initialize SHAP TreeExplainer.

        Args:
            model_predictor: Trained LightGBMTestPredictor instance.
        """
        if model_predictor.model is None:
            raise ValueError("Model predictor must be trained before initializing SHAP explainer.")

        self.model_predictor = model_predictor
        self.feature_names = list(FEATURE_NAMES)
        self.explainer = shap.TreeExplainer(self.model_predictor.model)
        ev = np.asarray(self.explainer.expected_value).flatten()
        self.expected_value = float(ev[1] if len(ev) > 1 else ev[0])
        logger.info(f"Initialized SHAP TreeExplainer (Base expected value: {self.expected_value:.4f})")

    def explain_instance(
        self,
        x_vector: np.ndarray,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Compute SHAP feature attribution breakdown for a single test case prediction.

        Args:
            x_vector: 1D feature array of shape (32,) or (1, 32).
            top_k: Number of top driving features to extract.

        Returns:
            Dictionary containing base value, prediction score, and top positive/negative feature drivers.
        """
        if x_vector.ndim == 1:
            x_vector = x_vector.reshape(1, -1)

        raw_shap = self.explainer.shap_values(x_vector)

        # Handle binary classifier list output [class_0_shap, class_1_shap] vs 2D array
        if isinstance(raw_shap, list) and len(raw_shap) > 1:
            shap_values = raw_shap[1][0]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            shap_values = raw_shap[0, :, 1]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 2:
            shap_values = raw_shap[0]
        else:
            shap_values = np.asarray(raw_shap).flatten()

        predicted_prob = float(self.model_predictor.predict_proba(x_vector)[0])

        feature_contributions = []
        for i, name in enumerate(self.feature_names):
            phi = float(shap_values[i])
            val = float(x_vector[0, i])
            feature_contributions.append({
                "feature": name,
                "feature_value": round(val, 4),
                "shap_attribution": round(phi, 4),
                "impact": "INCREASES_RISK" if phi > 0 else "DECREASES_RISK",
            })

        # Sort by absolute magnitude |phi|
        sorted_by_abs = sorted(feature_contributions, key=lambda x: abs(x["shap_attribution"]), reverse=True)
        top_positive = [f for f in sorted_by_abs if f["shap_attribution"] > 0][:top_k]
        top_negative = [f for f in sorted_by_abs if f["shap_attribution"] < 0][:top_k]

        return {
            "predicted_probability": round(predicted_prob, 4),
            "base_expected_value": round(self.expected_value, 4),
            "top_risk_increasing_features": top_positive,
            "top_risk_decreasing_features": top_negative,
            "all_attributions": feature_contributions,
        }

    def explain_dataset(self, X_matrix: np.ndarray) -> Dict[str, Any]:
        """Compute global mean absolute SHAP importance across a dataset."""
        raw_shap = self.explainer.shap_values(X_matrix)

        if isinstance(raw_shap, list) and len(raw_shap) > 1:
            shap_vals = raw_shap[1]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            shap_vals = raw_shap[:, :, 1]
        else:
            shap_vals = np.asarray(raw_shap)

        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        global_rankings = [
            {"feature": self.feature_names[i], "mean_abs_shap": round(float(mean_abs_shap[i]), 5)}
            for i in range(len(self.feature_names))
        ]
        global_rankings.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

        return {
            "num_samples": len(X_matrix),
            "global_feature_importance_shap": global_rankings,
        }
