"""
ConfTest Prediction Task Definition & Imbalance Management Module.

Formulates the regression test selection prediction task:
Given a commit diff c and candidate test case t, predict P(Test t Fails on c | x_{c,t}).

Provides label definitions, class weight balancing, and flaky-test sample down-weighting.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from conftest.logging_config import get_logger

logger = get_logger(__name__)


class LabelDefinition(str, Enum):
    """Supported ground-truth label formulations."""
    OPTION_A_DIRECT_FAILURE = "direct_failure"  # Positive if test failed on commit c
    OPTION_B_DESCENDANT_FAILURE = "descendant_failure"  # Positive if test failed on c or immediate descendant
    OPTION_C_REPLAY_ORACLE = "replay_oracle"  # Positive if test is part of failure-detecting oracle set


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate inverse frequency class weights to counteract severe class imbalance.

    Args:
        y: Binary target array (0 = pass, 1 = fail).

    Returns:
        Dictionary mapping class {0: w_neg, 1: w_pos}.
    """
    n_samples = len(y)
    if n_samples == 0:
        return {0: 1.0, 1: 1.0}

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    if n_pos == 0:
        logger.warning("Zero positive failure samples found in dataset.")
        return {0: 1.0, 1: 10.0}

    # Standard balanced class weighting: w_j = n_samples / (n_classes * n_samples_j)
    w_neg = n_samples / (2.0 * max(1, n_neg))
    w_pos = n_samples / (2.0 * max(1, n_pos))

    # Scale_pos_weight for LightGBM/XGBoost: n_neg / n_pos
    scale_pos_weight = n_neg / max(1, n_pos)

    logger.info(
        f"Class balance: {n_pos} positive (fail) / {n_neg} negative (pass) "
        f"({(n_pos/n_samples)*100:.2f}% failure rate). Scale pos weight: {scale_pos_weight:.2f}"
    )

    return {0: float(w_neg), 1: float(w_pos), "scale_pos_weight": float(scale_pos_weight)}


def compute_sample_weights(
    df: pd.DataFrame,
    label_col: str = "label_failed",
    flaky_col: str = "hist_flaky_score",
    flaky_discount: float = 0.5,
) -> np.ndarray:
    """
    Compute sample weights giving higher importance to true regression failures
    and down-weighting suspected flaky non-deterministic failures.

    Args:
        df: Pandas DataFrame containing dataset.
        label_col: Name of binary failure label column.
        flaky_col: Name of flaky score feature column.
        flaky_discount: Multiplier applied to samples with high flakiness.

    Returns:
        1D NumPy array of sample weights.
    """
    y = df[label_col].values.astype(int)
    weights_dict = compute_class_weights(y)

    sample_weights = np.ones(len(df), dtype=np.float32)

    for i in range(len(df)):
        cls = y[i]
        base_w = weights_dict.get(cls, 1.0)

        # Discount flaky test outcomes
        if flaky_col in df.columns:
            flaky_score = float(df.iloc[i][flaky_col])
            if flaky_score > 0.1:
                base_w *= flaky_discount

        sample_weights[i] = base_w

    return sample_weights
