"""
ConfTest Statistical Significance & Non-Parametric Hypothesis Testing Engine.

Implements Wilcoxon Signed-Rank tests, Cliff's Delta non-parametric effect sizes,
and Bootstrap 95% Confidence Intervals for empirical RTS evaluation.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

from conftest.logging_config import get_logger

logger = get_logger(__name__)


def compute_cliffs_delta(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """
    Compute Cliff's delta non-parametric effect size between two distributions.

    Formula:
        delta = ( #(x > y) - #(x < y) ) / (len(x) * len(y))

    Thresholds (Romano et al., 2006):
        |delta| < 0.147: Negligible
        0.147 <= |delta| < 0.330: Small
        0.330 <= |delta| < 0.474: Medium
        |delta| >= 0.474: Large

    Args:
        x: Sample distribution 1 (e.g. ConfTest metrics).
        y: Sample distribution 2 (e.g. Baseline metrics).

    Returns:
        Tuple of (delta_float, interpretation_str).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0, "Negligible"

    # Efficient pairwise comparison using matrix broadcasting
    greater = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])

    delta = float((greater - less) / (n_x * n_y))
    abs_d = abs(delta)

    if abs_d < 0.147:
        magnitude = "Negligible"
    elif abs_d < 0.330:
        magnitude = "Small"
    elif abs_d < 0.474:
        magnitude = "Medium"
    else:
        magnitude = "Large"

    return round(delta, 4), magnitude


def compute_wilcoxon_test(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = "two-sided",
) -> Tuple[float, float, bool]:
    """
    Compute Wilcoxon Signed-Rank test for paired non-parametric samples.

    Args:
        x: ConfTest metric array across commits.
        y: Baseline metric array across paired commits.
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        Tuple of (statistic_W, p_value, is_significant_at_p05).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    diff = x - y
    if np.all(diff == 0):
        return 0.0, 1.0, False

    # Filter zero differences for Wilcoxon
    non_zero_diff = diff[diff != 0]
    if len(non_zero_diff) < 5:
        # Insufficient non-zero pairs for asymptotic normal approximation
        return 0.0, 1.0, False

    try:
        res = stats.wilcoxon(x, y, alternative=alternative, zero_method="wilcox")
        stat_w = float(res.statistic)
        p_val = float(res.pvalue)
        is_sig = bool(p_val < 0.05)
        return round(stat_w, 4), round(p_val, 5), is_sig
    except Exception as exc:
        logger.warning(f"Wilcoxon calculation fallback ({exc}).")
        return 0.0, 1.0, False


def bootstrap_confidence_interval(
    data: np.ndarray,
    num_bootstraps: int = 1000,
    ci: float = 0.95,
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    random_seed: int = 42,
) -> Dict[str, float]:
    """
    Compute non-parametric percentile bootstrap confidence interval.

    Args:
        data: 1D array of sample values.
        num_bootstraps: Number of bootstrap resamples (default: 1000).
        ci: Confidence level (default: 0.95 for 95% CI).
        statistic_fn: Function computing summary statistic (default: mean).
        random_seed: Reproducibility seed.

    Returns:
        Dictionary with keys: mean, median, ci_lower, ci_upper.
    """
    data = np.asarray(data).ravel()
    n = len(data)
    if n == 0:
        return {"mean": 0.0, "median": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    rng = np.random.RandomState(random_seed)
    boot_stats = np.empty(num_bootstraps, dtype=np.float64)

    for i in range(num_bootstraps):
        resample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = statistic_fn(resample)

    alpha = (1.0 - ci) / 2.0
    low_pct = alpha * 100.0
    high_pct = (1.0 - alpha) * 100.0

    ci_lower = float(np.percentile(boot_stats, low_pct))
    ci_upper = float(np.percentile(boot_stats, high_pct))
    point_mean = float(statistic_fn(data))
    point_median = float(np.median(data))

    return {
        "mean": round(point_mean, 4),
        "median": round(point_median, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "confidence_level": ci,
    }


class StatisticalSignificanceTester:
    """Orchestrates comprehensive pairwise statistical significance testing across RTS baselines."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def evaluate_pairwise(
        self,
        conftest_metrics: Dict[str, np.ndarray],
        baseline_metrics: Dict[str, np.ndarray],
        baseline_name: str,
    ) -> Dict[str, Any]:
        """
        Evaluate ConfTest vs. a specific baseline across Failure Recall and Time Reduction.

        Args:
            conftest_metrics: Dict with keys 'failure_recall' and 'time_reduction'.
            baseline_metrics: Dict with keys 'failure_recall' and 'time_reduction'.
            baseline_name: Name of the baseline being compared.

        Returns:
            Structured statistical report dictionary.
        """
        results = {"baseline_name": baseline_name}

        for metric in ("failure_recall", "time_reduction"):
            c_vals = conftest_metrics.get(metric, np.array([]))
            b_vals = baseline_metrics.get(metric, np.array([]))

            w_stat, p_val, is_sig = compute_wilcoxon_test(c_vals, b_vals, alternative="two-sided")
            delta, magnitude = compute_cliffs_delta(c_vals, b_vals)
            c_boot = bootstrap_confidence_interval(c_vals, random_seed=self.random_seed)
            b_boot = bootstrap_confidence_interval(b_vals, random_seed=self.random_seed)

            results[metric] = {
                "wilcoxon_W": w_stat,
                "p_value": p_val,
                "statistically_significant_p05": is_sig,
                "cliffs_delta": delta,
                "effect_size": magnitude,
                "conftest_95ci": c_boot,
                "baseline_95ci": b_boot,
            }

        return results
