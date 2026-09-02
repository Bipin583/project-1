"""
ConfTest Uncertainty Calibration & Selective Risk-Coverage Evaluator.

Analyzes the relationship between epistemic uncertainty and model errors,
computes risk-coverage trade-offs, and validates uncertainty awareness.

Usage:
    python scripts/uncertainty_eval.py --ensemble models/ensembles/5_seed_lgbm --dataset data/splits/test.csv
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conftest.models.ensemble import EnsembleUncertaintyPredictor
from conftest.models.trainer import prepare_feature_arrays
from conftest.logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Uncertainty Evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ensemble",
        type=str,
        default="./models/ensembles/5_seed_lgbm",
        help="Directory containing trained ensemble checkpoints.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/splits/test.csv",
        help="Path to evaluation dataset CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./reports/uncertainty_analysis.json",
        help="Destination path for uncertainty evaluation metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ens_dir = Path(args.ensemble)
    data_path = Path(args.dataset)

    if not ens_dir.exists():
        logger.error(f"Ensemble directory not found: {ens_dir}. Run train_ensemble.py first.")
        sys.exit(1)
    if not data_path.exists():
        logger.error(f"Dataset file not found: {data_path}.")
        sys.exit(1)

    logger.info(f"Loading ensemble from {ens_dir}...")
    ensemble = EnsembleUncertaintyPredictor.load_ensemble(str(ens_dir))

    logger.info(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    X, y = prepare_feature_arrays(df)

    res = ensemble.predict_with_uncertainty(X)
    probs = res["mean_prob"]
    stds = res["epistemic_std"]
    entropy = res["predictive_entropy"]

    # Compute absolute prediction error: |y - p_hat|
    errors = np.abs(y - probs)

    # Compute correlation between uncertainty and prediction error
    corr = float(np.corrcoef(stds, errors)[0, 1]) if len(stds) > 1 else 0.0

    # Risk-Coverage Analysis: Sort by uncertainty and evaluate retained sample risk
    coverage_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    risk_coverage = []

    sorted_indices = np.argsort(stds)  # Lowest uncertainty first
    n = len(df)

    for cov in coverage_levels:
        k = max(1, int(n * cov))
        retained_idx = sorted_indices[:k]
        retained_errors = errors[retained_idx]
        mean_risk = float(np.mean(retained_errors))

        risk_coverage.append({
            "coverage_pct": f"{cov*100:.0f}%",
            "retained_samples": k,
            "mean_prediction_error": round(mean_risk, 4),
            "max_retained_uncertainty": round(float(stds[retained_idx[-1]]), 4),
        })

    analysis_report = {
        "num_samples": n,
        "mean_epistemic_uncertainty": round(float(np.mean(stds)), 4),
        "p95_epistemic_uncertainty": round(float(np.percentile(stds, 95)), 4),
        "mean_predictive_entropy": round(float(np.mean(entropy)), 4),
        "uncertainty_error_correlation": round(corr, 4),
        "risk_coverage_curve": risk_coverage,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis_report, f, indent=2)

    logger.info(f"Uncertainty analysis exported to: {out_path}")
    logger.info("\n=== Risk-Coverage Analysis (Lower Risk on Higher Confidence Subset) ===")
    logger.info(pd.DataFrame(risk_coverage).to_string(index=False))


if __name__ == "__main__":
    main()
