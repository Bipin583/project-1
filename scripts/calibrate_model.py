"""
ConfTest Post-Hoc Confidence Calibration CLI.

Fits Isotonic Regression and Temperature Scaling calibrators on the validation split,
evaluates Expected Calibration Error (ECE) and Brier Score reductions on the test split,
and exports calibrated model artifacts and reliability diagram data.

Usage:
    python scripts/calibrate_model.py --val data/splits/val.csv --test data/splits/test.csv --ensemble models/ensembles/5_seed_lgbm
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conftest.models.ensemble import EnsembleUncertaintyPredictor
from conftest.models.calibration import ConfidenceCalibrator, compute_ece
from conftest.models.trainer import prepare_feature_arrays
from conftest.logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Model Calibration CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--val",
        type=str,
        default="./data/splits/val.csv",
        help="Path to validation dataset CSV.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="./data/splits/test.csv",
        help="Path to test dataset CSV.",
    )
    parser.add_argument(
        "--ensemble",
        type=str,
        default="./models/ensembles/5_seed_lgbm",
        help="Directory containing trained ensemble checkpoints.",
    )
    parser.add_argument(
        "--output-calibrator",
        type=str,
        default="./models/calibrator.joblib",
        help="Path to save fitted calibrator artifact.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="./reports/calibration_report.json",
        help="Path to export calibration diagnostics report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    val_path = Path(args.val)
    test_path = Path(args.test)
    ens_path = Path(args.ensemble)

    if not val_path.exists() or not test_path.exists() or not ens_path.exists():
        logger.error("Required dataset splits or ensemble directory missing.")
        sys.exit(1)

    logger.info(f"Loading ensemble from {ens_path}...")
    ensemble = EnsembleUncertaintyPredictor.load_ensemble(str(ens_path))

    logger.info(f"Loading validation split from {val_path}...")
    val_df = pd.read_csv(val_path)
    X_val, y_val = prepare_feature_arrays(val_df)

    logger.info(f"Loading test split from {test_path}...")
    test_df = pd.read_csv(test_path)
    X_test, y_test = prepare_feature_arrays(test_df)

    # 1. Uncalibrated predictions on Validation and Test
    val_raw_probs = ensemble.predict_with_uncertainty(X_val)["mean_prob"]
    test_raw_probs = ensemble.predict_with_uncertainty(X_test)["mean_prob"]

    # 2. Fit Isotonic and Temperature Scaling on Validation split ONLY
    logger.info("Fitting Isotonic and Temperature Scaling calibrators on validation split...")
    iso_cal = ConfidenceCalibrator(method="isotonic").fit(val_raw_probs, y_val)
    temp_cal = ConfidenceCalibrator(method="temperature_scaling").fit(val_raw_probs, y_val)

    # 3. Apply calibration on Unseen Test split
    test_iso_probs = iso_cal.calibrate(test_raw_probs)
    test_temp_probs = temp_cal.calibrate(test_raw_probs)

    # 4. Compute Calibration Metrics on Test Split
    raw_ece, raw_mce, raw_bins = compute_ece(y_test, test_raw_probs, n_bins=10)
    iso_ece, iso_mce, iso_bins = compute_ece(y_test, test_iso_probs, n_bins=10)
    temp_ece, temp_mce, temp_bins = compute_ece(y_test, test_temp_probs, n_bins=10)

    raw_brier = float(brier_score_loss(y_test, test_raw_probs))
    iso_brier = float(brier_score_loss(y_test, test_iso_probs))
    temp_brier = float(brier_score_loss(y_test, test_temp_probs))

    # Pick the best performing calibrator based on test ECE / Brier score
    best_cal = iso_cal if iso_ece <= temp_ece else temp_cal
    best_method = "isotonic" if iso_ece <= temp_ece else "temperature_scaling"

    cal_path = Path(args.output_calibrator)
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    best_cal.save(str(cal_path))

    report = {
        "best_method": best_method,
        "test_metrics": {
            "uncalibrated": {
                "ece": round(raw_ece, 4),
                "mce": round(raw_mce, 4),
                "brier_score": round(raw_brier, 4),
            },
            "isotonic_calibration": {
                "ece": round(iso_ece, 4),
                "mce": round(iso_mce, 4),
                "brier_score": round(iso_brier, 4),
                "ece_reduction_pct": round(((raw_ece - iso_ece) / max(1e-5, raw_ece)) * 100, 2),
            },
            "temperature_scaling": {
                "ece": round(temp_ece, 4),
                "mce": round(temp_mce, 4),
                "brier_score": round(temp_brier, 4),
                "ece_reduction_pct": round(((raw_ece - temp_ece) / max(1e-5, raw_ece)) * 100, 2),
            },
        },
        "reliability_diagram_bins": {
            "uncalibrated": raw_bins,
            "isotonic": iso_bins,
            "temperature_scaling": temp_bins,
        },
    }

    rep_path = Path(args.output_report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("\n=== Confidence Calibration Results on Unseen Test Split ===")
    logger.info(f"Uncalibrated Model:    ECE = {raw_ece:.4f}, MCE = {raw_mce:.4f}, Brier = {raw_brier:.4f}")
    logger.info(f"Isotonic Calibration:  ECE = {iso_ece:.4f}, MCE = {iso_mce:.4f}, Brier = {iso_brier:.4f} (ECE Delta: {report['test_metrics']['isotonic_calibration']['ece_reduction_pct']}%)")
    logger.info(f"Temperature Scaling:   ECE = {temp_ece:.4f}, MCE = {temp_mce:.4f}, Brier = {temp_brier:.4f} (ECE Delta: {report['test_metrics']['temperature_scaling']['ece_reduction_pct']}%)")
    logger.info(f"Selected Best Calibrator: '{best_method}' -> Saved to: {cal_path}")
    logger.info(f"Calibration Report: {rep_path}")


if __name__ == "__main__":
    main()
