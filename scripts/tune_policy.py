"""
ConfTest Selective Policy Threshold Optimizer CLI.

Performs grid search on the validation split to determine optimal (tau_abstain, tau_conf) thresholds
that maximize regression test reduction while strictly preventing escaped failures.

Usage:
    python scripts/tune_policy.py --val data/splits/val.csv --test data/splits/test.csv --ensemble models/ensembles/5_seed_lgbm --calibrator models/calibrator.joblib
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conftest.models.ensemble import EnsembleUncertaintyPredictor
from conftest.models.calibration import ConfidenceCalibrator
from conftest.models.policy import SelectivePredictionPolicy, CostBenefitModel
from conftest.models.trainer import prepare_feature_arrays
from conftest.logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Policy Threshold Optimizer",
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
        "--calibrator",
        type=str,
        default="./models/calibrator.joblib",
        help="Path to fitted calibrator artifact.",
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default="./models/policy_config.json",
        help="Destination path for optimized policy JSON configuration.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.25,
        help="Target fast mode test selection budget.",
    )
    return parser.parse_args()


def evaluate_thresholds_on_dataset(
    df: pd.DataFrame,
    ensemble: EnsembleUncertaintyPredictor,
    calibrator: ConfidenceCalibrator,
    tau_abstain: float,
    tau_conf: float,
    budget_ratio: float,
) -> Dict[str, Any]:
    """Run selective policy simulation across a dataset and count escaped failures and time reduction."""
    policy = SelectivePredictionPolicy(
        tau_abstain=tau_abstain,
        tau_conf=tau_conf,
        budget_ratio=budget_ratio,
    )

    total_tests_available = 0
    total_tests_executed = 0
    total_failures_available = 0
    total_failures_detected = 0
    total_abstentions = 0
    escaped_commits = 0
    commits_count = df["commit_sha"].nunique()

    for sha, group in df.groupby("commit_sha"):
        X, y = prepare_feature_arrays(group)
        test_ids = list(group["test_id"].astype(str))

        preds = ensemble.predict_with_uncertainty(X)
        cal_probs = calibrator.calibrate(preds["mean_prob"])
        stds = preds["epistemic_std"]

        diff_files = int(group["diff_num_files_changed"].iloc[0]) if "diff_num_files_changed" in group.columns else 1
        diff_churn = int(group["diff_total_churn"].iloc[0]) if "diff_total_churn" in group.columns else 10

        decision = policy.evaluate_commit(
            commit_sha=sha,
            candidate_test_ids=test_ids,
            calibrated_confidences=cal_probs,
            epistemic_uncertainties=stds,
            num_changed_files=diff_files,
            total_churn_lines=diff_churn,
        )

        selected_set = set(decision.selected_test_ids)
        actual_failing_set = set(group[group["label_failed"] == 1]["test_id"].astype(str))

        detected = len(selected_set.intersection(actual_failing_set))
        missed = len(actual_failing_set) - detected

        total_tests_available += len(test_ids)
        total_tests_executed += len(decision.selected_test_ids)
        total_failures_available += len(actual_failing_set)
        total_failures_detected += detected

        if decision.abstained:
            total_abstentions += 1
        if missed > 0 and not decision.abstained:
            escaped_commits += 1

    trr = max(0.0, 1.0 - (total_tests_executed / max(1, total_tests_available))) * 100.0
    recall = (total_failures_detected / max(1, total_failures_available)) * 100.0
    abstention_rate = (total_abstentions / max(1, commits_count)) * 100.0

    return {
        "tau_abstain": tau_abstain,
        "tau_conf": tau_conf,
        "test_reduction_trr_pct": round(trr, 2),
        "failure_recall_pct": round(recall, 2),
        "abstention_rate_pct": round(abstention_rate, 2),
        "escaped_commits": escaped_commits,
        "missed_failures": total_failures_available - total_failures_detected,
    }


def main():
    args = parse_args()

    val_path = Path(args.val)
    test_path = Path(args.test)
    ens_path = Path(args.ensemble)
    cal_path = Path(args.calibrator)

    logger.info(f"Loading ensemble ({ens_path}) and calibrator ({cal_path})...")
    ensemble = EnsembleUncertaintyPredictor.load_ensemble(str(ens_path))
    calibrator = ConfidenceCalibrator.load(str(cal_path))

    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Grid Search Candidates
    tau_abstain_grid = [0.005, 0.010, 0.015, 0.020, 0.030, 0.050]
    tau_conf_grid = [0.10, 0.30, 0.50, 0.60, 0.70, 0.80]

    logger.info(f"Starting grid search over {len(tau_abstain_grid)*len(tau_conf_grid)} threshold pairs on validation set...")

    best_config = None
    best_trr = -1.0
    tuning_records = []

    for tau_a in tau_abstain_grid:
        for tau_c in tau_conf_grid:
            eval_res = evaluate_thresholds_on_dataset(
                df=val_df,
                ensemble=ensemble,
                calibrator=calibrator,
                tau_abstain=tau_a,
                tau_conf=tau_c,
                budget_ratio=args.budget,
            )
            tuning_records.append(eval_res)

            # Constraint: Zero escaped failures on validation split, maximize test reduction TRR
            if eval_res["escaped_commits"] == 0 and eval_res["test_reduction_trr_pct"] > best_trr:
                best_trr = eval_res["test_reduction_trr_pct"]
                best_config = (tau_a, tau_c)

    # Fallback to robust default if all had escaped or TRR was 0
    if best_config is None:
        best_config = (0.015, 0.50)

    best_tau_a, best_tau_c = best_config
    logger.info(f"\nOptimal Thresholds Found: tau_abstain = {best_tau_a:.4f}, tau_conf = {best_tau_c:.2f} (Val TRR: {best_trr:.1f}%)")

    # Evaluate optimal thresholds on unseen Test Split
    test_eval = evaluate_thresholds_on_dataset(
        df=test_df,
        ensemble=ensemble,
        calibrator=calibrator,
        tau_abstain=best_tau_a,
        tau_conf=best_tau_c,
        budget_ratio=args.budget,
    )

    policy = SelectivePredictionPolicy(
        tau_abstain=best_tau_a,
        tau_conf=best_tau_c,
        budget_ratio=args.budget,
    )
    policy.save(args.output_config)

    report = {
        "optimized_policy": {
            "tau_abstain": best_tau_a,
            "tau_conf": best_tau_c,
            "budget_ratio": args.budget,
        },
        "validation_evaluation": evaluate_thresholds_on_dataset(
            df=val_df,
            ensemble=ensemble,
            calibrator=calibrator,
            tau_abstain=best_tau_a,
            tau_conf=best_tau_c,
            budget_ratio=args.budget,
        ),
        "unseen_test_evaluation": test_eval,
    }

    rep_path = Path("./reports/policy_tuning_report.json")
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("\n=== Final Selective Policy Evaluation on Unseen Test Split ===")
    logger.info(f"Test Reduction Ratio (TRR):  {test_eval['test_reduction_trr_pct']}%")
    logger.info(f"Failure Recall (FR):         {test_eval['failure_recall_pct']}%")
    logger.info(f"Abstention Fallback Rate:   {test_eval['abstention_rate_pct']}%")
    logger.info(f"Escaped Commits:             {test_eval['escaped_commits']}")
    logger.info(f"Policy Saved to:             {args.output_config}")
    logger.info(f"Tuning Report:               {rep_path}")


if __name__ == "__main__":
    main()
