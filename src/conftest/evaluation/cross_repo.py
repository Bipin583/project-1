"""
ConfTest Multi-Repository Cross-Project Generalization Engine.

Evaluates Zero-Shot transferability and Leave-One-Project-Out (LOPO) cross-validation
to benchmark RTS model generalization across heterogeneous codebases.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from conftest.features.pipeline import FEATURE_NAMES
from conftest.models.lightgbm_model import LightGBMTestPredictor
from conftest.models.calibration import TemperatureScalingCalibrator, compute_ece
from conftest.models.trainer import evaluate_predictions
from conftest.logging_config import get_logger

logger = get_logger(__name__)


class CrossRepoEvaluator:
    """
    Evaluates RTS generalization across distinct repositories in Leave-One-Project-Out fashion.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def evaluate_lopo_transfer(
        self,
        repo_datasets: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, Any]:
        """
        Execute Leave-One-Project-Out (LOPO) transfer evaluation.

        For each target repo R:
            Train on all repos != R
            Calibrate on held-out validation of training repos
            Evaluate zero-shot on target repo R

        Args:
            repo_datasets: Dict mapping repo_name -> {'X': ndarray, 'y': ndarray}.

        Returns:
            Dictionary containing individual repo transfer results and macro-averaged metrics.
        """
        repo_names = list(repo_datasets.keys())
        if len(repo_names) < 2:
            raise ValueError("Cross-repo evaluation requires at least 2 repositories.")

        results: Dict[str, Any] = {"per_repository": {}, "macro_average": {}}
        macro_pr_aucs: List[float] = []
        macro_roc_aucs: List[float] = []
        macro_recalls: List[float] = []
        macro_eces: List[float] = []

        for target_repo in repo_names:
            logger.info(f"Evaluating LOPO transfer to target repository: {target_repo}...")

            # 1. Split training (all other repos) and target
            train_X_list, train_y_list = [], []
            for source_repo in repo_names:
                if source_repo != target_repo:
                    train_X_list.append(repo_datasets[source_repo]["X"])
                    train_y_list.append(repo_datasets[source_repo]["y"])

            X_train_combined = np.vstack(train_X_list)
            y_train_combined = np.concatenate(train_y_list)

            X_target = repo_datasets[target_repo]["X"]
            y_target = repo_datasets[target_repo]["y"]

            # Sub-split combined training for calibration (85% train, 15% cal)
            n_comb = len(y_train_combined)
            n_tr = int(n_comb * 0.85)
            X_tr, y_tr = X_train_combined[:n_tr], y_train_combined[:n_tr]
            X_cal, y_cal = X_train_combined[n_tr:], y_train_combined[n_tr:]

            # 2. Train model
            model = LightGBMTestPredictor(random_seed=self.random_seed, n_estimators=30)
            model.train(X_train=X_tr, y_train=y_tr)

            # 3. Fit Temperature Scaling calibrator
            calibrator = TemperatureScalingCalibrator()
            val_raw = model.predict_proba(X_cal)
            calibrator.fit(val_raw, y_cal)

            # 4. Zero-shot prediction on target repo
            target_raw = model.predict_proba(X_target)
            target_cal = calibrator.calibrate(target_raw)

            # Metrics
            metrics = evaluate_predictions(y_target, target_cal)
            ece_res = compute_ece(y_target, target_cal)

            k = max(1, int(len(y_target) * 0.25))
            top_k = np.argsort(target_cal)[-k:]
            total_fails = max(1, int(np.sum(y_target)))
            recall_25 = float(np.sum(y_target[top_k]) / total_fails)

            repo_res = {
                "train_samples": len(X_tr),
                "target_samples": len(X_target),
                "target_failure_rate": round(float(np.mean(y_target)), 4),
                "zero_shot_pr_auc": round(float(metrics["pr_auc"]), 4),
                "zero_shot_roc_auc": round(float(metrics["roc_auc"]), 4),
                "zero_shot_brier_score": round(float(metrics["brier_score"]), 4),
                "zero_shot_calibrated_ece": round(float(ece_res[0]), 4),
                "zero_shot_recall_at_25budget": round(recall_25, 4),
            }

            results["per_repository"][target_repo] = repo_res
            macro_pr_aucs.append(repo_res["zero_shot_pr_auc"])
            macro_roc_aucs.append(repo_res["zero_shot_roc_auc"])
            macro_recalls.append(repo_res["zero_shot_recall_at_25budget"])
            macro_eces.append(repo_res["zero_shot_calibrated_ece"])

        results["macro_average"] = {
            "mean_pr_auc": round(float(np.mean(macro_pr_aucs)), 4),
            "mean_roc_auc": round(float(np.mean(macro_roc_aucs)), 4),
            "mean_recall_at_25budget": round(float(np.mean(macro_recalls)), 4),
            "mean_calibrated_ece": round(float(np.mean(macro_eces)), 4),
            "total_repositories_evaluated": len(repo_names),
        }

        return results
