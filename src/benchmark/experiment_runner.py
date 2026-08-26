"""
ConfTest Benchmark Experiment Runner
Executes comprehensive comparative evaluation across 8 baselines and computes
TRR, ETR, Failure Recall (FR), Missed-Failure Rate (MFR), ECE, and Brier Score.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.benchmark.dataset_generator import BenchmarkDatasetGenerator
from src.models.lightgbm_model import TestFailureScorer
from src.models.calibration import TemperatureCalibrator, UncertaintyEstimator
from src.engine.selective_engine import SelectiveDecisionEngine

class ExperimentRunner:
    """
    Evaluates 8 baseline regression test selection strategies against ConfTest.
    """
    def __init__(self, n_commits: int = 500, n_tests: int = 50):
        self.generator = BenchmarkDatasetGenerator(n_commits=n_commits, n_tests=n_tests)
        self.feature_cols = TestFailureScorer.FEATURE_NAMES

    def run_all(self) -> pd.DataFrame:
        train_df, cal_df, test_df = self.generator.generate()

        X_train = train_df[self.feature_cols].values
        y_train = train_df["label"].values

        X_cal = cal_df[self.feature_cols].values
        y_cal = cal_df["label"].values

        X_test = test_df[self.feature_cols].values
        y_test = test_df["label"].values

        # 1. Train LightGBM Scorer
        scorer = TestFailureScorer(n_estimators=100, learning_rate=0.05)
        scorer.train(X_train, y_train, self.feature_cols)

        # 2. Fit Post-Hoc Temperature Calibration on Calibration Set
        cal_logits = scorer.predict_raw_logits(X_cal)
        calibrator = TemperatureCalibrator()
        temp = calibrator.fit(cal_logits, y_cal)
        print(f"[ConfTest Benchmark] Fitted Temperature Parameter: T = {temp:.4f}")

        # 3. Model Predictions on Unseen Test Set
        raw_test_probs = scorer.predict_proba(X_test)
        test_logits = scorer.predict_raw_logits(X_test)
        calibrated_test_probs = calibrator.predict_proba(test_logits)

        # Compute ECE & Brier
        raw_ece = UncertaintyEstimator.compute_ece(raw_test_probs, y_test)
        cal_ece = UncertaintyEstimator.compute_ece(calibrated_test_probs, y_test)
        raw_brier = float(np.mean((raw_test_probs - y_test) ** 2))
        cal_brier = float(np.mean((calibrated_test_probs - y_test) ** 2))

        print(f"[ConfTest Benchmark] Raw ECE: {raw_ece:.4f} -> Calibrated ECE: {cal_ece:.4f} ({(raw_ece - cal_ece)/raw_ece:.1%} error reduction)")

        # Evaluate across test commits
        unique_commits = test_df["commit_id"].unique()
        results: Dict[str, Dict[str, Any]] = {
            "1. Retest-All (Full Suite)": {"total_tests": 0, "selected_tests": 0, "total_time": 0.0, "selected_time": 0.0, "actual_fails": 0, "caught_fails": 0, "abstentions": 0},
            "2. Random Selection (50%)": {"total_tests": 0, "selected_tests": 0, "total_time": 0.0, "selected_time": 0.0, "actual_fails": 0, "caught_fails": 0, "abstentions": 0},
            "3. Changed-File Match": {"total_tests": 0, "selected_tests": 0, "total_time": 0.0, "selected_time": 0.0, "actual_fails": 0, "caught_fails": 0, "abstentions": 0},
            "4. Static Dependency RTS": {"total_tests": 0, "selected_tests": 0, "total_time": 0.0, "selected_time": 0.0, "actual_fails": 0, "caught_fails": 0, "abstentions": 0},
            "5. Historical-Failure Ranking": {"total_tests": 0, "selected_tests": 0, "total_time": 0.0, "selected_time": 0.0, "actual_fails": 0, "caught_fails": 0, "abstentions": 0},
            "6. Uncalibrated GBDT (Meta PTS)": {"total_tests": 0, "selected_tests": 0, "total_time": 0.0, "selected_time": 0.0, "actual_fails": 0, "caught_fails": 0, "abstentions": 0},
            "7. Calibrated (No Abstention)": {"total_tests": 0, "selected_tests": 0, "total_time": 0.0, "selected_time": 0.0, "actual_fails": 0, "caught_fails": 0, "abstentions": 0},
            "8. Proposed ConfTest": {"total_tests": 0, "selected_tests": 0, "total_time": 0.0, "selected_time": 0.0, "actual_fails": 0, "caught_fails": 0, "abstentions": 0},
        }

        engine = SelectiveDecisionEngine(tau_abstain=0.18, theta_select=0.08)

        for commit_id in unique_commits:
            c_mask = (test_df["commit_id"] == commit_id)
            c_df = test_df[c_mask]
            
            c_tests = c_df["test_name"].tolist()
            c_durs = c_df["avg_test_duration"].values
            c_labels = c_df["label"].values
            c_direct = c_df["direct_dependency_match"].values
            c_hist = c_df["historical_failure_rate"].values
            c_ood = bool(c_df["is_ood_refactoring"].iloc[0])

            c_probs_raw = raw_test_probs[c_mask.values]
            c_probs_cal = calibrated_test_probs[c_mask.values]
            
            # Uncertainty estimate: distance from extreme confidence + variance proxy
            c_uncertainties = 2.0 * np.abs(c_probs_cal - 0.5) * (0.15 if c_ood else 0.05)
            if c_ood:
                c_uncertainties = np.clip(c_uncertainties + 0.22, 0.0, 0.45)

            n_c_tests = len(c_tests)
            tot_dur = float(np.sum(c_durs))
            n_fails = int(np.sum(c_labels))

            def update_stat(name, selected_indices, is_abstain=False):
                r = results[name]
                r["total_tests"] += n_c_tests
                r["selected_tests"] += len(selected_indices)
                r["total_time"] += tot_dur
                r["selected_time"] += float(np.sum(c_durs[selected_indices])) if len(selected_indices) > 0 else 0.0
                r["actual_fails"] += n_fails
                caught = int(np.sum(c_labels[selected_indices])) if len(selected_indices) > 0 else 0
                r["caught_fails"] += caught
                if is_abstain:
                    r["abstentions"] += 1

            # 1. Retest-All
            update_stat("1. Retest-All (Full Suite)", list(range(n_c_tests)))

            # 2. Random 50%
            rand_idx = [i for i in range(n_c_tests) if np.random.rand() > 0.5]
            update_stat("2. Random Selection (50%)", rand_idx)

            # 3. Changed-File Match
            cf_idx = [i for i in range(n_c_tests) if c_direct[i] == 1]
            update_stat("3. Changed-File Match", cf_idx)

            # 4. Static Dependency RTS
            static_idx = [i for i in range(n_c_tests) if c_direct[i] == 1 or c_ood]
            update_stat("4. Static Dependency RTS", static_idx)

            # 5. Historical Failure Ranking (Top 30% most frequently failing)
            hist_thresh = np.percentile(c_hist, 70)
            hist_idx = [i for i in range(n_c_tests) if c_hist[i] >= hist_thresh]
            update_stat("5. Historical-Failure Ranking", hist_idx)

            # 6. Uncalibrated GBDT (Meta PTS equivalent, static cutoff 0.30)
            raw_idx = [i for i in range(n_c_tests) if c_probs_raw[i] >= 0.30]
            update_stat("6. Uncalibrated GBDT (Meta PTS)", raw_idx)

            # 7. Calibrated without Abstention
            cal_no_abs_idx = [i for i in range(n_c_tests) if c_probs_cal[i] >= 0.08]
            update_stat("7. Calibrated (No Abstention)", cal_no_abs_idx)

            # 8. Proposed ConfTest
            direct_dep_names = [c_tests[i] for i, d in enumerate(c_direct) if d == 1]
            decision = engine.decide(
                test_ids=c_tests,
                calibrated_probs=c_probs_cal,
                uncertainties=c_uncertainties,
                direct_dep_tests=direct_dep_names,
                is_ood_refactoring=c_ood
            )
            conftest_idx = [i for i, t in enumerate(c_tests) if t in decision.selected_tests]
            update_stat("8. Proposed ConfTest", conftest_idx, is_abstain=(decision.action == "ABSTAIN_SAFE_FALLBACK"))

        # Format Summary Table
        summary_rows = []
        for name, r in results.items():
            trr = (1.0 - r["selected_tests"] / max(r["total_tests"], 1)) * 100.0
            etr = (1.0 - r["selected_time"] / max(r["total_time"], 1)) * 100.0
            fr = (r["caught_fails"] / max(r["actual_fails"], 1)) * 100.0
            mfr = 100.0 - fr
            summary_rows.append({
                "Strategy / Baseline": name,
                "Test Reduction (TRR %)": f"{trr:.1f}%",
                "Time Reduction (ETR %)": f"{etr:.1f}%",
                "Failure Recall (FR %)": f"{fr:.1f}%",
                "Missed-Failure (MFR %)": f"{mfr:.1f}%",
                "Abstentions": r["abstentions"]
            })

        summary_df = pd.DataFrame(summary_rows)
        return summary_df

if __name__ == "__main__":
    runner = ExperimentRunner(n_commits=450, n_tests=50)
    df = runner.run_all()
    print("\n" + "="*80)
    print("CONFTEST BASELINE EXPERIMENTAL COMPARISON RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
