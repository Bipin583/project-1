"""
ConfTest Benchmark Dataset Generator
Generates realistic multi-commit software engineering benchmarks with authentic feature distributions,
bug injections, flakiness noise, and out-of-distribution refactorings with strict temporal splitting.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

class BenchmarkDatasetGenerator:
    """
    Generates realistic commit histories and test execution logs across 500+ commits and 50+ test suites.
    """
    def __init__(self, n_commits: int = 500, n_tests: int = 50, random_seed: int = 42):
        self.n_commits = n_commits
        self.n_tests = n_tests
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generates tabular dataset with strict chronological temporal split (70% train / 15% cal / 15% test).
        Returns: (train_df, cal_df, test_df)
        """
        records: List[Dict[str, Any]] = []

        test_names = [f"test_module_{i // 5:02d}.py::test_func_{i % 5:02d}" for i in range(self.n_tests)]
        test_durations = np.random.exponential(scale=1.5, size=self.n_tests) + 0.1
        test_base_fail_rates = np.random.beta(a=0.5, b=5.0, size=self.n_tests)

        for commit_id in range(self.n_commits):
            is_ood_refactoring = (commit_id % 35 == 0 and commit_id > 0)
            is_doc_only = (commit_id % 12 == 0 and not is_ood_refactoring)

            if is_ood_refactoring:
                churn = np.random.randint(800, 3000)
                n_mod_files = np.random.randint(15, 45)
                ast_delta = np.random.randint(100, 400)
                has_interface = 1
                has_import = 1
            elif is_doc_only:
                churn = np.random.randint(2, 20)
                n_mod_files = 1
                ast_delta = 0
                has_interface = 0
                has_import = 0
            else:
                churn = int(np.random.exponential(scale=35)) + 1
                n_mod_files = np.random.randint(1, 6)
                ast_delta = int(np.random.exponential(scale=8))
                has_interface = int(np.random.rand() > 0.8)
                has_import = int(np.random.rand() > 0.6)

            lines_added = int(churn * np.random.uniform(0.3, 0.7))
            lines_deleted = churn - lines_added

            n_impacted = min(self.n_tests, max(1, np.random.randint(1, 6)))
            impacted_indices = set(np.random.choice(self.n_tests, size=n_impacted, replace=False))

            for test_idx in range(self.n_tests):
                t_name = test_names[test_idx]
                t_dur = test_durations[test_idx]
                hist_fail = test_base_fail_rates[test_idx]
                flakiness = 0.05 if np.random.rand() > 0.9 else 0.0

                is_impacted = (test_idx in impacted_indices) or (is_ood_refactoring and np.random.rand() > 0.4)
                
                direct_dep = 1 if is_impacted and np.random.rand() > 0.25 else 0
                dep_overlap = np.random.uniform(0.4, 0.95) if direct_dep else np.random.uniform(0.0, 0.3)

                if is_doc_only:
                    label = 0
                elif is_impacted:
                    fail_prob = 0.75 if direct_dep else 0.45
                    label = 1 if np.random.rand() < fail_prob else 0
                else:
                    label = 1 if np.random.rand() < (hist_fail * 0.1) else 0

                records.append({
                    "commit_id": commit_id,
                    "test_name": t_name,
                    "lines_added": lines_added,
                    "lines_deleted": lines_deleted,
                    "total_churn": churn,
                    "modified_files_count": n_mod_files,
                    "ast_node_delta": ast_delta,
                    "has_interface_change": has_interface,
                    "has_import_change": has_import,
                    "direct_dependency_match": direct_dep,
                    "dependency_overlap_score": dep_overlap,
                    "historical_failure_rate": hist_fail,
                    "avg_test_duration": t_dur,
                    "flakiness_score": flakiness,
                    "is_ood_refactoring": int(is_ood_refactoring),
                    "is_doc_only": int(is_doc_only),
                    "label": label
                })

        df = pd.DataFrame(records)

        split_train = int(self.n_commits * 0.70)
        split_cal = int(self.n_commits * 0.85)

        train_df = df[df["commit_id"] < split_train].copy()
        cal_df = df[(df["commit_id"] >= split_train) & (df["commit_id"] < split_cal)].copy()
        test_df = df[df["commit_id"] >= split_cal].copy()

        return train_df, cal_df, test_df
