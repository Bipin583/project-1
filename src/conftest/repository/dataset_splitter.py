"""
ConfTest Temporal Dataset Partitioning & Anti-Leakage Splitting Engine.

Splits dataset strictly by chronological commit timestamp into Train (70%),
Validation/Calibration (15%), and Test/Evaluation (15%) splits without future-data leakage.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from conftest.logging_config import get_logger

logger = get_logger(__name__)


class TemporalDatasetSplitter:
    """Partitions commit-test datasets strictly by commit timestamp."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        timestamp_col: str = "commit_timestamp",
        sha_col: str = "commit_sha",
    ):
        """
        Initialize temporal splitter.

        Args:
            train_ratio: Fraction of earliest commits for training.
            val_ratio: Fraction of intermediate commits for calibration/validation.
            test_ratio: Fraction of latest commits for final evaluation.
            timestamp_col: Column containing commit timestamp.
            sha_col: Column containing unique commit SHA.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.timestamp_col = timestamp_col
        self.sha_col = sha_col

    def split_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Partition a DataFrame into temporal train, validation, and test sets.

        Args:
            df: Full tabular dataset containing commit_sha, commit_timestamp, and features.

        Returns:
            Tuple of (train_df, val_df, test_df, metadata_dict).
        """
        if df.empty:
            raise ValueError("Cannot split an empty DataFrame.")

        # Ensure timestamp is parsed as datetime
        df = df.copy()
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])

        # 1. Extract unique commits and sort chronologically (oldest -> newest)
        commits_df = (
            df[[self.sha_col, self.timestamp_col]]
            .drop_duplicates(subset=[self.sha_col])
            .sort_values(by=self.timestamp_col, ascending=True)
            .reset_index(drop=True)
        )

        n_commits = len(commits_df)
        if n_commits < 3:
            raise ValueError(f"At least 3 commits required for temporal splitting (got {n_commits}).")

        # 2. Compute commit index boundaries
        n_train = max(1, int(n_commits * self.train_ratio))
        n_val = max(1, int(n_commits * self.val_ratio))
        # Remaining commits go to test
        n_test = max(1, n_commits - n_train - n_val)

        train_commits = set(commits_df.iloc[:n_train][self.sha_col])
        val_commits = set(commits_df.iloc[n_train : n_train + n_val][self.sha_col])
        test_commits = set(commits_df.iloc[n_train + n_val :][self.sha_col])

        # 3. Filter DataFrame rows into splits
        train_df = df[df[self.sha_col].isin(train_commits)].copy().reset_index(drop=True)
        val_df = df[df[self.sha_col].isin(val_commits)].copy().reset_index(drop=True)
        test_df = df[df[self.sha_col].isin(test_commits)].copy().reset_index(drop=True)

        # 4. Verify temporal ordering (Zero future data leakage assertion)
        max_train_time = train_df[self.timestamp_col].max()
        min_val_time = val_df[self.timestamp_col].min()
        max_val_time = val_df[self.timestamp_col].max()
        min_test_time = test_df[self.timestamp_col].min()

        assert max_train_time <= min_val_time, "Temporal violation: train timestamp overlaps with validation!"
        assert max_val_time <= min_test_time, "Temporal violation: validation timestamp overlaps with test!"

        # 5. Build detailed metadata
        metadata = {
            "strategy": "STRICT_CHRONOLOGICAL_TEMPORAL_SPLIT",
            "total_commits": n_commits,
            "total_samples": len(df),
            "train": {
                "num_commits": len(train_commits),
                "num_samples": len(train_df),
                "min_timestamp": str(train_df[self.timestamp_col].min()),
                "max_timestamp": str(max_train_time),
                "positive_failure_samples": int(train_df.get("label_failed", 0).sum()) if "label_failed" in train_df.columns else 0,
                "failure_rate": float(train_df.get("label_failed", 0).mean()) if "label_failed" in train_df.columns else 0.0,
            },
            "val": {
                "num_commits": len(val_commits),
                "num_samples": len(val_df),
                "min_timestamp": str(min_val_time),
                "max_timestamp": str(max_val_time),
                "positive_failure_samples": int(val_df.get("label_failed", 0).sum()) if "label_failed" in val_df.columns else 0,
                "failure_rate": float(val_df.get("label_failed", 0).mean()) if "label_failed" in val_df.columns else 0.0,
            },
            "test": {
                "num_commits": len(test_commits),
                "num_samples": len(test_df),
                "min_timestamp": str(min_test_time),
                "max_timestamp": str(test_df[self.timestamp_col].max()),
                "positive_failure_samples": int(test_df.get("label_failed", 0).sum()) if "label_failed" in test_df.columns else 0,
                "failure_rate": float(test_df.get("label_failed", 0).mean()) if "label_failed" in test_df.columns else 0.0,
            },
        }

        logger.info(
            f"Temporal split complete:\n"
            f"  • Train: {len(train_commits)} commits ({len(train_df)} samples) [{train_df[self.timestamp_col].min()} -> {max_train_time}]\n"
            f"  • Val:   {len(val_commits)} commits ({len(val_df)} samples) [{min_val_time} -> {max_val_time}]\n"
            f"  • Test:  {len(test_commits)} commits ({len(test_df)} samples) [{min_test_time} -> {test_df[self.timestamp_col].max()}]"
        )

        return train_df, val_df, test_df, metadata
