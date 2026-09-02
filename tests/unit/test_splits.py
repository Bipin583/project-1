"""
Unit tests for Task Definition, Class Balancing, and Temporal Anti-Leakage Dataset Splitting.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from conftest.models.task_definition import (
    compute_class_weights,
    compute_sample_weights,
    LabelDefinition,
)
from conftest.repository.dataset_splitter import TemporalDatasetSplitter


def test_class_weight_computation():
    """Verify class weight calculations on imbalanced arrays."""
    # 90 pass (0), 10 fail (1)
    y = np.array([0] * 90 + [1] * 10)
    weights = compute_class_weights(y)

    assert weights[1] > weights[0]
    assert weights["scale_pos_weight"] == 9.0  # 90 / 10


def test_sample_weighting_with_flakiness():
    """Verify sample weighting down-weights flaky test outcomes."""
    df = pd.DataFrame({
        "label_failed": [1, 1, 0, 0],
        "hist_flaky_score": [0.0, 0.4, 0.0, 0.0],  # sample 1 is flaky
    })
    weights = compute_sample_weights(df, flaky_discount=0.5)

    assert len(weights) == 4
    # Non-flaky failure should have higher weight than flaky failure
    assert weights[0] > weights[1]


def test_temporal_dataset_splitter_anti_leakage():
    """Verify temporal dataset splitting maintains strict chronological ordering and zero leakage."""
    # Create 10 sequential commits
    rows = []
    base_time = datetime(2026, 1, 1, 0, 0, 0)

    for c_idx in range(10):
        c_time = base_time + timedelta(days=c_idx)
        sha = f"sha_{c_idx:02d}"
        for t_idx in range(4):
            rows.append({
                "commit_sha": sha,
                "commit_timestamp": c_time.isoformat(),
                "test_id": f"test_{t_idx}",
                "label_failed": 1 if (c_idx == 3 and t_idx == 0) else 0,
                "diff_lines_added": float(c_idx * 10),
            })

    df = pd.DataFrame(rows)

    splitter = TemporalDatasetSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    train_df, val_df, test_df, meta = splitter.split_dataframe(df)

    # 10 commits -> 7 train (0..6), 1 val (7), 2 test (8..9)
    assert len(train_df) == 28  # 7 commits * 4 tests
    assert len(val_df) >= 4     # at least 1 commit
    assert len(test_df) >= 4    # at least 1 commit
    assert len(train_df) + len(val_df) + len(test_df) == 40

    # Strict temporal boundary check
    train_max = pd.to_datetime(train_df["commit_timestamp"]).max()
    val_min = pd.to_datetime(val_df["commit_timestamp"]).min()
    val_max = pd.to_datetime(val_df["commit_timestamp"]).max()
    test_min = pd.to_datetime(test_df["commit_timestamp"]).min()

    assert train_max <= val_min
    assert val_max <= test_min

    # Verify metadata fields
    assert meta["strategy"] == "STRICT_CHRONOLOGICAL_TEMPORAL_SPLIT"
    assert "train" in meta
    assert "val" in meta
    assert "test" in meta


def test_temporal_splitter_rejects_insufficient_commits():
    """Verify splitter raises ValueError when fewer than 3 commits exist."""
    df = pd.DataFrame({
        "commit_sha": ["sha_1", "sha_2"],
        "commit_timestamp": ["2026-01-01T00:00:00", "2026-01-02T00:00:00"],
    })
    splitter = TemporalDatasetSplitter()
    with pytest.raises(ValueError):
        splitter.split_dataframe(df)
