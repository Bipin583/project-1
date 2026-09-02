"""
ConfTest Historical Execution & Anti-Leakage Feature Extraction Module.

Calculates test failure history, recent failure velocity, average runtimes,
and file churn strictly prior to the commit prediction timestamp to prevent future-data leakage.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session

from conftest.db.models import Commit, TestRun, TestCase, ChangedFile


def extract_history_features_from_db(
    db: Session,
    commit_timestamp: datetime,
    test_case_id: int,
    changed_file_paths: List[str],
) -> Dict[str, float]:
    """
    Extract historical features strictly using data recorded BEFORE commit_timestamp.

    Args:
        db: Database session.
        commit_timestamp: Upper temporal bound (exclusive).
        test_case_id: ID of the evaluated test case.
        changed_file_paths: List of modified file paths in the current commit.

    Returns:
        Dictionary of historical telemetry features.
    """
    # 1. Query prior test runs for this test case
    stmt_runs = (
        select(TestRun.status, TestRun.duration, TestRun.retry_count)
        .join(Commit, Commit.id == TestRun.commit_id)
        .where(
            and_(
                TestRun.test_case_id == test_case_id,
                Commit.timestamp < commit_timestamp,  # STRICT ANTI-LEAKAGE FILTER
            )
        )
        .order_by(Commit.timestamp.desc())
    )
    prior_runs = db.execute(stmt_runs).all()

    total_prior_runs = len(prior_runs)
    prior_failures = sum(1 for r in prior_runs if r.status in ("FAILED", "ERROR"))
    prior_retries = sum(r.retry_count for r in prior_runs)

    lifetime_failure_rate = (prior_failures / total_prior_runs) if total_prior_runs > 0 else 0.0

    # Recent failure velocity (last 10 executions)
    recent_10 = prior_runs[:10]
    recent_10_runs = len(recent_10)
    recent_10_failures = sum(1 for r in recent_10 if r.status in ("FAILED", "ERROR"))
    recent_failure_rate = (recent_10_failures / recent_10_runs) if recent_10_runs > 0 else 0.0

    # Average duration
    durations = [r.duration for r in prior_runs if r.duration > 0]
    avg_duration = (sum(durations) / len(durations)) if durations else 0.05

    # Flakiness heuristic (prior retries / runs)
    flaky_score = (prior_retries / total_prior_runs) if total_prior_runs > 0 else 0.0

    # 2. Historical churn on modified files
    total_prior_file_mods = 0
    if changed_file_paths:
        stmt_files = (
            select(func.count(ChangedFile.id))
            .join(Commit, Commit.id == ChangedFile.commit_id)
            .where(
                and_(
                    ChangedFile.file_path.in_(changed_file_paths),
                    Commit.timestamp < commit_timestamp,
                )
            )
        )
        total_prior_file_mods = db.execute(stmt_files).scalar() or 0

    return {
        "hist_total_prior_runs": float(total_prior_runs),
        "hist_prior_failures": float(prior_failures),
        "hist_lifetime_failure_rate": float(lifetime_failure_rate),
        "hist_recent_10_failure_rate": float(recent_failure_rate),
        "hist_avg_duration": float(avg_duration),
        "hist_flaky_score": float(flaky_score),
        "hist_has_ever_failed": 1.0 if prior_failures > 0 else 0.0,
        "hist_changed_files_prior_mod_count": float(total_prior_file_mods),
    }
