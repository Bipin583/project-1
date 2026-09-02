"""
Unit tests for ConfTest Core Selection Engine Orchestrator.
"""

from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session
import pytest

from conftest.engine.selector_engine import ConfTestEngine
from conftest.db import crud


def test_engine_initialization_with_heuristics():
    """Verify ConfTestEngine initializes cleanly even without model artifacts."""
    engine = ConfTestEngine(repo_root="./tests/sample_suite")
    assert engine.repo_root.endswith("sample_suite") or "sample_suite" in engine.repo_root
    assert engine.default_budget == 0.25


def test_engine_analyze_and_select_sample_suite():
    """Verify end-to-end analysis on sample test suite."""
    engine = ConfTestEngine(
        repo_root="./tests/sample_suite",
        ensemble_path="./models/ensembles/5_seed_lgbm",
        calibrator_path="./models/calibrator.joblib",
        policy_config_path="./models/policy_config.json",
    )

    diff = [{"file_path": "src_app/auth.py", "lines_added": 25, "lines_deleted": 5}]
    outcome = engine.analyze_and_select(
        commit_sha="test_sha_01",
        changed_files=diff,
        commit_message="fix: resolve auth security issue",
        budget_ratio=0.30,
        execute=False,
    )

    assert "decision_mode" in outcome
    assert outcome["total_count"] >= 5
    assert len(outcome["selected_test_ids"]) >= 1
    assert "top_confidence" in outcome
    assert "epistemic_uncertainty" in outcome
    assert "ranked_tests" in outcome


def test_engine_live_execution_and_db_logging(db_session: Session):
    """Verify live subprocess execution and DB entity logging via ConfTestEngine."""
    repo = crud.create_repository(
        db=db_session,
        full_name="test/engine-repo",
        url="https://github.com/test/engine-repo",
        local_path="./tests/sample_suite",
    )

    engine = ConfTestEngine(
        repo_root="./tests/sample_suite",
        ensemble_path="./models/ensembles/5_seed_lgbm",
        calibrator_path="./models/calibrator.joblib",
        policy_config_path="./models/policy_config.json",
    )

    diff = [{"file_path": "src_app/auth.py", "lines_added": 5, "lines_deleted": 0}]
    outcome = engine.analyze_and_select(
        commit_sha="commit_sha_exec",
        changed_files=diff,
        commit_message="test: verify engine execution",
        db=db_session,
        repository_id=repo.id,
        execute=True,
    )

    assert outcome["execution_outcome"] is not None
    assert outcome["execution_outcome"]["exit_code"] == 0

    # Verify DB persistence
    commit = crud.get_commit_by_sha(db_session, "commit_sha_exec")
    assert commit is not None

    preds = crud.get_predictions_for_commit(db_session, commit.id)
    assert len(preds) >= 1

    decision = crud.get_decision_for_commit(db_session, commit.id)
    assert decision is not None
    assert decision.mode in ("FAST_SELECTED", "SAFE_FULL_SUITE")

    runs = crud.get_test_runs_for_commit(db_session, commit.id)
    assert len(runs) >= 1
