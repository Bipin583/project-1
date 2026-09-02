"""
Unit tests for Diff, AST, Dependency Graph, and Historical Feature Extraction.
"""

from datetime import datetime, timedelta
import numpy as np
from sqlalchemy.orm import Session

from conftest.features.diff_features import extract_diff_features
from conftest.features.ast_features import (
    parse_ast_safely,
    estimate_cyclomatic_complexity,
    extract_ast_metrics_from_file,
)
from conftest.features.dependency_graph import DependencyGraphBuilder
from conftest.features.history_features import extract_history_features_from_db
from conftest.features.pipeline import FeatureExtractionPipeline, FEATURE_NAMES
from conftest.db import crud


def test_diff_feature_extraction():
    """Verify diff features calculation on added, deleted, and churn counts."""
    changed_files = [
        {"file_path": "src/auth.py", "lines_added": 40, "lines_deleted": 10, "is_test": False},
        {"file_path": "tests/test_auth.py", "lines_added": 20, "lines_deleted": 0, "is_test": True},
    ]
    msg = "fix: resolve authentication token expiry bug"
    feats = extract_diff_features(changed_files, msg)

    assert feats["diff_lines_added"] == 60.0
    assert feats["diff_lines_deleted"] == 10.0
    assert feats["diff_total_churn"] == 70.0
    assert feats["diff_num_files_changed"] == 2.0
    assert feats["diff_num_src_files"] == 1.0
    assert feats["diff_num_test_files"] == 1.0
    assert feats["diff_is_fix_commit"] == 1.0
    assert feats["diff_has_python"] == 1.0


def test_ast_complexity_and_metrics():
    """Verify AST parsing and complexity calculation."""
    sample_code = """
def sample_logic(x, y):
    if x > 0 and y > 0:
        for i in range(x):
            if i % 2 == 0:
                print(i)
    return x + y
"""
    tree = parse_ast_safely(sample_code)
    assert tree is not None
    complexity = estimate_cyclomatic_complexity(tree)
    # Base 1 + if + and + for + if = 5
    assert complexity >= 4.0


def test_dependency_graph_coupling():
    """Verify DependencyGraphBuilder calculates direct coupling and shortest path."""
    builder = DependencyGraphBuilder("./tests/sample_suite")
    feats = builder.compute_dependency_features(
        test_path="tests/test_auth.py",
        changed_file_paths=["src_app/auth.py"],
    )

    assert "dep_is_direct_import" in feats
    assert "dep_shortest_path_depth" in feats
    assert feats["dep_name_heuristic_coupled"] == 1.0


def test_history_anti_leakage_filter(db_session: Session):
    """Verify history feature extractor strictly ignores future commit runs."""
    repo = crud.create_repository(
        db=db_session,
        full_name="test/history-repo",
        url="https://github.com/test/history",
        local_path="./tests/sample_suite",
    )
    tc = crud.get_or_create_test_case(
        db=db_session,
        repository_id=repo.id,
        test_id="tests/test_sample.py::test_fn",
        test_path="tests/test_sample.py",
        test_function="test_fn",
    )

    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = datetime(2026, 1, 2, 12, 0, 0)
    t2 = datetime(2026, 1, 3, 12, 0, 0)

    # Commit 1 at t0: Test PASSED
    c1 = crud.create_commit(db_session, repo.id, sha="1"*40, timestamp=t0)
    crud.record_test_runs(db_session, c1.id, [{"test_case_id": tc.id, "status": "PASSED", "duration": 0.1}])

    # Commit 2 at t1: Test FAILED
    c2 = crud.create_commit(db_session, repo.id, sha="2"*40, timestamp=t1)
    crud.record_test_runs(db_session, c2.id, [{"test_case_id": tc.id, "status": "FAILED", "duration": 0.2}])

    # Commit 3 at t2: Future run
    c3 = crud.create_commit(db_session, repo.id, sha="3"*40, timestamp=t2)
    crud.record_test_runs(db_session, c3.id, [{"test_case_id": tc.id, "status": "FAILED", "duration": 0.5}])

    # Query history AS OF t1 (Should only see t0 run = 1 passed, 0 failed)
    hist_at_t1 = extract_history_features_from_db(
        db=db_session,
        commit_timestamp=t1,
        test_case_id=tc.id,
        changed_file_paths=[],
    )
    assert hist_at_t1["hist_total_prior_runs"] == 1.0
    assert hist_at_t1["hist_prior_failures"] == 0.0
    assert hist_at_t1["hist_lifetime_failure_rate"] == 0.0

    # Query history AS OF t2 (Should see t0 + t1 runs = 2 runs, 1 failed)
    hist_at_t2 = extract_history_features_from_db(
        db=db_session,
        commit_timestamp=t2,
        test_case_id=tc.id,
        changed_file_paths=[],
    )
    assert hist_at_t2["hist_total_prior_runs"] == 2.0
    assert hist_at_t2["hist_prior_failures"] == 1.0
    assert hist_at_t2["hist_lifetime_failure_rate"] == 0.5


def test_feature_pipeline_canonical_32_dimensions():
    """Verify FeatureExtractionPipeline generates exactly 32 non-NaN features."""
    pipeline = FeatureExtractionPipeline("./tests/sample_suite")
    feats = pipeline.extract_features_for_pair(
        test_path="tests/test_auth.py",
        test_function="test_password_hashing",
        changed_files=[{"file_path": "src_app/auth.py", "lines_added": 10, "lines_deleted": 2}],
        commit_message="feat: enhance password hash security",
    )

    assert len(feats) == 32
    for fname in FEATURE_NAMES:
        assert fname in feats
        assert isinstance(feats[fname], (int, float))
        assert not np.isnan(feats[fname])

    vector = pipeline.to_feature_vector(feats)
    assert vector.shape == (32,)
    assert vector.dtype == np.float32
