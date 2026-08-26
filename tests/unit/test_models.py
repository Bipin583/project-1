"""
Unit tests for ConfTest SQLAlchemy 2.0 ORM Models.
"""

from datetime import datetime
import pytest
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from conftest.db.models import (
    Repository,
    Commit,
    ChangedFile,
    TestCase,
    TestRun,
    FeatureRecord,
    Prediction,
    SelectionDecision,
    Outcome,
)


def test_repository_and_commit_relationship(db_session: Session):
    """Test creating a repository and associated commit with cascade delete."""
    repo = Repository(
        full_name="pallets/flask",
        url="https://github.com/pallets/flask",
        local_path="/tmp/repos/flask",
        language="python",
        default_branch="main",
    )
    db_session.add(repo)
    db_session.commit()

    commit = Commit(
        repository_id=repo.id,
        sha="a" * 40,
        timestamp=datetime.utcnow(),
        message="feat: add new routing rule",
        ci_status="passed",
        total_duration=12.5,
    )
    db_session.add(commit)
    db_session.commit()

    assert commit.id is not None
    assert commit.repository.full_name == "pallets/flask"
    assert len(repo.commits) == 1

    # Verify cascade delete
    db_session.delete(repo)
    db_session.commit()

    assert db_session.get(Commit, commit.id) is None


def test_unique_constraints(db_session: Session):
    """Verify unique constraint on repository name and commit sha."""
    repo1 = Repository(
        full_name="psf/requests",
        url="https://github.com/psf/requests",
        local_path="/tmp/repos/requests",
    )
    db_session.add(repo1)
    db_session.commit()

    # Attempt duplicate repo full_name
    repo2 = Repository(
        full_name="psf/requests",
        url="https://github.com/psf/requests-fork",
        local_path="/tmp/repos/requests-fork",
    )
    db_session.add(repo2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_full_pipeline_entities_persistence(db_session: Session):
    """Verify persistence across all 9 schema entities."""
    repo = Repository(
        full_name="tiangolo/fastapi",
        url="https://github.com/tiangolo/fastapi",
        local_path="/tmp/repos/fastapi",
    )
    db_session.add(repo)
    db_session.commit()

    commit = Commit(
        repository_id=repo.id,
        sha="b" * 40,
        timestamp=datetime.utcnow(),
        message="refactor: dependency injection core",
    )
    db_session.add(commit)
    db_session.commit()

    # 1. ChangedFile
    changed_file = ChangedFile(
        commit_id=commit.id,
        file_path="fastapi/routing.py",
        change_type="MODIFIED",
        lines_added=25,
        lines_deleted=10,
        cyclomatic_complexity=4.2,
    )
    db_session.add(changed_file)

    # 2. TestCase
    test_case = TestCase(
        repository_id=repo.id,
        test_id="tests/test_routing.py::test_custom_router",
        test_path="tests/test_routing.py",
        test_function="test_custom_router",
        average_duration=0.15,
    )
    db_session.add(test_case)
    db_session.commit()

    # 3. TestRun
    test_run = TestRun(
        commit_id=commit.id,
        test_case_id=test_case.id,
        status="PASSED",
        duration=0.14,
    )
    db_session.add(test_run)

    # 4. FeatureRecord
    feature_rec = FeatureRecord(
        commit_id=commit.id,
        test_case_id=test_case.id,
        feature_vector={"lines_changed": 35, "is_direct_dep": 1},
    )
    db_session.add(feature_rec)

    # 5. Prediction
    pred = Prediction(
        commit_id=commit.id,
        test_case_id=test_case.id,
        raw_score=0.88,
        uncertainty=0.04,
        calibrated_confidence=0.91,
        model_version="lgbm_v1",
    )
    db_session.add(pred)

    # 6. SelectionDecision
    decision = SelectionDecision(
        commit_id=commit.id,
        mode="FAST_SELECTED",
        abstained=False,
        uncertainty_score=0.04,
        threshold_used=0.15,
        selected_count=1,
        total_count=1,
        estimated_saving=0.75,
        reasons={"top_factor": "Direct import dependency on routing.py"},
    )
    db_session.add(decision)

    # 7. Outcome
    outcome = Outcome(
        commit_id=commit.id,
        actual_failures=0,
        detected_failures=0,
        missed_failures=0,
        full_duration=120.0,
        selected_duration=25.0,
        time_reduction_ratio=0.7917,
    )
    db_session.add(outcome)
    db_session.commit()

    # Verify all records loaded cleanly
    assert commit.selection_decision.mode == "FAST_SELECTED"
    assert commit.outcome.time_reduction_ratio == pytest.approx(0.7917, rel=1e-3)
    assert len(commit.changed_files) == 1
    assert len(commit.predictions) == 1
