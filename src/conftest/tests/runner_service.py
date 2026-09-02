"""
ConfTest Test Execution & Evaluation Service.

Orchestrates full-suite discovery, selective test subset execution,
database logging of test runs, and outcome evaluation calculation.
"""

from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from conftest.db import crud
from conftest.logging_config import get_logger
from conftest.tests.discovery import PytestDiscovery
from conftest.tests.executor import SafeTestExecutor, PytestExecutionResult

logger = get_logger(__name__)


class TestRunnerService:
    """Service managing test discovery, execution, and outcome evaluation."""

    __test__ = False  # Prevent pytest from treating service class as a test suite

    def __init__(self, repo_root: str, default_timeout: int = 60):
        """
        Initialize the runner service.

        Args:
            repo_root: Root directory of target repository.
            default_timeout: Execution timeout in seconds.
        """
        self.repo_root = repo_root
        self.discovery = PytestDiscovery(repo_root)
        self.executor = SafeTestExecutor(repo_root, default_timeout=default_timeout)

    def discover_repository_tests(self, db: Optional[Session] = None, repository_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Scan repository for all regression tests and optionally register in database.

        Returns:
            List of discovered test dictionaries.
        """
        tests = self.discovery.discover_via_pytest()
        logger.info(f"Discovered {len(tests)} test cases in {self.repo_root}")

        if db and repository_id:
            for t in tests:
                crud.get_or_create_test_case(
                    db=db,
                    repository_id=repository_id,
                    test_id=t["test_id"],
                    test_path=t["test_path"],
                    test_function=t["test_function"],
                    framework=t.get("framework", "pytest"),
                )
            db.commit()

        return tests

    def execute_and_evaluate(
        self,
        db: Session,
        commit_id: int,
        repository_id: int,
        selected_node_ids: Optional[List[str]] = None,
        run_full_suite_benchmark: bool = False,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute selected tests (or full suite) and record test runs and evaluation outcomes in DB.

        Args:
            db: SQLAlchemy database session.
            commit_id: Target commit ID.
            repository_id: Associated repository ID.
            selected_node_ids: Optional list of test node IDs to selectively run.
            run_full_suite_benchmark: If True, executes full suite first to establish ground truth.
            timeout: Subprocess timeout in seconds.

        Returns:
            Dictionary containing execution metrics and evaluation outcome.
        """
        # Ensure test cases are registered in DB
        all_tests = self.discover_repository_tests(db, repository_id)

        # 1. If full suite benchmark is enabled, run full suite to determine oracle failures
        full_res: Optional[PytestExecutionResult] = None
        if run_full_suite_benchmark:
            logger.info("Executing full test suite for ground truth baseline...")
            full_res = self.executor.run_tests(timeout=timeout)

        # 2. Execute selected tests
        logger.info(f"Executing {'SELECTED' if selected_node_ids else 'FULL'} suite...")
        selected_res = self.executor.run_tests(test_node_ids=selected_node_ids, timeout=timeout)

        # 3. Persist TestRun records
        test_run_payload = []
        for r in selected_res.test_runs:
            t_id = r["test_id"].lstrip("./")
            parts = t_id.split("::")
            test_path = parts[0]
            test_func = parts[-1] if len(parts) > 1 else "test"

            tc = crud.get_or_create_test_case(
                db=db,
                repository_id=repository_id,
                test_id=t_id,
                test_path=test_path,
                test_function=test_func,
            )
            test_run_payload.append({
                "test_case_id": tc.id,
                "status": r["status"],
                "duration": r["duration"],
                "source": "selective_run" if selected_node_ids else "full_suite",
            })

        if test_run_payload:
            crud.record_test_runs(db, commit_id, test_run_payload)

        # 4. Calculate evaluation metrics
        actual_failures = full_res.failed_count if full_res else selected_res.failed_count
        detected_failures = selected_res.failed_count
        missed_failures = max(0, actual_failures - detected_failures)
        full_duration = full_res.total_duration if full_res else selected_res.total_duration
        selected_duration = selected_res.total_duration

        reduction_ratio = max(
            0.0, 1.0 - (selected_duration / full_duration)
        ) if full_duration > 0 else 0.0

        # 5. Persist Outcome
        outcome = crud.save_outcome(
            db=db,
            commit_id=commit_id,
            actual_failures=actual_failures,
            detected_failures=detected_failures,
            missed_failures=missed_failures,
            full_duration=full_duration,
            selected_duration=selected_duration,
            time_reduction_ratio=reduction_ratio,
        )

        return {
            "exit_code": selected_res.exit_code,
            "selected_count": selected_res.total_count,
            "total_suite_count": len(all_tests),
            "actual_failures": actual_failures,
            "detected_failures": detected_failures,
            "missed_failures": missed_failures,
            "selected_duration": round(selected_duration, 3),
            "full_duration": round(full_duration, 3),
            "time_reduction_ratio": round(reduction_ratio, 4),
            "outcome_id": outcome.id,
        }
