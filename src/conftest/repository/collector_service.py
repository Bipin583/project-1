"""
ConfTest Data Collection Service Orchestrator.

Orchestrates repository ingestion from local Git repositories and optional
GitHub remote metadata with checkpointing, error handling, database persistence,
and data-quality reporting.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from conftest.db import crud
from conftest.github.client import GitHubClient
from conftest.logging_config import get_logger
from conftest.repository.git_collector import GitRepositoryMiner

logger = get_logger(__name__)


class CollectorService:
    """End-to-end data collector with checkpointing and data-quality reporting."""

    def __init__(
        self,
        repo_path: str,
        repo_name: str,
        github_token: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize collector service.

        Args:
            repo_path: Path to the local git repository.
            repo_name: Repository name (e.g. 'pallets/flask').
            github_token: Optional GitHub API token.
            output_dir: Destination folder for mined raw JSON data.
        """
        self.repo_path = Path(repo_path).resolve()
        self.repo_name = repo_name
        self.miner = GitRepositoryMiner(str(self.repo_path))
        self.github_client = GitHubClient(token=github_token)

        self.output_dir = Path(output_dir or "./data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.output_dir / f"checkpoint_{repo_name.replace('/', '_')}.json"

    def load_checkpoint(self) -> Dict[str, Any]:
        """Load processed commit hashes from checkpoint file."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning(f"Failed to read checkpoint file: {exc}")
        return {"processed_shas": [], "last_updated": None}

    def save_checkpoint(self, processed_shas: List[str]) -> None:
        """Persist processed commit SHAs to disk for seamless resume capability."""
        data = {
            "repository": self.repo_name,
            "processed_shas": processed_shas,
            "count": len(processed_shas),
            "last_updated": datetime.utcnow().isoformat(),
        }
        try:
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.warning(f"Failed to write checkpoint file: {exc}")

    def collect(
        self,
        db: Optional[Session] = None,
        max_commits: int = 100,
        branch: Optional[str] = None,
        fetch_github_ci: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute data collection pipeline.

        Args:
            db: Optional SQLAlchemy Session to persist mined records directly.
            max_commits: Maximum commits to collect.
            branch: Git branch name.
            fetch_github_ci: Whether to query GitHub API for CI statuses.

        Returns:
            Data quality report dictionary.
        """
        logger.info(f"Starting collection for {self.repo_name} (Max commits: {max_commits})...")
        checkpoint = self.load_checkpoint()
        processed_set = set(checkpoint.get("processed_shas", []))

        # Register repository in DB if provided
        repo_record = None
        if db:
            repo_record = crud.get_repository_by_name(db, self.repo_name)
            if not repo_record:
                repo_record = crud.create_repository(
                    db=db,
                    full_name=self.repo_name,
                    url=f"https://github.com/{self.repo_name}",
                    local_path=str(self.repo_path),
                )

        mined_records: List[Dict[str, Any]] = []
        newly_processed: List[str] = list(processed_set)
        total_churn = 0
        total_files_changed = 0

        for commit in self.miner.iter_commits(max_commits=max_commits, branch=branch, reverse=True):
            sha = commit.hexsha
            if sha in processed_set:
                continue

            record = self.miner.mine_commit_record(commit)

            # Query CI status if requested
            ci_status = "passed"
            if fetch_github_ci and "/" in self.repo_name:
                owner, repo = self.repo_name.split("/", 1)
                ci_status = self.github_client.get_commit_ci_status(owner, repo, sha)
            record["ci_status"] = ci_status

            mined_records.append(record)
            newly_processed.append(sha)
            processed_set.add(sha)

            total_churn += record["stats"]["lines_added"] + record["stats"]["lines_deleted"]
            total_files_changed += record["stats"]["files_changed"]

            # Persist to database if active session provided
            if db and repo_record:
                c_row = crud.get_commit_by_sha(db, sha)
                if not c_row:
                    c_row = crud.create_commit(
                        db=db,
                        repository_id=repo_record.id,
                        sha=record["sha"],
                        parent_sha=record["parent_sha"],
                        timestamp=record["timestamp"],
                        author_hash=record["author_hash"],
                        message=record["message"],
                        ci_status=record["ci_status"],
                    )
                    crud.add_changed_files(db, c_row.id, record["changed_files"])

            if len(mined_records) >= max_commits:
                break

        # Save checkpoint
        self.save_checkpoint(newly_processed)

        # Write raw dataset JSON
        raw_output_path = self.output_dir / f"mined_{self.repo_name.replace('/', '_')}.json"
        with open(raw_output_path, "w", encoding="utf-8") as f:
            # Serialize datetimes to isoformat
            serializable_records = []
            for r in mined_records:
                item = dict(r)
                item["timestamp"] = item["timestamp"].isoformat()
                serializable_records.append(item)
            json.dump({
                "metadata": {
                    "data_origin": "REAL_GIT_MINED",
                    "repository": self.repo_name,
                    "collected_at": datetime.utcnow().isoformat(),
                    "total_mined": len(mined_records),
                },
                "commits": serializable_records,
            }, f, indent=2)

        # Generate Data Quality Report
        report = {
            "repository": self.repo_name,
            "data_origin": "REAL_GIT_MINED",
            "commits_collected": len(mined_records),
            "total_checkpointed_commits": len(newly_processed),
            "total_line_churn": total_churn,
            "total_file_modifications": total_files_changed,
            "raw_dataset_path": str(raw_output_path),
            "status": "SUCCESS",
        }

        quality_report_path = self.output_dir / f"quality_report_{self.repo_name.replace('/', '_')}.json"
        with open(quality_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Collection complete: {len(mined_records)} commits mined. Report saved to {quality_report_path}")
        return report
