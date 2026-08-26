"""
ConfTest Local Git Mining and Diff Extraction Engine.

Mines Git commit history, file-level diffs, added/deleted line churn,
author hashes (anonymized), and commit timestamps directly from local repositories via GitPython.
"""

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import git
from git.exc import GitCommandError, InvalidGitRepositoryError

from conftest.logging_config import get_logger

logger = get_logger(__name__)


def anonymize_author(author_name_or_email: Optional[str]) -> str:
    """Produce a deterministic SHA-256 author hash for privacy compliance."""
    if not author_name_or_email:
        return "anonymous_author"
    return hashlib.sha256(author_name_or_email.strip().lower().encode("utf-8")).hexdigest()[:16]


class GitRepositoryMiner:
    """Extracts chronological commit histories and diff statistics from a Git repository."""

    def __init__(self, repo_path: str):
        """
        Initialize the Git repository miner.

        Args:
            repo_path: Absolute or relative path to the local Git repository.
        """
        self.repo_path = Path(repo_path).resolve()
        try:
            self.repo = git.Repo(str(self.repo_path))
        except InvalidGitRepositoryError:
            logger.error(f"Directory is not a valid Git repository: {self.repo_path}")
            raise

    def get_commit_count(self, branch: str = "main") -> int:
        """Return total commit count on the specified branch or HEAD."""
        try:
            return int(self.repo.git.rev_list("--count", branch))
        except GitCommandError:
            try:
                # Fallback to HEAD if branch name differs (e.g. master)
                return int(self.repo.git.rev_list("--count", "HEAD"))
            except GitCommandError:
                return 0

    def iter_commits(
        self,
        max_commits: Optional[int] = None,
        branch: Optional[str] = None,
        reverse: bool = True,
    ) -> Generator[git.Commit, None, None]:
        """
        Iterate through commits. By default, traverses in chronological order (oldest to newest).

        Args:
            max_commits: Optional upper bound on commits to yield.
            branch: Target branch name (defaults to active branch or HEAD).
            reverse: If True, yields in chronological order (oldest -> newest).
        """
        rev = branch or (self.repo.active_branch.name if not self.repo.head.is_detached else "HEAD")
        try:
            commits = list(self.repo.iter_commits(rev=rev, max_count=max_commits, reverse=reverse))
            for commit in commits:
                yield commit
        except GitCommandError as exc:
            logger.warning(f"Failed iterating commits on branch '{rev}': {exc}")
            try:
                # Fallback to HEAD
                commits = list(self.repo.iter_commits(rev="HEAD", max_count=max_commits, reverse=reverse))
                for commit in commits:
                    yield commit
            except GitCommandError:
                # Repository is empty / brand new with no commits
                logger.info("Repository has no commits yet.")
                return

    def extract_commit_diffs(self, commit: git.Commit) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Extract modified file paths, change types (ADDED, MODIFIED, DELETED),
        and added/deleted line counts for a given commit.

        Returns:
            Tuple of (list of changed file dicts, aggregate churn stats dict).
        """
        changed_files: List[Dict[str, Any]] = []
        total_added = 0
        total_deleted = 0

        # Handle root commit (no parents)
        if not commit.parents:
            for item in commit.tree.traverse():
                if item.type == "blob":
                    lines_count = len(item.data_stream.read().splitlines())
                    changed_files.append({
                        "file_path": item.path,
                        "change_type": "ADDED",
                        "lines_added": lines_count,
                        "lines_deleted": 0,
                        "is_test": "test" in item.path.lower(),
                    })
                    total_added += lines_count
            return changed_files, {"lines_added": total_added, "lines_deleted": total_deleted, "files_changed": len(changed_files)}

        parent = commit.parents[0]
        try:
            diff_index = parent.diff(commit, create_patch=True)
            for diff_item in diff_index:
                # Determine change type
                if diff_item.new_file:
                    change_type = "ADDED"
                    path = diff_item.b_path or diff_item.a_path
                elif diff_item.deleted_file:
                    change_type = "DELETED"
                    path = diff_item.a_path or diff_item.b_path
                elif diff_item.renamed_file:
                    change_type = "RENAMED"
                    path = diff_item.b_path or diff_item.a_path
                else:
                    change_type = "MODIFIED"
                    path = diff_item.b_path or diff_item.a_path

                if not path:
                    continue

                # Parse lines added / deleted from diff patch
                added = 0
                deleted = 0
                if diff_item.diff:
                    try:
                        patch_text = diff_item.diff.decode("utf-8", errors="replace")
                        for line in patch_text.splitlines():
                            if line.startswith("+") and not line.startswith("+++"):
                                added += 1
                            elif line.startswith("-") and not line.startswith("---"):
                                deleted += 1
                    except Exception:
                        pass

                total_added += added
                total_deleted += deleted

                changed_files.append({
                    "file_path": path.replace("\\", "/"),
                    "change_type": change_type,
                    "lines_added": added,
                    "lines_deleted": deleted,
                    "is_test": "test" in path.lower() or path.startswith("tests/"),
                })
        except Exception as exc:
            logger.warning(f"Error computing diff for commit {commit.hexsha[:8]}: {exc}")

        stats = {
            "lines_added": total_added,
            "lines_deleted": total_deleted,
            "files_changed": len(changed_files),
        }
        return changed_files, stats

    def mine_commit_record(self, commit: git.Commit) -> Dict[str, Any]:
        """Convert a Git commit object into a structured ConfTest dictionary."""
        changed_files, stats = self.extract_commit_diffs(commit)
        author_hash = anonymize_author(commit.author.email if commit.author else "unknown")
        commit_date = datetime.utcfromtimestamp(commit.committed_date)

        return {
            "sha": commit.hexsha,
            "parent_sha": commit.parents[0].hexsha if commit.parents else None,
            "timestamp": commit_date,
            "author_hash": author_hash,
            "message": commit.message.strip() if commit.message else "",
            "stats": stats,
            "changed_files": changed_files,
        }
