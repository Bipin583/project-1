"""
ConfTest Batch Feature Extraction CLI.

Extracts the 32-dimensional feature matrix for all (commit, test_case) pairs in a repository,
persists records to the SQLite database, and exports processed tabular CSV datasets.

Usage:
    python scripts/extract_features.py --repo-name synthetic/demo-project --output data/processed/features.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import select

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conftest.db.session import SessionLocal
from conftest.db import crud
from conftest.db.models import Repository, Commit, TestCase, ChangedFile
from conftest.features.pipeline import FeatureExtractionPipeline, FEATURE_NAMES
from conftest.logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Batch Feature Extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="synthetic/demo-project",
        help="Target repository name registered in database.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Local path to repository files for AST and graph extraction.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/processed/features.csv",
        help="Output path for exported feature dataset CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    db = SessionLocal()

    try:
        repo = crud.get_repository_by_name(db, args.repo_name)
        if not repo:
            # Fallback to first available repository in DB
            repos = crud.list_repositories(db, limit=1)
            if not repos:
                logger.error(f"No repositories found in database. Run collect_repository_data.py first.")
                sys.exit(1)
            repo = repos[0]
            logger.info(f"Repository '{args.repo_name}' not found. Using '{repo.full_name}' (ID: {repo.id})")

        logger.info(f"Starting feature extraction for {repo.full_name}...")
        pipeline = FeatureExtractionPipeline(repo_root=args.repo_root)

        commits = crud.list_commits_for_repo(db, repository_id=repo.id, limit=500)
        test_cases = crud.list_test_cases_for_repo(db, repository_id=repo.id, limit=500)

        logger.info(f"Loaded {len(commits)} commits and {len(test_cases)} test cases.")

        dataset_rows = []
        feature_records_payload = []

        for commit in commits:
            changed_files = [
                {
                    "file_path": cf.file_path,
                    "change_type": cf.change_type,
                    "lines_added": cf.lines_added,
                    "lines_deleted": cf.lines_deleted,
                    "cyclomatic_complexity": cf.cyclomatic_complexity,
                }
                for cf in commit.changed_files
            ]

            for tc in test_cases:
                feat_dict = pipeline.extract_features_for_pair(
                    test_path=tc.test_path,
                    test_function=tc.test_function,
                    changed_files=changed_files,
                    commit_message=commit.message or "",
                    commit_timestamp=commit.timestamp,
                    test_case_id=tc.id,
                    db=db,
                )

                # Fetch ground-truth outcome label if available
                test_run_stmt = select(crud.TestRun.status).where(
                    crud.TestRun.commit_id == commit.id,
                    crud.TestRun.test_case_id == tc.id,
                )
                run_status = db.execute(test_run_stmt).scalar_one_or_none()
                label = 1 if (run_status in ("FAILED", "ERROR")) else 0

                row = {
                    "commit_sha": commit.sha,
                    "commit_timestamp": commit.timestamp.isoformat(),
                    "test_id": tc.test_id,
                    "label_failed": label,
                    **feat_dict,
                }
                dataset_rows.append(row)

                feature_records_payload.append({
                    "test_case_id": tc.id,
                    "feature_vector": feat_dict,
                })

            # Save batch to feature_records table
            if feature_records_payload:
                crud.save_feature_records(db, commit.id, feature_records_payload)
                feature_records_payload = []

        # Export CSV
        df = pd.DataFrame(dataset_rows)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

        logger.info(f"Successfully extracted {len(df)} samples across 32 features.")
        logger.info(f"Dataset exported to: {out_path}")

        # Summary statistics
        logger.info("=== Feature Distribution Summary ===")
        summary_stats = df[FEATURE_NAMES].describe().T[["mean", "std", "min", "max"]]
        logger.info(f"\n{summary_stats.to_string()}")

    except Exception as exc:
        logger.error(f"Feature extraction failed: {exc}", exc_info=True)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
