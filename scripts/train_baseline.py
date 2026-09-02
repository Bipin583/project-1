"""
ConfTest 8-Baseline RTS Comparison Experiment Runner.

Evaluates all 8 RTS baselines under identical test budget constraints
across temporal test splits and exports comparison tables and metrics.

Usage:
    python scripts/train_baseline.py --dataset data/splits/test.csv --budget 0.25 --output reports/baseline_comparison.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conftest.evaluation.benchmark import BaselineBenchmarkRunner
from conftest.logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Baseline Comparison Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/splits/test.csv",
        help="Path to evaluation test split dataset CSV.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.25,
        help="Test budget fraction (e.g. 0.25 = top 25% tests).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./reports/baseline_comparison.csv",
        help="Destination path for benchmark results CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.dataset)

    # Fallback to processed features if split test.csv not present
    if not data_path.exists():
        fallback = Path("./data/processed/features.csv")
        if fallback.exists():
            logger.info(f"Test split {data_path} not found. Falling back to {fallback}...")
            data_path = fallback
        else:
            logger.error(f"Dataset file not found: {data_path}. Run extract_features.py or build_splits.py first.")
            sys.exit(1)

    logger.info(f"Loading benchmark dataset from {data_path}...")
    df = pd.read_csv(data_path)

    runner = BaselineBenchmarkRunner(budget_ratio=args.budget)
    results_df = runner.evaluate_dataset(df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    logger.info(f"Benchmark results exported to: {out_path}")
    logger.info("\n=== RTS Baseline Comparison Table (Budget: 25%) ===\n" + results_df.to_string(index=False))


if __name__ == "__main__":
    main()
