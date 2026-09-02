"""
ConfTest Temporal Dataset Splitting CLI.

Splits extracted tabular features into Train (70%), Validation (15%), and Test (15%) partitions
strictly by commit timestamp without future-data leakage.

Usage:
    python scripts/build_splits.py --input data/processed/features.csv --output-dir data/splits
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conftest.repository.dataset_splitter import TemporalDatasetSplitter
from conftest.logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Temporal Dataset Splitter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/processed/features.csv",
        help="Path to processed feature dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/splits",
        help="Destination directory for train.csv, val.csv, test.csv, and split_metadata.json.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion of earliest commits for training.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of intermediate commits for validation and calibration.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of newest commits for final evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input feature file not found: {input_path}. Run extract_features.py first.")
        sys.exit(1)

    logger.info(f"Loading feature dataset from {input_path}...")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} sample rows.")

    splitter = TemporalDatasetSplitter(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    train_df, val_df, test_df, metadata = splitter.split_dataframe(df)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"
    meta_path = out_dir / "split_metadata.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=== Temporal Splitting Succeeded ===")
    logger.info(f"Train set: {len(train_df)} rows -> {train_path}")
    logger.info(f"Val set:   {len(val_df)} rows -> {val_path}")
    logger.info(f"Test set:  {len(test_df)} rows -> {test_path}")
    logger.info(f"Metadata:  {meta_path}")


if __name__ == "__main__":
    main()
