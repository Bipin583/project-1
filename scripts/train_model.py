"""
ConfTest Gradient Boosted Model Training CLI.

Trains a LightGBM classification model on temporal train splits, evaluates on validation/test sets,
and exports model artifacts and feature importance reports.

Usage:
    python scripts/train_model.py --train data/splits/train.csv --val data/splits/val.csv --test data/splits/test.csv --output-dir models/ensembles --seed 42
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conftest.models.trainer import ModelTrainer
from conftest.logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Model Training CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train",
        type=str,
        default="./data/splits/train.csv",
        help="Path to training dataset CSV.",
    )
    parser.add_argument(
        "--val",
        type=str,
        default="./data/splits/val.csv",
        help="Path to validation dataset CSV.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="./data/splits/test.csv",
        help="Path to test dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/ensembles",
        help="Directory to save trained model artifacts and reports.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="lgbm_v1.0.0",
        help="Semantic model version tag.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    train_path = Path(args.train)
    val_path = Path(args.val)
    test_path = Path(args.test)

    if not train_path.exists():
        logger.error(f"Training split {train_path} not found. Run build_splits.py first.")
        sys.exit(1)

    logger.info(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) if val_path.exists() else None
    test_df = pd.read_csv(test_path) if test_path.exists() else None

    trainer = ModelTrainer(
        output_dir=args.output_dir,
        model_version=args.version,
        random_seed=args.seed,
    )

    report = trainer.train_and_evaluate(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    logger.info("\n=== Model Training Succeeded ===")
    logger.info(f"Model Artifact: {report['model_file']}")
    logger.info(f"Test PR-AUC:   {report['test_metrics'].get('pr_auc', 'N/A')}")
    logger.info(f"Test ROC-AUC:  {report['test_metrics'].get('roc_auc', 'N/A')}")
    logger.info(f"Test F1-Score: {report['test_metrics'].get('f1_score', 'N/A')}")
    logger.info(f"Brier Score:   {report['test_metrics'].get('brier_score', 'N/A')}")
    logger.info("\n=== Top 5 Risk-Predictive Features ===")
    for feat, gain in report["top_10_features_by_gain"][:5]:
        logger.info(f"  • {feat:<35} (Gain: {gain:.4f})")


if __name__ == "__main__":
    main()
