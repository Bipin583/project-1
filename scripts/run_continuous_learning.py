"""
ConfTest Continuous Learning & Concept Drift Adaptation CLI.

Simulates sequential streaming commits across CI/CD, triggers Page-Hinkley drift detection,
and executes online buffer replay retraining.

Usage:
    python scripts/run_continuous_learning.py --output reports/continuous_learning.json
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conftest.features.pipeline import FEATURE_NAMES
from conftest.models.continuous_learning import OnlineContinualLearner
from conftest.logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Continuous Learning & Drift Adaptation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-commits",
        type=int,
        default=40,
        help="Number of sequential commits to simulate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./reports/continuous_learning.json",
        help="Path to output JSON report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing Online Continual Learner...")

    rng = np.random.RandomState(42)
    n_feats = len(FEATURE_NAMES)

    # Initial historical training dataset (150 samples)
    X_init = rng.randn(150, n_feats).astype(np.float32)
    y_init = (rng.rand(150) < 0.08).astype(int)
    y_init[0] = 1
    y_init[1] = 1

    learner = OnlineContinualLearner(buffer_capacity=500, drift_threshold=2.0, random_seed=42)
    learner.initialize_base_model(X_init, y_init)

    timeline_events = []
    logger.info("\nStreaming commits through CI/CD pipeline...")

    for commit_idx in range(1, args.num_commits + 1):
        # Simulate 10 test executions per commit
        if commit_idx >= 20:
            # Inject concept drift: feature distribution shift & elevated failure rate
            X_comm = (rng.randn(10, n_feats) + 2.0).astype(np.float32)
            y_comm = (rng.rand(10) < 0.40).astype(int)
        else:
            X_comm = rng.randn(10, n_feats).astype(np.float32)
            y_comm = (rng.rand(10) < 0.05).astype(int)

        step_res = learner.process_streaming_commit(X_comm, y_comm)
        step_res["commit_sequence_id"] = commit_idx

        if step_res["drift_detected"]:
            logger.info(
                f"[Commit #{commit_idx:02d}] DRIFT DETECTED -> Triggered Retraining (Mean Error: {step_res['mean_commit_error']:.4f}, Total Adaptations: {step_res['total_adaptations']})"
            )

        timeline_events.append(step_res)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_streaming_commits": args.num_commits,
                "total_adaptations_performed": learner.adaptation_count,
                "timeline": timeline_events,
            },
            f,
            indent=2,
        )

    logger.info(f"\nContinual learning simulation complete. Adaptations: {learner.adaptation_count}. Report saved to: {out_path}")


if __name__ == "__main__":
    main()
