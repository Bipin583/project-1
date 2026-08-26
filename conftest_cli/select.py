"""
ConfTest CLI Selection Script
Invoked directly inside GitHub Actions / CI runners.
Example:
    python -m conftest_cli.select --base-sha abc1234 --head-sha def5678 --test-dir tests/ --output selected_tests.txt
"""
import argparse
import os
import sys
from typing import List

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.selective_engine import SelectiveDecisionEngine
import numpy as np

def discover_tests(test_dir: str) -> List[str]:
    """Discovers all test files in the specified directory."""
    test_files = []
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.startswith("test_") and f.endswith(".py"):
                test_files.append(os.path.relpath(os.path.join(root, f)))
    return sorted(test_files)

def main():
    parser = argparse.ArgumentParser(description="ConfTest CI Selective Test Runner")
    parser.add_argument("--base-sha", type=str, required=False, default="HEAD~1", help="Base commit SHA")
    parser.add_argument("--head-sha", type=str, required=False, default="HEAD", help="Head commit SHA")
    parser.add_argument("--test-dir", type=str, default="tests", help="Path to test directory")
    parser.add_argument("--risk-tolerance", type=float, default=0.18, help="Abstention uncertainty threshold tau")
    parser.add_argument("--output", type=str, default="selected_tests.txt", help="Output file for selected test paths")

    args = parser.parse_args()

    # 1. Discover all tests
    all_tests = discover_tests(args.test_dir)
    if not all_tests:
        print("[ConfTest] No tests found in directory:", args.test_dir)
        with open(args.output, "w") as f:
            f.write("")
        return

    print(f"[ConfTest] Discovered {len(all_tests)} total tests in {args.test_dir}")

    # 2. Simulated / Model predictions for prototype
    # (During full integration, this calls AST feature extractor and LightGBM model)
    n_tests = len(all_tests)
    mock_probs = np.random.uniform(0.01, 0.40, size=n_tests)
    mock_uncertainties = np.random.uniform(0.02, 0.15, size=n_tests)

    # 3. Execute Selective Decision Policy
    engine = SelectiveDecisionEngine(tau_abstain=args.risk_tolerance)
    decision = engine.decide(
        test_ids=all_tests,
        calibrated_probs=mock_probs,
        uncertainties=mock_uncertainties
    )

    print(f"[ConfTest Decision] Action: {decision.action}")
    print(f"[ConfTest Summary] Selected: {decision.selected_count}/{decision.total_tests_count} ({decision.test_reduction_ratio:.1%} reduction)")
    print(f"[ConfTest Reason] {decision.reason}")

    # 4. Write selected test paths to output file for pytest / test runner
    with open(args.output, "w") as f:
        f.write(" ".join(decision.selected_tests))

    print(f"[ConfTest] Selected test suite written to: {args.output}")

if __name__ == "__main__":
    main()
