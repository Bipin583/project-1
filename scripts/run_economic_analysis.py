"""
ConfTest Economic Cost-Benefit Analysis CLI.

Quantifies financial ROI, cloud CI compute cost savings, and developer productivity gains
across enterprise CI/CD teams.

Usage:
    python scripts/run_economic_analysis.py --developers 25 --output reports/economic_analysis.json
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conftest.evaluation.economic_model import EnterpriseEconomicConfig, EconomicCostBenefitModel
from conftest.logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Economic Cost-Benefit Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--developers", type=int, default=25, help="Number of active engineers.")
    parser.add_argument("--commits-per-dev", type=float, default=3.0, help="Commits per developer per day.")
    parser.add_argument("--suite-duration-mins", type=float, default=45.0, help="Full test suite duration in minutes.")
    parser.add_argument("--dev-hourly-rate", type=float, default=75.0, help="Developer hourly loaded cost in USD.")
    parser.add_argument("--bug-escape-cost", type=float, default=3500.0, help="Cost per production regression bug escape in USD.")
    parser.add_argument("--output", type=str, default="./reports/economic_analysis.json", help="Output path for JSON report.")
    return parser.parse_args()


def main():
    args = parse_args()

    config = EnterpriseEconomicConfig(
        num_developers=args.developers,
        commits_per_dev_per_day=args.commits_per_dev,
        full_suite_duration_minutes=args.suite_duration_mins,
        developer_hourly_rate_usd=args.dev_hourly_rate,
        cost_per_escaped_bug_usd=args.bug_escape_cost,
    )

    model = EconomicCostBenefitModel(config)
    baseline = model.calculate_full_suite_annual_cost()

    strategies = [
        ("Full Suite (Always Run All)", 0.00, 1.00, 0),
        ("Random-K (25% Budget)", 0.75, 0.28, 18),
        ("Changed File Heuristic", 0.68, 0.78, 6),
        ("Static Dependency Graph", 0.58, 0.89, 3),
        ("Uncalibrated ML Model", 0.75, 0.91, 2),
        ("ConfTest (Calibrated + Selective)", 0.686, 0.995, 0),
    ]

    eval_results = []
    for name, red_rate, rec_rate, escapes in strategies:
        res = model.evaluate_rts_strategy(
            strategy_name=name,
            test_reduction_rate=red_rate,
            failure_recall_rate=rec_rate,
            annual_escapes_with_strategy=escapes,
        )
        eval_results.append(res)

    logger.info("\n" + "=" * 105)
    logger.info("  ConfTest Economic Cost-Benefit & Enterprise Financial ROI Analysis")
    logger.info(f"  Team Size: {config.num_developers} Devs | Annual Commits: {model.total_annual_commits:,} | Suite: {config.full_suite_duration_minutes:.0f}m")
    logger.info("=" * 105)
    logger.info(f"{'Strategy Name':<34} | {'Test Red %':<10} | {'CI Savings':<12} | {'Dev Savings':<13} | {'Escape Cost':<12} | {'Net Benefit'}")
    logger.info("-" * 105)

    for item in eval_results:
        logger.info(
            f"{item['strategy_name']:<34} | "
            f"{item['test_reduction_rate_pct']:>8.1f}% | "
            f"${item['annual_ci_compute_savings_usd']:>10,.0f} | "
            f"${item['annual_developer_time_savings_usd']:>11,.0f} | "
            f"${item['annual_escape_cost_usd']:>10,.0f} | "
            f"${item['net_annual_benefit_usd']:>10,.0f}"
        )
    logger.info("=" * 105)

    out_data = {
        "enterprise_parameters": {
            "num_developers": config.num_developers,
            "total_annual_commits": model.total_annual_commits,
            "full_suite_duration_minutes": config.full_suite_duration_minutes,
            "ci_runner_cost_per_minute_usd": config.ci_runner_cost_per_minute_usd,
            "developer_hourly_rate_usd": config.developer_hourly_rate_usd,
            "cost_per_escaped_bug_usd": config.cost_per_escaped_bug_usd,
        },
        "baseline_full_suite_annual": baseline,
        "strategy_comparisons": eval_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    logger.info(f"\nEconomic analysis report saved to: {out_path}")


if __name__ == "__main__":
    main()
