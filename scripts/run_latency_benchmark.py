"""
ConfTest Latency Stress Benchmark & Profiling CLI.

Profiles microsecond execution latency across feature prep, ensemble inference,
temperature calibration, and selective policy decision stages.

Usage:
    python scripts/run_latency_benchmark.py --iterations 150 --output reports/latency_benchmark.json
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conftest.evaluation.benchmark_latency import LatencyBenchmarkSuite
from conftest.logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Latency Stress Benchmark CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iterations", type=int, default=150, help="Number of benchmark iterations.")
    parser.add_argument("--batch-size", type=int, default=50, help="Test cases per commit.")
    parser.add_argument("--output", type=str, default="./reports/latency_benchmark.json", help="Output JSON path.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing Latency Profiling Benchmark Engine...")

    suite = LatencyBenchmarkSuite(random_seed=42)
    report = suite.benchmark_pipeline(num_iterations=args.iterations, batch_size=args.batch_size)

    logger.info("\n" + "=" * 85)
    logger.info("  ConfTest Micro-Latency Profiling Breakdown (<100ms SLA)")
    logger.info(f"  Iterations: {report['num_iterations']} | Batch Size: {report['batch_size_tests']} Tests | SLA: {report['sla_under_100ms_compliance_pct']}%")
    logger.info("=" * 85)
    logger.info(f"{'Pipeline Stage':<28} | {'Mean (ms)':<10} | {'p50 (ms)':<10} | {'p90 (ms)':<10} | {'p99 (ms)'}")
    logger.info("-" * 85)

    for stage_name, s in report["stages"].items():
        logger.info(
            f"{stage_name:<28} | "
            f"{s['mean_ms']:>8.3f}ms | "
            f"{s['median_p50_ms']:>8.3f}ms | "
            f"{s['p90_ms']:>8.3f}ms | "
            f"{s['p99_ms']:>8.3f}ms"
        )
    logger.info("=" * 85)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nLatency benchmark report exported to: {out_path}")


if __name__ == "__main__":
    main()
