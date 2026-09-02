"""
Unit tests for Latency Stress Benchmark and Profiling module.
"""

import pytest
from conftest.evaluation.benchmark_latency import LatencyBenchmarkSuite


def test_latency_benchmark_suite_execution():
    """Verify latency benchmark measures all stages and computes valid stats."""
    suite = LatencyBenchmarkSuite(random_seed=42)
    report = suite.benchmark_pipeline(num_iterations=20, batch_size=20)

    assert "stages" in report
    assert "end_to_end_total" in report["stages"]
    assert "ensemble_inference" in report["stages"]

    total_stats = report["stages"]["end_to_end_total"]
    assert "mean_ms" in total_stats
    assert "p99_ms" in total_stats
    assert total_stats["mean_ms"] < 100.0  # Must be well below 100ms
    assert report["sla_under_100ms_compliance_pct"] == 100.0
