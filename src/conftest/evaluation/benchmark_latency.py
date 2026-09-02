"""
ConfTest Latency Stress Benchmark & Performance Profiling Engine.

Measures microsecond/millisecond execution latencies across feature mining,
ensemble inference, calibration, and selective prediction decision loops (<100ms SLA).
"""

import time
from typing import Any, Dict, List, Optional
import numpy as np

from conftest.features.pipeline import FEATURE_NAMES
from conftest.models.lightgbm_model import LightGBMTestPredictor
from conftest.models.calibration import TemperatureScalingCalibrator
from conftest.models.policy import SelectivePredictionPolicy
from conftest.logging_config import get_logger

logger = get_logger(__name__)


class LatencyBenchmarkSuite:
    """Micro-benchmarks and profiles latency across ConfTest pipeline stages."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def benchmark_pipeline(
        self,
        num_iterations: int = 150,
        batch_size: int = 50,  # 50 test cases per commit
    ) -> Dict[str, Any]:
        """
        Execute profiling runs measuring latencies across:
        1. Feature formatting
        2. Ensemble inference
        3. Temperature calibration
        4. Policy decision making
        """
        rng = np.random.RandomState(self.random_seed)
        n_feats = len(FEATURE_NAMES)

        # Pre-train ensemble model
        X_mock = rng.randn(100, n_feats).astype(np.float32)
        y_mock = (rng.rand(100) < 0.10).astype(int)
        y_mock[0] = 1

        models = []
        for seed in [42, 123, 456]:
            m = LightGBMTestPredictor(random_seed=seed, n_estimators=20)
            m.train(X_mock, y_mock)
            models.append(m)

        calibrator = TemperatureScalingCalibrator()
        calibrator.fit(models[0].predict_proba(X_mock), y_mock)
        policy = SelectivePredictionPolicy()

        feat_latencies_ms: List[float] = []
        infer_latencies_ms: List[float] = []
        cal_latencies_ms: List[float] = []
        dec_latencies_ms: List[float] = []
        total_latencies_ms: List[float] = []

        logger.info(f"Executing {num_iterations} latency benchmark iterations (Batch size: {batch_size} tests)...")

        for _ in range(num_iterations):
            X_batch = rng.randn(batch_size, n_feats).astype(np.float32)

            t_start = time.perf_counter()

            # 1. Feature processing
            t0 = time.perf_counter()
            _ = np.nan_to_num(X_batch, nan=0.0)
            t1 = time.perf_counter()

            # 2. Ensemble inference
            preds = [m.predict_proba(X_batch) for m in models]
            mean_probs = np.mean(preds, axis=0)
            epistemic_unc = np.std(preds, axis=0)
            t2 = time.perf_counter()

            # 3. Calibration
            cal_probs = calibrator.calibrate(mean_probs)
            t3 = time.perf_counter()

            # 4. Policy decision
            _ = policy.evaluate_commit(
                commit_sha="bench_commit",
                candidate_test_ids=[f"test_case_{i}" for i in range(batch_size)],
                calibrated_confidences=cal_probs,
                epistemic_uncertainties=epistemic_unc,
                num_changed_files=1,
                total_churn_lines=10,
            )
            t4 = time.perf_counter()

            total_ms = (t4 - t_start) * 1000.0
            feat_latencies_ms.append((t1 - t0) * 1000.0)
            infer_latencies_ms.append((t2 - t1) * 1000.0)
            cal_latencies_ms.append((t3 - t2) * 1000.0)
            dec_latencies_ms.append((t4 - t3) * 1000.0)
            total_latencies_ms.append(total_ms)

        def get_stats(arr: List[float]) -> Dict[str, float]:
            a = np.array(arr)
            return {
                "mean_ms": round(float(np.mean(a)), 3),
                "median_p50_ms": round(float(np.percentile(a, 50)), 3),
                "p90_ms": round(float(np.percentile(a, 90)), 3),
                "p95_ms": round(float(np.percentile(a, 95)), 3),
                "p99_ms": round(float(np.percentile(a, 99)), 3),
            }

        sla_under_100ms = float(np.mean(np.array(total_latencies_ms) < 100.0) * 100.0)

        return {
            "num_iterations": num_iterations,
            "batch_size_tests": batch_size,
            "sla_under_100ms_compliance_pct": round(sla_under_100ms, 2),
            "stages": {
                "feature_preparation": get_stats(feat_latencies_ms),
                "ensemble_inference": get_stats(infer_latencies_ms),
                "temperature_calibration": get_stats(cal_latencies_ms),
                "policy_decision": get_stats(dec_latencies_ms),
                "end_to_end_total": get_stats(total_latencies_ms),
            },
        }
