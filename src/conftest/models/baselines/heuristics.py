"""
ConfTest Rule-Based & Heuristic RTS Baselines.

Implements:
1. FullSuiteSelector (Oracle safety reference)
2. RandomKSelector (Budget-matched uniform random sampling)
3. ChangedFileSelector (Static file path string matching)
4. DependencyGraphSelector (Static call-graph reachability)
5. HistoricalFailureSelector (Historical failure frequency ranking)
"""

from pathlib import Path
import random
from typing import Any, Dict, List, Optional

from conftest.models.baselines.base import BaseSelector, SelectionResult


class FullSuiteSelector(BaseSelector):
    """Baseline 1: Executes all candidate tests (0% savings, 100% failure recall)."""

    def __init__(self):
        super().__init__(name="1. Full Test Suite")

    def select(
        self,
        candidate_tests: List[Dict[str, Any]],
        changed_files: List[Dict[str, Any]],
        budget_ratio: float = 1.0,
        **kwargs: Any,
    ) -> SelectionResult:
        all_ids = [t["test_id"] for t in candidate_tests]
        return SelectionResult(
            strategy_name=self.name,
            selected_tests=all_ids,
            total_tests=len(candidate_tests),
            abstained=False,
            mode="SAFE_FULL_SUITE",
            reasons=["Full suite execution requested."],
        )


class RandomKSelector(BaseSelector):
    """Baseline 2: Randomly selects tests up to budget limit."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name="2. Random-k Selection")
        self.rng = random.Random(random_seed)

    def select(
        self,
        candidate_tests: List[Dict[str, Any]],
        changed_files: List[Dict[str, Any]],
        budget_ratio: float = 0.25,
        **kwargs: Any,
    ) -> SelectionResult:
        total = len(candidate_tests)
        k = max(1, int(total * budget_ratio)) if total > 0 else 0
        all_ids = [t["test_id"] for t in candidate_tests]

        selected = self.rng.sample(all_ids, k=min(k, total)) if all_ids else []
        return SelectionResult(
            strategy_name=self.name,
            selected_tests=selected,
            total_tests=total,
            abstained=False,
            mode="FAST_SELECTED",
            reasons=[f"Uniform random sampling of {len(selected)} tests."],
        )


class ChangedFileSelector(BaseSelector):
    """Baseline 3: Selects tests directly associated with modified filenames."""

    def __init__(self):
        super().__init__(name="3. Changed-File Selection")

    def select(
        self,
        candidate_tests: List[Dict[str, Any]],
        changed_files: List[Dict[str, Any]],
        budget_ratio: float = 0.25,
        **kwargs: Any,
    ) -> SelectionResult:
        total = len(candidate_tests)
        k = max(1, int(total * budget_ratio)) if total > 0 else 0

        # Extract stems of changed source files
        changed_stems = {
            Path(f.get("file_path", "")).stem.lower().replace("test_", "").replace("_test", "")
            for f in changed_files
            if f.get("file_path")
        }

        matched_tests = []
        unmatched_tests = []

        for t in candidate_tests:
            t_path = t.get("test_path", t.get("test_id", "")).lower()
            t_stem = Path(t_path).stem.replace("test_", "").replace("_test", "")
            if t_stem in changed_stems or any(stem in t_path for stem in changed_stems):
                matched_tests.append(t["test_id"])
            else:
                unmatched_tests.append(t["test_id"])

        # Cap by budget
        selected = matched_tests[:k] if len(matched_tests) >= k else matched_tests

        return SelectionResult(
            strategy_name=self.name,
            selected_tests=selected,
            total_tests=total,
            abstained=False,
            mode="FAST_SELECTED",
            reasons=[f"Selected {len(selected)} tests matching changed file patterns."],
        )


class DependencyGraphSelector(BaseSelector):
    """Baseline 4: Selects tests using static call-graph reachability."""

    def __init__(self):
        super().__init__(name="4. Static AST Call-Graph Selection")

    def select(
        self,
        candidate_tests: List[Dict[str, Any]],
        changed_files: List[Dict[str, Any]],
        budget_ratio: float = 0.25,
        **kwargs: Any,
    ) -> SelectionResult:
        total = len(candidate_tests)
        k = max(1, int(total * budget_ratio)) if total > 0 else 0

        # Rank by shortest dependency depth (lowest depth first)
        def get_depth(t: Dict[str, Any]) -> float:
            return float(t.get("features", {}).get("dep_shortest_path_depth", 10.0))

        sorted_tests = sorted(candidate_tests, key=get_depth)
        # Select reachable tests (depth < 10.0) up to budget k
        reachable = [t["test_id"] for t in sorted_tests if get_depth(t) < 10.0]
        selected = reachable[:k] if reachable else [t["test_id"] for t in sorted_tests[:k]]

        return SelectionResult(
            strategy_name=self.name,
            selected_tests=selected,
            total_tests=total,
            abstained=False,
            mode="FAST_SELECTED",
            reasons=[f"Selected {len(selected)} tests within call-graph dependency path."],
        )


class HistoricalFailureSelector(BaseSelector):
    """Baseline 5: Selects tests ranked by historical failure frequency."""

    def __init__(self):
        super().__init__(name="5. Historical Failure Frequency")

    def select(
        self,
        candidate_tests: List[Dict[str, Any]],
        changed_files: List[Dict[str, Any]],
        budget_ratio: float = 0.25,
        **kwargs: Any,
    ) -> SelectionResult:
        total = len(candidate_tests)
        k = max(1, int(total * budget_ratio)) if total > 0 else 0

        # Rank by historical failure rate or failure count (descending)
        def get_hist_score(t: Dict[str, Any]) -> float:
            feats = t.get("features", {})
            return float(feats.get("hist_lifetime_failure_rate", 0.0) + feats.get("hist_prior_failures", 0.0) * 0.1)

        sorted_tests = sorted(candidate_tests, key=get_hist_score, reverse=True)
        selected = [t["test_id"] for t in sorted_tests[:k]]

        return SelectionResult(
            strategy_name=self.name,
            selected_tests=selected,
            total_tests=total,
            abstained=False,
            mode="FAST_SELECTED",
            reasons=[f"Selected {len(selected)} tests with highest historical failure counts."],
        )
