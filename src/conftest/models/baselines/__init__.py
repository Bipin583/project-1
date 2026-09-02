"""
ConfTest 8-Baseline Regression Test Selection Strategies.

Exports:
1. FullSuiteSelector
2. RandomKSelector
3. ChangedFileSelector
4. DependencyGraphSelector
5. HistoricalFailureSelector
6. UncalibratedMLSelector
7. CalibratedNoAbstentionSelector
8. ConfTestSelectiveSelector
"""

from conftest.models.baselines.base import BaseSelector, SelectionResult
from conftest.models.baselines.heuristics import (
    FullSuiteSelector,
    RandomKSelector,
    ChangedFileSelector,
    DependencyGraphSelector,
    HistoricalFailureSelector,
)
from conftest.models.baselines.ml_baseline import (
    UncalibratedMLSelector,
    CalibratedNoAbstentionSelector,
    ConfTestSelectiveSelector,
)

ALL_BASELINES = [
    FullSuiteSelector,
    RandomKSelector,
    ChangedFileSelector,
    DependencyGraphSelector,
    HistoricalFailureSelector,
    UncalibratedMLSelector,
    CalibratedNoAbstentionSelector,
    ConfTestSelectiveSelector,
]

__all__ = [
    "BaseSelector",
    "SelectionResult",
    "FullSuiteSelector",
    "RandomKSelector",
    "ChangedFileSelector",
    "DependencyGraphSelector",
    "HistoricalFailureSelector",
    "UncalibratedMLSelector",
    "CalibratedNoAbstentionSelector",
    "ConfTestSelectiveSelector",
    "ALL_BASELINES",
]
