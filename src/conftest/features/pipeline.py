"""
ConfTest Unified Feature Extraction Pipeline.

Combines diff metrics, AST syntactic structure, NetworkX dependency call-graphs,
and historical telemetry into a standardized 32-dimensional feature matrix.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
from sqlalchemy.orm import Session

from conftest.features.diff_features import extract_diff_features
from conftest.features.ast_features import extract_ast_metrics_from_file
from conftest.features.dependency_graph import DependencyGraphBuilder
from conftest.features.history_features import extract_history_features_from_db
from conftest.logging_config import get_logger

logger = get_logger(__name__)

# Canonical list of all 32 feature names in strict fixed order
FEATURE_NAMES: List[str] = [
    # 1. Diff & Churn Features (12)
    "diff_lines_added",
    "diff_lines_deleted",
    "diff_total_churn",
    "diff_num_files_changed",
    "diff_num_src_files",
    "diff_num_test_files",
    "diff_has_python",
    "diff_has_config",
    "diff_msg_length",
    "diff_msg_word_count",
    "diff_is_fix_commit",
    "diff_is_refactor_commit",
    # 2. AST Syntactic & Complexity Features (6)
    "ast_test_file_functions_count",
    "ast_test_file_classes_count",
    "ast_test_file_imports_count",
    "ast_test_file_complexity",
    "ast_test_is_parameterized",
    "ast_test_func_name_length",
    # 3. Structural Call-Graph & Dependency Features (6)
    "dep_is_direct_import",
    "dep_name_heuristic_coupled",
    "dep_shortest_path_depth",
    "dep_is_reachable",
    "dep_max_reverse_dependencies",
    "dep_test_total_out_degree",
    # 4. Historical Telemetry & Anti-Leakage Features (8)
    "hist_total_prior_runs",
    "hist_prior_failures",
    "hist_lifetime_failure_rate",
    "hist_recent_10_failure_rate",
    "hist_avg_duration",
    "hist_flaky_score",
    "hist_has_ever_failed",
    "hist_changed_files_prior_mod_count",
]


class FeatureExtractionPipeline:
    """Unified pipeline computing standard feature vectors for (commit, test_case) pairs."""

    def __init__(self, repo_root: str):
        """
        Initialize feature pipeline.

        Args:
            repo_root: Root path of target repository.
        """
        self.repo_root = repo_root
        self.dep_builder = DependencyGraphBuilder(repo_root)

    def extract_features_for_pair(
        self,
        test_path: str,
        test_function: str,
        changed_files: List[Dict[str, Any]],
        commit_message: str = "",
        commit_timestamp: Optional[datetime] = None,
        test_case_id: Optional[int] = None,
        db: Optional[Session] = None,
    ) -> Dict[str, float]:
        """
        Extract the complete 32-element feature dictionary for a (commit, test) pair.

        Returns:
            Dictionary mapping feature names to numerical float values.
        """
        feature_dict: Dict[str, float] = {}

        # 1. Diff & Churn Features
        diff_feats = extract_diff_features(changed_files, commit_message)
        feature_dict.update(diff_feats)

        # 2. AST Syntactic Features for the test file
        full_test_path = f"{self.repo_root}/{test_path}".replace("//", "/")
        ast_info = extract_ast_metrics_from_file(full_test_path)
        feature_dict.update({
            "ast_test_file_functions_count": float(ast_info.get("functions_count", 1)),
            "ast_test_file_classes_count": float(ast_info.get("classes_count", 0)),
            "ast_test_file_imports_count": float(ast_info.get("imports_count", 0)),
            "ast_test_file_complexity": float(ast_info.get("complexity", 1.0)),
            "ast_test_is_parameterized": 1.0 if ("[" in test_function and "]" in test_function) else 0.0,
            "ast_test_func_name_length": float(len(test_function)),
        })

        # 3. Structural Dependency Graph Features
        changed_file_paths = [f.get("file_path", "") for f in changed_files if f.get("file_path")]
        dep_feats = self.dep_builder.compute_dependency_features(test_path, changed_file_paths)
        feature_dict.update(dep_feats)

        # 4. Historical Telemetry Features
        if db and test_case_id and commit_timestamp:
            hist_feats = extract_history_features_from_db(
                db=db,
                commit_timestamp=commit_timestamp,
                test_case_id=test_case_id,
                changed_file_paths=changed_file_paths,
            )
            feature_dict.update(hist_feats)
        else:
            # Safe zero-filled fallback if history DB not provided
            feature_dict.update({
                "hist_total_prior_runs": 0.0,
                "hist_prior_failures": 0.0,
                "hist_lifetime_failure_rate": 0.0,
                "hist_recent_10_failure_rate": 0.0,
                "hist_avg_duration": 0.05,
                "hist_flaky_score": 0.0,
                "hist_has_ever_failed": 0.0,
                "hist_changed_files_prior_mod_count": 0.0,
            })

        # Ensure all 32 features are present with numerical float types and zero NaN
        for name in FEATURE_NAMES:
            if name not in feature_dict or np.isnan(feature_dict[name]):
                feature_dict[name] = 0.0

        return feature_dict

    def to_feature_vector(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """Convert a feature dictionary into a 1D NumPy array in canonical feature order."""
        return np.array([float(feature_dict.get(name, 0.0)) for name in FEATURE_NAMES], dtype=np.float32)
