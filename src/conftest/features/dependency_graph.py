"""
ConfTest Static Call-Graph & Dependency Relationship Engine.

Constructs module dependency graphs using NetworkX to calculate structural coupling,
shortest dependency path depth, direct import relationships, and reverse dependencies.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import networkx as nx

from conftest.features.ast_features import extract_ast_metrics_from_file
from conftest.logging_config import get_logger

logger = get_logger(__name__)


class DependencyGraphBuilder:
    """Constructs and queries static import call-graphs across Python repositories."""

    def __init__(self, repo_root: str):
        """
        Initialize the dependency graph builder.

        Args:
            repo_root: Root path of the repository to index.
        """
        self.repo_root = Path(repo_root).resolve()
        self.graph = nx.DiGraph()
        self.module_to_file_map: Dict[str, str] = {}
        self.file_to_imports_map: Dict[str, Set[str]] = {}
        self._build_initial_graph()

    def _normalize_module_name(self, file_rel_path: str) -> str:
        """Convert a file path like 'src/app/auth.py' to Python module string 'src.app.auth' and 'auth'."""
        clean = file_rel_path.replace("\\", "/").replace(".py", "")
        if clean.endswith("/__init__"):
            clean = clean[:-9]
        return clean.replace("/", ".")

    def _build_initial_graph(self) -> None:
        """Scan all Python files in the repository and build static dependency graph edges."""
        if not self.repo_root.exists():
            return

        py_files: List[Path] = []
        for root, _, files in os.walk(self.repo_root):
            for f in files:
                if f.endswith(".py"):
                    py_files.append(Path(root) / f)

        # 1. Map files to modules
        for pf in py_files:
            try:
                rel = pf.relative_to(self.repo_root).as_posix()
                mod_name = self._normalize_module_name(rel)
                self.module_to_file_map[mod_name] = rel
                self.module_to_file_map[pf.stem] = rel
                self.graph.add_node(rel)
            except Exception:
                pass

        # 2. Add dependency edges: A -> B means A imports B
        for pf in py_files:
            try:
                rel = pf.relative_to(self.repo_root).as_posix()
                ast_info = extract_ast_metrics_from_file(str(pf))
                imports = ast_info["imports"]
                self.file_to_imports_map[rel] = imports

                for imp in imports:
                    # Check if imported module matches a known project file
                    target_file = self.module_to_file_map.get(imp)
                    if not target_file:
                        # Check module prefix
                        base_mod = imp.split(".")[0]
                        target_file = self.module_to_file_map.get(base_mod)

                    if target_file and target_file != rel:
                        self.graph.add_edge(rel, target_file)
            except Exception as exc:
                logger.warning(f"Error parsing graph node {pf}: {exc}")

    def compute_dependency_features(
        self,
        test_path: str,
        changed_file_paths: List[str],
    ) -> Dict[str, float]:
        """
        Compute structural dependency features between a specific test file and changed files.

        Args:
            test_path: Relative path to candidate test file.
            changed_file_paths: List of relative paths to modified files in the commit.

        Returns:
            Dictionary with graph coupling features.
        """
        norm_test = test_path.replace("\\", "/")
        norm_changed = [p.replace("\\", "/") for p in changed_file_paths]

        is_direct = 0.0
        min_depth = 99.0
        max_reverse_deps = 0.0

        for ch in norm_changed:
            # 1. Direct import check
            if ch in self.graph:
                if self.graph.has_edge(norm_test, ch):
                    is_direct = 1.0
                    min_depth = min(min_depth, 1.0)
                elif norm_test in self.graph:
                    # 2. Shortest path in dependency graph (test -> ... -> changed_file)
                    try:
                        path_len = nx.shortest_path_length(self.graph, source=norm_test, target=ch)
                        min_depth = min(min_depth, float(path_len))
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass

                # 3. Reverse dependency count (how many files depend on this changed file)
                rev_deps = len(list(self.graph.predecessors(ch))) if ch in self.graph else 0
                max_reverse_deps = max(max_reverse_deps, float(rev_deps))

        # Check name heuristic coupling (e.g. tests/test_auth.py -> app/auth.py)
        test_stem = Path(norm_test).stem.replace("test_", "").replace("_test", "")
        name_coupled = float(any(test_stem in Path(ch).stem for ch in norm_changed))

        return {
            "dep_is_direct_import": is_direct,
            "dep_name_heuristic_coupled": name_coupled,
            "dep_shortest_path_depth": min_depth if min_depth != 99.0 else 10.0,
            "dep_is_reachable": 1.0 if min_depth < 99.0 else 0.0,
            "dep_max_reverse_dependencies": max_reverse_deps,
            "dep_test_total_out_degree": float(self.graph.out_degree(norm_test)) if norm_test in self.graph else 0.0,
        }
