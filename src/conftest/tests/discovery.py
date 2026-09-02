"""
ConfTest Pytest Discovery Engine.

Discovers regression test files, test classes, functions, and pytest node IDs
using pytest collection in isolated subprocesses and Python AST inspection.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from conftest.logging_config import get_logger

logger = get_logger(__name__)

# Strict regex pattern for validating pytest node IDs (avoids command injection)
NODE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\./\\]+(::[a-zA-Z0-9_\[\]\-\.\:]+)?$")


def validate_test_node_id(node_id: str) -> bool:
    """Validate that a test node ID contains only safe characters."""
    if not node_id or len(node_id) > 1024:
        return False
    return bool(NODE_ID_PATTERN.match(node_id))


class PytestDiscovery:
    """Discovers pytest test cases across a repository."""

    __test__ = False  # Prevent pytest from treating discovery class as a test suite

    def __init__(self, repo_root: str):
        """
        Initialize test discovery.

        Args:
            repo_root: Root directory of target repository to scan.
        """
        self.repo_root = Path(repo_root).resolve()

    def discover_via_pytest(self, test_dir: Optional[str] = None, timeout: int = 20) -> List[Dict[str, Any]]:
        """
        Discover test cases using `pytest --collect-only -q`.

        Args:
            test_dir: Specific test directory to scan (relative to repo_root).
            timeout: Subprocess timeout in seconds.

        Returns:
            List of discovered test dictionaries with node IDs, paths, and function names.
        """
        target_path = (self.repo_root / test_dir) if test_dir else self.repo_root
        if not target_path.exists():
            logger.warning(f"Target test path does not exist: {target_path}")
            return []

        cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", str(target_path)]
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error(f"Pytest collection timed out after {timeout}s on {target_path}")
            return self.discover_via_ast(test_dir)

        test_cases: List[Dict[str, Any]] = []
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            # Pytest -q collect-only outputs node IDs on each line, ending with summary
            if "::" in line and not line.startswith("<") and not line.endswith("collected"):
                node_id = line.split(" ")[0].replace("\\", "/")
                if validate_test_node_id(node_id):
                    parts = node_id.split("::")
                    file_path = parts[0]
                    func_name = parts[-1]
                    test_cases.append({
                        "test_id": node_id,
                        "test_path": file_path,
                        "test_function": func_name,
                        "framework": "pytest",
                    })

        if not test_cases:
            logger.info("Pytest subprocess collection yielded 0 tests. Falling back to AST scanning.")
            return self.discover_via_ast(test_dir)

        return test_cases

    def discover_via_ast(self, test_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fallback static discovery using Python's `ast` parser.
        Finds all files named `test_*.py` or `*_test.py` and functions starting with `test_`.
        """
        search_root = (self.repo_root / test_dir) if test_dir else self.repo_root
        test_cases: List[Dict[str, Any]] = []

        for root, _, files in os.walk(search_root):
            for file in files:
                if file.startswith("test_") and file.endswith(".py") or file.endswith("_test.py"):
                    full_path = Path(root) / file
                    try:
                        rel_path = full_path.relative_to(self.repo_root).as_posix()
                        with open(full_path, "r", encoding="utf-8") as f:
                            tree = ast.parse(f.read(), filename=str(full_path))

                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                                node_id = f"{rel_path}::{node.name}"
                                test_cases.append({
                                    "test_id": node_id,
                                    "test_path": rel_path,
                                    "test_function": node.name,
                                    "framework": "pytest",
                                })
                            elif isinstance(node, ast.ClassDef) and (node.name.startswith("Test") or node.name.endswith("Test")):
                                for method in node.body:
                                    if isinstance(method, ast.FunctionDef) and method.name.startswith("test_"):
                                        node_id = f"{rel_path}::{node.name}::{method.name}"
                                        test_cases.append({
                                            "test_id": node_id,
                                            "test_path": rel_path,
                                            "test_function": method.name,
                                            "framework": "pytest",
                                        })
                    except Exception as exc:
                        logger.warning(f"AST parse error in {full_path}: {exc}")

        return test_cases
