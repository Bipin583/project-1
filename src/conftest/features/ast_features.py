"""
ConfTest AST Parsing & Syntactic Code Analysis Module.

Extracts function definitions, class structures, import statements,
and cyclomatic complexity using Python's standard `ast` module.
"""

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from conftest.logging_config import get_logger

logger = get_logger(__name__)


def parse_ast_safely(source_code: str, filename: str = "<unknown>") -> Optional[ast.AST]:
    """Parse source code string into an AST tree safely without crashing on syntax errors."""
    try:
        return ast.parse(source_code, filename=filename)
    except SyntaxError as exc:
        logger.warning(f"Syntax error while parsing AST for {filename}: {exc}")
        return None
    except Exception as exc:
        logger.warning(f"Unexpected AST parse error for {filename}: {exc}")
        return None


def extract_imports_from_ast(tree: ast.AST) -> Set[str]:
    """Extract all imported module names and symbols from an AST tree."""
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if module:
                    imports.add(f"{module}.{alias.name}")
                    imports.add(module)
                else:
                    imports.add(alias.name)
    return imports


def extract_functions_and_classes(tree: ast.AST) -> Tuple[List[str], List[str]]:
    """Extract top-level and method function names and class names from AST."""
    functions = []
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
    return functions, classes


def estimate_cyclomatic_complexity(tree: ast.AST) -> float:
    """
    Estimate cyclomatic complexity from AST by counting branching decision points.
    Base complexity is 1, incremented for every If, For, While, ExceptHandler, And, Or.
    """
    complexity = 1.0
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With, ast.Assert)):
            complexity += 1.0
        elif isinstance(node, ast.BoolOp):
            complexity += max(0, len(node.values) - 1)
        elif isinstance(node, ast.IfExp):  # Ternary expression: a if b else c
            complexity += 1.0
    return complexity


def extract_ast_metrics_from_file(file_path: str) -> Dict[str, Any]:
    """Extract structural metrics (functions, classes, imports, complexity) from a Python file."""
    path = Path(file_path)
    if not path.exists() or not path.suffix == ".py":
        return {
            "functions_count": 0,
            "classes_count": 0,
            "imports_count": 0,
            "complexity": 1.0,
            "imports": set(),
        }

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()
        tree = parse_ast_safely(code, filename=str(path))
        if not tree:
            return {
                "functions_count": 0,
                "classes_count": 0,
                "imports_count": 0,
                "complexity": 1.0,
                "imports": set(),
            }

        funcs, classes = extract_functions_and_classes(tree)
        imports = extract_imports_from_ast(tree)
        complexity = estimate_cyclomatic_complexity(tree)

        return {
            "functions_count": len(funcs),
            "classes_count": len(classes),
            "imports_count": len(imports),
            "complexity": complexity,
            "imports": imports,
        }
    except Exception as exc:
        logger.warning(f"Failed extracting AST metrics for {file_path}: {exc}")
        return {
            "functions_count": 0,
            "classes_count": 0,
            "imports_count": 0,
            "complexity": 1.0,
            "imports": set(),
        }
