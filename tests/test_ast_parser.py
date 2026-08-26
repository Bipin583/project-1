"""
Unit tests for ConfTest AST & Dependency Feature Extractors.
"""
from src.features.ast_parser import ASTDiffAnalyzer
from src.features.dependency_graph import StaticDependencyGraph
from src.features.history_miner import HistoryMiner

def test_ast_diff_analyzer():
    analyzer = ASTDiffAnalyzer(language="python")
    sample_patch = """--- a/src/auth.py
+++ b/src/auth.py
@@ -1,3 +1,5 @@
+import jwt
+import bcrypt
 def login(user, pwd):
-    return False
+    return True
"""
    result = analyzer.parse_patch(sample_patch)
    assert result["total_added_lines"] == 3
    assert result["total_deleted_lines"] == 1
    assert result["total_churn"] == 4
    assert result["is_doc_only"] is False

def test_dependency_graph():
    graph = StaticDependencyGraph()
    sample_code = """
import os
from src.auth import login
import pytest
"""
    imports = graph.analyze_imports("tests/test_auth.py", sample_code)
    assert "os" in imports
    assert "src.auth" in imports

    assert graph.is_test_directly_dependent(imports, ["src/auth.py"]) is True
    assert graph.is_test_directly_dependent(imports, ["src/database/billing.py"]) is False

def test_history_miner():
    miner = HistoryMiner(decay_rate=0.05)
    miner.record_run("test_auth.py::test_login", outcome="FAIL", duration=1.2, age_days=1.0)
    miner.record_run("test_auth.py::test_login", outcome="FAIL", duration=1.4, age_days=2.0)
    miner.record_run("test_auth.py::test_login", outcome="PASS", duration=1.0, age_days=10.0)

    fail_rate = miner.compute_time_decayed_failure_rate("test_auth.py::test_login")
    assert fail_rate > 0.5  # High recent failure frequency
    assert miner.compute_average_duration("test_auth.py::test_login") > 1.0
