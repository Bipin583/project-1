# ConfTest System Architecture

## 1. Overview
ConfTest is a research-grade regression test selection (RTS) engine designed for CI/CD pipelines. It bridges the gap between fast test execution and safety by introducing **epistemic uncertainty estimation and post-hoc confidence calibration**.

```
+-------------------------------------------------------------------------------+
|                             GitHub CI/CD / PR                                 |
+---------------------------------------+---------------------------------------+
                                        | (Diff, Commit SHA, Changed Files)
                                        v
+-------------------------------------------------------------------------------+
|                       Ingestion & Static Code Analysis                        |
|  - Git Churn Mining (Lines +/-)                                               |
|  - AST Syntactic Parsing (Function-level modifications)                       |
|  - Static Dependency Call-Graph (NetworkX)                                    |
|  - Test Discovery (pytest --collect-only)                                     |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                        Feature Engineering Matrix                             |
|  - 32 Tabular Features (Diff + AST + Churn + History + Flakiness + Duration)  |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                        Machine Learning & Uncertainty                         |
|  - 5-Seed Diverse LightGBM / XGBoost Ensemble                                 |
|  - Epistemic Uncertainty (Ensemble Variance / Disagreement)                   |
|  - Post-Hoc Isotonic / Temperature Confidence Calibration                     |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                       Selective Policy & Decision Engine                      |
|                                                                               |
|       Is Uncertainty <= Threshold & Min Confidence >= Required?               |
|                      /                                \                       |
|                   [YES]                              [NO]                     |
|                    /                                    \                     |
|           FAST SELECTIVE MODE                    SAFE ABSTENTION MODE         |
|     - Rank tests by calibrated score       - Fall back to FULL test suite     |
|     - Execute top budget-matched subset    - Log uncertainty trigger cause    |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                       Execution & Reporting Interface                         |
|  - Subprocess Pytest Runner with Isolation & Timeout                          |
|  - SQLite / PostgreSQL Persistence (WAL mode)                                 |
|  - Streamlit Interactive Dashboard                                            |
|  - FastAPI REST Analytics API                                                 |
|  - GitHub Actions Pull Request Feedback Bot                                   |
+-------------------------------------------------------------------------------+
```
