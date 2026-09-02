# ConfTest Canonical 32-Feature Schema Specification

## 1. Overview
ConfTest represents each candidate $(c, t)$ pair (Commit $c$, Test Case $t$) as a standardized **32-dimensional continuous numerical vector** combining code churn, AST syntactic complexity, static call-graph reachability, and historical execution telemetry.

---

## 2. Feature Definitions

### Category A: Diff & Code Churn Features (12)
| # | Feature Identifier | Type | Range | Description |
| :-: | :--- | :--- | :--- | :--- |
| 1 | `diff_lines_added` | `Float` | $[0, \infty)$ | Total lines added in the commit |
| 2 | `diff_lines_deleted` | `Float` | $[0, \infty)$ | Total lines deleted in the commit |
| 3 | `diff_total_churn` | `Float` | $[0, \infty)$ | `lines_added + lines_deleted` |
| 4 | `diff_num_files_changed`| `Float` | $[0, \infty)$ | Count of all modified files |
| 5 | `diff_num_src_files` | `Float` | $[0, \infty)$ | Count of non-test source files modified |
| 6 | `diff_num_test_files` | `Float` | $[0, \infty)$ | Count of test files modified |
| 7 | `diff_has_python` | `Binary`| $\{0.0, 1.0\}$ | 1 if `.py` source files modified |
| 8 | `diff_has_config` | `Binary`| $\{0.0, 1.0\}$ | 1 if `.yaml`, `.toml`, `.json` configuration modified |
| 9 | `diff_msg_length` | `Float` | $[0, \infty)$ | Character length of commit message |
| 10| `diff_msg_word_count` | `Float` | $[0, \infty)$ | Total words in commit message |
| 11| `diff_is_fix_commit` | `Binary`| $\{0.0, 1.0\}$ | 1 if message contains 'fix', 'bug', 'patch' |
| 12| `diff_is_refactor_commit`| `Binary`| $\{0.0, 1.0\}$ | 1 if message contains 'refactor', 'clean' |

### Category B: AST Syntactic & Complexity Features (6)
| # | Feature Identifier | Type | Range | Description |
| :-: | :--- | :--- | :--- | :--- |
| 13| `ast_test_file_functions_count` | `Float` | $[1, \infty)$ | Total functions declared in test file |
| 14| `ast_test_file_classes_count` | `Float` | $[0, \infty)$ | Total test classes declared in test file |
| 15| `ast_test_file_imports_count` | `Float` | $[0, \infty)$ | Total imports in test file |
| 16| `ast_test_file_complexity` | `Float` | $[1.0, \infty)$ | Cyclomatic decision complexity of test file |
| 17| `ast_test_is_parameterized` | `Binary` | $\{0.0, 1.0\}$ | 1 if test has `pytest.mark.parametrize` signature |
| 18| `ast_test_func_name_length` | `Float` | $[1, \infty)$ | Character length of test function name |

### Category C: Static Dependency Call-Graph Features (6)
| # | Feature Identifier | Type | Range | Description |
| :-: | :--- | :--- | :--- | :--- |
| 19| `dep_is_direct_import` | `Binary` | $\{0.0, 1.0\}$ | 1 if test file directly imports a changed file |
| 20| `dep_name_heuristic_coupled` | `Binary` | $\{0.0, 1.0\}$ | 1 if test module name matches source module |
| 21| `dep_shortest_path_depth` | `Float` | $[1.0, 10.0]$ | Shortest path from test to changed file (10.0 = disconnected) |
| 22| `dep_is_reachable` | `Binary` | $\{0.0, 1.0\}$ | 1 if directed dependency path exists |
| 23| `dep_max_reverse_dependencies` | `Float` | $[0, \infty)$ | Max incoming dependents on changed files |
| 24| `dep_test_total_out_degree` | `Float` | $[0, \infty)$ | Total modules imported by candidate test |

### Category D: Historical Telemetry & Anti-Leakage Features (8)
*All Category D features are strictly filtered to executions occurring before $t_{\text{commit}}$.*
| # | Feature Identifier | Type | Range | Description |
| :-: | :--- | :--- | :--- | :--- |
| 25| `hist_total_prior_runs` | `Float` | $[0, \infty)$ | Total historical executions prior to commit |
| 26| `hist_prior_failures` | `Float` | $[0, \infty)$ | Historical failure count prior to commit |
| 27| `hist_lifetime_failure_rate` | `Float` | $[0.0, 1.0]$ | Lifetime failure fraction prior to commit |
| 28| `hist_recent_10_failure_rate` | `Float` | $[0.0, 1.0]$ | Failure fraction over last 10 prior runs |
| 29| `hist_avg_duration` | `Float` | $[0.0, \infty)$ | Average execution runtime in seconds |
| 30| `hist_flaky_score` | `Float` | $[0.0, 1.0]$ | Ratio of retry flips in test history |
| 31| `hist_has_ever_failed` | `Binary` | $\{0.0, 1.0\}$ | 1 if test has at least 1 prior historical failure |
| 32| `hist_changed_files_prior_mod_count`| `Float` | $[0, \infty)$ | Total prior commit modifications on changed files |
