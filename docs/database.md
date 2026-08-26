# ConfTest Database Schema & Data Models

## 1. Overview
ConfTest employs a relational schema designed with **SQLAlchemy 2.0**.
- Default Engine: **SQLite with Write-Ahead Logging (WAL)** and foreign keys enforced via connection PRAGMAs.
- Production/Cloud: PostgreSQL-compatible data types (`JSON`, `DateTime`, `Float`, `String(40)` for Git SHAs).

## 2. Table Specifications

### 1. `repositories`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | Primary Key, Auto-increment | Internal repository identifier |
| `full_name` | `VARCHAR(255)` | Unique, Not Null, Indexed | e.g. `pallets/flask` |
| `url` | `VARCHAR(512)` | Not Null | GitHub / Remote Clone URL |
| `language` | `VARCHAR(64)` | Default `'python'` | Repository primary language |
| `default_branch` | `VARCHAR(64)` | Default `'main'` | Default branch name |
| `local_path` | `VARCHAR(1024)` | Not Null | Absolute or relative local checkout path |
| `created_at` | `TIMESTAMP` | Default UTC | Registration timestamp |

### 2. `commits`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | Primary Key, Auto-increment | Internal commit identifier |
| `repository_id` | `INTEGER` | FK -> `repositories.id`, Cascade | Associated repository |
| `sha` | `VARCHAR(40)` | Unique, Not Null, Indexed | Full 40-character Git SHA-1 |
| `parent_sha` | `VARCHAR(40)` | Nullable | Parent Git SHA-1 |
| `timestamp` | `TIMESTAMP` | Not Null, Indexed | Commit creation time (used for temporal split) |
| `author_hash` | `VARCHAR(64)` | Nullable | Anonymized SHA-256 author identifier |
| `message` | `TEXT` | Nullable | Git commit message |
| `ci_status` | `VARCHAR(32)` | Default `'pending'` | Overall CI outcome: `passed`, `failed`, `error` |
| `total_duration`| `FLOAT` | Default 0.0 | Full test suite duration in seconds |

### 3. `changed_files`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | Primary Key, Auto-increment | Internal diff record ID |
| `commit_id` | `INTEGER` | FK -> `commits.id`, Cascade | Associated commit |
| `file_path` | `VARCHAR(1024)` | Not Null | Relative path to modified file |
| `change_type` | `VARCHAR(16)` | Default `'MODIFIED'` | `'ADDED'`, `'MODIFIED'`, `'DELETED'` |
| `lines_added` | `INTEGER` | Default 0 | Added line count |
| `lines_deleted`| `INTEGER` | Default 0 | Removed line count |
| `cyclomatic_complexity` | `FLOAT` | Default 0.0 | Radon complexity delta |

### 4. `test_cases`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | Primary Key, Auto-increment | Internal test case ID |
| `repository_id` | `INTEGER` | FK -> `repositories.id`, Cascade | Associated repository |
| `test_id` | `VARCHAR(1024)` | Not Null, Indexed | Unique node ID (e.g. `tests/test_x.py::test_y`) |
| `test_path` | `VARCHAR(512)` | Not Null | Path to test file |
| `test_function`| `VARCHAR(256)` | Not Null | Name of test function/method |
| `framework` | `VARCHAR(32)` | Default `'pytest'` | Test framework |
| `average_duration` | `FLOAT` | Default 0.0 | Historical running time in seconds |
| `flaky_indicator` | `FLOAT` | Default 0.0 | Flakiness score $[0.0, 1.0]$ |

### 5. `test_runs`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | Primary Key, Auto-increment | Execution instance ID |
| `commit_id` | `INTEGER` | FK -> `commits.id`, Cascade | Evaluated commit |
| `test_case_id` | `INTEGER` | FK -> `test_cases.id`, Cascade | Evaluated test case |
| `status` | `VARCHAR(32)` | Not Null | `'PASSED'`, `'FAILED'`, `'SKIPPED'`, `'ERROR'` |
| `duration` | `FLOAT` | Default 0.0 | Actual execution time in seconds |
| `retry_count` | `INTEGER` | Default 0 | Retries before final status |
| `source` | `VARCHAR(32)` | Default `'ci'` | `'ci'`, `'local'`, `'replay'` |

### 6. `feature_records`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | Primary Key, Auto-increment | Feature vector record ID |
| `commit_id` | `INTEGER` | FK -> `commits.id`, Cascade | Evaluated commit |
| `test_case_id` | `INTEGER` | FK -> `test_cases.id`, Cascade | Evaluated test case |
| `feature_vector` | `JSON` | Not Null | 32-element extracted feature dictionary |

### 7. `predictions`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | Primary Key, Auto-increment | Model output record ID |
| `commit_id` | `INTEGER` | FK -> `commits.id`, Cascade | Evaluated commit |
| `test_case_id` | `INTEGER` | FK -> `test_cases.id`, Cascade | Evaluated test case |
| `raw_score` | `FLOAT` | Not Null | Uncalibrated failure score $[0, 1]$ |
| `uncertainty` | `FLOAT` | Not Null | Epistemic disagreement $\sigma$ |
| `calibrated_confidence` | `FLOAT` | Not Null | Post-hoc calibrated probability $P(\text{fail})$ |
| `model_version` | `VARCHAR(64)` | Not Null | Model tag (e.g. `'lgbm_ensemble_v1.0'`) |

### 8. `selection_decisions`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | Primary Key, Auto-increment | Decision record ID |
| `commit_id` | `INTEGER` | FK -> `commits.id`, Unique | Evaluated commit |
| `mode` | `VARCHAR(32)` | Not Null | `'FAST_SELECTED'` or `'SAFE_FULL_SUITE'` |
| `abstained` | `BOOLEAN` | Not Null | True if full suite was triggered due to uncertainty |
| `uncertainty_score` | `FLOAT` | Not Null | Max/Mean epistemic uncertainty |
| `threshold_used` | `FLOAT` | Not Null | Configured $\tau_{\text{abstain}}$ |
| `selected_count` | `INTEGER` | Not Null | Number of tests selected |
| `total_count` | `INTEGER` | Not Null | Total tests in test suite |
| `estimated_saving` | `FLOAT` | Default 0.0 | Estimated time reduction ratio |
| `reasons` | `JSON` | Nullable | Traceable SHAP and dependency rationale |

### 9. `outcomes`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | Primary Key, Auto-increment | Outcome audit record ID |
| `commit_id` | `INTEGER` | FK -> `commits.id`, Unique | Evaluated commit |
| `actual_failures` | `INTEGER` | Default 0 | Total real failing tests |
| `detected_failures` | `INTEGER` | Default 0 | Real failing tests captured in selected set |
| `missed_failures` | `INTEGER` | Default 0 | Failing tests missed by selected set |
| `full_duration` | `FLOAT` | Default 0.0 | Full suite runtime in seconds |
| `selected_duration`| `FLOAT` | Default 0.0 | Selected suite runtime in seconds |
| `time_reduction_ratio` | `FLOAT` | Default 0.0 | $1 - \frac{\text{selected}}{\text{full}}$ |
