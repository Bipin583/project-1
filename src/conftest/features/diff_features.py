"""
ConfTest Diff & Churn Feature Extraction Module.

Extracts change metrics from Git diffs including added/deleted lines,
file churn, modified file counts, and commit message characteristics.
"""

from typing import Any, Dict, List


def extract_diff_features(
    changed_files: List[Dict[str, Any]],
    commit_message: str = "",
) -> Dict[str, float]:
    """
    Extract change and churn features from a list of modified files in a commit.

    Args:
        changed_files: List of file dictionaries with lines_added, lines_deleted, change_type, file_path.
        commit_message: Raw commit message string.

    Returns:
        Dictionary of numerical diff features.
    """
    lines_added = sum(f.get("lines_added", 0) for f in changed_files)
    lines_deleted = sum(f.get("lines_deleted", 0) for f in changed_files)
    total_churn = lines_added + lines_deleted
    num_files = len(changed_files)

    num_test_files = sum(1 for f in changed_files if f.get("is_test") or "test" in f.get("file_path", "").lower())
    num_src_files = max(0, num_files - num_test_files)

    has_python_change = float(any(f.get("file_path", "").endswith(".py") for f in changed_files))
    has_config_change = float(any(
        f.get("file_path", "").endswith((".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"))
        for f in changed_files
    ))

    # Commit message length and keywords
    msg_len = len(commit_message.strip())
    msg_words = len(commit_message.split())
    msg_lower = commit_message.lower()
    is_fix = float("fix" in msg_lower or "bug" in msg_lower or "patch" in msg_lower)
    is_refactor = float("refactor" in msg_lower or "clean" in msg_lower or "restructure" in msg_lower)

    return {
        "diff_lines_added": float(lines_added),
        "diff_lines_deleted": float(lines_deleted),
        "diff_total_churn": float(total_churn),
        "diff_num_files_changed": float(num_files),
        "diff_num_src_files": float(num_src_files),
        "diff_num_test_files": float(num_test_files),
        "diff_has_python": has_python_change,
        "diff_has_config": has_config_change,
        "diff_msg_length": float(msg_len),
        "diff_msg_word_count": float(msg_words),
        "diff_is_fix_commit": is_fix,
        "diff_is_refactor_commit": is_refactor,
    }
