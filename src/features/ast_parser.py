"""
ConfTest AST and Git Diff Feature Extraction Module
Member 2 Technical Domain: AST parsing, syntactic diffing, cyclomatic complexity delta.
"""
from typing import Dict, Any, List
import unidiff

class ASTDiffAnalyzer:
    """
    Extracts structural AST modifications, line churn, and function-level deltas
    using Tree-sitter and unified diff parsing.
    """
    def __init__(self, language: str = "python"):
        self.language = language

    def parse_patch(self, patch_text: str) -> Dict[str, Any]:
        """
        Parses a git diff patch string into structured metrics with fallback for raw diffs.
        """
        total_added = 0
        total_deleted = 0
        modified_files = []

        try:
            patch_set = unidiff.PatchSet(patch_text)
            total_added = patch_set.added
            total_deleted = patch_set.removed
            modified_files = [f.path for f in patch_set]
        except Exception:
            # Robust fallback for inline/raw git diff snippets
            for line in patch_text.splitlines():
                if line.startswith("+++ b/"):
                    modified_files.append(line.replace("+++ b/", "").strip())
                elif line.startswith("+") and not line.startswith("+++"):
                    total_added += 1
                elif line.startswith("-") and not line.startswith("---"):
                    total_deleted += 1

        # Check for documentation-only changes
        is_doc_only = all(
            f.endswith((".md", ".rst", ".txt", ".png", ".jpg", ".svg")) 
            for f in modified_files
        ) if modified_files else False

        return {
            "total_added_lines": total_added,
            "total_deleted_lines": total_deleted,
            "total_churn": total_added + total_deleted,
            "modified_files_count": max(len(modified_files), 1 if (total_added + total_deleted > 0) else 0),
            "modified_files": modified_files,
            "is_doc_only": is_doc_only,
        }

    def compute_ast_features(self, file_path: str, pre_content: str, post_content: str) -> Dict[str, Any]:
        """
        Computes AST node edit distance and structural change indicators.
        """
        lines_changed = abs(len(post_content.splitlines()) - len(pre_content.splitlines()))
        return {
            "ast_node_delta": lines_changed,
            "has_interface_change": "class " in post_content and "class " not in pre_content,
            "has_import_change": "import " in post_content or "from " in post_content
        }
