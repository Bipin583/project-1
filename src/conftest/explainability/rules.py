"""
ConfTest Rule-Based Natural Language Explanation Engine.

Translates continuous feature attributions and selective prediction metrics
into transparent, human-readable reason cards and PR Markdown summaries for developers.
"""

from typing import Any, Dict, List, Optional


class RuleBasedExplainer:
    """Generates structured natural language explanation cards for CI developers."""

    @staticmethod
    def generate_test_reason_card(
        test_id: str,
        feature_dict: Dict[str, float],
        shap_drivers: Optional[List[Dict[str, Any]]] = None,
        is_selected: bool = True,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate a human-readable reason card for a single test case.

        Args:
            test_id: Unique test node identifier.
            feature_dict: 32-feature dictionary for the (commit, test) pair.
            shap_drivers: Optional top SHAP feature drivers.
            is_selected: Whether test was selected for execution.
            confidence: Predicted failure confidence probability.

        Returns:
            Structured dictionary with decision summary, bullet reasons, and risk level.
        """
        reasons: List[str] = []

        # 1. Dependency & Coupling Rules
        if feature_dict.get("dep_is_direct_import", 0.0) == 1.0:
            reasons.append("Direct Dependency: Test file directly imports a modified source module.")
        elif feature_dict.get("dep_shortest_path_depth", 10.0) <= 2.0:
            depth = int(feature_dict.get("dep_shortest_path_depth", 2.0))
            reasons.append(f"Call-Graph Proximity: Test is within {depth} hops of modified functions in the static dependency graph.")
        elif feature_dict.get("dep_name_heuristic_coupled", 0.0) == 1.0:
            reasons.append("Module Coupling: Test module name pattern directly corresponds to changed source file.")

        # 2. Historical Telemetry Rules
        hist_runs = feature_dict.get("hist_total_prior_runs", 0.0)
        hist_fail_rate = feature_dict.get("hist_lifetime_failure_rate", 0.0)
        recent_10_fail = feature_dict.get("hist_recent_10_failure_rate", 0.0)

        if recent_10_fail > 0.0:
            reasons.append(f"Recent Regression History: Test failed in {recent_10_fail*100:.0f}% of its last 10 executions.")
        elif hist_fail_rate > 0.15:
            reasons.append(f"Historical Flakiness/Risk: Lifetime failure rate is {hist_fail_rate*100:.1f}% over {int(hist_runs)} runs.")

        # 3. Code Churn & Complexity Rules
        total_churn = feature_dict.get("diff_total_churn", 0.0)
        if total_churn > 100:
            reasons.append(f"High Code Churn: Commit introduces extensive modifications ({int(total_churn)} lines added/deleted).")

        if feature_dict.get("diff_is_fix_commit", 0.0) == 1.0:
            reasons.append("Bugfix Patch Context: Commit message indicates a bug fix or regression patch.")

        # Fallback default reason if no high-signal rules fired
        if not reasons:
            if is_selected:
                reasons.append("Baseline Coverage: Test selected based on global risk ranking within allocated budget.")
            else:
                reasons.append("Low Regression Risk: Test does not import modified files and has zero recent failures.")

        # SHAP attribution summary
        shap_summary = []
        if shap_drivers:
            for d in shap_drivers[:3]:
                shap_summary.append(f"{d['feature']} (value: {d['feature_value']}, SHAP: {d['shap_attribution']:+.3f})")

        risk_level = "HIGH" if (confidence and confidence > 0.60) else "MEDIUM" if (confidence and confidence > 0.20) else "LOW"

        return {
            "test_id": test_id,
            "status": "SELECTED" if is_selected else "OMITTED",
            "risk_level": risk_level,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "primary_reasons": reasons,
            "top_shap_drivers": shap_summary,
        }

    @staticmethod
    def generate_commit_markdown_summary(
        commit_sha: str,
        decision_dict: Dict[str, Any],
        top_tests: List[Dict[str, Any]],
    ) -> str:
        """Generate formatted GitHub PR Markdown comment summarizing test selection decisions."""
        mode = decision_dict.get("decision_mode", "FAST_SELECTED")
        abstained = decision_dict.get("abstained", False)
        saved_pct = decision_dict.get("test_reduction_pct", 0.0)
        selected_count = decision_dict.get("selected_count", len(top_tests))
        total_count = decision_dict.get("total_count", len(top_tests))
        uncertainty = decision_dict.get("epistemic_uncertainty", 0.0)
        confidence = decision_dict.get("top_confidence", 0.0)

        badge_color = "red" if abstained else "green"
        badge_text = "SAFE FALLBACK (100% SUITE)" if abstained else f"FAST RTS ({saved_pct:.1f}% SAVINGS)"

        md = []
        md.append(f"## 🛡️ ConfTest CI Regression Test Selection Report")
        md.append(f"**Commit:** `{commit_sha[:8]}` | **Status:** `{badge_text}`\n")

        md.append("### 📊 Decision Summary")
        md.append(f"- **Execution Mode:** `{mode}`")
        md.append(f"- **Selected Tests:** **{selected_count} / {total_count}** ({saved_pct:.1f}% test execution reduction)")
        md.append(f"- **Model Confidence:** `{confidence:.2%}` | **Epistemic Uncertainty:** `{uncertainty:.4f}`")

        if abstained:
            md.append("\n> ⚠️ **Safe Fallback Triggered:**")
            for r in decision_dict.get("reasons", []):
                md.append(f"> - {r}")
        else:
            md.append("\n> ⚡ **Selective Fast Execution:** High confidence with minimal uncertainty.")

        md.append("\n### 🔍 Top High-Risk Selected Tests & Rationale")
        md.append("| Test Case | Risk Level | Calibrated Confidence | Primary Rationale |")
        md.append("| :--- | :---: | :---: | :--- |")

        for t in top_tests[:10]:
            t_id = t.get("test_id", "")
            risk = t.get("risk_level", "MEDIUM")
            conf = f"{t.get('confidence', 0.0):.1%}" if t.get('confidence') else "N/A"
            reasons_str = "<br>".join(t.get("primary_reasons", ["Ranked risk within budget."])[:2])
            md.append(f"| `{t_id}` | `{risk}` | `{conf}` | {reasons_str} |")

        md.append("\n---\n*Generated automatically by ConfTest Confidence-Calibrated RTS Engine.*")
        return "\n".join(md)
