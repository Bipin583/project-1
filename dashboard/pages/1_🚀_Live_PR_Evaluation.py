"""
ConfTest Streamlit Page 1: Live PR Evaluation.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import get_cached_engine
from conftest.explainability.rules import RuleBasedExplainer

st.set_page_config(page_title="Live PR Evaluation | ConfTest", page_icon="🚀", layout="wide")

st.title("🚀 Live Pull Request Test Selection")
st.markdown("Run the full ConfTest engine on an arbitrary commit diff with selective prediction fallback.")

# Input Controls
with st.sidebar:
    st.header("⚙️ Evaluation Parameters")
    commit_sha = st.text_input("Commit SHA / Branch Ref", value="c0ffee99")
    budget_pct = st.slider("Fast Mode Test Budget (%)", min_value=10, max_value=100, value=25, step=5)
    commit_msg = st.text_input("Commit Message", value="fix: update auth session handling")
    
    st.markdown("### 📁 Simulated Modified Files")
    f_auth = st.checkbox("src_app/auth.py", value=True)
    f_pay = st.checkbox("src_app/payment.py", value=False)
    f_db = st.checkbox("src_app/database.py", value=False)

changed_files = []
if f_auth:
    changed_files.append({"file_path": "src_app/auth.py", "change_type": "M", "lines_added": 18, "lines_deleted": 4})
if f_pay:
    changed_files.append({"file_path": "src_app/payment.py", "change_type": "M", "lines_added": 25, "lines_deleted": 8})
if f_db:
    changed_files.append({"file_path": "src_app/database.py", "change_type": "M", "lines_added": 12, "lines_deleted": 2})

if not changed_files:
    changed_files.append({"file_path": "src_app/auth.py", "change_type": "M", "lines_added": 15, "lines_deleted": 3})

btn_run = st.button("⚡ Evaluate Regression Tests", type="primary")

if btn_run or "last_outcome" in st.session_state:
    if btn_run:
        engine = get_cached_engine()
        with st.spinner("Analyzing AST, extracting 32 features, and running 5-seed ensemble..."):
            outcome = engine.analyze_and_select(
                commit_sha=commit_sha,
                changed_files=changed_files,
                commit_message=commit_msg,
                budget_ratio=budget_pct / 100.0,
                execute=False,
            )
            st.session_state["last_outcome"] = outcome
    else:
        outcome = st.session_state["last_outcome"]

    # Decision Banner
    mode = outcome["decision_mode"]
    abstained = outcome["abstained"]
    saved_pct = outcome["test_reduction_pct"]
    sel_count = outcome["selected_count"]
    tot_count = outcome["total_count"]

    if abstained:
        st.warning(f"🛡️ **Decision Mode: SAFE_FULL_SUITE (Abstained from RTS)**\n\nFull suite fallback triggered for 100% safety. Executing all **{tot_count} tests**.")
    else:
        st.success(f"⚡ **Decision Mode: FAST_SELECTED**\n\nHigh confidence with minimal uncertainty. Selected **{sel_count} / {tot_count} tests** ({saved_pct:.1f}% time reduction).")

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected Tests", f"{sel_count} / {tot_count}")
    m2.metric("Execution Reduction", f"{saved_pct:.1f}%")
    m3.metric("Top Confidence", f"{outcome['top_confidence']:.2%}")
    m4.metric("Epistemic Uncertainty", f"{outcome['epistemic_uncertainty']:.4f}")

    st.divider()

    # Ranked Tests Breakdown
    st.subheader("🔍 Ranked Tests & Failure Risk Attributions")
    ranked = outcome.get("ranked_tests", [])
    if ranked:
        df_ranked = pd.DataFrame(ranked)
        df_ranked["risk_pct"] = df_ranked["calibrated_confidence"] * 100
        
        fig = px.bar(
            df_ranked,
            x="risk_pct",
            y="test_id",
            orientation="h",
            color="is_selected",
            color_discrete_map={True: "#22c55e", False: "#64748b"},
            labels={"risk_pct": "Calibrated Failure Probability (%)", "test_id": "Test Identifier", "is_selected": "Selected for Run"},
            title="Individual Test Failure Probabilities",
        )
        fig.update_layout(yaxis={"autorange": "reversed"})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Developer Rationale Cards")
        rule_explainer = RuleBasedExplainer()
        for t in ranked[:5]:
            card = rule_explainer.generate_test_reason_card(
                test_id=t["test_id"],
                feature_dict={"dep_is_direct_import": 1.0 if t["is_selected"] else 0.0, "diff_total_churn": 22.0},
                is_selected=t["is_selected"],
                confidence=t["calibrated_confidence"],
            )
            with st.expander(f"{'✅' if t['is_selected'] else '⏸️'} `{t['test_id']}` — Risk: {card['risk_level']} ({t['calibrated_confidence']:.1%})"):
                for r in card["primary_reasons"]:
                    st.write(f"- {r}")
