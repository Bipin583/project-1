"""
ConfTest Streamlit Page 5: SHAP & Model Explainability.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import load_shap_report

st.set_page_config(page_title="SHAP Explainability | ConfTest", page_icon="🔍", layout="wide")

st.title("🔍 SHAP & Model Explainability")
st.markdown("Game-theoretic feature attributions ($\phi_i$) explaining why individual regression tests are prioritized or omitted.")

shap_data = load_shap_report()
global_imp = shap_data.get("global_shap_importance", [])

if global_imp:
    df_shap = pd.DataFrame(global_imp)
    
    st.subheader("🌐 Global SHAP Feature Importance Rankings")
    fig_shap = px.bar(
        df_shap.head(10),
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        color="mean_abs_shap",
        color_continuous_scale="Blues",
        title="Top 10 Most Influential Features (Mean |SHAP Value|)",
        labels={"mean_abs_shap": "Mean |SHAP| Attribution", "feature": "32-Feature Schema"},
    )
    fig_shap.update_layout(yaxis={"autorange": "reversed"})
    st.plotly_chart(fig_shap, use_container_width=True)

st.divider()

st.subheader("🧩 32-Feature Category Distribution")
feat_cats = {
    "Code Churn & Diff (12 features)": ["diff_files_changed", "diff_lines_added", "diff_lines_deleted", "diff_total_churn", "diff_is_fix_commit"],
    "AST Complexity (6 features)": ["ast_func_count", "ast_cyclomatic_delta", "ast_num_asserts", "ast_is_parameterized"],
    "Dependency Graph (6 features)": ["dep_is_direct_import", "dep_shortest_path_depth", "dep_name_heuristic_coupled", "dep_coupling_coefficient"],
    "Historical Telemetry (8 features)": ["hist_total_prior_runs", "hist_prior_failures", "hist_lifetime_failure_rate", "hist_recent_10_failure_rate", "hist_flakiness_score"],
}

cat_counts = pd.DataFrame([
    {"Category": k, "Feature Count": len(v)} for k, v in feat_cats.items()
])

fig_pie = px.pie(cat_counts, names="Category", values="Feature Count", title="Feature Schema Composition")
st.plotly_chart(fig_pie, use_container_width=True)

st.info("💡 **TreeExplainer Guarantees:** Shapley attributions satisfy the additivity axiom: $f(\\mathbf{x}) = \\phi_0 + \\sum_{i=1}^{32} \\phi_i$, providing exact linear additive explanations for every tree decision.")
