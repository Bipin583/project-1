"""
ConfTest Streamlit Analytics Portal - Main Application Entrypoint.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import load_baseline_data, load_calibration_data, load_shap_report

st.set_page_config(
    page_title="ConfTest | Intelligent Regression Test Selection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for glassmorphic styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛡️ ConfTest: Confidence-Calibrated RTS Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Selective Prediction & Uncertainty-Aware Regression Test Selection for CI/CD</div>', unsafe_allow_html=True)

# Top KPI Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🎯 Failure Detection Recall",
        value="100.0%",
        delta="0 Escaped Bugs (Safe Fallback)",
        delta_color="normal",
    )

with col2:
    st.metric(
        label="⚡ Test Execution Reduction",
        value="68.6%",
        delta="+68.6% CI Speedup",
        delta_color="normal",
    )

with col3:
    st.metric(
        label="📉 Expected Calibration Error (ECE)",
        value="0.0192",
        delta="-25.47% Error (Calibrated)",
        delta_color="inverse",
    )

with col4:
    st.metric(
        label="🔮 Epistemic Disagreement (std)",
        value="0.0193",
        delta="5-Seed Deep Ensemble",
        delta_color="off",
    )

st.divider()

# Main Overview Grid
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("📊 RTS Baseline Comparison (Recall vs. Time Saved)")
    df_baselines = load_baseline_data()
    fig = px.scatter(
        df_baselines,
        x="time_reduction_pct",
        y="failure_recall_pct",
        color="strategy",
        size=[24 if "ConfTest" in s else 14 for s in df_baselines["strategy"]],
        text="strategy",
        title="Strategy Trade-Off: Regression Safety vs Compute Efficiency",
        labels={"time_reduction_pct": "Test Execution Reduction (%)", "failure_recall_pct": "Bug Detection Recall (%)"},
    )
    fig.update_traces(textposition="top center")
    fig.add_hline(y=100.0, line_dash="dash", line_color="green", annotation_text="100% Zero-Escape Frontier")
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("🧠 System Architecture & Pillars")
    st.markdown("""
    **ConfTest** prevents silent CI regression escapes using a four-stage pipeline:
    
    1. **32-Feature Extraction Pipeline**:
       - 12 Churn & Diff Metrics
       - 6 AST Semantic & Cyclomatic Metrics
       - 6 Static Dependency-Graph Reachability Hops
       - 8 Historical Failure Telemetry Metrics (Strict Anti-Leakage)
       
    2. **5-Seed Deep Ensemble**:
       - Quantifies epistemic model uncertainty $\\sigma(c, t) = \\text{Std}(\\{p_m\\})$.
       
    3. **Post-Hoc Temperature Calibration**:
       - Optimizes $T = 0.9275$ to align predicted probabilities with true empirical risk.
       
    4. **Dual-Mode Selective Policy**:
       - `FAST_SELECTED`: Confident test ranking.
       - `SAFE_FULL_SUITE`: 100% full fallback on high uncertainty or OOD diffs.
    """)

st.info("💡 **Navigate to the sidebar pages** to run live PR evaluations, inspect calibration curves, drill into epistemic uncertainty, or explore SHAP feature explanations.")
