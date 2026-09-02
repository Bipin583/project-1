"""
ConfTest Streamlit Page 4: RTS Baseline Benchmark Comparisons.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import load_baseline_data

st.set_page_config(page_title="RTS Baseline Comparison | ConfTest", page_icon="📊", layout="wide")

st.title("📊 RTS Baseline Comparison Benchmark")
st.markdown("Comparative evaluation across **8 Regression Test Selection strategies** under budget-matched testing constraints.")

df = load_baseline_data()

highlight_cols = [c for c in ["failure_recall_pct", "time_reduction_pct", "safety_score"] if c in df.columns]
st.dataframe(
    df.style.highlight_max(subset=highlight_cols, color="#1e3a8a") if highlight_cols else df,
    use_container_width=True,
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Failure Detection Recall (%)")
    fig_rec = px.bar(
        df,
        x="failure_recall_pct",
        y="strategy",
        orientation="h",
        color="strategy",
        title="Bug Recall across Strategies (Target: 100%)",
        labels={"failure_recall_pct": "Recall (%)", "strategy": "RTS Strategy"},
    )
    fig_rec.add_vline(x=100.0, line_dash="dash", line_color="green")
    fig_rec.update_layout(showlegend=False, yaxis={"autorange": "reversed"})
    st.plotly_chart(fig_rec, use_container_width=True)

with col2:
    st.subheader("⚡ Compute Time Reduction (%)")
    fig_time = px.bar(
        df,
        x="time_reduction_pct",
        y="strategy",
        orientation="h",
        color="strategy",
        title="Test Execution Reduction across Strategies",
        labels={"time_reduction_pct": "Time Saved (%)", "strategy": "RTS Strategy"},
    )
    fig_time.update_layout(showlegend=False, yaxis={"autorange": "reversed"})
    st.plotly_chart(fig_time, use_container_width=True)

st.info("🏆 **Key Finding:** ConfTest is the **only selective RTS system** achieving **100.0% Failure Recall** while delivering **68.6% test execution reduction**, eliminating regression escapes through selective full-suite fallback.")
