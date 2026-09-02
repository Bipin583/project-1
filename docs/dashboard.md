# ConfTest Streamlit Analytics Dashboard Specification

## 1. Overview
The ConfTest Streamlit Visual Analytics Dashboard is a comprehensive, interactive multi-page developer portal for monitoring CI regression test selection, exploring confidence calibration, analyzing ensemble epistemic uncertainty, and inspecting per-test SHAP explanations.

- **Launch Command:**
  ```bash
  streamlit run dashboard/app.py
  ```

---

## 2. Multi-Page Architecture

### 🏠 Main Portal (`dashboard/app.py`)
- **System KPIs:** 100% Zero-Escape Recall, 68.6% Test Execution Reduction, 25.47% ECE Reduction, and 0.0193 Epistemic Disagreement.
- **RTS Frontier Scatter Plot:** Interactive visualization showing the Pareto frontier of regression safety vs. compute time savings.

### 🚀 Page 1: Live PR Evaluation (`1_🚀_Live_PR_Evaluation.py`)
- Live interactive commit evaluator.
- Adjust test budget ratio slider ($10\% - 100\%$).
- Select modified files and trigger `ConfTestEngine`.
- Displays real-time `FAST_SELECTED` vs `SAFE_FULL_SUITE` banners, confidence bars, and expandable developer reason cards.

### 📉 Page 2: Confidence Calibration (`2_📉_Confidence_Calibration.py`)
- Reliability diagram with perfect calibration diagonal line ($y=x$).
- Before vs. after Temperature Scaling ($T=0.9275$) curve overlay.
- ECE, MCE, and Brier Score diagnostics.

### 🔮 Page 3: Uncertainty Drilldown (`3_🔮_Uncertainty_Drilldown.py`)
- Epistemic uncertainty ($\sigma$) across 5 diverse random seeds.
- Risk-Coverage curve demonstrating empirical error reduction on high-confidence subsets (34% error drop from 100% to 90% coverage).
- Scatter plot of commit-level uncertainty vs. failure risk.

### 📊 Page 4: RTS Baseline Comparisons (`4_📊_Baseline_Comparison.py`)
- 8-strategy RTS benchmark comparison:
  1. Full Suite
  2. Random-K
  3. Changed File
  4. Dependency Graph
  5. Historical Failure
  6. Uncalibrated ML
  7. Calibrated No-Abstain
  8. ConfTest Selective (Ours)
- Bar charts comparing Failure Recall % and Compute Time Saved %.

### 🔍 Page 5: SHAP Explainability (`5_🔍_SHAP_Explainability.py`)
- Global mean absolute SHAP feature importance rankings.
- 32-feature category composition pie chart (Churn, AST, Graph, History).
