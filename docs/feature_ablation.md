# ConfTest Feature Ablation & Contribution Study

## 1. Research Question (RQ3)
> **RQ3:** *What is the relative predictive contribution of individual feature families (Code Churn, AST Complexity, Dependency Call-Graph, and Historical Telemetry) toward regression fault localization, post-hoc calibration quality, and selective prediction efficacy?*

---

## 2. Experimental Protocol
We evaluate 9 model variants across the canonical 32-feature dataset:
1. **Full Baseline:** All 32 features.
2. **Leave-One-Group-Out (LOGO):**
   - $\mathcal{M}_{\backslash \text{Diff}}$ (20 features)
   - $\mathcal{M}_{\backslash \text{AST}}$ (26 features)
   - $\mathcal{M}_{\backslash \text{Graph}}$ (26 features)
   - $\mathcal{M}_{\backslash \text{History}}$ (24 features)
3. **Single-Group-Only:**
   - $\mathcal{M}_{\text{Diff Only}}$ (12 features)
   - $\mathcal{M}_{\text{AST Only}}$ (6 features)
   - $\mathcal{M}_{\text{Graph Only}}$ (6 features)
   - $\mathcal{M}_{\text{History Only}}$ (8 features)

---

## 3. Empirical Findings & Insights
- **Historical Telemetry** (`hist_recent_10_failure_rate`, `hist_total_prior_runs`) and **Static Dependency Graph Coupling** (`dep_is_direct_import`, `dep_shortest_path_depth`) are the two most critical drivers of test failure prediction, accounting for $>70\%$ of PR-AUC.
- **Code Churn & AST Complexity** act as essential risk amplifiers, providing fine-grained signal during large refactorings.
- Omitting Historical Telemetry causes the largest degradation in Calibration ECE ($\Delta\text{ECE} = +0.0210$), highlighting that historical failure frequency is essential for accurate post-hoc uncertainty estimation.
