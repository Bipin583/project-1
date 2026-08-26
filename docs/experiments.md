# ConfTest Empirical Experiment Protocols

## 1. Budget-Matched Baselines
ConfTest is benchmarked against 7 standard RTS strategies:
1. **Full Suite Execution (Oracle Safety)**
2. **Random-$k$ Selection:** Uniform random sampling of tests at equal budget.
3. **Changed-File Selection:** Selects tests whose filename matches modified source files.
4. **AST Call-Graph Selection:** Selects tests with static dependency call paths to modified symbols.
5. **Historical Failure Frequency:** Selects tests with highest historical failure counts.
6. **Uncalibrated ML (LightGBM/XGBoost):** Standard regression test prioritization without calibration.
7. **Calibrated ML (No Abstention):** Calibrated probabilities without abstention fallback.
8. **ConfTest (Proposed):** Calibrated confidence with ensemble uncertainty selective abstention.

## 2. Evaluation Metrics
- **Test Reduction Ratio (TRR):** $1 - \frac{|\mathcal{T}_{\text{selected}}|}{|\mathcal{T}_{\text{total}}|}$
- **Execution Time Reduction (ETR):** $1 - \frac{\text{Duration}(\mathcal{T}_{\text{selected}})}{\text{Duration}(\mathcal{T}_{\text{total}})}$
- **Failure Recall (FR):** $\frac{\text{Failures Detected by Selected Suite}}{\text{Total Failures in Full Suite}}$
- **Missed-Failure Rate (MFR):** $1 - \text{Failure Recall}$
- **Expected Calibration Error (ECE):** $\sum_{b=1}^B \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$
- **Abstention Rate (AR):** Percentage of commits routed to full suite due to high uncertainty.
