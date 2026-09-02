# ConfTest Statistical Significance & Empirical Methodology

## 1. Academic Rationale
Software engineering empirical benchmarks frequently suffer from high variance across repositories, commit sizes, and testing frameworks. To establish that ConfTest's performance is statistically significant and not an artifact of random test suite noise, we implement non-parametric statistical hypothesis testing and effect size quantification.

---

## 2. Hypothesis Formulation
For each RTS baseline $B \in \{\text{Random-K, Changed-File, AST-Graph, History, Uncalibrated-ML, Calibrated-No-Abstain}\}$:
- **Null Hypothesis ($H_0$):** There is no significant difference in Failure Recall ($FR$) between ConfTest and baseline $B$:
  $$H_0: \tilde{\mu}_{\text{ConfTest}} = \tilde{\mu}_B$$
- **Alternative Hypothesis ($H_1$):** ConfTest achieves a strictly superior median Failure Recall:
  $$H_1: \tilde{\mu}_{\text{ConfTest}} > \tilde{\mu}_B$$

---

## 3. Statistical Testing Suite

### A. Wilcoxon Signed-Rank Test (Non-Parametric Paired)
Because commit-level execution times and failure counts are non-normally distributed, standard paired Student's $t$-tests violate normality assumptions. The Wilcoxon Signed-Rank test ranks the absolute differences $|x_i - y_i|$ and computes:
$$W = \min(W^+, W^-)$$
Significance threshold is set at $\alpha = 0.05$ ($p < 0.05$).

### B. Cliff's Delta ($\delta$) Effect Size
To measure the practical magnitude of improvement beyond statistical significance, we compute non-parametric Cliff's $\delta$:
$$\delta = \frac{\#(\text{ConfTest} > B) - \#(\text{ConfTest} < B)}{m \times n}$$
- $|\delta| < 0.147$: **Negligible**
- $0.147 \le |\delta| < 0.330$: **Small**
- $0.330 \le |\delta| < 0.474$: **Medium**
- $|\delta| \ge 0.474$: **Large**

### C. 1,000-Iteration Percentile Bootstrap Confidence Intervals
We construct 95% non-parametric bootstrap intervals for both Failure Recall and Time Reduction % across 1,000 resamples:
$$\text{CI}_{95\%} = \left[ \hat{\theta}^*_{(2.5\%)}, \; \hat{\theta}^*_{(97.5\%)} \right]$$
