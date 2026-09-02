# ConfTest Epistemic Uncertainty Estimation & Out-of-Distribution Protocol

## 1. Motivation: Epistemic vs. Aleatoric Uncertainty
In Regression Test Selection (RTS), traditional machine learning models output overconfident point probabilities $\hat{p} \in [0, 1]$ even on unfamiliar code modifications, novel third-party libraries, or massive refactoring diffs.

ConfTest explicitly separates two modes of uncertainty:
1. **Aleatoric Uncertainty (Data Noise / Non-Determinism):** Inherent flakiness in network calls, concurrency, or environment races (mitigated via historical flakiness discounting).
2. **Epistemic Uncertainty (Model Ignorance / Out-of-Distribution):** Lack of knowledge due to sparse training data in regions of feature space $\mathbf{x} \in \mathbb{R}^{32}$. Epistemic uncertainty is high for unseen architectural changes, new source files, or cross-cutting refactoring.

---

## 2. Mathematical Formulation: 5-Seed Deep Ensemble

Let $\mathcal{M} = \{f_1, f_2, f_3, f_4, f_5\}$ be an ensemble of $M=5$ gradient boosted decision trees trained with distinct random seeds $s \in \{42, 101, 2024, 777, 999\}$ and stochastic row/column bagging.

### A. Mean Ensemble Prediction
For candidate test $t$ on commit $c$:
$$\bar{p}(c, t) = \frac{1}{M} \sum_{m=1}^M f_m(\mathbf{x}_{c, t})$$

### B. Epistemic Uncertainty (Model Disagreement)
$$\sigma(c, t) = \sqrt{\frac{1}{M} \sum_{m=1}^M \left( f_m(\mathbf{x}_{c, t}) - \bar{p}(c, t) \right)^2}$$
- When all 5 models agree ($\hat{p}_m \approx 0.95$), epistemic uncertainty is low ($\sigma \approx 0.01$).
- When models disagree due to sparse training coverage ($\hat{p}_1 = 0.90, \hat{p}_2 = 0.10$), epistemic uncertainty spikes ($\sigma \approx 0.35$).

### C. Predictive Entropy
$$\mathcal{H}(\bar{p}) = -\bar{p} \log_2(\bar{p}) - (1 - \bar{p}) \log_2(1 - \bar{p})$$

---

## 3. Commit-Level Aggregation & Abstention Policy

To protect CI pipelines from silent test escapes, ConfTest computes the commit-level risk metric:
$$U(c) = \max_{t \in \mathcal{T}} \sigma(c, t)$$

The **Selective Prediction Policy** operates as follows:
$$\text{Mode}(c) = \begin{cases} \text{SAFE\_FULL\_SUITE} & \text{if } U(c) > \tau_{\text{abstain}} \text{ or } \text{is\_OOD}(c) \\ \text{FAST\_SELECTED} & \text{otherwise (run top budget-matched tests)} \end{cases}$$
