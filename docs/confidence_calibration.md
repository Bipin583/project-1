# ConfTest Post-Hoc Confidence Calibration Specification

## 1. The Calibration Imperative in CI/CD Test Selection
Standard decision tree models (LightGBM, XGBoost) and neural networks are notorious for **empirical miscalibration**: their raw output scores do not reflect the true posterior probability of failure.

A model is defined as **well-calibrated** if:
$$P(Y = 1 \mid \hat{p} = p) = p, \quad \forall p \in [0, 1]$$
*Example:* Among all test executions assigned a predicted failure risk of $\hat{p} = 0.80$, exactly $80\%$ must detect a true regression failure.

---

## 2. Post-Hoc Calibration Algorithms

### A. Isotonic Regression (Non-Parametric Monotonic Step Mapping)
Fits a piecewise non-decreasing step function $m: [0, 1] \to [0, 1]$ by minimizing square error on the validation split:
$$\min_m \sum_{i=1}^{N_{\text{val}}} (y_i - m(p_i))^2 \quad \text{subject to } m(p_i) \le m(p_j) \text{ whenever } p_i \le p_j$$

### B. Temperature Scaling / Platt Scaling (Parametric Logit Transformation)
Operates on the uncalibrated log-odds $z_i = \text{logit}(p_i) = \log\left(\frac{p_i}{1 - p_i}\right)$ and optimizes a single scalar temperature $T > 0$ via Negative Log-Likelihood:
$$\hat{p}_{\text{cal}} = \sigma\left(\frac{z}{T}\right) = \frac{1}{1 + e^{-z / T}}$$
- If $T > 1$: The uncalibrated model is overconfident; temperature scaling softens probabilities toward the base rate.
- If $T < 1$: The model is underconfident; temperature scaling sharpens probabilities.

---

## 3. Calibration Evaluation Metrics

### 1. Expected Calibration Error (ECE)
Partitions the range $[0, 1]$ into $B = 10$ equal-width probability bins $B_b = \left(\frac{b-1}{B}, \frac{b}{B}\right]$:
$$\text{ECE} = \sum_{b=1}^B \frac{|B_b|}{N} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$
where $\text{acc}(B_b) = \frac{1}{|B_b|} \sum_{i \in B_b} y_i$ and $\text{conf}(B_b) = \frac{1}{|B_b|} \sum_{i \in B_b} \hat{p}_i$.

### 2. Maximum Calibration Error (MCE)
Measures worst-case bin deviation:
$$\text{MCE} = \max_{b \in \{1, \dots, B\}} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$

### 3. Brier Score (Mean Squared Probability Error)
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$

---

## 4. Strict Temporal Anti-Leakage Protocol
Calibration models are strictly fitted on the **Validation Split** (`val.csv`) and evaluated on the **Unseen Test Split** (`test.csv`). Calibrators are never trained on the base training set or evaluated on calibration data.
