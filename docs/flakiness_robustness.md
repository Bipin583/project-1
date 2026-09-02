# ConfTest Flakiness Robustness & Noise Stress Study

## 1. Motivation: The Flaky Test Dilemma in ML-RTS
Flaky tests—test cases that intermittently fail without code changes—constitute between $4\%$ and $26\%$ of failures in industrial CI/CD systems (Luo et al., 2014; Gruber et al., 2021).

When machine learning models are trained on raw CI test outcomes without flakiness awareness:
1. Flaky failures inject label noise, causing standard gradient boosting to overfit to intermittent patterns.
2. Uncalibrated risk estimates cause false-positive test selections, reducing compute time savings.
3. Genuine regression failures are overlooked if flaky tests dominate the top ranking.

---

## 2. ConfTest Robustness Mechanisms
ConfTest counteracts test flakiness through dual defenses:
1. **Flakiness Downweighting:**
   Sample weight for commit-test pair $(c_i, t_j)$:
   $$w_{ij} = \max\left(0.1, \; 1.0 - \alpha \cdot \text{hist\_flakiness\_score}_j\right)$$
   downweighting high-flakiness test runs during loss minimization.
2. **Epistemic Abstention Fallback:**
   When flakiness creates high ensemble divergence $\sigma(c, t) > \tau_{\text{abstain}}$, ConfTest abstains from aggressive subsetting and falls back to full execution for regression safety.

---

## 3. Stress Test Protocol
We inject synthetic label noise at controlled rates $\eta \in [0\%, 5\%, 10\%, 20\%, 30\%]$ into training labels and compare:
- **Standard Unweighted ML** (naive baseline).
- **ConfTest Robust Model** (sample downweighting + temperature calibration).
