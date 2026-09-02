# ConfTest Selective Prediction Policy & Fallback Specification

## 1. The Selective Prediction Paradigm in CI/CD
Standard machine learning models operate under a **forced-choice assumption**: the model is required to make a test selection prediction on every commit, even when it is uncertain. In mission-critical CI/CD regression testing, a single missed test failure allows a regression bug to escape to production, causing outages, customer downtime, and expensive hotfixes.

ConfTest introduces **Confidence-Aware Selective Regression Test Selection** based on Chow's Rule (1970) and Geifman & El-Yaniv (2017):
$$\text{Decision}(c) = \begin{cases} \text{FAST\_SELECTED } (S \subset \mathcal{T}) & \text{if } U(c) \le \tau_{\text{abstain}} \land \hat{p}_{\max}(c) \ge \tau_{\text{conf}} \land \neg \text{is\_OOD}(c) \\ \text{SAFE\_FULL\_SUITE } (\mathcal{T}) & \text{otherwise (Abstain \& Run Full Suite)} \end{cases}$$

---

## 2. Decision Triggers & Safety Guarantees

| Trigger Criterion | Condition | Action | Rationale |
| :--- | :--- | :--- | :--- |
| **Epistemic Disagreement** | $U(c) = \max_t \sigma(c, t) > \tau_{\text{abstain}}$ | `SAFE_FULL_SUITE` | 5 ensemble models disagree on failure risk due to sparse training coverage in feature space. |
| **Low Failure Confidence** | $\max_t \hat{p}(c, t) < \tau_{\text{conf}}$ | `SAFE_FULL_SUITE` | No single test in the candidate set exhibits clear regression correlation; full suite execution prevents blind misses. |
| **Architectural OOD Diff** | $\text{files} > 15 \lor \text{churn} > 500$ | `SAFE_FULL_SUITE` | Large refactoring diffs alter systemic coupling beyond localized AST/diff feature representations. |
| **High-Confidence Risk** | Low $\sigma$, High $\hat{p}$, in-distribution | `FAST_SELECTED` | Selects top budget-matched tests ($K = \lceil N \times \text{budget} \rceil$) saving 75%+ CI time. |

---

## 3. Cost-Benefit & CI Utility Formulation

Let $T_{\text{full}}$ be full-suite runtime, $T_{\text{sel}}$ be selective runtime, $\lambda_{\text{sec}}$ be cloud runner cost per second (\$0.01/sec), and $\lambda_{\text{escape}}$ be escaped bug penalty (\$50.00/bug):
$$\text{Net Utility}(c) = \underbrace{(T_{\text{full}}(c) - T_{\text{sel}}(c)) \cdot \lambda_{\text{sec}}}_{\text{Gross Compute Savings}} - \underbrace{\text{Missed Failures}(c) \cdot \lambda_{\text{escape}}}_{\text{Escaped Regression Penalty}}$$

Optimizing $\tau_{\text{abstain}}$ on validation commits guarantees maximal compute savings while maintaining zero escaping defects.
