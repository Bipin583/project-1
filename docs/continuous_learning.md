# ConfTest Online Continuous Learning & Concept Drift Adaptation

## 1. Problem Formulation: Evolving Test Suites & Concept Drift
Software development repositories are inherently non-stationary environments. Codebases experience:
- **Structural Shifts:** Migration to new libraries, architectural refactorings.
- **Developer Churn:** Changes in commit patterns and coding styles.
- **Test Flakiness Evolution:** New flaky tests introduced over time.

These changes induce **concept drift** in the joint probability distribution $P(X, y)$, degrading static model performance over time.

---

## 2. Continual Learning Architecture
ConfTest implements a dual-stage streaming adaptation mechanism:
1. **Page-Hinkley Drift Detector:**
   Monitors streaming test error deviation:
   $$m_t = \alpha m_{t-1} + (1-\alpha) e_t$$
   $$U_t = \sum_{i=1}^t (e_i - m_i - \delta)$$
   $$PH_t = U_t - \min_{1 \le k \le t} U_k$$
   When $PH_t > \lambda$, statistical concept drift is declared.

2. **Experience Replay Buffer:**
   Maintains a sliding circular buffer of recent commit-test outcomes ($W = 1,000$) to refresh tree split thresholds while preventing catastrophic forgetting.
