# ConfTest Dataset & Data Collection Protocol

## 1. Dataset Collection Strategy
ConfTest evaluates Regression Test Selection on real and curated open-source Python repositories (e.g. `flask`, `requests`, `scikit-learn`, `fastapi`).

## 2. Temporal Data Splitting (Anti-Leakage Protocol)
To ensure research validity and eliminate future-data leakage:
- Commits are ordered strictly chronologically: $t_1 < t_2 < \dots < t_N$.
- **Training Set (70%):** Earliest commits $[t_1, t_{train}]$.
- **Validation Set (15%):** Middle commits $(t_{train}, t_{val}]$ used exclusively for hyperparameter tuning and post-hoc confidence calibration fitting.
- **Test / Evaluation Set (15%):** Most recent commits $(t_{val}, t_N]$ used exclusively for final out-of-sample evaluation.
- *Strict Rule:* Never use random K-fold CV across commit timelines, as future commit patterns leak into past evaluations.

## 3. Label Definition
For a commit $c$ and test $t$:
- $y_{c, t} = 1$ if test $t$ failed or encountered an error on commit $c$.
- $y_{c, t} = 0$ if test $t$ passed or was unaffected.
