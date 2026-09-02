# ConfTest Prediction Task Formulation & Anti-Leakage Protocol

## 1. Formal Problem Definition
Let $\mathcal{C} = \{c_1, c_2, \dots, c_N\}$ be a chronologically ordered sequence of repository commits with timestamps $T(c_1) < T(c_2) < \dots < T(c_N)$.
For each commit $c_j$, let $\mathcal{T}_j = \{t_1, t_2, \dots, t_K\}$ be the candidate regression test suite.

For each pair $(c_j, t_i)$, ConfTest predicts the **posterior probability of test failure / regression detection**:
$$\hat{p}(c_j, t_i) = P(\text{Test } t_i \text{ Fails on Commit } c_j \mid \mathbf{x}_{c_j, t_i})$$
where $\mathbf{x}_{c_j, t_i} \in \mathbb{R}^{32}$ is the continuous tabular feature representation.

---

## 2. Ground-Truth Labeling Strategies

### Option A: Direct Commit Failure (Chosen Primary Standard)
$$y_{c_j, t_i} = \begin{cases} 1 & \text{if test } t_i \text{ failed or errored during CI execution of commit } c_j \\ 0 & \text{if test } t_i \text{ passed or was unaffected} \end{cases}$$
- **Advantages:** Unambiguous ground truth directly observable in CI runner logs.
- **Trade-off:** Passing tests on un-executed modules are labeled 0 (unaffected).

### Option C: Replay Oracle Set (Benchmark Mode)
In retrospective replay experiments where the full test suite $\mathcal{T}$ is executed on every commit $c_j$, $y_{c_j, t_i} = 1$ if and only if $t_i$ detected an actual bug in the oracle run.

---

## 3. Addressing Extreme Class Imbalance

In standard CI/CD workloads, regression failure rates typically range between **1% and 10%** ($90–99\%$ of tests pass). ConfTest addresses class imbalance via:
1. **Inverse Class Frequency Loss Weighting:**
   $$\text{scale\_pos\_weight} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$$
2. **Flakiness Down-Weighting:**
   Suspected non-deterministic tests with flip rates $> 0.10$ receive a discount multiplier $w_i \leftarrow w_i \times 0.50$ to prevent models from overfitting to flaky noise.

---

## 4. Strict Temporal Partitioning (Zero Future-Data Leakage)

To mirror realistic production CI deployment:
- **Train Split (Earliest 70% Commits):** Used for base tree training $[0, T_{\text{train}}]$.
- **Validation Split (Intermediate 15% Commits):** Used strictly for hyperparameter optimization and post-hoc confidence calibration $(T_{\text{train}}, T_{\text{val}}]$.
- **Test Split (Latest 15% Commits):** Evaluated strictly once as unseen future commits $(T_{\text{val}}, T_{\text{test}}]$.

$$\max T(\text{Train}) < \min T(\text{Val}) \le \max T(\text{Val}) < \min T(\text{Test})$$
*Rule:* Random $K$-fold cross validation across commit timelines is strictly prohibited as it causes future-data leakage.
