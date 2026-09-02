# ConfTest Model Explainability Specification

## 1. The Need for Developer-Centric Interpretability
When a test selection system decides to execute or omit a specific regression test, CI engineers require transparent explanations to build developer trust, prevent accidental bug escapes, and audit model behavior.

ConfTest combines two synergistic explainability layers:
1. **Game-Theoretic SHAP Attributions:** Exact numerical contribution of each of the 32 tabular features.
2. **Rule-Based Developer Cards:** Human-readable explanations mapping syntactic, graph, and historical signals into clear reason cards.

---

## 2. SHAP (SHapley Additive exPlanations)

ConfTest implements Lundberg & Lee's `TreeExplainer` on the gradient-boosted decision tree ensemble:
$$f(\mathbf{x}) = \phi_0 + \sum_{i=1}^{32} \phi_i(\mathbf{x})$$
where $\phi_0 = \mathbb{E}[f(\mathbf{x})]$ is the expected baseline failure probability and $\phi_i$ is the exact Shapley attribution for feature $i$.
- **$\phi_i > 0$:** Feature increased predicted failure risk (e.g. `dep_is_direct_import = 1.0`, `hist_recent_10_failure_rate = 0.40`).
- **$\phi_i < 0$:** Feature decreased predicted failure risk (e.g. `dep_shortest_path_depth = 10.0` [disconnected], `diff_total_churn = 4`).

---

## 3. Rule-Based Developer Reason Cards

The `RuleBasedExplainer` translates continuous features and SHAP drivers into domain-specific developer bullets:
- **Direct Coupling:** `"Test directly imports modified module src/auth.py"`
- **Call-Graph Proximity:** `"Test is within 2 dependency hops of modified functions in the static AST graph"`
- **Historical Velocity:** `"Test failed in 30% of its last 10 executions"`
- **High Code Churn:** `"Commit introduces 240 lines of churn in coupled source modules"`
- **Epistemic Safety Fallback:** `"5 ensemble models disagreed on risk (uncertainty: 0.045 > tau_abstain 0.030)"`

---

## 4. GitHub Actions PR Comment Synthesis
The engine automatically formats PR comments with badges, test reduction stats, and top-risk rationale tables directly in developer pull requests.
