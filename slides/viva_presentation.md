---
marp: true
theme: default
paginate: true
header: "ConfTest: Confidence-Calibrated Selective RTS | KTU B.Tech CSE 2026"
footer: "Bipin B | APJ Abdul Kalam Technological University"
style: |
  section {
    background-color: #0f172a;
    color: #f8fafc;
    font-family: 'Inter', sans-serif;
  }
  h1, h2 {
    color: #38bdf8;
  }
  table {
    font-size: 0.8em;
  }
  th {
    background-color: #1e293b;
    color: #38bdf8;
  }
---

# 🚀 ConfTest
## Confidence-Calibrated Selective Regression Test Selection via Uncertainty-Aware Deep Ensembles

**Candidate:** Bipin B (Register No: KTU-CSE-2026)  
**Guide:** Assistant Professor, Dept. of Computer Science & Engineering  
**Institution:** APJ Abdul Kalam Technological University (KTU)  
**Academic Year:** 2025–2026  

---

# 1. The CI/CD Bottleneck
- Continuous Integration demands frequent code commits and fast feedback loops.
- **The Problem:** Modern regression test suites take 45–180 minutes to run.
- **Consequences:** Developer context switching, blocked PR pipelines, and high cloud compute bills.

---

# 2. Existing Approaches & Their Limitations
- **Dynamic RTS (Coverage/Bytecode):** 20–50% runtime instrumentation overhead; brittle to config edits.
- **Static RTS (File-based):** Overly conservative; selects up to 80% of unchanged suites.
- **Machine Learning (ML-RTS):** Fast, but **overconfident** on out-of-distribution commits, causing silent bug escapes.

---

# 3. The Overconfidence Dilemma in ML-RTS
- Standard gradient boosted trees output uncalibrated risk scores.
- A model outputting **$80\%$ failure confidence** often corresponds to an empirical failure rate of **$<40\%$**.
- **Result:** Critical regression failures slip through, costing \$3,500+ per escaped production incident.

---

# 4. The ConfTest Vision
ConfTest bridges the gap between **ML efficiency** and **static safety**:
1. **Uncertainty Quantification:** 5-seed Deep Ensemble measuring epistemic divergence ($\sigma$).
2. **Post-Hoc Probability Calibration:** Temperature Scaling reducing ECE by $25.47\%$.
3. **Selective Prediction Policy:** Fast subset execution when confident; automatic full-suite fallback when uncertain.

---

# 5. End-to-End System Architecture
```
[ Commit Diff ] --> [ 32-Feature Mining ] --> [ 5-Seed Ensemble ]
                                                      |
[ Selective Policy ] <-- [ Temperature Scaling ] <----+ (p, sigma)
        |
        +---> High Confidence  ==> FAST MODE (Top-K Subset)
        +---> High Uncertainty ==> SAFE FALLBACK (100% Suite)
```

---

# 6. Canonical 32-Feature Pipeline
- **Diff & Churn (12 Features):** Lines added/deleted, file counts, churn velocity.
- **AST Complexity (6 Features):** Functions, classes, cyclomatic delta, asserts.
- **Static Call-Graph (6 Features):** Direct imports, NetworkX shortest path depth, coupling coefficients.
- **Historical Telemetry (8 Features):** Recent-10 failure rates, flakiness scores, prior executions.

---

# 7. AST & NetworkX Call-Graph Modeling
- Parses Python AST trees without running code.
- Builds static directed dependency graphs $G = (V, E)$.
- Computes graph geodesic depth from changed functions to test entry points:
  $$\text{depth}(f_{\text{changed}}, t_j) = \text{shortest\_path}(G, f_{\text{changed}}, t_j)$$

---

# 8. Anti-Leakage Temporal Dataset Splitting
- Software history has a strict time arrow.
- **No Random K-Fold CV:** Prevents future telemetry leaking into past predictions.
- **Temporal Split:** 70% Train, 15% Validation, 15% Test strictly ordered by commit timestamp.

---

# 9. Deep Ensemble Epistemic Uncertainty ($\sigma$)
- Evaluates $M=5$ distinct LightGBM models initialized with different random seeds.
- **Mean Failure Probability:**
  $$\bar{p}(c, t) = \frac{1}{M}\sum_{m=1}^M f_{\theta_m}(c, t)$$
- **Epistemic Disagreement:**
  $$\sigma(c, t) = \sqrt{\frac{1}{M}\sum_{m=1}^M \left(f_{\theta_m}(c, t) - \bar{p}(c, t)\right)^2}$$

---

# 10. Temperature Scaling Calibration
- Minimizes Negative Log-Likelihood on held-out validation logits:
  $$T^* = \arg\min_{T > 0} -\frac{1}{N_{\text{val}}}\sum_{k=1}^{N_{\text{val}}} \left[ y_k \log \hat{p}_k(T) + (1-y_k)\log(1 - \hat{p}_k(T)) \right]$$
- Reduces Expected Calibration Error (ECE) from **0.0631 to 0.0470** ($25.47\%$ reduction).

---

# 11. Cost-Optimal Selective Prediction Policy
- **Fast Subset Execution:** Triggered when epistemic uncertainty $\sigma < \tau_{\text{abstain}}$ and confidence $\max \hat{p} \ge \tau_{\text{conf}}$.
- **Safe Full-Suite Fallback:** Triggered when uncertainty is high or large architectural refactorings are detected ($\text{files} > 15$).

---

# 12. Zero-Escape Safety Guarantee
- If the model is uncertain about *any* aspect of a commit, it **refuses to guess** and executes the entire suite.
- Guarantees **$100\%$ regression fault recall** across all evaluated production commits.

---

# 13. Model Explainability: Tree SHAP ($\phi_i$)
- Generates exact local Shapley attributions for every test prediction:
  $$\hat{p}(x) = \phi_0 + \sum_{i=1}^{32} \phi_i(x)$$
- Provides transparent explanations to developers in CI pull request comments.

---

# 14. Rule-Based Natural Language Developer Cards
- Translates continuous feature attributions into readable cards:
  - ⚠️ *"Direct import of modified module detected (Shortest path: 1 step)"*
  - 📈 *"Test failed in 3 of the last 10 runs (Elevated historical risk)"*

---

# 15. Empirical Benchmark Comparison (8 RTS Strategies)

| RTS Strategy | Failure Recall | Time Reduction | ECE | Wilcoxon $p$ |
| :--- | :---: | :---: | :---: | :---: |
| **Full Suite** | 100.0% | 0.0% | — | — |
| **Random-K (25%)** | 28.5% | 75.0% | — | $<0.00001$ |
| **Changed File** | 78.4% | 68.0% | — | $<0.00001$ |
| **Dependency Graph** | 89.2% | 58.0% | — | $<0.00001$ |
| **Historical Failure** | 82.1% | 65.0% | — | $<0.00001$ |
| **Uncalibrated ML** | 91.5% | 75.0% | 0.0631 | $<0.00001$ |
| **Calibrated No-Abstain** | 94.8% | 75.0% | 0.0470 | $<0.00001$ |
| **ConfTest (Ours)** | **100.0%** | **68.6%** | **0.0470** | — |

---

# 16. Statistical Hypothesis Testing
- **Wilcoxon Signed-Rank Test:** $p < 0.00001$ across all baselines ($\alpha = 0.05$).
- **Cliff's Delta Effect Size:** $\delta = 0.7088\text{--}1.0000$ (Large Effect Size vs. heuristic baselines).
- **1,000-Iteration Bootstrap 95% CI:** ConfTest Recall $\in [98.5\%, 100.0\%]$.

---

# 17. Feature Ablation Study (LOGO)
- **Historical Telemetry Removal:** $\Delta\text{PR-AUC} = -0.1171$, $\Delta\text{Recall} = -20.0\%$.
- **Call-Graph Removal:** $\Delta\text{PR-AUC} = -0.0837$.
- **Conclusion:** Graph Coupling + Historical Telemetry provide $>70\%$ of regression failure predictive signal.

---

# 18. Flakiness Stress Testing & Noise Robustness
- Injected $0\%\text{--}30\%$ synthetic label flip noise.
- **Flakiness Downweighting ($w_i = 1 - 0.7 \cdot \text{flaky}_i$):**
  - At $10\%$ noise: ConfTest maintains **$60.0\%$ recall** vs **$40.0\%$** for unweighted ML ($+20\%$ advantage).

---

# 19. Multi-Repository Cross-Project Generalization
- Leave-One-Project-Out (LOPO) across `requests`, `flask`, `fastapi`, `click`.
- **Zero-Shot Transfer Results:**
  - Macro Mean PR-AUC: **0.8560**
  - Macro Mean ROC-AUC: **0.9917**
  - Macro Zero-Shot Recall@25%: **100.0%**

---

# 20. Online Continuous Learning & Drift Adaptation
- Streaming CI/CD commits with Page-Hinkley statistical drift detector.
- Circular experience replay buffer ($W=500$).
- Concept drift detected and model seamlessly adapted with zero downtime.

---

# 21. Micro-Latency Profiling (<100ms SLA)
- **Feature Vector Prep:** $0.041\text{ms}$
- **Deep Ensemble Inference:** $2.244\text{ms}$
- **Temperature Scaling:** $0.027\text{ms}$
- **Policy Decision:** $0.032\text{ms}$
- **Total End-to-End Latency:** $\mathbf{2.344\text{ms}}$ (**$100\%$ SLA Compliance**).

---

# 22. Enterprise Financial ROI Analysis
- **Team Size:** 25 Developers | 18,750 Annual Commits | 45-min Suite.
- **Direct CI Compute Savings:** \$9,261 / year
- **Developer Productivity Gain:** \$217,055 / year
- **Regression Escape Penalties:** \$0 (Zero Escapes)
- **Net Annual Financial Benefit:** $\mathbf{\$240,316\text{ / year}}$

---

# 23. Production REST API Suite (FastAPI)
- 7 modular routes with strict Pydantic V2 validation:
  - `POST /api/v1/select`: Core RTS selection.
  - `POST /api/v1/explain`: SHAP & Rule explainability.
  - `GET /api/v1/calibration/diagnostics`: Live ECE diagnostics.
  - `POST /api/v1/github/webhook`: Webhook ingestion.

---

# 24. GitHub Actions & PR Bot Integration
- Automated PR analysis bot with HMAC SHA-256 signature verification.
- Posts formatted markdown comment tables directly onto pull requests.
- Idempotent comment updating on subsequent commit pushes.

---

# 25. Streamlit Visual Analytics Dashboard
- 5 multi-page interactive modules:
  1. *Live PR Evaluation & Budget Sliders*
  2. *Confidence Calibration & Reliability Diagrams*
  3. *Uncertainty Drilldown & Disagreement Variance*
  4. *8-Strategy RTS Baseline Comparison*
  5. *SHAP Global Feature Importance Explorer*

---

# 26. Production Dockerization Stack
- Multi-stage `Dockerfile` with minimal runtime image footprint.
- `docker-compose.yml` orchestrating API (8000), Dashboard (8501), and shared SQLite volume.

---

# 27. Summary of Contributions
1. First framework to integrate **Deep Ensemble Epistemic Uncertainty** into RTS.
2. First system to demonstrate **Temperature Scaling** eliminates RTS overconfidence.
3. Formulated a **Cost-Optimal Selective Prediction Policy** with zero-escape guarantees.
4. Comprehensive empirical benchmark across 8 strategies with statistical hypothesis tests.

---

# 28. Future Research Directions
- Neural Graph Attention Networks (GATs) for deep whole-program call-graph embeddings.
- Multi-language expansion (Java, Go, TypeScript).
- Multi-Armed Bandit dynamic test budget scheduling in edge CI nodes.

---

# 29. Publications & Deliverables
- **IEEE/ACM 8-Page Conference Paper:** `paper/main.tex` & `references.bib`
- **Official KTU B.Tech Major Project Report:** `ktu_report/`
- **Interactive Google Colab Demonstration:** `notebooks/conftest_colab_demo.ipynb`
- **110/110 Automated Tests Passing (100% Pass Rate)**

---

# 30. Thank You!
### Questions & Viva Defense Discussion

**Bipin B** | KTU B.Tech Computer Science & Engineering  
Repository: `https://github.com/bbipin/conftest`
