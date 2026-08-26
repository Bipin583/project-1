# ConfTest System Limitations & Honest Scope Disclosures

## 1. Zero Formal Bug-Free Guarantee
- ConfTest reduces regression testing latency by predicting failure probability based on historical patterns, diff churn, and static dependencies.
- It does **not** provide a mathematical proof or guarantee that unselected tests will never fail.
- For safety-critical systems, users can set $\tau_{\text{abstain}} = 0$ to force full-suite execution or configure specific mandatory test paths.

## 2. Dynamic Language Limitations
- Python allows dynamic dispatch (`getattr`, `eval`, `importlib`), dynamic monkey-patching, and runtime mocking. Static AST parsing may miss dynamic invocations; ConfTest mitigates this through history mining and ensemble uncertainty.

## 3. Cold-Start for New Tests and Unseen Modules
- Newly authored tests lack historical failure telemetry. ConfTest assigns a conservative default uncertainty to novel tests until historical runs accumulate.
