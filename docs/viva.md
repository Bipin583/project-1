# ConfTest Major Project Viva Defense & Technical Q&A

## Core Concept in 30 Seconds
"ConfTest is a confidence-calibrated regression test selection system for CI/CD optimization. Instead of naively running a reduced test subset with unquantified risk, ConfTest estimates prediction uncertainty using an ensemble of trees and calibrates probabilities using isotonic regression. If the model is confident, it runs only the high-risk tests saving 50–80% CI build time; if uncertain, it safely falls back to the full suite."

## Top Viva Questions & Crisp Answers

### 1. What is the fundamental research gap ConfTest addresses?
Standard RTS models predict test failure without uncertainty awareness. When an unseen or out-of-distribution code change occurs, standard models silently drop failing tests, causing critical regressions to escape. ConfTest introduces selective prediction with confidence calibration and full-suite fallback.

### 2. Why is raw classifier score not equal to confidence?
Modern non-linear classifiers (like deep trees or neural nets) output uncalibrated scores that do not reflect true posterior probabilities. Calibration adjusts scores such that a prediction with confidence $0.90$ fails $90\%$ of the time empirically.

### 3. Why use temporal splitting instead of K-Fold Cross-Validation?
Software repositories evolve over time. Random K-fold sampling creates future-to-past data leakage. ConfTest uses strict chronological splitting: earlier commits train the model, intermediate commits fit calibration, and future commits evaluate performance.
