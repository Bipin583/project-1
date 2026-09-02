# ConfTest Cross-Repository Generalization & Transferability

## 1. Motivation: Cold-Start and Multi-Repo Generalization
A major limitation of traditional ML-based RTS is the **cold-start problem**: newly onboarded repositories lack sufficient historical execution data ($N_{\text{runs}} \approx 0$) to train custom failure prediction models.

ConfTest overcomes cold-start via **Cross-Project Generalization**:
1. Semantic AST complexity and static call-graph dependency coupling features (`dep_is_direct_import`, `dep_shortest_path_depth`) are programming-language structural invariants that transfer zero-shot across Python codebases.
2. Models trained on diverse multi-project corpora can predict regression risks in unseen repositories immediately upon onboarding.

---

## 2. Leave-One-Project-Out (LOPO) Protocol
Given $K$ repositories $\mathcal{R} = \{R_1, R_2, \dots, R_K\}$:
- For each target $R_i \in \mathcal{R}$:
  - Train: $\bigcup_{j \neq i} R_j$
  - Validate / Calibrate: Held-out validation split of source repositories.
  - Zero-Shot Test: Evaluate exclusively on $R_i$ without any fine-tuning on $R_i$'s history.
