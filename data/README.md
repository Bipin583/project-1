# ConfTest Data Directory

This directory stores datasets at various stages of processing:
- `raw/`: Raw mined commit logs, pull request diffs, and CI execution traces.
- `interim/`: Extracted AST trees, dependency call-graphs, and intermediate representations.
- `processed/`: Unified tabular feature matrices with ground-truth test outcome labels.
- `splits/`: Strictly temporal train/validation/test splits preventing future-data leakage.
