# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview
HGT (Heterogeneous Graph Transformer) for Drug–Protein–Disease link prediction (drug repositioning). Academic "đồ án cơ sở" — prioritize working end-to-end pipelines over SOTA reproduction. See `ai_agent_context_hgt_drug_protein_disease.md` for full context.

## Stack
Python, PyTorch, PyG (torch-geometric), scikit-learn, NumPy, NetworkX
**Note**: DGL doesn't support Python 3.13 — switched to PyG. Use `torch_geometric.nn.HGTConv` instead of `dgl.nn.HGTConv`.

## Critical Non-Obvious Rules
- **AMDGT repo is reference only** — model class is named `AMNTDDA` (not AMDGT); repo hardcodes CUDA, has no val split, and selects best epoch on test set (data leakage). Do NOT copy these patterns.
- **Train/val/test split is mandatory** — AMDGT repo only does train/test k-fold with best epoch chosen on test AUC. This project MUST use a proper val split; evaluate on test only once.
- **Negative samples are assumed negatives** — unknown drug–disease pairs treated as negatives are not confirmed; document this limitation explicitly.
- **Reverse edges required** — AMDGT repo only declares 3 forward edge types. Add reverse edges (`protein→drug`, `disease→protein`, `disease→drug`) for proper bidirectional message passing.
- **Two-tier approach** — Tier 1: HGT-only baseline on heterogeneous graph. Tier 2 (if time permits): add similarity graphs + fusion inspired by AMDGT.
- **Feature dimensions from AMDGT data**: Drug_mol2vec=300d, DiseaseFeature=64d, Protein_ESM=320d — alignment/projection layers needed.
- **Target datasets**: C-dataset (663 drugs, 409 diseases, 993 proteins) or F-dataset (593 drugs, 313 diseases, 2741 proteins). B-dataset exists in repo but not in README.
- **Metrics**: AUC + AUPR minimum. AUPR is more important for imbalanced data.

## Code Structure Convention
```
data_loader.py → preprocess.py → feature_builder.py → graph_builder.py → split.py
baseline.py → model_hgt.py → trainer.py → evaluator.py → main.py
```

## Communication
Explain in Vietnamese, Feynman-style (simple→advanced), avoid jumping to math/code too early.
