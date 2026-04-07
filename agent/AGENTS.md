# Project Debug Rules (Non-Obvious Only)

- **DGL not available**: DGL doesn't support Python 3.13. Project uses PyG (torch-geometric) instead. AMDGT repo patterns using DGL must be translated to PyG equivalents (`HeteroData`, `HGTConv` from `torch_geometric.nn`).
- **Silent shape errors**: Mismatched feature dimensions (300/64/320) won't always throw errors — DGL may broadcast or truncate silently. Always assert tensor shapes after projection layers.
- **CUDA OOM on heterographs**: Full C-dataset graph (~2000 nodes) fits in GPU, but F-dataset (~3600 nodes) with dense similarity edges can OOM on small GPUs. Test with CPU first, then move to GPU.
- **Negative sampling reproducibility**: Set `random.seed`, `np.random.seed`, and `torch.manual_seed` in `split.py`. Different negative samples across runs make debugging impossible.
- **NaN in attention scores**: HGT attention can produce NaN when node degree is 0 (isolated nodes). Check for isolated nodes after graph construction and either remove them or add self-loops.
- **Metric computation**: AUPR requires `precision_recall_curve` from sklearn, not manual computation. Using `average_precision_score` directly is correct; manual trapezoid integration introduces errors.
- **K-fold confusion**: AMDGT repo's k-fold has no val set — if you see suspiciously high metrics matching AMDGT paper, you likely replicated their test-set leakage bug.
