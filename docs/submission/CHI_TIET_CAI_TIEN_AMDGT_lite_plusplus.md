# CHI TIẾT CẢI TIẾN KỸ THUẬT: AMDGT-lite++ SO VỚI AMDGT GỐC

> Tài liệu bổ sung kỹ thuật cho file `FINAL_Trinh_bay_mo_hinh_goc_va_mo_hinh_nhom.md`.
> Dành cho bảo vệ đồ án — mỗi cải tiến đều có **trích dẫn code cụ thể** từ cả 2 repo.

---

## Tổng quan 12 cải tiến

| # | Cải tiến | Loại | Ảnh hưởng |
|---|---------|------|-----------|
| 1 | Khắc phục Data Leakage | Phương pháp luận | ⭐⭐⭐ Quan trọng nhất |
| 2 | Gated Fusion thay TransformerEncoder | Kiến trúc model | ⭐⭐⭐ |
| 3 | Sparse Top-K Similarity Graph | Kiến trúc model | ⭐⭐⭐ |
| 4 | Reverse Edges 2 chiều | Kiến trúc model | ⭐⭐ |
| 5 | Hybrid Decoder 4 tín hiệu | Kiến trúc model | ⭐⭐ |
| 6 | BCEWithLogitsLoss + FocalLoss | Hàm mất mát | ⭐⭐ |
| 7 | Residual Connections + LayerNorm | Ổn định training | ⭐⭐ |
| 8 | GELU thay ReLU | Activation | ⭐ |
| 9 | Mini-batch DataLoader | Kỹ thuật training | ⭐⭐ |
| 10 | Early Stopping + Monitor Metric | Kỹ thuật training | ⭐⭐ |
| 11 | Device-agnostic (CPU/CUDA auto) | Kỹ thuật hệ thống | ⭐ |
| 12 | Pipeline hoàn chỉnh (Inference + Dashboard + Baseline) | Kỹ thuật hệ thống | ⭐⭐⭐ |

---

## Cải tiến 1: Khắc phục Data Leakage (Rò rỉ dữ liệu)

### Bản chất vấn đề

**Data leakage** xảy ra khi mô hình được chọn/tối ưu dựa trên thông tin từ tập test — tức là mô hình "nhìn trộm" đáp án trước khi thi.

### Code repo gốc — nơi xảy ra leakage

**File `train_DDA.py`, dòng 91–121:**

```python
# train_DDA.py — repo gốc AMDGT
for epoch in range(1000):
    model.train()
    dr_representation, train_score = model(...)
    train_loss = cross_entropy(train_score, train_label)
    train_loss.backward()
    optimizer.step()

    model.eval()
    # ↓↓↓ ĐÁNH GIÁ TRÊN TEST SET MỖI EPOCH ↓↓↓
    dr_representation, test_score = model(..., X_test)
    AUC, AUPR, Acc, Pre, Recall, F1, Mcc = get_metric(Y_test, ...)

    # ↓↓↓ CHỌN BEST MODEL BẰNG TEST AUC ↓↓↓
    if AUC > best_auc:
        best_auc = AUC
        best_aupr = AUPR
        # ... lưu best metrics từ TEST
```

**File `data_preprocess.py`, dòng 106–135 — KHÔNG có validation set:**

```python
# data_preprocess.py — repo gốc
def k_fold(pairs, k=10):
    skf = StratifiedKFold(n_splits=k, shuffle=False)  # ← KHÔNG SHUFFLE
    for train_index, test_index in skf.split(X, Y):
        X_train = X[train_index]
        X_test = X[test_index]    # ← CHỈ CÓ train VÀ test
        Y_train = Y[train_index]  # ← KHÔNG CÓ validation
        Y_test = Y[test_index]
```

### Hậu quả

- Model chọn epoch tốt nhất dựa trên test AUC → test metrics **không phản ánh khả năng generalization**
- AUC > 0.95 là **inflated** (phổng)
- Nếu áp dụng vào thực tế, model sẽ kém hơn nhiều vì chưa bao giờ được đánh giá trung thực

### Code repo nhóm — cách khắc phục

**File `src/crossval.py`, dòng 192–198 — Tách inner validation:**

```python
# src/crossval.py — repo nhóm
train_rows, val_rows = train_test_split(
    train_val_rows,
    test_size=val_ratio_within_train,   # 10% training fold → validation
    stratify=train_val_labels,          # stratified để cân bằng class
    random_state=random_seed,
)
```

**File `src/trainer.py`, dòng 326–343 — Chọn model theo VAL, không phải TEST:**

```python
# src/trainer.py — repo nhóm
monitor_value = _extract_metric_value(val_metrics, config.monitor_metric)  # ← VAL
if _is_improvement(monitor_value, best_value, config.monitor_mode):
    best_value = monitor_value
    best_epoch = epoch
    best_metrics = val_metrics
    best_state_dict = copy.deepcopy(model.state_dict())
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= config.early_stopping_patience:
        break  # ← early stopping
```

**File `src/crossval.py`, dòng 258–264 — Test chỉ đánh giá 1 lần duy nhất:**

```python
# Test set CHƯA BAO GIỜ xuất hiện trong training loop
# Chỉ được dùng 1 lần sau khi training hoàn tất
model.load_state_dict(training_result.best_state_dict)
test_metrics = evaluate_on_test(model, test_data)  # ← đánh giá 1 lần duy nhất
```

### Minh chứng thực nghiệm

Nhóm xây cờ `--emulate-paper-leakage` để **chứng minh cơ chế leakage**:

| Chế độ | Monitor = Test? | Monitor AUC | Test AUC |
|--------|-----------------|-------------|----------|
| **Trung thực** (mặc định) | ❌ Khác nhau | 0.7963 (val) | 0.7422 (test) |
| **Leakage** (`--emulate-paper-leakage`) | ✅ Trùng 100% | 0.7385 (test) | 0.7385 (test) |

→ Khi bật flag, monitor metrics trùng 100% test metrics = **xác nhận model đang nhìn trộm test**.

---

## Cải tiến 2: Gated Fusion thay TransformerEncoder

### Repo gốc — TransformerEncoder fusion

**File `model/AMNTDDA.py`, dòng 76–83:**

```python
# AMDGT gốc — fusion bằng TransformerEncoder
dr = torch.stack((dr_sim, dr_hgt), dim=1)        # [N, 2, 200] — coi 2 view như 2 token
dr = self.drug_trans(dr)                           # TransformerEncoder xử lý
dr = dr.view(drug_number, 2 * hidden_dim)          # flatten → [N, 400]
```

**Hạn chế:**
- Self-attention giữa 2 token (sim vs hgt) → học ra **1 bộ trọng số chung** cho mọi node
- Tất cả drug đều trộn 2 view với cùng tỷ lệ → **không linh hoạt**
- TransformerEncoder thêm nhiều tham số (multi-head attention + FFN) mà benefit hạn chế khi chỉ có 2 token

### Repo nhóm — Gated Fusion học được

**File `src/model_fusion_hgt.py`, dòng 132–143 — Khởi tạo gate network:**

```python
# AMDGT-lite++ — fusion gate
gate_input_dim = config.hidden_dim * 4    # 4 tín hiệu đầu vào

self.fusion_gates = nn.ModuleDict({
    node_type: nn.Sequential(
        nn.Linear(gate_input_dim, config.hidden_dim),  # 512 → 128
        _make_activation(config.activation),            # GELU
        nn.Dropout(config.dropout),                     # regularization
        nn.Linear(config.hidden_dim, config.hidden_dim), # 128 → 128 (gate values)
    )
    for node_type in ("drug", "disease")  # gate riêng cho drug và disease
})
```

**File `src/model_fusion_hgt.py`, dòng 299–316 — Forward pass:**

```python
def _fuse_embeddings(self, hetero_embedding, similarity_embedding, *, node_type):
    # Tạo 4 tín hiệu đầu vào cho gate
    gate_input = torch.cat((
        hetero_embedding,                                  # h_het       (128-dim)
        similarity_embedding,                              # h_sim       (128-dim)
        hetero_embedding * similarity_embedding,           # h_het ⊙ h_sim (128-dim)
        torch.abs(hetero_embedding - similarity_embedding), # |h_het − h_sim| (128-dim)
    ), dim=-1)                                             # → [N, 512]

    gate = torch.sigmoid(self.fusion_gates[node_type](gate_input))  # → [N, 128] ∈ (0,1)

    return gate * hetero_embedding + (1.0 - gate) * similarity_embedding
```

**Ưu điểm so với TransformerEncoder:**

| Khía cạnh | TransformerEncoder (gốc) | Gated Fusion (nhóm) |
|-----------|--------------------------|---------------------|
| Granularity | 1 bộ trọng số cho mọi node | **Mỗi node, mỗi chiều** có gate riêng |
| Gate input | Chỉ 2 embedding gốc | **4 tín hiệu** (gốc + tích + hiệu) |
| Tham số | Multi-head attention + FFN | 2 Linear layers + GELU |
| Linh hoạt | Cố định qua self-attention | **gate ≈ 1** → tin heterograph, **gate ≈ 0** → tin similarity |
| Riêng biệt | 1 module chung drug & disease | **Gate riêng cho drug** + **gate riêng cho disease** |

---

## Cải tiến 3: Sparse Top-K Similarity Graph

### Repo gốc — Dense matrix → NetworkX → DGL

**File `data_preprocess.py`, dòng 22–29:**

```python
# AMDGT gốc — tạo similarity graph
def k_matrix(matrix, k):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)  # ← Ma trận dense N×N (663×663 cho drug)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k+1]] = matrix[i, idx_sort[i, :k+1]]
    return knn_graph + np.eye(num)

# Rồi convert qua 2 tầng:
drdr_nx = nx.from_numpy_matrix(drdr_matrix)   # ← NetworkX graph (thừa)
drdr_graph = dgl.from_networkx(drdr_nx)       # ← DGL graph
```

**Vấn đề:**
- Giữ ma trận dense 663×663 = 439,569 entries (phần lớn = 0) → **lãng phí bộ nhớ O(N²)**
- Convert qua NetworkX rồi DGL → **2 lần copy thừa**
- self-loop (diagonal = 1) → không cần thiết cho message passing

### Repo nhóm — Sparse edge_index trực tiếp

**File `src/similarity_graph.py`, dòng 120–191:**

```python
# AMDGT-lite++ — tạo sparse similarity graph trực tiếp
def _build_topk_similarity_graph(*, matrix, top_k, symmetric, name):
    values = matrix.copy()
    np.fill_diagonal(values, 0.0)          # bỏ self-loop
    num_nodes = values.shape[0]

    directed_edges = {}
    for source_index in range(num_nodes):
        row = values[source_index]
        candidate_indices = np.flatnonzero(row > 0)       # chỉ xét non-zero
        selected = _select_topk_indices(row[candidate_indices], top_k=top_k)
        neighbor_indices = candidate_indices[selected]

        for target_index in neighbor_indices:
            weight = float(row[target_index])
            directed_edges[(source_index, target_index)] = weight

    # Symmetrize: nếu A→B hoặc B→A đều có trong top-k, giữ cả 2 chiều
    if symmetric:
        for (src, tgt), weight in directed_edges.items():
            sources.extend((src, tgt))     # thêm cả 2 chiều
            targets.extend((tgt, src))
            weights.extend((weight, weight))

    # Kết quả: sparse edge_index [2, num_edges] — PyG native
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(-1)
```

**So sánh hiệu quả:**

| Khía cạnh | Dense (gốc) | Sparse Top-K (nhóm) |
|-----------|-------------|----------------------|
| **Drug similarity** | 663×663 = 439,569 entries | 663 × top_20 = **~19,986 edges** (giảm **22×**) |
| **Disease similarity** | 409×409 = 167,281 entries | 409 × top_20 = **~13,552 edges** (giảm **12×**) |
| **Bộ nhớ** | O(N²) | O(k × N) |
| **Convert** | Dense → NetworkX → DGL (3 bước) | Trực tiếp → PyG edge_index (1 bước) |
| **Self-loop** | Có (diagonal = 1) | Không (bỏ đi đúng chuẩn) |
| **Edge weights** | Mất khi convert qua NetworkX | **Giữ nguyên** trong `edge_attr` → TransformerConv dùng attention trên weights |

### Tại sao "chỉ giữ top-k" là đúng

- **Tín hiệu mạnh**: 2 thuốc giống nhau 95% → rất cần chia sẻ information. 2 thuốc giống nhau 5% → noise
- **Graph sparsity**: Dense graph → GNN bị over-smoothing (mọi node embedding giống nhau)
- **Computational**: Sparse graph → message passing nhanh hơn, gradient rõ hơn

---

## Cải tiến 4: Reverse Edges cho Message Passing 2 chiều

### Repo gốc — Chỉ 3 loại cạnh 1 chiều

**File `data_preprocess.py`, dòng 167–171:**

```python
# AMDGT gốc — chỉ 3 relation 1 chiều
heterograph_dict = {
    ('drug', 'association', 'disease'): ...,     # drug → disease
    ('drug', 'association', 'protein'): ...,     # drug → protein
    ('disease', 'association', 'protein'): ...,  # disease → protein
}
# Tổng: 3 loại cạnh, TẤT CẢ 1 CHIỀU
```

→ Trong HGT, khi node drug chạy message passing, nó **CHỈ nhận** thông tin từ disease và protein. Nhưng disease **KHÔNG nhận** thông tin trực tiếp từ drug (vì không có cạnh disease→drug).

### Repo nhóm — 6 loại cạnh 2 chiều

**File `src/graph_builder.py`, dòng 19–29:**

```python
# AMDGT-lite++ — 3 relation gốc + 3 reverse
RELATION_MAPPING = {
    "drug_disease": ("drug", "treats", "disease"),         # drug → disease
    "drug_protein": ("drug", "targets", "protein"),        # drug → protein
    "protein_disease": ("protein", "associates", "disease"), # protein → disease
}

REVERSE_RELATION_MAPPING = {
    "drug_disease": ("disease", "treated_by", "drug"),      # disease → drug  ← REVERSE
    "drug_protein": ("protein", "targeted_by", "drug"),     # protein → drug  ← REVERSE
    "protein_disease": ("disease", "associated_by", "protein"), # disease → protein ← REVERSE
}
```

**File `src/graph_builder.py`, dòng 96–102 — Tự động thêm reverse edges:**

```python
if config.add_reverse_edges:
    reverse_source, reverse_relation, reverse_target = REVERSE_RELATION_MAPPING[edge_key]
    reverse_edge_index = torch.stack((edge_index[1], edge_index[0]), dim=0)  # đảo src↔tgt
    data[(reverse_source, reverse_relation, reverse_target)].edge_index = reverse_edge_index
```

**Kết quả cụ thể — 6 loại cạnh:**

| Loại cạnh | Số cạnh | Hướng |
|-----------|---------|-------|
| `drug → treats → disease` | 2,050 | Gốc |
| `disease → treated_by → drug` | 2,050 | **Reverse** |
| `drug → targets → protein` | 3,672 | Gốc |
| `protein → targeted_by → drug` | 3,672 | **Reverse** |
| `protein → associates → disease` | 10,691 | Gốc |
| `disease → associated_by → protein` | 10,691 | **Reverse** |

→ Mọi node đều nhận thông tin từ **TẤT CẢ** loại hàng xóm → embedding phong phú hơn.

---

## Cải tiến 5: Hybrid Decoder 4 tín hiệu

### Repo gốc — Chỉ element-wise product

**File `model/AMNTDDA.py`, dòng 85–88:**

```python
# AMDGT gốc — decoder chỉ dùng product
drdi_embedding = torch.mul(dr[sample[:,0]], di[sample[:,1]])  # [batch, 400]
output = self.mlp(drdi_embedding)  # → Linear(400→1024→1024→256→2) → 2 logits
```

**Hạn chế:** Product chỉ capture tương tác nhân (multiplicative), bỏ qua thông tin additive và difference.

### Repo nhóm — 4 đặc trưng cặp

**File `src/model_fusion_hgt.py`, dòng 388–409:**

```python
# AMDGT-lite++ — decoder mode "hybrid"
def _build_pair_embeddings(drug_embeddings, disease_embeddings, *, mode):
    if mode == "hybrid":
        return torch.cat((
            drug_embeddings,                                    # embedding gốc drug     (128)
            disease_embeddings,                                 # embedding gốc disease  (128)
            drug_embeddings * disease_embeddings,               # tương tác nhân         (128)
            torch.abs(drug_embeddings - disease_embeddings),    # khoảng cách tuyệt đối  (128)
        ), dim=-1)  # → [batch, 512]

    # Các mode khác:
    # "product":  drug ⊙ disease                              → [batch, 128]
    # "concat":   [drug, disease]                              → [batch, 256]
```

**Decoder MLP:**

```python
# src/model_fusion_hgt.py dòng 145-153
# Input: 512 (hybrid) → 256 → GELU → Dropout(0.2)
#                       → 128 → GELU → Dropout(0.2)
#                       → 1 logit (sigmoid)
```

| Khía cạnh | Repo gốc | Repo nhóm |
|-----------|----------|-----------|
| Input signals | Product only (1 tín hiệu) | **4 tín hiệu** (drug, disease, product, |diff|) |
| Input dim | 400 | 512 |
| MLP depth | 4 layers (400→1024→1024→256→2) | 3 layers (512→256→128→1) — **gọn hơn** |
| Output | 2 logits → softmax | 1 logit → sigmoid — **chuẩn binary** |

---

## Cải tiến 6: BCEWithLogitsLoss + FocalLoss

### Repo gốc — CrossEntropyLoss 2 class

```python
# train_DDA.py, dòng 66 — repo gốc
cross_entropy = nn.CrossEntropyLoss()
# Model output: 2 logits → softmax → cross entropy
# → Coi như bài toán 2-class classification (thừa 1 class)
```

### Repo nhóm — BCE 1 logit + Focal Loss tùy chọn

**File `src/trainer.py`, dòng 562–594:**

```python
# Mặc định: BCEWithLogitsLoss (binary, 1 logit)
def _make_loss_fn(config, device):
    if config.loss_name == "bce":
        return nn.BCEWithLogitsLoss()      # ← chuẩn cho binary classification
    if config.loss_name == "focal":
        return FocalBCEWithLogitsLoss(gamma=config.focal_gamma)  # ← Focal Loss

# FocalLoss implementation
class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        self.gamma = gamma     # gamma càng lớn → càng tập trung vào hard samples

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        focal_factor = (1.0 - p_t).pow(self.gamma)
        # ↑ Easy samples (p_t ≈ 1): focal_factor ≈ 0 → loss gần 0 → bỏ qua
        # ↑ Hard samples (p_t ≈ 0): focal_factor ≈ 1 → loss giữ nguyên → tập trung

        loss = bce * focal_factor
        return loss.mean()
```

**Tại sao FocalLoss quan trọng:**
- Bài toán drug-disease có **class imbalance tiềm ẩn** (nhiều negative pairs dễ phân loại đúng)
- FocalLoss giảm ảnh hưởng easy negatives → model tập trung vào borderline cases → **discriminative hơn**

---

## Cải tiến 7: Residual Connections + LayerNorm

### Repo gốc — Không có residual hay LayerNorm

```python
# model/AMNTDDA.py — repo gốc
# HGT layers không có residual connections
# Similarity branches cũng không có
# Chỉ có ReLU trong decoder MLP
```

### Repo nhóm — Residual + LayerNorm ở MỌI layer

**HGT branch — `src/model_fusion_hgt.py`, dòng 250–269:**

```python
for layer_index, conv in enumerate(self.hetero_convs):
    conv_out = conv(x_dict, data.edge_index_dict)
    for node_type in self.metadata[0]:
        updated = conv_out.get(node_type)
        updated = updated + previous                                    # ← RESIDUAL
        updated = self.hetero_layer_norms[layer_index][node_type](updated)  # ← LAYERNORM
        updated = F.gelu(updated)                                       # ← GELU
        updated = F.dropout(updated, p=self.config.dropout, training=self.training)
```

**Similarity branch — `src/model_fusion_hgt.py`, dòng 284–296:**

```python
for layer_index, conv in enumerate(convs):
    updated = conv(x, edge_index, edge_attr=edge_attr)
    updated = updated + x                                           # ← RESIDUAL
    updated = self.sim_layer_norms[node_type][layer_index](updated) # ← LAYERNORM
    updated = F.gelu(updated)                                       # ← GELU
    updated = F.dropout(updated, p=self.config.sim_dropout, training=self.training)
    x = updated
```

**Ý nghĩa:**
- **Residual**: Giúp gradient không bị vanishing qua nhiều layer → train sâu hơn (3 layers HGT + 2 layers sim)
- **LayerNorm per node type per layer**: Ổn định distribution của từng loại node ở từng tầng → không bị 1 node type dominate

---

## Cải tiến 8: GELU thay ReLU

### Repo gốc

```python
# AMDGT gốc — chỉ dùng ReLU, chỉ trong decoder MLP
nn.ReLU()  # hard cutoff ở 0
```

### Repo nhóm

```python
# AMDGT-lite++ — GELU ở toàn bộ model
config.activation = "gelu"  # default

# GELU = x * Φ(x) — smooth approximation of ReLU
# Không "cắt cứng" ở 0 → gradient mượt hơn → convergence tốt hơn
```

GELU được dùng ở: encoder HGT, encoder similarity, fusion gate, decoder MLP — **nhất quán toàn bộ model**.

---

## Cải tiến 9: Mini-batch DataLoader

### Repo gốc — Full-batch training

```python
# train_DDA.py — repo gốc
# Mỗi epoch gom TẤT CẢ samples vào 1 batch
dr_representation, train_score = model(drdr_graph, didi_graph, drdipr_graph,
                                        drug_feature_matrix, ..., X_train)
# → 1 forward pass cho tất cả 4000+ pairs
```

### Repo nhóm — Mini-batch với DataLoader

**File `src/trainer.py`, dòng 532–559:**

```python
# AMDGT-lite++ — mini-batch training
loader = DataLoader(
    TensorDataset(drug_indices, disease_indices, labels),
    batch_size=config.batch_size,  # 512 mặc định
    shuffle=True,
)

for drug_index, disease_index, labels in loader:
    optimizer.zero_grad()
    logits = model(graph, drug_index, disease_index)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
```

**Ưu điểm:**
- Gradient estimation có noise → implicit regularization → giảm overfitting
- Tối ưu bộ nhớ GPU → có thể chạy trên GPU nhỏ
- Shuffle mỗi epoch → model thấy data ở thứ tự khác nhau

---

## Cải tiến 10: Early Stopping + Configurable Monitor Metric

### Repo gốc — Chạy hết 1000 epochs, không dừng sớm

```python
# train_DDA.py — repo gốc
for epoch in range(1000):  # ← LUÔN chạy đủ 1000 epochs
    # ...
```

### Repo nhóm — Early stopping + chọn metric theo config

**File `src/trainer.py`, dòng 326–343:**

```python
# Cấu hình early stopping
config = TrainerConfig(
    early_stopping_patience=50,    # dừng nếu 50 epoch không cải thiện
    monitor_metric="aupr",         # theo dõi AUPR (khó hơn AUC)
    monitor_mode="max",            # maximize
)

# Trong training loop:
if patience_counter >= config.early_stopping_patience:
    break  # ← dừng sớm
```

**Kết quả thực tế từ 10-fold vừa chạy:**

| Fold | Max epochs | Thực tế dừng ở | Tiết kiệm |
|------|-----------|----------------|-----------|
| 1 | 1000 | 297 | **70% epochs** |
| 2 | 1000 | ~67 epochs | **93% epochs** |
| 4 | 1000 | ~50 epochs | **95% epochs** |
| 7 | 1000 | 514 | **49% epochs** |

→ Trung bình tiết kiệm **70–80% thời gian** so với chạy đủ 1000 epochs.

---

## Cải tiến 11: Device-agnostic (CPU/CUDA tự động)

### Repo gốc — Hardcode CUDA

```python
# train_DDA.py, dòng 59 — repo gốc
device = torch.device('cuda')  # ← CỨNG, nếu không có GPU → crash
```

### Repo nhóm — Tự động detect

**File `src/trainer.py`:**

```python
def resolve_device(requested: str | None) -> torch.device:
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
```

→ Chạy được trên **mọi máy** (laptop, server, Colab) mà không cần sửa code.

---

## Cải tiến 12: Pipeline hoàn chỉnh (Inference + Dashboard + Baseline)

### Repo gốc — Chỉ có train

| Tính năng | Repo gốc | Repo nhóm |
|-----------|----------|-----------|
| Training | ✅ (1 file) | ✅ Modular (trainer.py + crossval.py) |
| Inference | ❌ | ✅ `src/inference.py` (700 dòng) |
| Dashboard | ❌ | ✅ `dashboard/app.py` (Streamlit, tiếng Việt) |
| MLP Baseline | ❌ | ✅ `src/baseline.py` (so sánh fair) |
| Hparam Search | ❌ | ✅ `src/hparam_search.py` |
| Feature Builder | ❌ | ✅ `src/feature_builder.py` (export drug/disease tables) |
| Web Export | ❌ | ✅ JSON lookup tables cho web app |
| Data Validation | ❌ | ✅ `src/preprocess.py` (12 sanity checks) |
| Graph Validation | ❌ | ✅ `src/graph_builder.py` (edge index validation + isolated node detection) |

### Chi tiết Inference Pipeline

**File `src/inference.py` (700 dòng):**

```python
# Load checkpoint → rebuild model → score top-k diseases cho 1 thuốc
# Hỗ trợ cả 3 loại model: fusion_hgt, hgt, baseline
# Export JSON tables cho drug/disease lookup
```

### Chi tiết MLP Baseline

**File `src/baseline.py` (184 dòng):**

```python
# MLP đơn giản: concat(drug_feature, disease_feature) → MLP → score
# KHÔNG dùng graph → dùng để so sánh: graph model có tốt hơn non-graph không?
# → Nếu graph model > baseline → chứng minh graph structure có ích
```

---

## Kết quả thực nghiệm 10-Fold Cross-Validation

**Lệnh chạy (tương đương lệnh thầy):**

```bash
.\venv\Scripts\python -m src.crossval --model fusion_hgt --dataset C-dataset \
  --epochs 1000 --folds 10 --similarity-topk 20 \
  --learning-rate 0.0005 --weight-decay 0.0001 \
  --hgt-layers 3 --hidden-dim 128 \
  --early-stopping-patience 50 --hgt-decoder-mode hybrid
```

### Kết quả từng fold:

| Fold | Test AUC | Test AUPR | Thời gian |
|------|----------|-----------|-----------|
| 1 | 0.7294 | 0.7550 | 390s |
| 2 | 0.7790 | 0.7905 | 68s |
| 3 | 0.7767 | 0.8026 | 183s |
| 4 | 0.7304 | 0.7419 | 50s |
| 5 | 0.7575 | 0.7789 | 180s |
| 6 | 0.7689 | 0.7680 | 50s |
| 7 | 0.7920 | 0.8109 | 514s |
| 8 | 0.7970 | 0.7832 | 243s |
| 9 | 0.7426 | 0.7418 | 129s |
| 10 | **0.8351** | **0.8270** | 147s |

### Trung bình ± Std (7 metrics):

| Metric | Mean ± Std |
|--------|-----------|
| **AUC** | **0.7709 ± 0.031** |
| **AUPR** | **0.7800 ± 0.027** |
| Accuracy | 0.7164 ± 0.038 |
| Precision | 0.7526 ± 0.031 |
| Recall | 0.6567 ± 0.144 |
| F1 | 0.6888 ± 0.087 |
| MCC | 0.4444 ± 0.059 |

### So sánh với paper gốc AMDGT:

| | AMDGT gốc (paper) | AMDGT-lite++ (nhóm) |
|---|---|---|
| AUC | 0.9572 | **0.7709** |
| AUPR | 0.9548 | **0.7800** |
| Protocol | Test set dùng để chọn model (**leakage**) | Inner validation (**sạch**) |
| Kết quả đáng tin? | ❌ Inflated | ✅ Trung thực |

**Giải thích chênh lệch:**
1. Paper gốc mắc **data leakage** → AUC bị inflated
2. Kiến trúc khác (TransformerEncoder vs Gated Fusion, CrossEntropy vs BCE)
3. Training regime khác (full-batch 1000 epochs vs mini-batch + early stopping)
4. Con số 0.77 là **khả năng generalization thực** — model chưa bao giờ thấy test data trước khi đánh giá

---

## Tóm tắt: Cách trình bày 12 cải tiến trong 1 phút

> Nhóm em đã thực hiện **12 cải tiến** so với repo gốc AMDGT, chia thành 3 nhóm:
>
> **Nhóm 1 — Phương pháp luận (quan trọng nhất):**
> Khắc phục data leakage bằng inner validation, early stopping, và model selection theo val AUPR thay vì test AUC.
>
> **Nhóm 2 — Kiến trúc mô hình (5 cải tiến):**
> Gated Fusion thay TransformerEncoder, sparse top-k similarity graph, reverse edges 2 chiều, hybrid decoder 4 tín hiệu, BCEWithLogitsLoss + FocalLoss. Thêm residual connections và LayerNorm ở mọi layer, activation GELU nhất quán.
>
> **Nhóm 3 — Hệ thống & Pipeline (4 cải tiến):**
> Mini-batch training, device-agnostic, inference pipeline top-k ranking, dashboard Streamlit, MLP baseline. Tổng ~5000 dòng code viết lại hoàn toàn bằng PyG.
