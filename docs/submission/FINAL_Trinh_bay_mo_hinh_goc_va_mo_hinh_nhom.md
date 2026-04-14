# TÀI LIỆU TRÌNH BÀY VỚI GIẢNG VIÊN (BẢN CHI TIẾT)

## Đề tài

Heterogeneous Graph Transformer áp dụng cho mạng Drug - Protein - Disease và hướng phát triển thành mô hình mới AMDGT-lite++

## Hai repo cần phân biệt ngay từ đầu

- Repo gốc để tham chiếu: `https://github.com/JK-Liu7/AMDGT`
- Repo của nhóm em: `https://github.com/DucTri2207/do-an-drug-disease-hgt`

Cách nói đúng là:

- Repo AMDGT là repo gốc để tham chiếu bài toán, dữ liệu và kiến trúc gần nhất.
- Repo của nhóm là nơi nhóm em tự triển khai, cải tiến và trình bày kết quả.

---

## 1. Câu mở đầu nên nói với thầy

Thưa thầy, để trình bày chính xác thì em xin phân biệt theo 3 tầng.

- Tầng nền tảng kiến trúc là **Heterogeneous Graph Transformer, gọi tắt là HGT**.
- Tầng công trình tham chiếu trực tiếp là **AMDGT**, tức repo gốc và paper mà nhóm bám theo.
- Tầng triển khai hiện tại của nhóm là **AMDGT-lite++**, tức mô hình mới do nhóm phát triển từ repo của mình.

Nói ngắn gọn:

- **HGT là nền tảng**
- **AMDGT là mô hình gốc để tham chiếu**
- **AMDGT-lite++ là mô hình mới nhóm đang làm**

---

## 2. Bài toán nhóm đang giải là gì

Bài toán của nhóm là **dự đoán liên kết thuốc - bệnh** trên một **đồ thị dị thể** gồm 3 loại nút:

- `drug` (663 thuốc, feature 300 chiều từ Mol2Vec)
- `protein` (993 protein, feature 320 chiều từ ESM)
- `disease` (409 bệnh, feature 64 chiều)

Các liên kết đã biết trong dữ liệu là:

- `drug - disease` (2532 cạnh)
- `drug - protein` (3672 cạnh)
- `protein - disease` (10691 cạnh)

Mục tiêu cuối cùng là:

- cho mô hình học từ các liên kết đã biết
- sau đó dự đoán xem một cặp `drug - disease` chưa biết có khả năng liên kết hay không

Đây là bài toán:

- trong AI: **link prediction**
- trong biomedical AI: **drug repositioning / drug-disease association prediction**

---

## 3. Vì sao phải dùng đồ thị dị thể

Nếu chỉ nhìn thuốc và bệnh như hai bảng dữ liệu rời nhau thì mô hình sẽ bỏ qua cầu nối sinh học quan trọng là **protein**.

Trong thực tế:

- thuốc tác động lên protein
- protein có liên quan đến bệnh
- nên có thể suy ra thuốc có tiềm năng điều trị bệnh

Chính vì vậy, bài toán này phù hợp với **heterogeneous graph** hơn là mô hình tabular thông thường.

---

## 4. Mô hình gốc nền tảng: HGT là gì

### 4.1 HGT dùng để làm gì

HGT là viết tắt của **Heterogeneous Graph Transformer**.

Mô hình này được thiết kế cho đồ thị có:

- nhiều loại nút
- nhiều loại cạnh

Điểm mạnh của HGT là:

- không coi mọi nút là giống nhau
- không coi mọi quan hệ là giống nhau
- dùng attention để học xem hàng xóm nào quan trọng hơn

### 4.2 HGT hoạt động từng bước như thế nào

#### Bước 1: Nhận feature đầu vào của từng loại node

Trong dữ liệu của đề tài:

- `drug` có feature 300 chiều (Mol2Vec fingerprint)
- `protein` có feature 320 chiều (ESM protein language model)
- `disease` có feature 64 chiều

Ba loại feature này khác kích thước nên chưa thể đưa thẳng vào cùng một lớp.

#### Bước 2: Chiếu các feature về cùng một không gian ẩn

Mỗi loại node có một lớp chiếu riêng (Linear layer) để đưa về cùng `hidden_dim = 128`.

```python
# src/model_hgt.py dòng 48-53
self.input_projection = nn.ModuleDict({
    node_type: nn.Linear(input_dim, config.hidden_dim)
    for node_type, input_dim in self.input_dims.items()
})
```

Ý nghĩa: tạo ra một "ngôn ngữ chung" cho đồ thị.

#### Bước 3: Message passing trên đồ thị dị thể

Sau khi chiếu xong, mỗi node bắt đầu nhận thông tin từ các node lân cận.

```python
# src/model_hgt.py dòng 101-120
for layer_idx, conv in enumerate(self.convs):
    conv_out = conv(x_dict, data.edge_index_dict)
    for node_type in self.metadata[0]:
        updated = conv_out.get(node_type)
        updated = updated + previous        # residual connection
        updated = self.layer_norms[...](updated)  # LayerNorm
        updated = F.gelu(updated)            # activation
        updated = F.dropout(updated, ...)    # regularization
```

#### Bước 4: Attention quyết định hàng xóm nào quan trọng

HGT không lấy trung bình đơn giản. Nó học xem:

- quan hệ nào quan trọng hơn
- hàng xóm nào đáng chú ý hơn
- message nào nên được ưu tiên

Đây chính là phần "transformer" trong HGT.

#### Bước 5: Cập nhật embedding cho từng node

Sau vài lớp HGT, mỗi node có một embedding mới không chỉ chứa feature gốc mà còn chứa ngữ cảnh từ các quan hệ xung quanh.

#### Bước 6: Dự đoán liên kết drug - disease

Sau khi có embedding của `drug` và `disease`, mô hình kết hợp chúng lại qua decoder rồi tạo ra một score. Score này càng cao thì khả năng tồn tại liên kết `drug - disease` càng lớn.

---

## 5. Mô hình tham chiếu trực tiếp: AMDGT

### 5.1 AMDGT gồm những phần nào

AMDGT là mô hình lai gồm 3 nhánh:

- Nhánh 1: **Drug Similarity** — dùng custom GraphTransformer trên drug similarity graph
- Nhánh 2: **Disease Similarity** — dùng custom GraphTransformer trên disease similarity graph
- Nhánh 3: **Heterogeneous Graph** — dùng DGL HGTConv trên đồ thị drug-protein-disease

Sau đó dùng **TransformerEncoder** để kết hợp (fusion) các nhánh.

### 5.2 AMDGT hoạt động cụ thể theo code

```python
# model/AMNTDDA.py — forward()
def forward(self, drdr_graph, didi_graph, drdipr_graph, ...):
    # Nhánh 1: Drug similarity
    dr_sim = self.gt_drug(drdr_graph)           # → [num_drugs, 200]

    # Nhánh 2: Disease similarity
    di_sim = self.gt_disease(didi_graph)         # → [num_diseases, 200]

    # Nhánh 3: HGT trên heterograph
    g = dgl.to_homogeneous(drdipr_graph, ...)    # convert sang homogeneous
    for layer in self.hgt:
        feature = layer(g, feature, g.ndata['_TYPE'], g.edata['_TYPE'])
    dr_hgt = feature[:drug_number, :]            # → [num_drugs, 200]
    di_hgt = feature[drug_number:drug_number+disease_number, :]

    # Fusion: TransformerEncoder
    dr = torch.stack((dr_sim, dr_hgt), dim=1)    # [N, 2, 200]
    dr = self.drug_trans(dr)                     # TransformerEncoder
    dr = dr.view(drug_number, 400)               # flatten

    # Decoder: element-wise product → MLP 4 tầng → 2-class output
    drdi_embedding = torch.mul(dr[sample[:,0]], di[sample[:,1]])
    output = self.mlp(drdi_embedding)            # → Linear(400→1024→1024→256→2)
```

---

## 6. Mô hình của nhóm: AMDGT-lite++

### 6.1 Kiến trúc tổng quát

```
┌─────────────────────────────────────┐
│  Input Projection (shared)           │
│  drug(300) → 128                     │
│  protein(320) → 128                  │
│  disease(64) → 128                   │
├──────────┬──────────┬───────────────┤
│ Nhánh 1  │ Nhánh 2  │ Nhánh 3       │
│ HGT      │ Drug Sim │ Disease Sim   │
│ (PyG     │ Transf-  │ TransformerC. │
│  HGTConv)│ ormerConv│ (sparse topk) │
│ 3 layers │ 2 layers │ 2 layers      │
│ +residual│ +residual│ +residual     │
│ +LN+GELU│ +LN+GELU│ +LN+GELU     │
├──────────┴──────────┴───────────────┤
│  Gated Fusion (learnable gate)      │
│  gate_input = [h_het, h_sim,        │
│                h_het⊙h_sim,         │
│                |h_het − h_sim|]     │
│  gate = σ(MLP(gate_input))          │
│  fused = gate·h_het + (1−gate)·h_sim│
├─────────────────────────────────────┤
│  Hybrid Decoder:                    │
│  [drug, disease, drug⊙disease,     │
│   |drug−disease|]                   │
│  → Linear(512→256) → GELU → Drop   │
│  → Linear(256→128) → GELU → Drop   │
│  → Linear(128→1)                    │
│  → BCEWithLogitsLoss                │
└─────────────────────────────────────┘
```

---

## 7. SÁU CẢI TIẾN CỤ THỂ SO VỚI AMDGT GỐC

### Cải tiến 1: Khắc phục rò rỉ dữ liệu (Data Leakage)

**Đây là cải tiến quan trọng nhất về mặt phương pháp luận.**

**Vấn đề của repo gốc:**

Trong `train_DDA.py` (dòng 91–121), mỗi epoch đều đánh giá trên **test set** rồi chọn best model theo **test AUC**:

```python
# train_DDA.py — repo gốc
for epoch in range(1000):
    model.train()
    # ... train trên X_train ...

    model.eval()
    dr_representation, test_score = model(..., X_test)   # ← đánh giá trên TEST
    AUC, AUPR, ... = get_metric(Y_test, ...)

    if AUC > best_auc:       # ← chọn best model bằng TEST AUC
        best_auc = AUC
```

Thêm vào đó, hàm `k_fold()` trong `data_preprocess.py` (dòng 106–135) chỉ tạo `X_train` và `X_test`, **KHÔNG có `X_val`**:

```python
# data_preprocess.py — repo gốc
skf = StratifiedKFold(n_splits=k, shuffle=False)
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]   # ← KHÔNG có validation
```

=> Model được chọn dựa trên performance trên test set → test metrics được báo cáo **không phải đánh giá khách quan** → AUC > 0.95 là kết quả bị "phổng" (inflated).

**Cách nhóm em khắc phục:**

Trong `src/crossval.py` (dòng 192–198), mỗi training fold được tách thêm **inner validation**:

```python
# src/crossval.py — repo nhóm
train_rows, val_rows = train_test_split(
    train_val_rows,
    test_size=val_ratio_within_train,   # 10% của training fold làm val
    stratify=train_val_labels,
)
```

Trong `src/trainer.py` (dòng 340), best model được chọn theo **val AUPR**, KHÔNG theo test:

```python
# src/trainer.py — repo nhóm (chế độ mặc định)
monitor_value = _extract_metric_value(val_metrics, config.monitor_metric)  # ← val, KHÔNG PHẢI test
if _is_improvement(monitor_value, best_value, ...):
    best_state_dict = copy.deepcopy(model.state_dict())
```

Test set chỉ được đánh giá **DUY NHẤT 1 LẦN** sau khi training kết thúc (dòng 258–264 crossval.py).

**Minh chứng thực nghiệm:**

Nhóm đã xây dựng cờ `--emulate-paper-leakage` để chứng minh cơ chế leakage:

| Chế độ | Monitor = Test? | Best Monitor AUC | Test AUC |
|--------|-----------------|-------------------|----------|
| **Trung thực** (mặc định) | ❌ Khác nhau | 0.7963 (val) | 0.7422 (test) |
| **Leakage** (`--emulate-paper-leakage`) | ✅ Trùng 100% | 0.7385 (test) | 0.7385 (test) |

---

### Cải tiến 2: Gated Fusion thay TransformerEncoder

**Vấn đề của repo gốc:**

AMDGT gốc dùng `nn.TransformerEncoder` để "trộn" 2 view (similarity + hgt):

```python
# model/AMNTDDA.py dòng 76-83 — repo gốc
dr = torch.stack((dr_sim, dr_hgt), dim=1)   # shape [N, 2, 200]
dr = self.drug_trans(dr)                     # TransformerEncoder xử lý 2 token
dr = dr.view(drug_number, 2 * 200)           # flatten → [N, 400]
```

Cách này có hạn chế:
- Coi mọi node đều trộn 2 view theo cùng một quy tắc cố định
- Không cho phép mỗi node tự quyết định view nào quan trọng hơn

**Cách nhóm em cải tiến:**

Nhóm dùng **Gated Fusion** (cổng học được) trong `src/model_fusion_hgt.py` (dòng 299–316):

```python
# src/model_fusion_hgt.py dòng 306-316
def _fuse_embeddings(self, hetero_embedding, similarity_embedding, *, node_type):
    gate_input = torch.cat((
        hetero_embedding,                             # h_het
        similarity_embedding,                         # h_sim
        hetero_embedding * similarity_embedding,      # h_het ⊙ h_sim
        torch.abs(hetero_embedding - similarity_embedding),  # |h_het − h_sim|
    ), dim=-1)                                        # → [N, 4 × hidden_dim]

    gate = torch.sigmoid(self.fusion_gates[node_type](gate_input))  # → [N, hidden_dim]

    return gate * hetero_embedding + (1.0 - gate) * similarity_embedding
```

**Ý nghĩa:**
- `gate ≈ 1`: tin vào heterograph view
- `gate ≈ 0`: tin vào similarity view
- gate đầu vào gồm 4 tín hiệu: 2 embedding gốc + tích + độ lệch tuyệt đối → giúp gate "nhìn thấy" sự tương đồng và khác biệt giữa 2 view
- Mỗi node, mỗi chiều đều có gate riêng → linh hoạt hơn TransformerEncoder cố định

---

### Cải tiến 3: Sparse Top-K Similarity Graph

**Vấn đề của repo gốc:**

Trong `data_preprocess.py` (dòng 22–29, 138–144), AMDGT cũng lấy top-k neighbors nhưng lưu trữ dưới dạng **ma trận dense NxN** rồi convert qua NetworkX → DGL:

```python
# data_preprocess.py — repo gốc
def k_matrix(matrix, k):
    knn_graph = np.zeros(matrix.shape)        # ← ma trận dense N×N (663×663 cho drug)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k+1]] = matrix[i, idx_sort[i, :k+1]]
    return knn_graph + np.eye(num)

drdr_nx = nx.from_numpy_matrix(drdr_matrix)   # ← convert qua NetworkX
drdr_graph = dgl.from_networkx(drdr_nx)       # ← rồi mới qua DGL
```

Cách này tốn bộ nhớ O(N²) và phải đi qua 2 lần convert thừa.

**Cách nhóm em cải tiến:**

Trong `src/similarity_graph.py`, nhóm tạo **trực tiếp sparse edge_index** dạng PyG:

```python
# src/similarity_graph.py — repo nhóm
# Chỉ giữ top-k neighbors → tạo trực tiếp edge_index [2, num_edges]
# Không bao giờ tạo ma trận dense N×N
# Drug: 663 nodes × top_20 = ~19,986 edges (thay vì 663×663 = 439,569 entries)
# Disease: 409 nodes × top_20 = ~13,552 edges (thay vì 409×409 = 167,281 entries)
```

**Hiệu quả:**
- Bộ nhớ: O(k × N) thay vì O(N²) — giảm ~20 lần cho drug similarity
- Không cần convert qua NetworkX → nhanh hơn
- Edge weights được giữ nguyên trong `edge_attr` → `TransformerConv` có thể dùng attention trên trọng số cạnh

---

### Cải tiến 4: Reverse Edges cho Message Passing 2 chiều

**Vấn đề của repo gốc:**

Trong `data_preprocess.py` (dòng 167–171), đồ thị chỉ có **3 loại cạnh 1 chiều**:

```python
# data_preprocess.py — repo gốc
heterograph_dict = {
    ('drug', 'association', 'disease'): ...,
    ('drug', 'association', 'protein'): ...,
    ('disease', 'association', 'protein'): ...,
}
# → CHỈ CÓ 3 loại cạnh, tất cả 1 chiều
# → disease KHÔNG GỬI message ngược lại cho drug
# → protein KHÔNG GỬI message cho drug qua cạnh trực tiếp
```

**Cách nhóm em cải tiến:**

Trong `src/graph_builder.py`, nhóm **tự động thêm 3 reverse edges**:

```python
# src/graph_builder.py — repo nhóm
# Đồ thị nhóm có 6 loại cạnh:
# drug  → treats       → disease  (1772 cạnh)
# disease → treated_by  → drug     (1772 cạnh) ← REVERSE
# drug  → targets      → protein  (3672 cạnh)
# protein → targeted_by → drug     (3672 cạnh) ← REVERSE
# protein → associates  → disease  (10691 cạnh)
# disease → associated_by → protein (10691 cạnh) ← REVERSE
```

**Ý nghĩa:**
- Với HGT, message passing theo cạnh. Nếu không có reverse edges, thông tin chỉ đi 1 chiều.
- Ví dụ: nếu chỉ có `drug → disease`, thì embedding của drug được cập nhật từ disease, nhưng **embedding của disease KHÔNG được cập nhật từ drug**.
- Thêm reverse edges giúp mọi node đều nhận thông tin từ mọi loại hàng xóm → embedding phong phú hơn.

---

### Cải tiến 5: Hybrid Decoder + Focal Loss

**Repo gốc:**

Decoder chỉ dùng **element-wise product** + MLP + **CrossEntropyLoss 2-class**:

```python
# model/AMNTDDA.py dòng 85-88 — repo gốc
drdi_embedding = torch.mul(dr[sample[:,0]], di[sample[:,1]])  # ← chỉ product
output = self.mlp(drdi_embedding)                            # → 2 logits (softmax)

# train_DDA.py dòng 66
cross_entropy = nn.CrossEntropyLoss()                        # ← 2-class classification
```

**Repo nhóm:**

Decoder dùng **4 tín hiệu** + **BCEWithLogitsLoss** hoặc **FocalLoss**:

```python
# src/model_fusion_hgt.py dòng 399-408 — mode "hybrid"
def _build_pair_embeddings(drug_embeddings, disease_embeddings, *, mode):
    return torch.cat((
        drug_embeddings,                              # drug embedding gốc
        disease_embeddings,                           # disease embedding gốc
        drug_embeddings * disease_embeddings,         # element-wise product (tương tác)
        torch.abs(drug_embeddings - disease_embeddings),  # absolute difference (khác biệt)
    ), dim=-1)  # → [batch, 4 × hidden_dim = 512]

# src/trainer.py dòng 547-548 — FocalLoss cho class imbalance
class FocalBCEWithLogitsLoss(nn.Module):
    def forward(self, logits, targets):
        focal_factor = (1.0 - p_t).pow(self.gamma)   # giảm loss cho easy samples
        loss = bce * focal_factor                     # tập trung vào hard samples
```

**Chi tiết sự khác biệt:**

| Khía cạnh | Repo gốc | Repo nhóm |
|-----------|----------|-----------|
| Decoder input | `drug ⊙ disease` (128-dim) | `[drug, disease, drug⊙disease, |drug−disease|]` (512-dim) |
| Output | 2 logits → softmax → CrossEntropy | 1 logit → sigmoid → BCE/Focal |
| Class imbalance | Không xử lý | FocalLoss (`γ=2.0`) giảm ảnh hưởng easy negatives |

---

### Cải tiến 6: Pipeline hoàn chỉnh (Inference + Dashboard)

**Repo gốc:** Chỉ có code train, KHÔNG có inference pipeline hay demo.

**Repo nhóm:**

| Module | File | Chức năng |
|--------|------|-----------|
| **Inference** | `src/inference.py` (700 dòng) | Load checkpoint → predict top-k diseases cho 1 thuốc |
| **Dashboard** | `dashboard/app.py` (311 dòng) | UI Streamlit tiếng Việt: tìm thuốc → xem liên kết đã biết → dự đoán bệnh mới |
| **Hparam Search** | `src/hparam_search.py` (234 dòng) | Random search tự động trên không gian hyperparameter |
| **Cross-val** | `src/crossval.py` (615 dòng) | 10-fold outer + inner validation |
| **Web Export** | `src/inference.py` | Export JSON lookup tables cho drug/disease/protein |
| **MLP Baseline** | `src/baseline.py` (184 dòng) | So sánh fair: graph model vs non-graph model |

---

## 8. Bảng so sánh tổng hợp AMDGT gốc vs AMDGT-lite++

| Khía cạnh | AMDGT gốc | AMDGT-lite++ (nhóm) |
|-----------|-----------|---------------------|
| **Framework** | DGL | PyG |
| **HGT** | `dgl.nn.HGTConv` (cần convert homogeneous) | `torch_geometric.nn.HGTConv` (native heterogeneous) |
| **Similarity encoder** | Custom GraphTransformer (code tự viết) | `TransformerConv` (PyG, có residual + LayerNorm) |
| **Similarity graph** | Ma trận dense NxN → NetworkX → DGL | Sparse edge_index top-k trực tiếp |
| **Fusion** | `nn.TransformerEncoder` (cố định) | **Gated Fusion** σ(MLP) (học được, per-node per-dim) |
| **Reverse edges** | ❌ Không (3 loại cạnh 1 chiều) | ✅ Có (6 loại cạnh 2 chiều) |
| **Residual connections** | ❌ Không | ✅ Mọi layer (HGT + Sim branches) |
| **LayerNorm** | ❌ Không | ✅ Per node type per layer |
| **Activation** | ReLU (chỉ decoder) | GELU (toàn bộ model, cấu hình được) |
| **Decoder input** | `drug ⊙ disease` | `[drug, disease, drug⊙disease, |drug−disease|]` |
| **Decoder output** | 2 logits (softmax, CrossEntropy) | 1 logit (sigmoid, BCE hoặc FocalLoss) |
| **Validation set** | ❌ KHÔNG CÓ | ✅ Inner val chia từ training fold |
| **Model selection** | Test AUC mỗi epoch (**data leakage**) | Val AUPR + early stopping |
| **Early stopping** | ❌ Chạy đủ 1000 epochs | ✅ Patience configurable |
| **Batch training** | Full-batch (gom tất cả) | Mini-batch DataLoader |
| **Device** | `torch.device('cuda')` cứng | Auto-detect CPU/CUDA |
| **Inference** | ❌ Không có | ✅ Load checkpoint → top-k ranking |
| **Dashboard** | ❌ Không có | ✅ Streamlit tiếng Việt |
| **Baseline** | ❌ Không có | ✅ MLP baseline để so sánh fair |
| **Hparam search** | ❌ Không có | ✅ Random search tự động |
| **Tổng code** | ~700 dòng, 6 file | ~5000+ dòng, 17 file |

---

## 9. Kịch bản hỏi đáp phản biện

### Nếu thầy hỏi: "Mô hình của em có mạnh hơn mô hình gốc không?"

> Thưa thầy, nếu so pure metric thì repo gốc báo cáo AUC > 0.95. Tuy nhiên trong quá trình đọc code repo gốc, nhóm em phát hiện repo gốc mắc lỗi **data leakage**: cụ thể ở file `train_DDA.py`, mỗi epoch đều đánh giá trên test set và chọn best model theo test AUC (dòng 118: `if AUC > best_auc`), mà KHÔNG có validation set tách riêng. Hàm `k_fold()` cũng chỉ tạo `X_train` và `X_test`, không tạo `X_val`.
>
> Nhóm em đã xây dựng cờ `--emulate-paper-leakage` để chứng minh cơ chế này: khi bật cờ, monitor metrics trùng 100% test metrics, xác nhận model đang nhìn trộm test.
>
> Vì vậy, con số 0.95 của paper không phản ánh khả năng generalization thực tế. Nhóm em dùng inner validation để chọn model, nên kết quả test AUC ~0.74 là **con số trung thực**, phản ánh đúng khả năng của model trên dữ liệu chưa thấy.
>
> Cải tiến của nhóm em không chỉ nằm ở metric, mà ở **6 điểm thiết kế**: gated fusion, sparse similarity, reverse edges, hybrid decoder, focal loss, và protocol đánh giá sạch.

### Nếu thầy hỏi: "Em dùng code gốc hay tự viết?"

> Thưa thầy, nhóm em **viết lại hoàn toàn từ đầu**. Repo gốc viết bằng DGL (~700 dòng, 6 file), repo nhóm viết bằng PyG (~5000 dòng, 17 file). Không có dòng code nào được copy. Kiến trúc model khác (Gated Fusion vs TransformerEncoder), framework khác (PyG vs DGL), loss function khác (BCE vs CrossEntropy), decoder khác (hybrid 4-feature vs product 1-feature).

### Nếu thầy hỏi: "Gated Fusion cụ thể hoạt động thế nào?"

> Thưa thầy, sau khi có 2 embedding cho mỗi node (1 từ heterograph, 1 từ similarity graph), nhóm em nối 4 tín hiệu: 2 embedding gốc, tích element-wise, và hiệu tuyệt đối. Đưa qua MLP rồi sigmoid để ra vector gate ∈ (0,1). Gatepered  = gate × heterograph + (1−gate) × similarity. Mỗi node, mỗi chiều đều có gate riêng, nên model tự quyết định tin view nào hơn tùy context.

### Nếu thầy hỏi: "Reverse edges là gì, tại sao cần?"

> Thưa thầy, trong HGT, thông tin truyền theo hướng cạnh. Nếu chỉ có cạnh drug→disease mà không có disease→drug, thì khi chạy message passing, drug nhận thông tin từ disease nhưng disease KHÔNG nhận ngược lại từ drug. Nhóm em tự động thêm 3 reverse relations (treated_by, targeted_by, associated_by) với 3 relation gốc, tổng cộng 6 loại cạnh. Nhờ đó mọi node đều nhận thông tin đa chiều.

---

## 10. Tài liệu và code nên viện dẫn nếu thầy hỏi sâu

### Repo gốc tham chiếu

- AMDGT repo: `https://github.com/JK-Liu7/AMDGT`
- Paper: "AMDGT: Attention Aware Multi-Modal Fusion Using a Dual Graph Transformer for Drug–Disease Associations Prediction"

### Repo nhóm

- Repo nhóm: `https://github.com/DucTri2207/do-an-drug-disease-hgt`

### File chính trong repo nhóm

| File | Vai trò | Dòng code |
|------|---------|-----------|
| `src/model_fusion_hgt.py` | Mô hình AMDGT-lite++ (3 nhánh + gated fusion) | 428 |
| `src/model_hgt.py` | Mô hình HGT lõi (baseline graph) | 258 |
| `src/similarity_graph.py` | Xây sparse top-k similarity graph | 199 |
| `src/graph_builder.py` | Xây đồ thị PyG HeteroData + reverse edges | 283 |
| `src/trainer.py` | Training loop + early stopping + FocalLoss | 660 |
| `src/crossval.py` | 10-fold outer + inner validation | 620 |
| `src/evaluator.py` | 7 metrics (AUC, AUPR, Acc, Prec, Rec, F1, MCC) | 137 |
| `src/inference.py` | Load checkpoint → predict top-k | 700 |
| `src/baseline.py` | MLP baseline (không dùng graph) | 184 |
| `src/hparam_search.py` | Random search hyperparameter | 234 |
| `dashboard/app.py` | Streamlit dashboard tiếng Việt | 311 |

### Lệnh chạy demo nhanh

```bash
# Train AMDGT-lite++ trung thực (30 epochs)
.\venv\Scripts\python src\main.py --model fusion_hgt --epochs 30 --result-json results\demo.json

# Chứng minh data leakage
.\venv\Scripts\python src\main.py --model fusion_hgt --emulate-paper-leakage --epochs 30

# Dashboard
.\venv\Scripts\python -m streamlit run dashboard\app.py
```

---

## 11. Kết luận cuối cùng

Nếu thầy yêu cầu trình bày rất ngắn:

> Mô hình nền tảng là HGT vì dữ liệu là đồ thị dị thể drug-protein-disease. Repo gốc tham chiếu là AMDGT (DGL, TransformerEncoder fusion, 2-class CrossEntropy). Mô hình mới của nhóm là AMDGT-lite++ (PyG, Gated Fusion, sparse top-k similarity, reverse edges, hybrid decoder, FocalLoss). Nhóm viết lại hoàn toàn ~5000 dòng code, khắc phục lỗi data leakage của paper gốc, và xây pipeline hoàn chỉnh từ train đến inference và dashboard demo.
