# Kế hoạch triển khai chuẩn kỹ sư: HGT Drug-Protein-Disease dựa trên AMDGT

## 1. Mục tiêu kỹ thuật

Xây một pipeline sạch, có thể bảo trì, theo thứ tự:

1. HGT-only baseline chạy được end-to-end trên `C-dataset`
2. Đánh giá đúng bằng `train/val/test` và `AUC + AUPR`
3. Tách rõ data, graph, model, train, eval, inference
4. Chỉ sau khi baseline ổn mới cân nhắc nhánh AMDGT-inspired

## 2. Nguyên tắc reuse từ AMDGT

### 2.1. Reuse trực tiếp hoặc gần trực tiếp

- Cấu trúc dữ liệu và tên file trong `data/`
- Ý nghĩa các feature:
  - `Drug_mol2vec`
  - `DiseaseFeature`
  - `Protein_ESM`
- Ý nghĩa các relation:
  - `DrugDiseaseAssociationNumber`
  - `DrugProteinAssociationNumber`
  - `ProteinDiseaseAssociationNumber`
- Ý tưởng similarity graph:
  - `DrugFingerprint`, `DrugGIP`
  - `DiseasePS`, `DiseaseGIP`
- Ý tưởng decoder:
  - lấy embedding `drug` và `disease`
  - kết hợp rồi qua MLP để phân loại
- Bộ metric cơ bản trong `metric.py`

### 2.2. Reuse ý tưởng nhưng phải rewrite

- `data_preprocess.py`
  - tách thành `data_loader.py`, `preprocess.py`, `graph_builder.py`, `split.py`
- `train_DDA.py`
  - tách thành `trainer.py`, `evaluator.py`, `main.py`
- `model/AMNTDDA.py`
  - chỉ mượn ý tưởng high-level
  - không bê nguyên sang repo hiện tại
- `model/gt_net_drug.py`, `model/gt_net_disease.py`, `model/graph_transformer_layer.py`
  - chỉ dùng nếu sau này làm Tier 2 similarity branch
  - nếu dùng thì phải port hoặc rewrite cho stack hiện tại

### 2.3. Không reuse

- Hardcode `torch.device('cuda')`
- DGL stack cũ
- K-fold không có validation set riêng
- Chọn best epoch theo test AUC
- Heterograph chỉ có 3 edge forward, không có reverse edges
- Mọi giả định “repo AMDGT là chuẩn đánh giá cuối cùng”

## 3. Kết luận kiến trúc sau khi đọc repo AMDGT

Repo AMDGT nên được xem như:

- **reference về schema dữ liệu**
- **reference về kiến trúc hybrid nâng cao**
- **reference về decoder + metric**

Repo AMDGT **không nên** là template train/eval để copy.

## 4. Mapping từ AMDGT repo sang repo hiện tại

| AMDGT repo | Vai trò gốc | Repo hiện tại nên map sang |
|---|---|---|
| `data_preprocess.py` | load data + negative sampling + graph DGL | `data_loader.py`, `preprocess.py`, `graph_builder.py`, `split.py` |
| `metric.py` | tính metric | `evaluator.py` |
| `train_DDA.py` | train loop | `trainer.py`, `main.py` |
| `model/AMNTDDA.py` | hybrid model | `model_hgt.py` cho Tier 1, module extension riêng cho Tier 2 |
| `model/gt_net_drug.py` | drug similarity transformer | module extension Tier 2 |
| `model/gt_net_disease.py` | disease similarity transformer | module extension Tier 2 |
| `model/graph_transformer_layer.py` | transformer layer cho similarity graph | module extension Tier 2 |

## 5. Những phát hiện quan trọng từ AMDGT cần giữ trong đầu

1. `AMNTDDA.py` đang là model hybrid, không phải HGT thuần.
2. `train_DDA.py` dùng test split để theo dõi epoch tốt nhất, đây là data leakage.
3. `data_preprocess.py` sample negative trước, ý này giữ lại được.
4. Repo dùng DGL và `dgl.nn.pytorch.conv.HGTConv`, nhưng đồ án hiện tại phải dùng PyG.
5. `AMNTDDA.py` đang ngầm buộc disease feature 64 chiều vào `hgt_in_dim=64`, đây không phải thiết kế tốt để giữ nguyên.
6. Nhánh similarity graph là phần mở rộng, chưa phải ưu tiên của baseline.

## 6. Kế hoạch thực hiện theo pha

### Pha A: Nền tảng dữ liệu

#### Bước 0. Setup môi trường
- Mục tiêu: Python, PyTorch, PyG, sklearn chạy ổn
- Trạng thái: đã làm
- Done khi:
  - import được `torch`
  - import được `torch_geometric`
  - chạy được smoke test cơ bản

#### Bước 1. Download data từ AMDGT repo
- Mục tiêu: có `B/C/F-dataset` trong `data/`
- Trạng thái: đã làm
- Done khi:
  - xác nhận đủ file feature, edge, similarity

#### Bước 2. Data loader và preprocessing
- Mục tiêu: đọc dữ liệu vào cấu trúc canonical
- Trạng thái: đã làm bản đầu
- Done khi:
  - loader dùng được cho `B/C/F-dataset`
  - preprocessing phát hiện mismatch và cảnh báo đúng

### Pha B: Graph và split

#### Bước 3. Xây dựng đồ thị
- Mục tiêu: dựng `HeteroData` cho PyG
- Không dùng DGL
- Việc cần làm:
  - tạo node stores: `drug`, `disease`, `protein`
  - gán `x` cho từng loại node
  - tạo edge types:
    - `drug -> protein`
    - `protein -> drug`
    - `protein -> disease`
    - `disease -> protein`
    - `drug -> disease`
    - `disease -> drug`
  - support hai mode:
    - full graph cho inference
    - train graph chỉ chứa positive `drug-disease` của train
- AMDGT reuse:
  - chỉ reuse ý tưởng graph từ `data_preprocess.py`
  - rewrite 100% bằng PyG
- Done khi:
  - graph build pass shape checks
  - reverse edges đúng

#### Bước 4. Train/Val/Test split
- Mục tiêu: chia chuẩn, không rò rỉ test
- Trạng thái: đã làm bản đầu
- Việc còn lại:
  - nối split với graph builder
  - chuẩn hóa output cho trainer
- AMDGT reuse:
  - giữ ý tưởng sample negative từ unknown pairs
  - bỏ hoàn toàn k-fold no-val
- Done khi:
  - train graph không chứa positive của val/test
  - split report log rõ số positive/negative

### Pha C: Baseline và model chính

#### Bước 5. Baseline MLP
- Mục tiêu: có baseline tối thiểu để so sánh
- Thiết kế:
  - input = concat(`drug_feature`, `disease_feature`)
  - MLP nhị phân
- AMDGT reuse:
  - không reuse code
  - reuse decoder intuition và metric
- Done khi:
  - train/eval được
  - có AUC/AUPR baseline

#### Bước 6. HGT model
- Mục tiêu: model chính của đồ án
- Thiết kế Tier 1:
  - `drug_linear: 300 -> hidden_dim`
  - `disease_linear: 64 -> hidden_dim`
  - `protein_linear: 320 -> hidden_dim`
  - `HGTConv` x 2 hoặc x 3
  - decoder: element-wise product hoặc concat + MLP
- AMDGT reuse:
  - giữ ý tưởng projection + decoder
  - bỏ DGL HGT implementation
  - bỏ nhánh similarity ở giai đoạn đầu
- Done khi:
  - forward pass ổn trên `C-dataset`
  - shape pass
  - train không NaN

### Pha D: Huấn luyện và đánh giá

#### Bước 7. Training loop
- Mục tiêu: train ổn, có early stopping
- Thiết kế:
  - optimizer Adam
  - loss `BCEWithLogitsLoss`
  - early stopping theo `val AUPR`
  - checkpoint best model
- AMDGT reuse:
  - chỉ reuse hyperparameter tham khảo rất nhẹ
  - không reuse loop chọn best theo test
- Done khi:
  - lưu được best checkpoint
  - log train/val rõ ràng

#### Bước 8. Evaluation
- Mục tiêu: đánh giá test đúng 1 lần sau cùng
- Thiết kế:
  - AUC
  - AUPR
  - có thể thêm accuracy/f1/mcc như phụ
- AMDGT reuse:
  - `metric.py` có thể lấy ý tưởng gần trực tiếp
  - nhưng nên ưu tiên `average_precision_score` cho AUPR để rõ ràng hơn
- Done khi:
  - có bảng kết quả baseline vs HGT
  - có file kết quả trong `results/`

### Pha E: Inference và chuẩn bị web

#### Bước 9. Inference top-k
- Mục tiêu: nhập 1 thuốc, trả top-k disease
- Thiết kế:
  - map tên thuốc -> local id
  - score với mọi disease
  - sort giảm dần
  - đánh dấu known/unknown
- AMDGT reuse:
  - không có module riêng để reuse
  - tự thiết kế theo output của model hiện tại
- Done khi:
  - chạy CLI inference được

#### Bước 10. Export cho website
- Mục tiêu: đóng gói model và artifacts để backend gọi
- Thiết kế:
  - checkpoint model
  - mappings:
    - drug id -> tên
    - disease id -> tên
  - config infer
  - script API-facing
- Done khi:
  - backend có thể load model và dự đoán mà không train lại

## 7. Thứ tự ưu tiên thực hiện thực tế

### Sprint 1
- Bước 3: `graph_builder.py`
- Hoàn thiện wiring với bước 4
- Viết test/smoke test graph + split

### Sprint 2
- Bước 5: baseline MLP
- Bước 7: trainer tối thiểu cho baseline
- Bước 8: evaluator

### Sprint 3
- Bước 6: HGT model
- Bước 7: trainer cho HGT
- Bước 8: báo cáo baseline vs HGT

### Sprint 4
- Bước 9: inference top-k
- Bước 10: export artifacts cho web

### Sprint 5 nếu còn thời gian
- Tier 2: similarity branch kiểu AMDGT-lite

## 8. Definition of Done cho bản đồ án đạt chuẩn

- `C-dataset` chạy end-to-end
- Có baseline MLP
- Có HGT model
- Có `train/val/test` sạch
- Chọn best theo `val AUPR`
- Test chỉ đánh giá sau cùng
- Có top-k inference
- Có artifact đủ để tích hợp web

## 9. Kế hoạch “lấy gì từ AMDGT” cực ngắn

### Lấy ngay
- schema dữ liệu
- ý nghĩa feature
- ý nghĩa relation
- metric ideas
- decoder intuition

### Lấy sau
- similarity branch
- graph transformer cho drug/disease similarity
- fusion giữa similarity view và heterograph view

### Không lấy
- DGL runtime
- train protocol cũ
- CUDA hardcode
- best-on-test selection
