# AI Agent Context — Đồ án cơ sở HGT cho mạng Drug–Protein–Disease

## 1) Mục đích của file này

File này là **ngữ cảnh chuẩn hoá** để một AI agent hiểu đúng bài toán, nguồn tham khảo, mức độ đồ án, phạm vi nên làm, những gì có thể tái sử dụng từ paper/repo, và những bẫy cần tránh.

Mục tiêu của agent **không phải** chỉ tóm tắt paper, mà là dùng toàn bộ ngữ cảnh dưới đây để hỗ trợ triển khai một đồ án cơ sở có thể hoàn thành được, trình bày được, và có giá trị học thuật cơ bản.

---

## 2) Người dùng đang thực sự cần gì?

### Mục tiêu thực tế của người dùng
Người dùng đang làm **đồ án cơ sở** với đề tài xoay quanh:

> **Heterogeneous Graph Transformer (HGT) áp dụng cho mạng Drug–Protein–Disease để dự đoán liên kết Drug–Disease (drug repositioning / drug repurposing).**

### Diễn giải ngắn gọn
- Có 3 loại thực thể chính: **Drug**, **Protein**, **Disease**.
- Có nhiều loại quan hệ: thuốc–protein, protein–bệnh, thuốc–bệnh.
- Người dùng muốn xây dựng một mô hình học sâu trên **đồ thị dị thể** để dự đoán xem **thuốc nào có khả năng liên quan/điều trị bệnh nào**.

### Ràng buộc thực tế
- **Python là ngôn ngữ chính**.
- Mức độ là **đồ án cơ sở**, nên ưu tiên:
  - mô hình chạy được end-to-end,
  - pipeline rõ ràng,
  - có baseline tối thiểu,
  - có đánh giá bằng metric phù hợp,
  - có thể giải thích và thuyết trình được.
- Không nên bắt đầu bằng việc cố tái hiện đầy đủ toàn bộ AMDGT nếu chưa có baseline HGT cơ bản.

### Cách agent nên hiểu nhiệm vụ
Agent nên hiểu đây là một bài toán **link prediction trên heterogeneous graph**, trong đó:
- graph là: `Drug`, `Protein`, `Disease`
- mục tiêu chính là: dự đoán cạnh `Drug -> Disease`
- HGT là kiến trúc trọng tâm
- AMDGT repo là **nguồn tham khảo rất quan trọng**, nhưng **không nhất thiết là đích tái hiện 1:1** ở giai đoạn đầu của đồ án.

---

## 3) Tóm tắt điều hướng chiến lược

### Phiên bản mục tiêu phù hợp nhất cho đồ án cơ sở
Ưu tiên triển khai theo hai tầng:

#### Tầng 1 — Phiên bản lõi bắt buộc
- Xây heterogeneous graph gồm `Drug`, `Protein`, `Disease`
- Dùng **HGT** để học embedding
- Dự đoán liên kết `Drug-Disease`
- Đánh giá bằng **AUC** và **AUPR**
- Có ít nhất 1 baseline đơn giản để so sánh

#### Tầng 2 — Phiên bản mở rộng nếu còn thời gian
- Thêm multi-modal feature như AMDGT
- Thêm similarity graph cho drug–drug và disease–disease
- Thêm module fusion/attention giữa các modality
- Thêm ablation study nhỏ
- Thêm case study/top-k prediction

### Kết luận chiến lược
**HGT cơ bản trước, AMDGT-inspired extension sau.**

---

## 4) Nguồn tham khảo chính đã có

## 4.1 Primary paper — AMDGT (ScienceDirect / Knowledge-Based Systems, 2024)
**Title:** AMDGT: Attention aware multi-modal fusion using a dual graph transformer for drug–disease associations prediction  
**Venue:** Knowledge-Based Systems, Volume 284, 25 January 2024, Article 111329  
**DOI:** 10.1016/j.knosys.2023.111329  
**GitHub in paper:** https://github.com/JK-Liu7/AMDGT

### Ý chính của paper
Paper này đề xuất một framework đa mô thức cho dự đoán drug–disease associations. Điểm cốt lõi là:
- không chỉ dùng similarity data,
- mà còn kết hợp thông tin sinh học/hóa học phức tạp,
- dùng **dual graph transformer** để học từ **hai góc nhìn**:
  1. **similarity networks** (homogeneous graphs),
  2. **biochemical heterogeneous association network**.
- sau đó dùng một **attention-aware modality interaction module** để fusion biểu diễn từ các modality khác nhau.

### 4 đóng góp chính của paper
1. **Construct multimodal networks** để biểu diễn dữ liệu drug/disease/protein từ nhiều nguồn.
2. Dùng **dual graph transformers** để học biểu diễn từ cả homogeneous và heterogeneous network.
3. Dùng **modality interaction module** để fusion features từ nhiều modality.
4. Thực nghiệm cho thấy mô hình **vượt state-of-the-art** trên benchmark datasets, và còn có case study + molecular docking.

### Datasets được paper dùng
Paper mô tả 3 benchmark datasets:
- **B-dataset**: 269 drugs, 598 diseases, 1021 proteins, 18,416 DDAs, density cao hơn.
- **C-dataset**: 663 drugs, 409 diseases, 993 proteins, 2,532 known DDAs.
- **F-dataset**: 593 drugs, 313 diseases, 2741 proteins, 1,933 known DDAs.

Paper cũng nêu rằng negative samples được tạo bằng cách random pair các drug–disease không có known association và cân bằng theo số lượng dương/âm.

### Features paper dùng
Paper AMDGT dùng 3 loại feature chính:
- **Drug_mol2vec**: embedding hóa học của drug (300 chiều)
- **DiseaseFeature**: embedding disease từ MeSH / semantic info (64 chiều)
- **Protein_ESM**: embedding protein từ ESM-2 (320 chiều)

Ngoài ra còn dùng similarity matrices:
- `DrugFingerprint`
- `DrugGIP`
- `DiseasePS`
- `DiseaseGIP`

### Mô hình paper về mặt ý tưởng
Paper không phải là “HGT thuần”. Nó là một mô hình lai gồm:
- graph transformer cho **drug similarity graph**,
- graph transformer cho **disease similarity graph**,
- HGT trên **heterogeneous association graph**,
- transformer encoder để fusion hai view,
- classifier cuối để dự đoán DDA.

### Baselines được so sánh trong paper
Paper so sánh với 6 phương pháp:
- deepDR
- HNet-DNN
- DRHGCN
- HINGRL
- DRWBNCF
- DDAGDL

### Kết quả đáng chú ý
Paper báo cáo AMDGT đạt kết quả tốt nhất trên cả 3 benchmark. Ví dụ:
- **C-dataset**: AUC ~ 0.9681, AUPR ~ 0.9698
- **F-dataset**: AUC ~ 0.9598, AUPR ~ 0.9617

### Ý nghĩa với đồ án của người dùng
AMDGT là một **nguồn tham khảo rất tốt** để hiểu:
- cách tổ chức dữ liệu,
- cách dùng nhiều modality,
- cách kết hợp homogeneous + heterogeneous graph,
- cách thiết kế pipeline drug–protein–disease.

Tuy nhiên, với mức **đồ án cơ sở**, việc tái hiện đầy đủ AMDGT có thể là quá nặng. Do đó AMDGT nên được dùng như:
- **kim chỉ nam thiết kế**, và/hoặc
- **phiên bản mở rộng sau khi HGT cơ bản chạy ổn**.

---

## 4.2 GitHub repository — JK-Liu7/AMDGT
**Repo:** https://github.com/JK-Liu7/AMDGT

### Repo structure (quan trọng)
Repo có các thành phần chính:
- `data/`
- `model/`
- `README.md`
- `data_preprocess.py`
- `metric.py`
- `train_DDA.py`

Trong `model/` có:
- `AMNTDDA.py`
- `graph_transformer_layer.py`
- `gt_net_drug.py`
- `gt_net_disease.py`

### Lưu ý quan trọng
Tên class/model trong code là **AMNTDDA**, không phải AMDGT. Đây có thể là khác biệt tên gọi nội bộ trong repo.

### Requirements theo README
- python 3.9.13
- cudatoolkit 11.3.1
- pytorch 1.10.0
- dgl 0.9.0
- networkx 2.8.4
- numpy 1.23.1
- scikit-learn 0.24.2

### Data files theo README
Repo nêu các file dữ liệu cần thiết:
- `Drug_mol2vec`
- `DrugFingerprint`, `DrugGIP`
- `DiseaseFeature`
- `DiseasePS`, `DiseaseGIP`
- `Protein_ESM`
- `DrugDiseaseAssociationNumber`
- `DrugProteinAssociationNumber`
- `ProteinDiseaseAssociationNumber`

### Điểm đáng chú ý về data folder
README chỉ nói dữ liệu chứa **C-dataset** và **F-dataset**, nhưng repo thực tế có cả:
- `B-dataset`
- `C-dataset`
- `F-dataset`

=> Đây là một **discrepancy nhỏ giữa README và repo tree**.

---

## 5) Repo AMDGT thực sự triển khai gì? (rất quan trọng để agent không hiểu sai)

### 5.1 Data loading và preprocessing
File `data_preprocess.py` cho thấy pipeline repo hoạt động như sau:

#### Input được load
- similarity matrices của drug và disease
- association matrices:
  - drug–disease
  - drug–protein
  - protein–disease
- node features:
  - drugfeature
  - diseasefeature
  - proteinfeature

#### Cách negative sampling
- Từ ma trận `drug-disease`, repo lấy toàn bộ positive pairs.
- Các pair không có liên kết được xem là ứng viên negative.
- Sau đó random lấy một lượng negative theo `negative_rate * num_positive`.
- Mặc định `negative_rate = 1.0` → dataset train/test được cân bằng dương/âm.

#### Cách similarity fusion trong preprocessing
Repo tạo:
- `drs = mean(DrugFingerprint, DrugGIP)` theo logic thay thế zero bằng giá trị còn lại
- `dis = mean(DiseasePS, DiseaseGIP)` với logic tương tự

#### K-fold split
- Dùng `StratifiedKFold`
- Mặc định `k_fold = 10`
- Split trực tiếp trên toàn bộ sample dương + âm
- Không có validation set riêng trong script gốc

### 5.2 Graph construction trong repo
Repo thực chất xây **hai loại graph**:

#### (A) Homogeneous similarity graphs
- drug–drug graph từ `drs`
- disease–disease graph từ `dis`
- tạo k-nearest-neighbour graph bằng `k_matrix(...)`

#### (B) Heterogeneous association graph
`dgl_heterograph(...)` xây graph với 3 node types:
- `drug`
- `disease`
- `protein`

và 3 edge groups:
- `('drug', 'association', 'disease')`
- `('drug', 'association', 'protein')`
- `('disease', 'association', 'protein')`

### 5.3 Model architecture trong repo
Từ `AMNTDDA.py`, mô hình repo gồm các phần chính:

#### 1. Graph Transformer cho similarity views
- `gt_net_drug.GraphTransformer(...)`
- `gt_net_disease.GraphTransformer(...)`

Hai module này học biểu diễn từ:
- drug similarity graph
- disease similarity graph

#### 2. HGTConv cho heterogeneous association graph
Repo dùng `dgl.nn.pytorch.conv.HGTConv` trên graph dị thể sau khi convert sang homogeneous bằng `dgl.to_homogeneous(...)` và dùng type ids của node/edge.

#### 3. Fusion bằng Transformer encoder
Sau khi có:
- biểu diễn từ similarity view
- biểu diễn từ heterogeneous association view

repo stack 2 embedding này lại và đưa qua:
- `nn.TransformerEncoder`

để fusion thành embedding cuối của drug và disease.

#### 4. Prediction head
- lấy embedding của drug và disease theo sample pair
- dùng **element-wise multiplication**
- đưa qua MLP nhiều tầng
- output 2 logits cho binary classification

### 5.4 Training protocol trong repo
`train_DDA.py` cho thấy:
- mặc định chạy trên `cuda`
- mặc định `epochs = 1000`
- loss = `CrossEntropyLoss`
- optimizer = `Adam`
- metrics gồm:
  - AUC
  - AUPR
  - Accuracy
  - Precision
  - Recall
  - F1
  - MCC

### 5.5 Những điểm mạnh của repo
- code nhỏ, dễ theo dõi
- data schema rõ ràng
- cho thấy cách tổ chức benchmark datasets
- cho thấy cách kết hợp similarity graphs + heterogeneous graph
- khá phù hợp để học pipeline tổng thể

### 5.6 Những điểm yếu / rủi ro / caveat trong repo
Đây là các điểm AI agent **phải biết rõ**:

1. **Hard-code CUDA**
   - `device = torch.device('cuda')`
   - nếu chạy máy không có GPU sẽ lỗi nếu không sửa.

2. **Không có validation split riêng**
   - repo train/test theo k-fold, nhưng không có val set riêng để chọn model.

3. **Chọn best epoch theo test AUC**
   - trong `train_DDA.py`, model theo dõi metric trên test split mỗi epoch và chọn best AUC trên test.
   - đây là **data leakage / optimistic evaluation** nếu dùng lại y nguyên cho báo cáo chuẩn.

4. **Negative sampling là giả định**
   - pair chưa biết không chắc là negative thật.
   - chỉ là negative tạm thời.

5. **Heterograph gốc không có reverse edges rõ ràng**
   - repo chỉ khai báo 3 hướng association, không thêm reverse relations riêng.
   - nếu muốn HGT mạnh hơn cho đồ án riêng, có thể cân nhắc thêm edge ngược.

6. **Đây không phải baseline HGT thuần**
   - đây là mô hình hybrid khá phức tạp.
   - nếu người dùng muốn đồ án cơ sở rõ ràng, nên có một phiên bản **HGT-only baseline** trước.

### Kết luận về repo
Repo AMDGT nên được dùng như:
- **reference implementation về pipeline dữ liệu**, và
- **reference architecture cho phiên bản mở rộng**,

chứ không nên là điểm bắt đầu duy nhất nếu mục tiêu là một đồ án cơ sở sạch, dễ giải thích.

---

## 6) Review paper tổng quát (Biomolecules 2022) cho biết điều gì?

**Title:** Drug-Disease Association Prediction Using Heterogeneous Networks for Computational Drug Repositioning  
**Type:** review/survey + comparative experiment

### 6.1 Khung khái niệm quan trọng
Review này rất hữu ích để giải thích nền tảng đồ án:
- drug repositioning giúp giảm thời gian và chi phí phát triển thuốc
- heterogeneous networks là cách biểu diễn tự nhiên cho dữ liệu y sinh nhiều loại thực thể
- network-based DDA prediction thường tích hợp:
  - drug–drug similarities
  - disease–disease similarities
  - gene/protein similarities
  - drug–disease associations
  - drug–gene associations
  - disease–gene associations

### 6.2 3 nhóm phương pháp lớn trong literature
Review chia các phương pháp thành 3 nhóm:
1. **Graph mining**
2. **Matrix factorization / matrix completion**
3. **Deep learning**

### 6.3 Ý nghĩa cho đồ án hiện tại
Điều này rất quan trọng:
- HGT/AMDGT chỉ là **một hướng trong nhánh deep learning / graph deep learning**.
- Nó không phải toàn bộ bức tranh.
- Khi viết báo cáo, nên đặt HGT trong bối cảnh broader literature này.

### 6.4 Bài học thực nghiệm từ review
Review chỉ ra rằng trên benchmark của họ:
- graph mining và matrix factorization/completion thường rất mạnh,
- deep learning không phải lúc nào cũng thắng,
- dữ liệu DDA rất **sparse** và **imbalanced**,
- AUC và AUPR là metric quan trọng,
- họ còn dùng **AUPR*** để phản ánh mất cân bằng tốt hơn.

### 6.5 Ý nghĩa thực tế cho agent
Agent nên tránh khẳng định kiểu “deep learning chắc chắn tốt hơn”.  
Thay vào đó, nên giữ quan điểm:
- HGT là mô hình hiện đại, hợp lý cho heterogeneous graph,
- nhưng để đánh giá công bằng cần có baseline,
- và phải chú ý split + negative sampling + metric.

---

## 7) Các paper tham khảo khác trong file zip và vai trò của chúng

Dưới đây là cách AI agent nên hiểu vai trò của từng paper tham khảo trong hệ sinh thái bài toán.

### 7.1 PREDICT (2011)
**Vai trò:** baseline kinh điển / nguyên lý similarity-based cổ điển  
**Ý tưởng cốt lõi:** similar drugs tend to treat similar diseases.

**Ý nghĩa với đồ án:**
- rất tốt để viết phần lịch sử bài toán,
- cho thấy gốc tư duy similarity-based trước kỷ nguyên GNN/Transformer.

### 7.2 MBiRW (2016)
**Vai trò:** graph-based classical method  
**Ý tưởng:** comprehensive similarities + bi-random walk.

**Ý nghĩa với đồ án:**
- là baseline thuộc nhóm graph mining,
- có thể dùng để giải thích vì sao heterogeneous network + random walk là hướng đi mạnh trước deep learning.

### 7.3 LAGCN (2021)
**Vai trò:** GCN-based deep learning baseline  
**Ý tưởng:** heterogeneous network + GCN + layer attention.

**Ý nghĩa với đồ án:**
- là cầu nối giữa graph neural networks cổ điển và transformer-based models,
- rất phù hợp để dùng làm baseline so sánh nếu muốn có 1 baseline GNN trước HGT.

### 7.4 AMDGT (2024)
**Vai trò:** reference paper gần nhất và sát nhất với pipeline đa mô thức  
**Ý tưởng:** dual graph transformers + modality interaction.

**Ý nghĩa với đồ án:**
- tham khảo mạnh cho thiết kế dữ liệu, feature, benchmark, code structure,
- nhưng bản thân nó đã là một mô hình nâng cao hơn mức “HGT cơ sở”.

### 7.5 HGTDR
**Vai trò:** reference paper rất gần với đề tài HGT thuần  
**Ý tưởng:** heterogeneous graph transformer cho drug repurposing, nhấn mạnh khả năng end-to-end trên dữ liệu dị thể.

**Ý nghĩa với đồ án:**
- rất phù hợp để làm nền lý thuyết cho phần “vì sao chọn HGT”,
- đặc biệt hữu ích nếu người dùng muốn đề tài mang màu sắc “HGT chính thống” hơn là AMDGT hybrid.

### 7.6 MRDDA
**Vai trò:** multi-relational GNN reference  
**Ý tưởng:** khai thác nhiều loại quan hệ, meta-path, layer-wise attention.

**Ý nghĩa với đồ án:**
- giúp mở rộng phần related work hiện đại,
- cho thấy xu hướng mô hình hóa nhiều quan hệ/quan điểm khác nhau trên biomedical graph.

### 7.7 DD-HGNN+
**Vai trò:** advanced hypergraph/multimodal reference  
**Ý tưởng:** hypergraph neural network + hierarchical contrastive learning + cross-attention.

**Ý nghĩa với đồ án:**
- cho thấy xu hướng mới hơn: high-order relation, hypergraph, contrastive learning,
- nhưng quá nặng để là mục tiêu chính của đồ án cơ sở.

---

## 8) Mô hình tinh thần đúng cho AI agent

### Điều agent phải hiểu đúng
Người dùng **không chỉ muốn summary literature**. Người dùng cần một agent hiểu rằng:

1. đây là **đồ án cơ sở**, không phải bài reproduce SOTA full-scale;
2. chủ đề gốc là **HGT trên Drug–Protein–Disease graph**;
3. AMDGT paper + repo là nguồn tham khảo chính;
4. nhưng cần **dịch** từ AMDGT / literature sang một kế hoạch triển khai khả thi.

### Do đó agent phải ưu tiên
- tính thực thi,
- tính mô-đun,
- giải thích rõ ràng,
- đánh giá đúng,
- bám sát yêu cầu của người dùng hơn là cố show quá nhiều SOTA.

---

## 9) Bản đồ phạm vi đề tài nên chốt

## 9.1 Phạm vi chính thức khuyến nghị

### Tên đề tài triển khai nên hiểu là:
> Xây dựng mô hình Heterogeneous Graph Transformer cho mạng Drug–Protein–Disease nhằm dự đoán liên kết Drug–Disease.

### Bài toán học máy
- **Task:** link prediction / binary classification trên cặp `(drug, disease)`
- **Input:** heterogeneous graph + node features
- **Output:** xác suất tồn tại liên kết drug–disease

### Các loại node
- `drug`
- `protein`
- `disease`

### Các loại edge tối thiểu
- `drug -> protein`
- `protein -> disease`
- `drug -> disease` (known training associations)

### Các edge mở rộng nếu cần
- edge ngược tương ứng
- `drug -> drug` similarity
- `disease -> disease` similarity

### Các feature tối thiểu nên có
- drug feature
- protein feature
- disease feature

### Mô hình tối thiểu nên có
- 1 baseline đơn giản (MLP / GCN / GAT nhẹ)
- 1 mô hình chính: **HGT**

### Metric tối thiểu
- **AUC**
- **AUPR**

### Mở rộng tốt nếu còn thời gian
- ablation nhỏ
- case study top-k prediction
- visualization subgraph
- so sánh có/không protein edges

---

## 9.2 Những gì KHÔNG nên coi là bắt buộc
- tái hiện đầy đủ 100% AMDGT paper
- tái hiện molecular docking
- triển khai quá nhiều SOTA baselines
- fine-tune protein language model lớn
- benchmark quá nhiều dataset trong giai đoạn đầu
- dùng quá nhiều kỹ thuật nặng như contrastive learning, hypergraph, multi-stage pretraining

---

## 10) Khuyến nghị kỹ thuật cho agent khi đề xuất code/kiến trúc

### 10.1 Nguyên tắc triển khai
1. **Data pipeline trước, model sau.**
2. **Baseline trước, HGT sau.**
3. **Train/val/test sạch trước, tuning sau.**
4. **Giữ code modular.**
5. **Không đánh giá theo cách rò rỉ test set.**

### 10.2 Cấu trúc code nên ưu tiên
- `data_loader.py`
- `preprocess.py`
- `feature_builder.py`
- `graph_builder.py`
- `split.py`
- `baseline.py`
- `model_hgt.py`
- `trainer.py`
- `evaluator.py`
- `main.py`

### 10.3 Split protocol tốt hơn repo gốc
Không nên y nguyên repo AMDGT trong chuyện chọn best epoch theo test.

Khuyến nghị:
- train split
- validation split
- test split
- best model chọn theo **validation AUC/AUPR**
- test chỉ dùng 1 lần cuối

### 10.4 Negative sampling
Nên ghi rõ trong báo cáo:
- negative pairs là **unknown pairs treated as negatives**,
- không phải negative được xác nhận sinh học.

### 10.5 Reverse edges
Nếu AI agent hỗ trợ viết HGT thuần, nên cân nhắc thêm:
- `protein -> drug`
- `disease -> protein`
- `disease -> drug`

Lý do: message passing hai chiều thường hợp lý hơn cho heterogeneous graph.

### 10.6 Nếu muốn “AMDGT-lite”
Agent có thể đề xuất lộ trình 2 bước:

#### Version A — HGT core
- chỉ heterogeneous graph
- HGT encoder
- link predictor

#### Version B — AMDGT-inspired extension
- thêm similarity graphs cho drug/disease
- thêm fusion giữa similarity embedding và heterograph embedding

---

## 11) Cách AI agent nên giải thích cho người dùng

Người dùng đã yêu cầu giải thích theo kiểu đơn giản, có xu hướng “Feynman” và cấu trúc từ cơ bản đến nâng cao. Vì vậy agent nên:
- ưu tiên tiếng Việt rõ ràng,
- giải thích thuật ngữ bằng ví dụ dễ hiểu,
- tránh nhảy thẳng vào toán quá sớm,
- sau đó mới chuyển sang mức code/triển khai.

### Trật tự giải thích khuyến nghị
1. graph là gì
2. heterogeneous graph là gì
3. GNN là gì
4. attention là gì
5. transformer là gì
6. HGT là gì
7. link prediction là gì
8. áp dụng vào Drug–Protein–Disease ra sao
9. pipeline code nên đi như thế nào

---

## 12) Quy tắc làm việc khuyến nghị cho AI agent

### Khi hỗ trợ viết code
- ưu tiên Python thuần + PyTorch + DGL/PyG
- viết code tối giản trước
- luôn nêu rõ input/output shape
- tách riêng preprocess / graph build / model / trainer / evaluator
- không nhồi hết mọi thứ vào một file

### Khi hỗ trợ báo cáo
- đặt AMDGT trong bối cảnh related work
- nhấn mạnh đồ án chọn HGT vì phù hợp với dữ liệu dị thể nhiều loại node/cạnh
- nói rõ phạm vi đồ án là **core HGT for DDA prediction**
- nếu dùng AMDGT repo, nói rõ đâu là phần tham khảo, đâu là phần tự thiết kế

### Khi hỗ trợ đánh giá
- ưu tiên AUC + AUPR
- nếu dữ liệu mất cân bằng nặng, nhấn mạnh AUPR
- nếu muốn nâng cao, có thể nhắc tới AUPR* như review paper

### Khi hỗ trợ thuyết trình
Nên chuẩn bị được tối thiểu:
- sơ đồ graph
- sơ đồ pipeline
- bảng kết quả baseline vs HGT
- một case study dự đoán tiêu biểu

---

## 13) Rủi ro chính cần theo dõi

1. **Ôm quá nhiều ngay từ đầu**
   - HGT + similarity graph + multimodal fusion + too many baselines = dễ vỡ tiến độ.

2. **Hiểu nhầm AMDGT là HGT thuần**
   - AMDGT là hybrid model, không phải baseline HGT thuần túy.

3. **Data leakage khi đánh giá**
   - repo gốc chọn best epoch theo test AUC.
   - không nên giữ nguyên nếu làm báo cáo nghiêm túc.

4. **Feature alignment lỗi**
   - index mapping giữa feature matrix và node ids có thể sai nếu không chuẩn hoá kỹ.

5. **Hard dependency vào GPU / DGL version cũ**
   - repo gốc dùng stack cũ, cần cân nhắc tương thích môi trường hiện tại.

6. **Negative samples không phải negative thật**
   - cần diễn đạt cẩn thận trong báo cáo và phần limitations.

---

## 14) Quyết định thiết kế khuyến nghị cho phiên bản đồ án cơ sở

### Khuyến nghị mạnh
- **Start with C-dataset hoặc F-dataset**
- **Build HGT-only baseline first**
- **Use train/val/test properly**
- **Use AUC + AUPR**
- **Keep one baseline simple**
- **Add one ablation if possible**

### Khuyến nghị vừa phải
- lấy cảm hứng từ AMDGT cho feature set
- chưa cần reproduce toàn bộ dual graph transformer ở giai đoạn 1

### Nếu còn thời gian
- thêm similarity graph branch như AMDGT
- thêm transformer-based fusion

---

## 15) Câu mô tả chuẩn mà AI agent có thể dùng về đề tài

> Đây là một bài toán dự đoán liên kết drug–disease trên đồ thị dị thể Drug–Protein–Disease. Mục tiêu là xây dựng một mô hình Heterogeneous Graph Transformer để học biểu diễn từ nhiều loại nút và nhiều loại quan hệ sinh học, từ đó suy ra các liên kết drug–disease tiềm năng phục vụ drug repositioning.

---

## 16) Short glossary cho agent

- **Drug repositioning / repurposing**: tìm chỉ định/công dụng mới cho thuốc đã có.
- **Drug–Disease Association (DDA)**: liên kết giữa thuốc và bệnh.
- **Heterogeneous graph**: đồ thị có nhiều loại nút/cạnh.
- **Link prediction**: dự đoán một cạnh có nên tồn tại hay không.
- **HGT**: Transformer dành cho heterogeneous graph.
- **Similarity graph**: graph đồng loại như drug–drug hoặc disease–disease.
- **Association graph**: graph dị thể như drug–protein–disease.
- **Modality**: một nguồn thông tin/kiểu feature riêng.
- **AUC**: area under ROC.
- **AUPR**: area under precision-recall curve.

---

## 17) Hướng dẫn cuối cùng cho AI agent

Nếu bạn là AI agent đọc file này, hãy làm theo ưu tiên sau:

1. **Luôn nhớ mục tiêu chính là giúp người dùng hoàn thành một đồ án cơ sở HGT cho Drug–Protein–Disease.**
2. Dùng paper AMDGT và GitHub AMDGT làm nguồn tham khảo thiết kế, **nhưng không ép người dùng phải reproduce toàn bộ paper**.
3. Khi đề xuất pipeline/code, hãy bắt đầu từ phiên bản HGT tối giản và sạch.
4. Khi đề xuất đánh giá, tránh rò rỉ test set.
5. Khi viết hoặc giải thích, ưu tiên tiếng Việt rõ ràng, có cấu trúc từ cơ bản đến nâng cao.
6. Khi có nhiều hướng khả thi, ưu tiên hướng:
   - dễ chạy,
   - dễ giải thích,
   - đủ đẹp để báo cáo,
   - phù hợp mức đồ án cơ sở.

---

## 18) Source provenance

### Primary sources
1. **AMDGT paper (Knowledge-Based Systems, 2024)**  
   - ScienceDirect article / uploaded PDF version in local zip.
2. **GitHub repo: JK-Liu7/AMDGT**  
   - README, `train_DDA.py`, `data_preprocess.py`, `metric.py`, `model/*`
3. **Review paper (Biomolecules, 2022)**  
   - “Drug-Disease Association Prediction Using Heterogeneous Networks for Computational Drug Repositioning”

### Additional uploaded references
- PREDICT
- MBiRW
- LAGCN
- AMDGT PDF
- HGTDR
- MRDDA
- DD-HGNN+

