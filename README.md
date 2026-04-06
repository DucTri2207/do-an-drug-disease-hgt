# HGT Drug–Protein–Disease Link Prediction

## Mô tả
Xây dựng mô hình **Heterogeneous Graph Transformer (HGT)** cho mạng Drug–Protein–Disease nhằm dự đoán liên kết Drug–Disease (drug repositioning).

## Mục tiêu
- Dự đoán mối liên kết thuốc–bệnh dựa trên đồ thị dị thể
- Trả về top-k kết quả với confidence score
- Tích hợp vào website tra cứu (giai đoạn sau)

## Cấu trúc dự án
```
├── data/                    # Dữ liệu (C-dataset từ AMDGT repo)
├── notebooks/               # Jupyter notebooks khám phá dữ liệu
├── src/                     # Source code chính
│   ├── data_loader.py       # Load raw data files
│   ├── preprocess.py        # Chuẩn hóa và tiền xử lý
│   ├── feature_builder.py   # Tổ chức feature tensors
│   ├── graph_builder.py     # Xây DGL heterograph
│   ├── split.py             # Train/val/test split
│   ├── baseline.py          # MLP baseline model
│   ├── model_hgt.py         # HGT model chính
│   ├── trainer.py           # Training loop
│   ├── evaluator.py         # Metrics và evaluation
│   ├── inference.py         # Top-k prediction
│   └── main.py              # Entry point
├── checkpoints/             # Saved model weights
├── results/                 # Evaluation outputs
├── plans/                   # Planning documents
├── requirements.txt         # Python dependencies
└── README.md
```

## Cài đặt

### 1. Tạo virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 2. Cài thư viện
```bash
pip install -r requirements.txt
```

### 3. Cài DGL (chọn theo GPU/CPU)
```bash
# CPU only:
pip install dgl -f https://data.dgl.ai/wheels/repo.html

# CUDA 11.8:
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

# CUDA 12.1:
pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html
```

## Sử dụng

### Training
```bash
python src/main.py --mode train --dataset C-dataset
```

### Evaluation
```bash
python src/main.py --mode evaluate --checkpoint checkpoints/best_model.pt
```

### Prediction (top-k)
```bash
python src/main.py --mode predict --drug "Metformin" --top-k 10
```

## Dataset
Sử dụng C-dataset từ [AMDGT repo](https://github.com/JK-Liu7/AMDGT):
- 663 drugs, 409 diseases, 993 proteins
- 2,532 known drug-disease associations
- Features: Drug_mol2vec (300d), DiseaseFeature (64d), Protein_ESM (320d)

## Tham khảo
- AMDGT paper: Knowledge-Based Systems, 2024
- HGT paper: Hu et al., WWW 2020
- Review: Biomolecules, 2022
