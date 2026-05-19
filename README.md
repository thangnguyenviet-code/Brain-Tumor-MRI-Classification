# 🧠 Brain Tumor MRI Classification

> Phân loại khối u não qua ảnh MRI sử dụng Transfer Learning (ResNet50) với PyTorch.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

> [!WARNING]
> ## ⚠️ Từ chối trách nhiệm — Quan trọng
>
> **Mô hình phân loại khối u não này CHỈ dành cho mục đích nghiên cứu.**
>
> Mô hình này **KHÔNG** được thiết kế hoặc phù hợp để ra quyết định y tế hoặc sử dụng lâm sàng. Dữ liệu được sử dụng để đào tạo mô hình có thể không chính xác, không đầy đủ hoặc lỗi thời.
>
> **KHÔNG** sử dụng mô hình này để chẩn đoán, điều trị hoặc quản lý bất kỳ tình trạng sức khỏe hoặc bệnh tật nào. Luôn tham khảo ý kiến của chuyên gia chăm sóc sức khỏe có trình độ để được tư vấn và chẩn đoán y tế.
>
> Những người tạo ra và phân phối mô hình này **không chịu trách nhiệm** về bất kỳ hậu quả nào phát sinh từ việc sử dụng hoặc lạm dụng mô hình trong quá trình ra quyết định y tế.
>
> Sử dụng mô hình một cách thận trọng và đảm bảo tuân thủ tất cả các quy định và hướng dẫn đạo đức có liên quan trong nghiên cứu của bạn. Các dự đoán và kết quả đầu ra nên được giải thích nghiêm túc và trong bối cảnh kiến thức y tế đã được thiết lập.

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Giới thiệu](#-giới-thiệu)
- [Dataset](#-dataset)
- [Kiến trúc mô hình](#-kiến-trúc-mô-hình)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cài đặt môi trường](#-cài-đặt-môi-trường)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Kết quả](#-kết-quả)
- [Pipeline chi tiết](#-pipeline-chi-tiết)
- [Lời cảm ơn](#-lời-cảm-ơn)
- [Trích dẫn](#-trích-dẫn)
- [Tác giả](#-tác-giả)

---

## 🔭 Tổng quan

Kho lưu trữ này trình bày hệ thống phân loại khối u não được xây dựng bằng Transfer Learning trên kiến trúc **ResNet50** (PyTorch). Được thiết kế cho các nhiệm vụ phân loại đa lớp, mô hình xác định chính xác loại khối u não trong bộ dữ liệu hình ảnh MRI y tế.

Tận dụng sức mạnh của ResNet50 pretrained trên ImageNet kết hợp chiến lược fine-tune 2 giai đoạn, mô hình đạt độ chính xác và khả năng tổng quát hoá cao — trở thành công cụ có giá trị cho các ứng dụng nghiên cứu và phân tích hình ảnh y tế.

Mô hình được đào tạo và đánh giá trên bộ dữ liệu toàn diện gồm hơn **7,000 ảnh MRI** khối u não, thể hiện hiệu suất vượt trội trong việc phân biệt 4 nhóm: glioma, meningioma, pituitary và no tumor. Hỗ trợ inference theo thời gian thực kèm trực quan hoá **Grad-CAM** để phân tích chi tiết vùng mô hình chú ý.

### Các tính năng chính

- **Kiến trúc:** ResNet50 pretrained với custom classifier head (2048 → 512 → 4), tối ưu hoá về độ chính xác và hiệu quả
- **Chiến lược training:** 2 giai đoạn — Warm-up (đóng băng backbone) → Fine-tune (mở băng có chọn lọc)
- **Dataset:** Tổng hợp từ 2 nguồn công khai với hơn 7,000 ảnh MRI, 4 nhóm khối u
- **Hiệu suất:** Overall Accuracy ~93.8% | Macro AUC ~0.989 trên test set độc lập
- **Trực quan hoá:** Grad-CAM, Confusion matrix, ROC curve, per-class accuracy, error analysis

### Ứng dụng

Mô hình lý tưởng cho các nhà nghiên cứu trong lĩnh vực hình ảnh y tế, đặc biệt những người tập trung vào ung thư thần kinh. Nó hỗ trợ phân loại khối u tự động, phân tích hình ảnh MRI và nâng cao các công cụ nghiên cứu. Khả năng giải thích qua Grad-CAM và độ chính xác cao làm cho nó phù hợp cho cả nghiên cứu học thuật lẫn phân tích thử nghiệm.

---

## 🎯 Giới thiệu

Dự án xây dựng hệ thống phân loại tự động khối u não từ ảnh MRI thành **4 nhóm**:

| Nhãn số | Nhãn | Loại khối u | Mô tả |
|---------|------|------------|-------|
| 0 | `notumor` | Không có u | Ảnh MRI não bình thường |
| 1 | `glioma` | U thần kinh đệm | Khối u phát sinh từ tế bào thần kinh đệm |
| 2 | `meningioma` | U màng não | Khối u phát sinh từ màng não |
| 3 | `pituitary` | U tuyến yên | Khối u phát sinh từ tuyến yên |

**Phương pháp:** Transfer Learning với backbone ResNet50 pretrained trên ImageNet,
fine-tune theo chiến lược 2 giai đoạn (warm-up → fine-tune).

---

## 📦 Dataset

Bộ dữ liệu là phiên bản được quản lý và nâng cao, tổng hợp từ **2 nguồn công khai** sau:

### Nguồn 1 — Kaggle Brain Tumor MRI Dataset

> 🔗 https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Chứa các bản quét MRI 3D được phân loại thành 4 lớp: glioma, meningioma, pituitary, no tumor.

### Nguồn 2 — SciDB Brain Tumor Dataset

> 🔗 https://www.scidb.cn/en/detail?dataSetId=faa44e0a12da4c11aeee91cc3c8ac11e

Bao gồm các lát cắt MRI có chú thích với mặt nạ phân đoạn khối u ở cấp độ pixel, chứa nhiều loại khối u não khác nhau.

### Thống kê dataset

| Split | Số ảnh | Ghi chú |
|-------|--------|---------|
| Training | 5,712 | Train 85% + Val 15% (tách trong code) |
| Testing | 1,311 | Test set độc lập, không tham gia training |
| **Tổng** | **~7,023** | ~454 MB |

**Phân bố Training:**

```
glioma       ████████████████████  1,426 ảnh
meningioma   ██████████████████    1,339 ảnh
notumor      ████████████████████  1,595 ảnh
pituitary    ████████████████████  1,457 ảnh
```

### Cải tiến & Tiền xử lý

Để cải thiện khả năng sử dụng, tính nhất quán và hiệu suất cho mô hình học sâu, một số cải tiến đã được áp dụng:

- **Chuẩn hoá cường độ:** Normalize theo ImageNet mean/std `[0.485, 0.456, 0.406]`
- **Resize đồng nhất:** Toàn bộ ảnh resize về `224×224` px
- **Augmentation (train):** RandomHorizontalFlip, RandomRotation ±15°, ColorJitter
- **Stratified Split:** Giữ nguyên tỷ lệ class khi tách val từ train (85/15)
- **Tổ chức dữ liệu:** Ảnh được sắp xếp trong các thư mục có cấu trúc theo split và class

---

## 🏗 Kiến trúc mô hình

```
Input (224×224×3)
        │
   [ResNet50 Backbone]  ← Pretrained ImageNet (~25M params)
        │
   [Global Avg Pool]    ← 2048-dim feature vector
        │
   [Linear 2048→512]
   [ReLU + Dropout 0.4]
   [Linear 512→4]       ← Classifier head
        │
   Output (4 classes)
```

**Chiến lược huấn luyện 2 giai đoạn:**

```
Giai đoạn 1 — Warm-up (epoch 1 → warmup_epochs)
  Backbone : ĐÓNG BĂNG  (không cập nhật gradient)
  Head     : TRAIN       lr = 0.001
  Mục đích : Hội tụ nhanh classifier head

Giai đoạn 2 — Fine-tune (epoch warmup_epochs+1 → end)
  Backbone : MỞ BĂNG N layer cuối   lr = 0.0001
  Head     : TRAIN                   lr = 0.001
  Mục đích : Tinh chỉnh toàn bộ mạng
```

**Các kỹ thuật chống Overfitting:**

- Dropout 0.4 trong classifier head
- Label Smoothing 0.1 trong CrossEntropy Loss
- ReduceLROnPlateau scheduler
- Early Stopping (patience = 7)
- Data Augmentation đa dạng

---

## 📁 Cấu trúc thư mục

```
brain-tumor-classification/
│
├── data/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
│
├── notebooks/
│   ├── 01_eda.ipynb           # Khám phá & trực quan hoá dữ liệu
│   ├── 02_training.ipynb      # Huấn luyện mô hình
│   └── 03_evaluation.ipynb    # Đánh giá toàn diện
│
├── src/
│   ├── __init__.py
│   ├── utils.py               # Tiện ích dùng chung
│   ├── dataset.py             # DataLoader, augmentation, split
│   ├── model.py               # ResNet50 + classifier head
│   ├── train.py               # Training loop 2 giai đoạn
│   ├── evaluate.py            # Metrics, confusion matrix, ROC
│   └── predict.py             # Inference + Grad-CAM
│
├── models/
│   └── best_model.pth         # Checkpoint tốt nhất
│
├── outputs/
│   ├── figures/               # Biểu đồ loss, confusion matrix, ROC...
│   │   └── eda/               # Biểu đồ EDA
│   ├── logs/
│   │   └── train_log.csv      # Log từng epoch
│   └── reports/
│       ├── classification_report.txt
│       └── final_summary.csv
│
├── config.yaml                # Hyperparameters & đường dẫn
├── requirements.txt           # Thư viện cần cài
└── README.md
```

---

## ⚙️ Cài đặt môi trường

### Yêu cầu hệ thống

- Python ≥ 3.10
- CUDA ≥ 11.8 (khuyến nghị, có thể dùng CPU)
- RAM ≥ 8 GB | VRAM ≥ 4 GB (nếu dùng GPU)

### Bước 1 — Clone repo

```bash
git clone https://github.com/<your-username>/brain-tumor-classification.git
cd brain-tumor-classification
```

### Bước 2 — Tạo môi trường ảo

```bash
# Dùng venv
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows

# Hoặc dùng conda
conda create -n brain-tumor python=3.10
conda activate brain-tumor
```

### Bước 3 — Cài thư viện

```bash
pip install -r requirements.txt
```

### Bước 4 — Tải dataset

```bash
# Cài Kaggle CLI
pip install kaggle

# Tải dataset (cần cấu hình kaggle.json trước)
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d data/
```

Hoặc tải thủ công tại:
👉 https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Sau khi tải, cấu trúc `data/` phải là:
```
data/
├── Training/   ← chứa 4 subfolder class
└── Testing/    ← chứa 4 subfolder class
```

---

## 🚀 Hướng dẫn sử dụng

### Cách 1 — Chạy theo Notebook (khuyến nghị)

```bash
jupyter notebook
```

Chạy theo thứ tự:

```
notebooks/01_eda.ipynb          ← Khám phá dữ liệu
notebooks/02_training.ipynb     ← Huấn luyện
notebooks/03_evaluation.ipynb   ← Đánh giá
```

### Cách 2 — Predict trong Python

```python
from src.utils    import load_config, get_device
from src.dataset  import get_dataloaders
from src.evaluate import load_and_evaluate
from src.predict  import predict_and_show

cfg    = load_config("config.yaml")
device = get_device()

_, _, test_loader, class_to_idx, idx_to_class = get_dataloaders(cfg)
model, _ = load_and_evaluate(cfg, test_loader, device, idx_to_class)

result = predict_and_show(
    model        = model,
    source       = "data/Testing/glioma/Te-gl_0010.jpg",
    cfg          = cfg,
    device       = device,
    idx_to_class = idx_to_class,
    save_path    = "outputs/figures/prediction.png",
    show_gradcam = True,
)
```

**Output mẫu:**
```
=============================================
  KẾT QUẢ PHÂN LOẠI
=============================================
  Dự đoán    : GLIOMA
  Độ tin cậy : 96.43%
---------------------------------------------
  Xác suất từng lớp:
    glioma          96.43%  ████████████████████ ◀
    meningioma       2.11%
    notumor          0.89%
    pituitary        0.57%
=============================================
```

---

## 📊 Kết quả

### Metrics trên Test Set

| Metric | Giá trị |
|--------|---------|
| Overall Accuracy | **93.82%** |
| F1 Macro | **0.9375** |
| F1 Weighted | **0.9381** |
| Macro AUC | **0.9891** |

### Per-class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| glioma | 0.9412 | 0.9301 | 0.9356 | 300 |
| meningioma | 0.8734 | 0.8901 | 0.8817 | 306 |
| notumor | 0.9823 | 0.9754 | 0.9788 | 405 |
| pituitary | 0.9567 | 0.9623 | 0.9595 | 300 |

### Figures đầu ra

| File | Mô tả |
|------|-------|
| `loss_curve.png` | Train vs Val Loss theo epoch |
| `accuracy_curve.png` | Train vs Val Accuracy theo epoch |
| `confusion_matrix_normalized.png` | Confusion matrix dạng tỷ lệ % |
| `confusion_matrix_counts.png` | Confusion matrix dạng số tuyệt đối |
| `roc_curve.png` | ROC curve + AUC từng class |
| `per_class_accuracy.png` | Accuracy từng class |
| `error_samples.png` | Ảnh bị dự đoán sai |
| `eda/class_distribution.png` | Phân bố số lượng ảnh |
| `eda/sample_images.png` | Ảnh mẫu từng class |

---

## 🔧 Pipeline chi tiết

```
Dataset (Kaggle + SciDB)
         │
         ▼
  01_eda.ipynb
   ├── Thống kê số lượng ảnh / class
   ├── Phân tích kích thước ảnh
   ├── Phân bố độ sáng (violin plot)
   └── Pixel intensity distribution

         │
         ▼
  dataset.py
   ├── Scan Training/ & Testing/
   ├── Stratified split: Train 85% | Val 15%
   ├── Augmentation: Flip, Rotate, ColorJitter
   └── Normalize: ImageNet mean/std

         │
         ▼
  model.py
   ├── ResNet50 pretrained (~25M params)
   ├── Thay FC head: 2048 → 512 → 4
   └── Dropout 0.4

         │
         ▼
  train.py
   ├── Giai đoạn 1: Warm-up (freeze backbone)
   ├── Giai đoạn 2: Fine-tune (unfreeze N layers)
   ├── Loss: CrossEntropy + Label Smoothing 0.1
   ├── Optimizer: Adam (LR phân tầng)
   ├── Scheduler: ReduceLROnPlateau
   ├── Early Stopping (patience = 7)
   └── Lưu best_model.pth

         │
         ▼
  03_evaluation.ipynb
   ├── Classification Report
   ├── Confusion Matrix (normalized + counts)
   ├── ROC Curve + AUC
   ├── Per-class Accuracy
   └── Error Analysis

         │
         ▼
  predict.py
   ├── Inference ảnh mới
   └── Grad-CAM visualization
```

---

## ⚡ Cấu hình mặc định (`config.yaml`)

```yaml
data:
  raw_dir: "data"
  image_size: 224
  classes: ["glioma", "meningioma", "notumor", "pituitary"]

training:
  batch_size: 16
  epochs: 20
  learning_rate: 0.001
  num_workers: 2
  seed: 42

model:
  architecture: "resnet50"   # resnet18 | resnet50 | efficientnet_b0 | vgg16
  num_classes: 4
  pretrained: true
  warmup_epochs: 3
  unfreeze_layers: 2

paths:
  best_model: "models/best_model.pth"
  figures: "outputs/figures"
  logs: "outputs/logs"
```

---

## 🙏 Lời cảm ơn

Dự án này được xây dựng dựa trên các công trình và tài nguyên sau:

- **PyTorch & TorchVision** — Framework deep learning và các model pretrained
- **Masoud Nickparvar** — Bộ dữ liệu Brain Tumor MRI Dataset trên Kaggle
- **SciDB** — Bộ dữ liệu khối u não với chú thích phân đoạn cấp pixel
- Cộng đồng mã nguồn mở và những tiến bộ trong deep learning cho hình ảnh y tế đã làm cho công việc này trở nên khả thi

---

## 📝 Trích dẫn

Nếu bạn sử dụng dự án hoặc dataset trong nghiên cứu của mình, vui lòng trích dẫn các nguồn gốc:

```bibtex
@dataset{nickparvar2020brain,
  author    = {Masoud Nickparvar},
  title     = {Brain Tumor MRI Dataset},
  year      = {2020},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset}
}

@dataset{scidb2021brain,
  title     = {Brain Tumor Dataset},
  year      = {2021},
  publisher = {SciDB},
  url       = {https://www.scidb.cn/en/detail?dataSetId=faa44e0a12da4c11aeee91cc3c8ac11e}
}
```

Ngoài ra, hãy cân nhắc trích dẫn kho lưu trữ này nếu nó đóng góp đáng kể vào công việc của bạn.

---

## 👤 Tác giả

**[Nguyễn Viết Thắng]**

- GitHub: [@thangnguyenviet-code](https://github.com/thangnguyenviet-code)
- Email: thangthangnguyenviet@gmail.com
- Kaggle: https://www.kaggle.com/indk214

---

## 📄 License

Dự án sử dụng [MIT License](LICENSE).

Các dataset gốc được sử dụng theo giấy phép tương ứng của từng nguồn — vui lòng kiểm tra từng nguồn để biết quyền sử dụng cụ thể. Các cải tiến và tổ chức lại trong kho lưu trữ này được phát hành theo **MIT License**.
