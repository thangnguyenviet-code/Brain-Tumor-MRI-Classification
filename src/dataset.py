# src/dataset.py
import os
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class BrainTumorDataset(Dataset):
    """Dataset tùy chỉnh để tải ảnh MRI u não"""

    def __init__(self, image_paths: list, labels: list, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def _scan_directory(data_dir: Path, classes: list, class_to_idx: dict):
    """
    Quét một thư mục (Training hoặc Testing) để lấy paths và labels.
    Tự động khớp tên folder không phân biệt hoa/thường.
    """
    image_paths, labels = [], []

    # Map tên folder thực tế → tên class trong config (không phân biệt hoa/thường)
    folder_map = {}
    if data_dir.exists():
        for folder in data_dir.iterdir():
            if folder.is_dir():
                folder_lower = folder.name.lower()
                for cls_name in classes:
                    if cls_name.lower() == folder_lower:
                        folder_map[folder.name] = cls_name

    for folder_name, cls_name in folder_map.items():
        cls_dir = data_dir / folder_name
        count = 0
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(str(cls_dir / img_name))
                labels.append(class_to_idx[cls_name])
                count += 1
        print(f"  [{cls_name}] {count} ảnh")

    return image_paths, labels


def get_class_distribution(labels: list, class_to_idx: dict):
    """In phân bố số lượng ảnh theo từng class"""
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    counter = Counter(labels)
    print("\n[Phân bố dữ liệu]")
    for idx, count in sorted(counter.items()):
        print(f"  {idx_to_class[idx]:<20} {count} ảnh")
    print()


def get_dataloaders(cfg: dict):
    """
    Đọc cấu hình, quét data từ thư mục Training,
    chia Stratified (70/15/15) và tạo DataLoaders.
    Thư mục Testing được giữ nguyên làm test set độc lập.
    """
    # ------------------------------------------------------------------ #
    # FIX #1: Trỏ đúng vào thư mục Training (không phải thư mục data gốc)
    # ------------------------------------------------------------------ #
    base_dir    = Path(cfg["data"]["raw_dir"])   # "data"
    train_dir   = base_dir / "Training"
    test_dir    = base_dir / "Testing"

    img_size    = cfg["data"]["image_size"]
    batch_size  = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    seed        = cfg["training"]["seed"]
    classes     = cfg["data"]["classes"]

    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # 1. Quét thư mục Training → dùng để split train/val
    print("[Dataset] Quét thư mục Training...")
    train_paths, train_labels = _scan_directory(train_dir, classes, class_to_idx)
    print(f"  → Tổng: {len(train_paths)} ảnh")

    get_class_distribution(train_labels, class_to_idx)

    # 2. Quét thư mục Testing → test set độc lập (KHÔNG tham gia training)
    print("[Dataset] Quét thư mục Testing...")
    test_paths, test_labels = _scan_directory(test_dir, classes, class_to_idx)
    print(f"  → Tổng: {len(test_paths)} ảnh\n")

    if len(train_paths) == 0:
        raise ValueError(
            f"Không tìm thấy ảnh trong {train_dir}. "
            f"Kiểm tra lại đường dẫn raw_dir trong config.yaml "
            f"và tên các class: {classes}"
        )

    # 3. Tách val từ Training (Stratified): 85% train — 15% val
    X_train, X_val, y_train, y_val = train_test_split(
        train_paths, train_labels,
        test_size=0.15,
        random_state=seed,
        stratify=train_labels
    )

    print(f"[Split] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(test_paths)}")

    # 4. Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # thêm: giả lập điều kiện chụp khác nhau
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 5. Dataset objects
    train_dataset = BrainTumorDataset(X_train, y_train, transform=train_transforms)
    val_dataset   = BrainTumorDataset(X_val,   y_val,   transform=eval_transforms)
    test_dataset  = BrainTumorDataset(test_paths, test_labels, transform=eval_transforms)

    # FIX #2: pin_memory chỉ bật khi có GPU
    use_pin_memory = torch.cuda.is_available()

    # 6. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=use_pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=use_pin_memory)

    return train_loader, val_loader, test_loader, class_to_idx, idx_to_class