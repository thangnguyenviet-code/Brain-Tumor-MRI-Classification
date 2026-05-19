# src/utils.py
import os
import random
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """Đọc file config.yaml"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    """Cố định random seed để kết quả reproducible"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Tự động chọn GPU nếu có, không thì dùng CPU"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Đang dùng: {device}")
    return device


def ensure_dirs(cfg: dict):
    """Tạo các thư mục output nếu chưa tồn tại"""
    dirs = [
        cfg["paths"]["figures"],
        cfg["paths"]["logs"],
        "models",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_loss_curve(train_losses: list, val_losses: list, save_path: str):
    """Vẽ và lưu biểu đồ loss"""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses,   label="Val Loss",   marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Saved] Loss curve → {save_path}")


def save_accuracy_curve(train_accs: list, val_accs: list, save_path: str):
    """Vẽ và lưu biểu đồ accuracy"""
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="Train Acc", marker="o")
    plt.plot(val_accs,   label="Val Acc",   marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Saved] Accuracy curve → {save_path}")