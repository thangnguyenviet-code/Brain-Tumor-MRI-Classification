# src/evaluate.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)


# ------------------------------------------------------------------ #
# 1. Thu thập dự đoán
# ------------------------------------------------------------------ #
@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple:
    """
    Chạy inference trên toàn bộ loader.
    Trả về:
        all_preds  : list nhãn dự đoán
        all_labels : list nhãn thật
        all_probs  : array [N, num_classes] xác suất softmax
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)

        outputs = model(images)                        # [B, C]
        probs   = torch.softmax(outputs, dim=1)        # [B, C]
        preds   = probs.argmax(dim=1)                  # [B]

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


# ------------------------------------------------------------------ #
# 2. Confusion Matrix
# ------------------------------------------------------------------ #
def plot_confusion_matrix(
    y_true, y_pred,
    class_names: list,
    save_path: str,
    normalize: bool = True,
):
    """
    Vẽ Confusion Matrix.
    normalize=True → hiển thị tỷ lệ % thay vì số tuyệt đối
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, vmax = ".2%", 1.0
        title = "Confusion Matrix (Normalized)"
    else:
        cm_plot = cm
        fmt, vmax = "d", cm.max()
        title = "Confusion Matrix (Counts)"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0, vmax=vmax,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(title,              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Saved] Confusion matrix → {save_path}")

    # In raw counts kèm theo
    print("\n[Confusion Matrix - Raw Counts]")
    header = f"{'':>15}" + "".join(f"{c:>15}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:>15}" + "".join(f"{v:>15}" for v in row)
        print(row_str)


# ------------------------------------------------------------------ #
# 3. ROC Curve (One-vs-Rest, đa lớp)
# ------------------------------------------------------------------ #
def plot_roc_curve(
    y_true, y_probs,
    class_names: list,
    save_path: str,
):
    """
    Vẽ ROC Curve cho từng class theo chiến lược One-vs-Rest.
    Tính AUC macro-average.
    """
    num_classes = len(class_names)
    colors      = plt.cm.tab10(np.linspace(0, 1, num_classes))

    fig, ax = plt.subplots(figsize=(8, 6))

    auc_scores = []
    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        # Nhị phân hoá: class i vs phần còn lại
        y_bin  = (y_true == i).astype(int)
        y_prob = y_probs[:, i]

        fpr, tpr, _ = roc_curve(y_bin, y_prob)
        roc_auc     = auc(fpr, tpr)
        auc_scores.append(roc_auc)

        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls_name} (AUC = {roc_auc:.3f})")

    # Đường random baseline
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

    macro_auc = np.mean(auc_scores)
    ax.set_title(f"ROC Curve — Macro AUC: {macro_auc:.3f}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Saved] ROC curve → {save_path}")

    return macro_auc


# ------------------------------------------------------------------ #
# 4. Per-class Accuracy Bar Chart
# ------------------------------------------------------------------ #
def plot_per_class_accuracy(
    y_true, y_pred,
    class_names: list,
    save_path: str,
):
    """Vẽ biểu đồ accuracy từng class"""
    cm  = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    colors = ["#2ecc71" if a >= 0.85 else
              "#f39c12" if a >= 0.70 else
              "#e74c3c" for a in per_class_acc]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(class_names, per_class_acc * 100, color=colors, edgecolor="white")

    # Ghi giá trị lên đầu mỗi cột
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{acc*100:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Legend màu
    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="≥ 85%  Tốt"),
        mpatches.Patch(color="#f39c12", label="≥ 70%  Trung bình"),
        mpatches.Patch(color="#e74c3c", label="< 70%  Cần cải thiện"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")

    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-class Accuracy", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Saved] Per-class accuracy → {save_path}")


# ------------------------------------------------------------------ #
# 5. Hàm tổng hợp — gọi 1 lần ra đủ kết quả
# ------------------------------------------------------------------ #
def evaluate(
    model,
    test_loader,
    cfg: dict,
    device,
    idx_to_class: dict,
):
    """
    Đánh giá toàn diện trên test set.
    Xuất: classification report, confusion matrix,
          ROC curve, per-class accuracy.
    """
    class_names  = [idx_to_class[i] for i in range(len(idx_to_class))]
    figures_dir  = cfg["paths"]["figures"]
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Thu thập dự đoán
    print("\n[Evaluate] Đang inference test set...")
    y_pred, y_true, y_probs = collect_predictions(model, test_loader, device)

    # 2. Classification Report
    print("\n" + "="*60)
    print("  CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )
    print(report)

    # Lưu report ra file txt
    report_path = os.path.join(figures_dir, "..", "reports", "classification_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Saved] Report → {report_path}")

    # 3. Confusion Matrix (cả 2 dạng)
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=os.path.join(figures_dir, "confusion_matrix_normalized.png"),
        normalize=True,
    )
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=os.path.join(figures_dir, "confusion_matrix_counts.png"),
        normalize=False,
    )

    # 4. ROC Curve
    macro_auc = plot_roc_curve(
        y_true, y_probs, class_names,
        save_path=os.path.join(figures_dir, "roc_curve.png"),
    )

    # 5. Per-class Accuracy
    plot_per_class_accuracy(
        y_true, y_pred, class_names,
        save_path=os.path.join(figures_dir, "per_class_accuracy.png"),
    )

    # 6. Tổng kết số liệu
    overall_acc = (y_pred == y_true).mean() * 100
    print("\n" + "="*60)
    print("  KẾT QUẢ TỔNG HỢP")
    print("="*60)
    print(f"  Overall Accuracy : {overall_acc:.2f}%")
    print(f"  Macro AUC        : {macro_auc:.4f}")
    print("="*60 + "\n")

    return {
        "overall_acc": overall_acc,
        "macro_auc":   macro_auc,
        "y_pred":      y_pred,
        "y_true":      y_true,
        "y_probs":     y_probs,
    }


# ------------------------------------------------------------------ #
# 6. Load checkpoint và evaluate
# ------------------------------------------------------------------ #
def load_and_evaluate(cfg: dict, test_loader, device, idx_to_class: dict):
    """
    Tiện ích: load best_model.pth rồi evaluate luôn.
    Dùng trong notebook 03_evaluation.ipynb
    """
    from src.model import build_model

    model_path = cfg["paths"]["best_model"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model tại: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model      = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"[Load] Model từ epoch {checkpoint['epoch']} "
          f"— val_loss: {checkpoint['val_loss']:.4f} "
          f"| val_acc: {checkpoint['val_acc']:.2f}%")

    results = evaluate(model, test_loader, cfg, device, idx_to_class)
    return model, results