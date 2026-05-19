# src/predict.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path


# ------------------------------------------------------------------ #
# 1. Transform cho ảnh inference (giống eval_transforms trong dataset.py)
# ------------------------------------------------------------------ #
def get_inference_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ------------------------------------------------------------------ #
# 2. Load ảnh từ đường dẫn hoặc PIL Image
# ------------------------------------------------------------------ #
def load_image(source) -> Image.Image:
    """
    Nhận vào: đường dẫn file (str/Path) hoặc PIL Image sẵn có.
    Luôn trả về PIL Image ở chế độ RGB.
    """
    if isinstance(source, (str, Path)):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Không tìm thấy ảnh: {source}")
        image = Image.open(source).convert("RGB")
    elif isinstance(source, Image.Image):
        image = source.convert("RGB")
    else:
        raise TypeError(f"Kiểu không hỗ trợ: {type(source)}. "
                        f"Dùng str, Path, hoặc PIL.Image.")
    return image


# ------------------------------------------------------------------ #
# 3. Grad-CAM — trực quan hoá vùng mô hình chú ý
# ------------------------------------------------------------------ #
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Highlight vùng ảnh ảnh hưởng nhiều nhất đến quyết định của mô hình.
    Hỗ trợ: resnet50, resnet18 → target layer: model.layer4[-1]
             efficientnet_b0   → target layer: model.features[-1]
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Sinh ra heatmap Grad-CAM cho class_idx.
        Trả về numpy array [H, W] giá trị trong [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)           # forward pass

        self.model.zero_grad()
        # Backprop chỉ theo class cần xem
        output[0, class_idx].backward()

        # Tính trọng số gradient (Global Average Pooling theo H, W)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam     = (weights * self.activations).sum(dim=1).squeeze()  # [H, W]

        # ReLU + normalize về [0, 1]
        cam = torch.clamp(cam, min=0)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()


def get_gradcam_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    """Lấy đúng layer cuối cùng tùy theo kiến trúc"""
    arch = architecture.lower()
    if arch in ("resnet50", "resnet18"):
        return model.layer4[-1]
    elif arch == "efficientnet_b0":
        return model.features[-1]
    elif arch == "vgg16":
        return model.features[-1]
    else:
        raise ValueError(f"Grad-CAM chưa hỗ trợ architecture: {architecture}")


# ------------------------------------------------------------------ #
# 4. Hàm predict chính
# ------------------------------------------------------------------ #
@torch.no_grad()
def predict(
    model: nn.Module,
    source,
    cfg: dict,
    device,
    idx_to_class: dict,
) -> dict:
    """
    Inference 1 ảnh MRI.

    Args:
        model       : mô hình đã load
        source      : đường dẫn ảnh (str/Path) hoặc PIL Image
        cfg         : config dict
        device      : torch.device
        idx_to_class: dict {0: 'glioma', 1: 'meningioma', ...}

    Returns:
        dict chứa:
            predicted_class : tên lớp dự đoán
            predicted_idx   : index lớp dự đoán
            confidence      : độ tin cậy (%)
            probabilities   : dict {class_name: prob%} cho tất cả lớp
    """
    img_size  = cfg["data"]["image_size"]
    transform = get_inference_transform(img_size)

    # Load và transform ảnh
    image        = load_image(source)
    input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs   = torch.softmax(outputs, dim=1)[0]  # [num_classes]

    predicted_idx   = probs.argmax().item()
    predicted_class = idx_to_class[predicted_idx]
    confidence      = probs[predicted_idx].item() * 100

    probabilities = {
        idx_to_class[i]: round(probs[i].item() * 100, 2)
        for i in range(len(idx_to_class))
    }

    return {
        "predicted_class": predicted_class,
        "predicted_idx":   predicted_idx,
        "confidence":      round(confidence, 2),
        "probabilities":   probabilities,
        "image":           image,
        "input_tensor":    input_tensor,
    }


# ------------------------------------------------------------------ #
# 5. Visualize kết quả + Grad-CAM
# ------------------------------------------------------------------ #
def visualize_prediction(
    result: dict,
    cfg: dict,
    model: nn.Module,
    device,
    save_path: str = None,
    show_gradcam: bool = True,
):
    """
    Vẽ ảnh gốc + Grad-CAM heatmap + biểu đồ xác suất.
    """
    image           = result["image"]
    predicted_class = result["predicted_class"]
    confidence      = result["confidence"]
    probabilities   = result["probabilities"]
    input_tensor    = result["input_tensor"]
    predicted_idx   = result["predicted_idx"]

    # Màu theo độ tin cậy
    bar_color = ("#2ecc71" if confidence >= 85 else
                 "#f39c12" if confidence >= 60 else
                 "#e74c3c")

    n_cols = 3 if show_gradcam else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    # ── Cột 1: Ảnh gốc ──────────────────────────────────────────── #
    axes[0].imshow(image)
    axes[0].set_title("Ảnh MRI gốc", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # ── Cột 2: Grad-CAM overlay ─────────────────────────────────── #
    if show_gradcam:
        try:
            architecture = cfg["model"]["architecture"]
            target_layer = get_gradcam_target_layer(model, architecture)
            gradcam      = GradCAM(model, target_layer)

            # Grad-CAM cần gradient → không dùng torch.no_grad()
            input_grad = input_tensor.clone().requires_grad_(True)
            heatmap    = gradcam.generate(input_grad, predicted_idx)

            # Resize heatmap về kích thước ảnh gốc
            heatmap_img = Image.fromarray(
                (heatmap * 255).astype(np.uint8)
            ).resize(image.size, Image.BILINEAR)
            heatmap_np  = np.array(heatmap_img) / 255.0

            # Overlay: ảnh gốc + colormap
            img_np      = np.array(image) / 255.0
            colormap    = cm.jet(heatmap_np)[..., :3]   # [H, W, 3]
            overlay     = 0.55 * img_np + 0.45 * colormap
            overlay     = np.clip(overlay, 0, 1)

            axes[1].imshow(overlay)
            axes[1].set_title("Grad-CAM\n(vùng mô hình chú ý)",
                               fontsize=12, fontweight="bold")
            axes[1].axis("off")

        except Exception as e:
            axes[1].text(0.5, 0.5, f"Grad-CAM lỗi:\n{e}",
                         ha="center", va="center", transform=axes[1].transAxes)
            axes[1].axis("off")

    # ── Cột cuối: Biểu đồ xác suất ──────────────────────────────── #
    ax_prob = axes[2] if show_gradcam else axes[1]
    classes = list(probabilities.keys())
    probs   = list(probabilities.values())
    colors  = [bar_color if c == predicted_class else "#bdc3c7" for c in classes]

    bars = ax_prob.barh(classes, probs, color=colors, edgecolor="white", height=0.5)
    for bar, prob in zip(bars, probs):
        ax_prob.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{prob:.1f}%", va="center", fontsize=10, fontweight="bold")

    ax_prob.set_xlim(0, 115)
    ax_prob.set_xlabel("Xác suất (%)", fontsize=11)
    ax_prob.set_title(
        f"Dự đoán: {predicted_class.upper()}\nĐộ tin cậy: {confidence:.1f}%",
        fontsize=12, fontweight="bold", color=bar_color
    )
    ax_prob.grid(axis="x", alpha=0.3)

    plt.suptitle("Brain Tumor MRI — Kết quả phân loại",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] Kết quả → {save_path}")

    plt.show()


# ------------------------------------------------------------------ #
# 6. Tiện ích: predict + visualize gộp làm 1
# ------------------------------------------------------------------ #
def predict_and_show(
    model: nn.Module,
    source,
    cfg: dict,
    device,
    idx_to_class: dict,
    save_path: str = None,
    show_gradcam: bool = True,
) -> dict:
    """
    Gọi 1 hàm duy nhất: inference + in kết quả + vẽ biểu đồ.
    """
    result = predict(model, source, cfg, device, idx_to_class)

    # In kết quả ra console
    print("\n" + "="*45)
    print("  KẾT QUẢ PHÂN LOẠI")
    print("="*45)
    print(f"  Dự đoán    : {result['predicted_class'].upper()}")
    print(f"  Độ tin cậy : {result['confidence']:.2f}%")
    print("-"*45)
    print("  Xác suất từng lớp:")
    for cls, prob in sorted(result["probabilities"].items(),
                            key=lambda x: x[1], reverse=True):
        bar  = "█" * int(prob / 5)
        flag = " ◀" if cls == result["predicted_class"] else ""
        print(f"    {cls:<15} {prob:>6.2f}%  {bar}{flag}")
    print("="*45 + "\n")

    visualize_prediction(result, cfg, model, device, save_path, show_gradcam)
    return result