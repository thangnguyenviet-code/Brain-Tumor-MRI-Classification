# src/model.py
import torch
import torch.nn as nn
from torchvision import models


def build_model(cfg: dict) -> nn.Module:
    """
    Xây dựng mô hình Transfer Learning từ config.
    Hỗ trợ: resnet50, resnet18, efficientnet_b0, vgg16
    """
    architecture = cfg["model"]["architecture"].lower()
    num_classes  = cfg["model"]["num_classes"]       # 4
    pretrained   = cfg["model"]["pretrained"]         # True

    weights = "IMAGENET1K_V1" if pretrained else None

    # ------------------------------------------------------------------ #
    # Chọn backbone
    # ------------------------------------------------------------------ #
    if architecture == "resnet50":
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features          # 2048
        model.fc = _build_classifier(in_features, num_classes)

    elif architecture == "resnet18":
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features          # 512
        model.fc = _build_classifier(in_features, num_classes)

    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features   # 1280
        model.classifier = _build_classifier(in_features, num_classes)

    elif architecture == "vgg16":
        model = models.vgg16(weights=weights)
        in_features = model.classifier[6].in_features   # 4096
        model.classifier[6] = _build_classifier(in_features, num_classes)

    else:
        raise ValueError(
            f"Kiến trúc '{architecture}' chưa được hỗ trợ. "
            f"Chọn một trong: resnet50, resnet18, efficientnet_b0, vgg16"
        )

    print(f"[Model] Kiến trúc : {architecture}")
    print(f"[Model] Pretrained : {pretrained}")
    print(f"[Model] Num classes: {num_classes}")

    return model


def _build_classifier(in_features: int, num_classes: int) -> nn.Sequential:
    """
    Classifier head thay thế lớp cuối của backbone.
    Thêm Dropout để giảm Overfitting.
    """
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(512, num_classes)
    )


def freeze_backbone(model: nn.Module, architecture: str):
    """
    Đóng băng toàn bộ backbone, chỉ train classifier head.
    Dùng ở giai đoạn đầu (warm-up) để head hội tụ nhanh.
    """
    # Đóng băng tất cả
    for param in model.parameters():
        param.requires_grad = False

    # Mở băng phần classifier head
    if architecture in ("resnet50", "resnet18"):
        for param in model.fc.parameters():
            param.requires_grad = True

    elif architecture == "efficientnet_b0":
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif architecture == "vgg16":
        for param in model.classifier[6].parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Freeze] Chỉ train classifier — {trainable:,} params")


def unfreeze_backbone(model: nn.Module, architecture: str, unfreeze_layers: int = 2):
    # Mở tất cả trước
    for param in model.parameters():
        param.requires_grad = True

    if architecture in ("resnet50", "resnet18"):
        all_layers = [model.layer1, model.layer2, model.layer3, model.layer4]
        freeze_until = len(all_layers) - unfreeze_layers
        for i, layer in enumerate(all_layers):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False
        for param in list(model.conv1.parameters()) + list(model.bn1.parameters()):
            param.requires_grad = False

    elif architecture == "efficientnet_b0":
        total = len(model.features)
        for i, block in enumerate(model.features):
            if i < total - unfreeze_layers:
                for param in block.parameters():
                    param.requires_grad = False

    elif architecture == "vgg16":
        # VGG16 có 31 layer trong features (index 0-30)
        # Chỉ mở băng N block cuối, đóng băng phần đầu
        total = len(model.features)
        freeze_until = total - (unfreeze_layers * 4)  # mỗi block ~4 layer
        freeze_until = max(0, freeze_until)            # không âm
        for i, layer in enumerate(model.features):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False
        for param in model.classifier[:6].parameters():
            param.requires_grad = False

    else:
        raise ValueError(
            f"unfreeze_backbone chưa hỗ trợ '{architecture}'. "
            f"Thêm nhánh elif trước khi dùng."
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[Unfreeze] Fine-tune {unfreeze_layers} layers cuối — "
          f"{trainable:,} / {total:,} params ({100*trainable/total:.1f}%)")


def get_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """
    Tạo optimizer với learning rate khác nhau cho backbone và head.
    Head dùng LR cao hơn backbone 10x vì backbone đã pretrained.
    """
    architecture = cfg["model"]["architecture"].lower()
    base_lr      = cfg["training"]["learning_rate"]   # 0.001

    if architecture in ("resnet50", "resnet18"):
        head_params     = list(model.fc.parameters())
    elif architecture == "efficientnet_b0":
        head_params     = list(model.classifier.parameters())
    elif architecture == "vgg16":
        head_params     = list(model.classifier[6].parameters())
    else:
        head_params     = []

    head_ids    = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters()
                       if id(p) not in head_ids and p.requires_grad]

    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": base_lr * 0.1},  # backbone: lr nhỏ hơn
        {"params": head_params,     "lr": base_lr},         # head: lr gốc
    ])

    print(f"[Optimizer] Adam — backbone lr={base_lr*0.1} | head lr={base_lr}")
    return optimizer


def count_parameters(model: nn.Module):
    """In tổng số params và số params đang được train"""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Params] Tổng      : {total:>12,}")
    print(f"[Params] Trainable : {trainable:>12,}")
    print(f"[Params] Frozen    : {total - trainable:>12,}\n")