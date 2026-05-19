# src/train.py
import os
import time
import csv
from pathlib import Path
import wandb
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    """
    Chạy 1 epoch training.
    Trả về: (avg_loss, accuracy%)
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping — tránh gradient bùng nổ khi fine-tune
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        # In tiến trình mỗi 20 batch
        if (batch_idx + 1) % 20 == 0:
            batch_acc = (preds == labels).float().mean().item() * 100
            print(f"    Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f} | Acc: {batch_acc:.1f}%")

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device) -> tuple[float, float]:
    """
    Chạy 1 epoch đánh giá (val hoặc test).
    Trả về: (avg_loss, accuracy%)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs     = model(images)
        loss        = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy


def _init_csv_log(log_path: str):
    """Khởi tạo file CSV để ghi log từng epoch"""
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "phase",
                         "train_loss", "train_acc",
                         "val_loss",   "val_acc",
                         "lr"])


def _append_csv_log(log_path: str, row: list):
    """Ghi thêm 1 dòng vào CSV log"""
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def train(model, train_loader, val_loader, cfg: dict, device) -> dict:
    # --- BẮT ĐẦU THEO DÕI BẰNG W&B ---
    wandb.init(
        project="brain-tumor-classification", 
        name=f"{cfg['model']['architecture']}_run_01",
        config=cfg 
    )
    """
    Vòng lặp training chính.

    Chiến lược 2 giai đoạn:
      Giai đoạn 1 (warm-up): backbone bị đóng băng, chỉ train classifier head
      Giai đoạn 2 (fine-tune): mở băng N layer cuối backbone, train toàn bộ

    Trả về dict history chứa các list loss/acc để vẽ biểu đồ.
    """
    from src.model import freeze_backbone, unfreeze_backbone, get_optimizer

    architecture   = cfg["model"]["architecture"]
    epochs         = cfg["training"]["epochs"]
    warmup_epochs  = cfg["model"]["warmup_epochs"]
    unfreeze_layers= cfg["model"].get("unfreeze_layers", 2)
    model_path     = cfg["paths"]["best_model"]
    log_path       = os.path.join(cfg["paths"]["logs"], "train_log.csv")

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Loss function: Label Smoothing giảm overconfidence
    # ------------------------------------------------------------------ #
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ------------------------------------------------------------------ #
    # Giai đoạn 1: Warm-up — đóng băng backbone
    # ------------------------------------------------------------------ #
    freeze_backbone(model, architecture)
    optimizer = get_optimizer(model, cfg)

    # Learning rate scheduler — giảm LR khi val_loss không cải thiện
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  factor=0.5, patience=3)

    # Khởi tạo log
    _init_csv_log(log_path)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    best_val_loss  = float("inf")
    epochs_no_improve = 0
    EARLY_STOP_PATIENCE = 7   # dừng nếu val_loss không cải thiện sau N epoch

    print("\n" + "="*60)
    print(f"  BẮT ĐẦU TRAINING — {epochs} epochs | warmup: {warmup_epochs}")
    print("="*60)

    for epoch in range(1, epochs + 1):
        start = time.time()

        # ------------------------------------------------------------------ #
        # Chuyển sang Giai đoạn 2 tại epoch warmup_epochs + 1
        # ------------------------------------------------------------------ #
        if epoch == warmup_epochs + 1:
            print("\n" + "-"*60)
            print(f"  [Epoch {epoch}] Chuyển sang Fine-tune — mở băng backbone")
            print("-"*60)
            torch.cuda.empty_cache() # Thêm dòng này để giải phóng VRAM rác
            unfreeze_backbone(model, architecture, unfreeze_layers)
            # Tạo lại optimizer với learning rate phân tầng
            optimizer = get_optimizer(model, cfg)
            scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                          factor=0.5, patience=3)

        phase = "warm-up" if epoch <= warmup_epochs else "fine-tune"

        print(f"\nEpoch [{epoch:02d}/{epochs}] — {phase}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = evaluate_one_epoch(
            model, val_loader, criterion, device
        )

        # Cập nhật scheduler dựa trên val_loss
        scheduler.step(val_loss)

        # Lấy LR hiện tại (lấy của head — group index 1)
        current_lr = optimizer.param_groups[-1]["lr"]

        elapsed = time.time() - start

        print(f"  Train — Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Val   — Loss: {val_loss:.4f}   | Acc: {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f} | Time: {elapsed:.1f}s")

        # Ghi history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Ghi CSV log
        _append_csv_log(log_path, [
            epoch, phase,
            round(train_loss, 5), round(train_acc, 3),
            round(val_loss, 5),   round(val_acc, 3),
            current_lr
        ])
        # --- GỬI ĐIỂM SỐ LÊN W&B ---
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        # ------------------------------------------------------------------ #
        # Lưu model tốt nhất dựa trên val_loss
        # ------------------------------------------------------------------ #
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch":      epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":   val_loss,
                "val_acc":    val_acc,
                "cfg":        cfg,
            }, model_path)
            print(f"  ✅ Lưu best model — val_loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  ⏳ Không cải thiện {epochs_no_improve}/{EARLY_STOP_PATIENCE}")

        # ------------------------------------------------------------------ #
        # Early Stopping
        # ------------------------------------------------------------------ #
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n🛑 Early Stopping tại epoch {epoch} "
                  f"— val_loss không cải thiện sau {EARLY_STOP_PATIENCE} epochs")
            break

    print("\n" + "="*60)
    print(f"  TRAINING HOÀN TẤT")
    print(f"  Best val_loss : {best_val_loss:.4f}")
    print(f"  Log saved     : {log_path}")
    print(f"  Model saved   : {model_path}")
    print("="*60 + "\n")
    wandb.finish()
    return history