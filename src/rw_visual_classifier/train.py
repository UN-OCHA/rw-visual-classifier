"""Fine-tune MobileNetV4-small on Map / Infographic / Other."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from rw_visual_classifier.config import (
    BEST_CHECKPOINT,
    CHECKPOINTS_DIR,
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_NUM_WORKERS,
    DEFAULT_WEIGHT_DECAY,
    EARLY_STOP_PATIENCE,
    METRICS_DIR,
    MODEL_NAME,
)
from rw_visual_classifier.data import PreviewDataset, load_split_frames
from rw_visual_classifier.transforms import build_eval_transform, build_train_transform


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_model(num_classes: int = len(CLASS_NAMES), pretrained: bool = True):
    return timm.create_model(
        MODEL_NAME,
        pretrained=pretrained,
        num_classes=num_classes,
    )


@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += float(loss.item()) * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_preds)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return {
        "loss": total_loss / max(len(y_true), 1),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": {
            name: float(score) for name, score in zip(CLASS_NAMES, per_class)
        },
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    running = 0.0
    seen = 0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running += float(loss.item()) * images.size(0)
        seen += images.size(0)
    return running / max(seen, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "model_name": MODEL_NAME,
            "class_names": list(CLASS_NAMES),
        },
        path,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train RW visual classifier")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint to resume from",
    )
    args = parser.parse_args(argv)

    device = pick_device()
    print(f"Device: {device}")

    frames = load_split_frames()
    for split, frame in frames.items():
        print(f"{split}: {len(frame)} images")
        print(frame["label"].value_counts().to_string())

    train_ds = PreviewDataset(frames["train"], transform=build_train_transform())
    val_ds = PreviewDataset(frames["val"], transform=build_eval_transform())

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = create_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    start_epoch = 0
    best_f1 = -1.0
    patience_left = args.patience
    history: list[dict] = []

    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_f1 = float(ckpt.get("metrics", {}).get("macro_f1", -1.0))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = run_eval(model, val_loader, device)
        scheduler.step()
        elapsed = time.perf_counter() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
        }
        history.append(row)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"per_class={val_metrics['per_class_f1']} "
            f"({elapsed:.1f}s)"
        )

        epoch_path = CHECKPOINTS_DIR / f"epoch_{epoch:03d}.pt"
        save_checkpoint(epoch_path, model, optimizer, epoch, val_metrics)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            patience_left = args.patience
            save_checkpoint(BEST_CHECKPOINT, model, optimizer, epoch, val_metrics)
            print(f"  saved best → {BEST_CHECKPOINT} (macro_f1={best_f1:.4f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    history_path = METRICS_DIR / "train_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Wrote {history_path}")
    print(f"Best val macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
