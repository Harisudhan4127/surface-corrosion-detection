#!/usr/bin/env python3
"""
src/train.py
=============
Main training script for NEU Surface Defect Detection.

Two-phase fine-tuning:
  Phase 1 (freeze_epochs)  : Train only the classification head. 
  Phase 2 (remaining epochs): Unfreeze backbone, train end-to-end with lower LR.

Usage:
    python src/train.py
    python src/train.py --arch resnet50 --epochs 60 --batch_size 32
    python src/train.py --arch efficientnet_b3 --epochs 40
    python src/train.py --resume models/checkpoints/best_model.pth
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import NEUSurfaceDataset
from data.transforms import get_train_transform, get_val_transform
from models.model import build_model
from utils.logger import setup_logger
from utils.metrics import MetricsTracker


# ── Args ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train NEU Surface Defect CNN")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--arch",       default=None,
                   choices=["resnet50", "resnet18", "efficientnet_b3", "custom_cnn"])
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--resume",     type=str,   default=None)
    p.add_argument("--data_dir",   type=str,   default=None)
    p.add_argument("--no_pretrain",action="store_true")
    return p.parse_args()


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── One epoch ────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc="train" if training else "val  ", bar_format='{l_bar}{bar:20}{r_bar} | {n_fmt}/{total_fmt} [{percentage:3.0f}%]', leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    n        = len(loader.dataset)
    avg_loss = total_loss / n
    acc      = np.mean(np.array(all_preds) == np.array(all_labels))
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1, all_preds, all_labels


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg  = load_cfg(ROOT / args.config)

    # CLI overrides
    if args.arch:       cfg["model"]["architecture"]  = args.arch
    if args.epochs:     cfg["training"]["epochs"]     = args.epochs
    if args.batch_size: cfg["training"]["batch_size"] = args.batch_size
    if args.lr:         cfg["training"]["lr"]         = args.lr
    if args.data_dir:   cfg["data"]["split_dir"]      = args.data_dir
    if args.no_pretrain:cfg["model"]["pretrained"]    = False

    logger = setup_logger("train", cfg["logging"]["log_dir"])
    logger.info(f"Architecture : {cfg['model']['architecture']}")
    logger.info(f"Config       : {json.dumps(cfg['training'], indent=2)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device       : {device}")
    if device.type == "cuda":
        logger.info(f"GPU          : {torch.cuda.get_device_name(0)}")

    # ── Datasets ─────────────────────────────────────────────────
    img_size = cfg["data"]["image_size"]
    train_tf = get_train_transform(img_size)
    val_tf   = get_val_transform(img_size)

    split_dir = ROOT / cfg["data"]["split_dir"]
    train_ds  = NEUSurfaceDataset(split_dir, "train", transform=train_tf)
    val_ds    = NEUSurfaceDataset(split_dir, "val",   transform=val_tf)

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    logger.info(f"Class counts (train): {train_ds.class_counts()}")

    nw = cfg["training"]["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                               shuffle=True,  num_workers=nw, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"],
                               shuffle=False, num_workers=nw, pin_memory=False)

    # ── Model ────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    if hasattr(model, "trainable_params"):
        logger.info(f"Trainable params: {model.trainable_params():,}")
    else:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable params: {total:,}")

    # ── Loss ─────────────────────────────────────────────────────
    # Dataset is balanced (300/class) so uniform weights;
    # label smoothing helps generalisation on small datasets
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg["training"].get("label_smoothing", 0.1)
    )

    # ── Optimiser & Scheduler ────────────────────────────────────
    def make_optimizer(lr): 
        return optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=cfg["training"]["weight_decay"],
        )

    epochs       = cfg["training"]["epochs"]
    freeze_epochs= cfg["training"].get("freeze_epochs", 5) \
                   if cfg["model"]["pretrained"] else 0
    base_lr      = cfg["training"]["lr"]

    optimizer  = make_optimizer(base_lr)
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── Resume ───────────────────────────────────────────────────
    start_epoch    = 0
    best_val_f1    = 0.0
    best_val_loss  = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_f1 = ckpt.get("val_f1", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}")

    # ── TensorBoard ──────────────────────────────────────────────
    tb_dir = ROOT / cfg["logging"]["tensorboard_dir"]
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    ckpt_dir = ROOT / cfg["training"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Early stopping ───────────────────────────────────────────
    patience   = cfg["training"]["patience"]
    es_counter = 0
    save_every = cfg["training"].get("save_every", 5)

    # ── Training loop ─────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("  TRAINING START")
    logger.info("="*60)

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # ── Phase switch: unfreeze backbone after freeze_epochs ───
        if epoch == freeze_epochs and freeze_epochs > 0 and hasattr(model, "unfreeze_backbone"):
            logger.info(f"\n>>> Epoch {epoch+1}: Unfreezing backbone (phase 2 fine-tuning)")
            model.unfreeze_backbone()
            # Rebuild optimizer with lower backbone LR
            optimizer = optim.AdamW([
                {"params": model.backbone.parameters(), "lr": base_lr * 0.1},
                {"params": model.head.parameters(),     "lr": base_lr},
            ], weight_decay=cfg["training"]["weight_decay"])
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs - epoch, eta_min=1e-7)

        # ── Forward / backward ───────────────────────────────────
        tr_loss, tr_acc, tr_f1, _, _            = run_epoch(model, train_loader, criterion, optimizer, device, True)
        vl_loss, vl_acc, vl_f1, vl_preds, vl_gt = run_epoch(model, val_loader,   criterion, None,      device, False)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Ep [{epoch+1:>3}/{epochs}] "
            f"TrLoss={tr_loss:.4f} TrAcc={tr_acc:.3f} TrF1={tr_f1:.3f} | "
            f"VlLoss={vl_loss:.4f} VlAcc={vl_acc:.3f} VlF1={vl_f1:.3f} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
        )

        # TensorBoard
        for tag, val in [
            ("loss/train", tr_loss), ("loss/val", vl_loss),
            ("acc/train",  tr_acc),  ("acc/val",  vl_acc),
            ("f1/train",   tr_f1),   ("f1/val",   vl_f1),
            ("lr", optimizer.param_groups[0]["lr"]),
        ]:
            writer.add_scalar(tag, val, epoch)

        # ── Checkpointing ────────────────────────────────────────
        state = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss":             vl_loss,
            "val_acc":              vl_acc,
            "val_f1":               vl_f1,
        }

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            es_counter  = 0
            torch.save(state, ckpt_dir / "best_model.pth")
            logger.info(f"  ↳ ✅ Best model saved  val_f1={vl_f1:.4f}")
        else:
            es_counter += 1

        if (epoch + 1) % save_every == 0:
            torch.save(state, ckpt_dir / f"ckpt_ep{epoch+1:03d}.pth")

        if es_counter >= patience:
            logger.info(f"\n⏹  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    # ── Final report ─────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("  TRAINING COMPLETE")
    logger.info(f"  Best Val F1: {best_val_f1:.4f}")
    logger.info("="*60)

    logger.info("\nFinal validation classification report:")
    report = classification_report(vl_gt, vl_preds,
                                   target_names=train_ds.class_names, zero_division=0)
    logger.info(f"\n{report}")
    (ROOT / "results" / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "reports" / "final_train_report.txt").write_text(report)

    writer.close()
    logger.info(f"\nRun TensorBoard:  tensorboard --logdir {tb_dir}")
    logger.info(f"Best checkpoint:  {ckpt_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
