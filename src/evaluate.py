#!/usr/bin/env python3
"""
src/evaluate.py
================
Full test-set evaluation with:
  • Per-class precision / recall / F1
  • Macro & weighted averages
  • Confusion matrix (heatmap)
  • ROC curves (One-vs-Rest)
  • Top-k accuracy
  • Saves metrics.json + all plots to results/reports/

Usage:
    python src/evaluate.py --checkpoint models/checkpoints/best_model.pth
    python src/evaluate.py --checkpoint models/checkpoints/best_model.pth --tta
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, f1_score,
)
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.dataset   import NEUSurfaceDataset
from data.transforms import get_val_transform, get_tta_transforms
from models.model   import build_model
from utils.logger   import setup_logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--config",      default="configs/config.yaml")
    p.add_argument("--data_dir",    default=None)
    p.add_argument("--split",       default="test", choices=["val","test"])
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--tta",         action="store_true", help="Test-Time Augmentation")
    p.add_argument("--output_dir",  default="results/reports")
    return p.parse_args()


def load_cfg(path):
    with open(path) as f: return yaml.safe_load(f)


# ── Inference ────────────────────────────────────────────────────────────────
def run_inference(model, loader, device, transforms_list=None):
    """
    If transforms_list is provided (TTA), averages softmax over all transforms.
    Returns (probs [N×C], preds [N], labels [N])
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            if transforms_list:
                # TTA: average probs over augmented views
                batch_probs = []
                for tf in transforms_list:
                    # Re-apply transform on CPU
                    imgs_aug = torch.stack([tf(
                        __import__("torchvision").transforms.functional.to_pil_image(
                            img.cpu())
                    ) for img in images])
                    logits = model(imgs_aug.to(device))
                    batch_probs.append(F.softmax(logits, dim=1).cpu())
                probs = torch.stack(batch_probs).mean(0)
            else:
                logits = model(images.to(device))
                probs  = F.softmax(logits, dim=1).cpu()

            all_probs.append(probs)
            all_labels.extend(labels.numpy())

    all_probs  = torch.cat(all_probs, dim=0).numpy()
    all_labels = np.array(all_labels)
    all_preds  = all_probs.argmax(axis=1)
    return all_probs, all_preds, all_labels


# ── Plots ────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in [
        (axes[0], cm,      "d",     "Confusion Matrix (counts)"),
        (axes[1], cm_norm, ".2f",   "Confusion Matrix (normalised)"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True",      fontsize=11)
        ax.set_title(title,        fontsize=12)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_roc_curves(y_true, y_prob, class_names, out_path):
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors  = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, (cls, col) in enumerate(zip(class_names, colors)):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=col,
                label=f"{cls}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.01])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — One-vs-Rest (per class)", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_per_class_f1(y_true, y_pred, class_names, out_path):
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    colors = ["#e74c3c" if f < 0.85 else "#2ecc71" for f in f1s]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(class_names, f1s, color=colors, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, labels=[f"{f:.3f}" for f in f1s], padding=3, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score")
    ax.tick_params(axis="x", rotation=25)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args    = parse_args()
    logger  = setup_logger("evaluate")
    cfg     = load_cfg(ROOT / args.config)
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Dataset
    split_dir = ROOT / (args.data_dir or cfg["data"]["split_dir"])
    val_tf    = get_val_transform(cfg["data"]["image_size"])
    ds        = NEUSurfaceDataset(split_dir, args.split, transform=val_tf)
    loader    = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    class_names = ds.class_names
    logger.info(f"Evaluating {len(ds)} images ({args.split} split)")

    # Model
    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    sd    = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # TTA transforms
    tta_tfs = get_tta_transforms(cfg["data"]["image_size"]) if args.tta else None

    # Inference
    probs, preds, labels = run_inference(model, loader, device, tta_tfs)

    # ── Metrics ──────────────────────────────────────────────────
    report = classification_report(labels, preds, target_names=class_names, zero_division=0)
    logger.info(f"\n{report}")

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    top1_acc = np.mean(preds == labels)

    # Top-2 accuracy
    top2 = np.sort(probs, axis=1)[:, -2:]
    top2_labels = np.argsort(probs, axis=1)[:, -2:]
    top2_acc = np.mean([labels[i] in top2_labels[i] for i in range(len(labels))])

    metrics = {
        "top1_accuracy": round(float(top1_acc), 4),
        "top2_accuracy": round(float(top2_acc), 4),
        "macro_f1":      round(float(macro_f1), 4),
        "per_class_f1": {
            cls: round(float(f), 4)
            for cls, f in zip(class_names,
                              f1_score(labels, preds, average=None, zero_division=0))
        },
    }
    logger.info(f"\nMetrics:\n{json.dumps(metrics, indent=2)}")

    # ── Save outputs ─────────────────────────────────────────────
    (out_dir / "classification_report.txt").write_text(report)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_confusion_matrix(labels, preds, class_names, out_dir / "confusion_matrix.png")
    plot_roc_curves(labels, probs, class_names, out_dir / "roc_curves.png")
    plot_per_class_f1(labels, preds, class_names, out_dir / "per_class_f1.png")

    logger.info(f"\nAll evaluation outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
