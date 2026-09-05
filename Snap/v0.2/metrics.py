"""
metrics.py — Evaluation metrics, confusion matrix, and reporting utilities
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
)


class MetricsTracker:
    """Accumulates predictions over an epoch and computes metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._preds  = []
        self._labels = []
        self._probs  = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        probs  = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds  = logits.argmax(dim=1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        self._probs.extend(probs.tolist())
        self._preds.extend(preds.tolist())
        self._labels.extend(labels.tolist())

    def compute(self):
        y_true = np.array(self._labels)
        y_pred = np.array(self._preds)
        y_prob = np.array(self._probs)

        metrics = {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
        }
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = float("nan")

        return metrics


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix → {save_path}")
    return fig


def plot_roc_curve(y_true, y_prob, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label="Train")
    axes[0].plot(val_losses,   label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(train_accs, label="Train")
    axes[1].plot(val_accs,   label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig
