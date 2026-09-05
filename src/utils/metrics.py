"""src/utils/metrics.py"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self._preds  = []
        self._labels = []
        self._probs  = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        import torch.nn.functional as F
        probs  = F.softmax(logits, dim=1).detach().cpu().numpy()
        preds  = logits.argmax(dim=1).detach().cpu().numpy()
        self._probs.extend(probs.tolist())
        self._preds.extend(preds.tolist())
        self._labels.extend(labels.cpu().numpy().tolist())

    def compute(self) -> dict:
        y_true = np.array(self._labels)
        y_pred = np.array(self._preds)
        y_prob = np.array(self._probs)
        metrics = {
            "accuracy":  float(accuracy_score(y_true, y_pred)),
            "macro_f1":  float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        try:
            metrics["macro_auc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except Exception:
            metrics["macro_auc"] = float("nan")
        return metrics
