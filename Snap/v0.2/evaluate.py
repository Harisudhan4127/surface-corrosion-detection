"""
evaluate.py — Full evaluation on the test split with metrics, plots, and report
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report

from data.dataset import CorrosionDataset
from utils.metrics import (
    MetricsTracker,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
)
from utils.logger import setup_logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data_dir",    default="data/augmented")
    p.add_argument("--model_type",  default="resnet", choices=["resnet", "custom_cnn"])
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--output_dir",  default="results/reports")
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("evaluate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_ds = CorrosionDataset(root_dir=args.data_dir, split="test", transform=tf)
    loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    logger.info(f"Test samples: {len(test_ds)}")

    # Load model
    if args.model_type == "resnet":
        from models.resnet_model import CorrosionResNet
        model = CorrosionResNet(num_classes=2, pretrained=False)
    else:
        from models.cnn_model import CorrosionCNN
        model = CorrosionCNN(num_classes=2)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = model.to(device).eval()

    tracker = MetricsTracker()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            tracker.update(logits, labels)

    metrics = tracker.compute()
    logger.info(f"\n{json.dumps(metrics, indent=2)}")

    y_true = np.array(tracker._labels)
    y_pred = np.array(tracker._preds)
    y_prob = np.array(tracker._probs)

    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        class_names=test_ds.class_names,
        save_path=out_dir / "confusion_matrix.png",
    )

    # ROC
    try:
        plot_roc_curve(y_true, y_prob, save_path=out_dir / "roc_curve.png")
    except Exception as e:
        logger.warning(f"ROC plot failed: {e}")

    # Full classification report
    report = classification_report(y_true, y_pred, target_names=test_ds.class_names)
    logger.info(f"\n{report}")
    (out_dir / "classification_report.txt").write_text(report)

    # Save metrics JSON
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(f"Evaluation artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
