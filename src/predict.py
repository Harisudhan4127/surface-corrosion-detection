#!/usr/bin/env python3
"""
src/predict.py
===============
Inference on a single image or folder — with Grad-CAM explainability.

Usage:
    # Single image
    python src/predict.py --checkpoint models/checkpoints/ckpt_ep010.pth \
                          --input data/splits/test/Crazing/crazing_10.jpg --gradcam

    # Folder
    python src/predict.py --checkpoint models/checkpoints/best_model.pth \
                          --input data/splits/test/ --gradcam --save_json
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.transforms import get_val_transform, denormalize
from models.model    import build_model
from utils.logger    import setup_logger


CLASS_NAMES = [
    "Crazing", "Inclusion", "Patches",
    "Pitted_surface", "Rolled-in_scale", "Scratches"
]

# Colour per class for overlay titles
CLASS_COLORS = {
    "Crazing":          "#e74c3c",
    "Inclusion":        "#e67e22",
    "Patches":          "#f1c40f",
    "Pitted_surface":   "#2ecc71",
    "Rolled-in_scale":  "#3498db",
    "Scratches":        "#9b59b6",
}


# ── Grad-CAM ─────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self._grads      = None
        self._acts       = None
        target_layer.register_forward_hook(self._save_acts)
        target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _, __, out):  self._acts  = out.detach()
    def _save_grads(self, _, gi, go): self._grads = go[0].detach()

    def generate(self, inp, class_idx=None):
        self.model.eval()
        out = self.model(inp)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(out)
        one_hot[0, class_idx] = 1.0
        out.backward(gradient=one_hot)

        w   = self._grads.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((w * self._acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=inp.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def get_gradcam_layer(model):
    """Return the last conv layer suitable for Grad-CAM."""
    # ResNet-50: last layer of layer4
    if hasattr(model, "backbone"):
        children = list(model.backbone.children())
        for child in reversed(children):
            if hasattr(child, "children"):
                subchildren = list(child.children())
                if subchildren:
                    return subchildren[-1]
    # Custom CNN
    if hasattr(model, "features"):
        return list(model.features.children())[-1]
    raise RuntimeError("Cannot auto-detect target layer for Grad-CAM")


# ── Inference helpers ─────────────────────────────────────────────────────────
def load_model(ckpt_path, cfg, device):
    model = build_model(cfg).to(device)
    import numpy._core.multiarray
    torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd    = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval()
    return model

def predict_image(model, img_path, transform, device, gradcam=None):
    img     = Image.open(img_path).convert("RGB")
    tensor  = transform(img).unsqueeze(0).to(device)

    if gradcam is not None:
        cam, pred_idx = gradcam.generate(tensor)
        probs = F.softmax(model(tensor), dim=1).squeeze().detach().cpu().numpy()
    else:
        with torch.no_grad():
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = int(probs.argmax())
        cam = None

    result = {
        "image":       str(img_path),
        "prediction":  CLASS_NAMES[pred_idx],
        "confidence":  round(float(probs[pred_idx]), 4),
        "probabilities": {cls: round(float(p), 4)
                          for cls, p in zip(CLASS_NAMES, probs)},
    }
    return result, cam, np.array(img.resize((224, 224)))


def save_gradcam_figure(orig_rgb, cam, result, save_path):
    """Save a 3-panel Grad-CAM figure: original | heatmap | overlay."""
    heatmap = mpl_cm.jet(cam)[:, :, :3]
    overlay = 0.5 * (orig_rgb / 255.0) + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Original", "Grad-CAM Heatmap", "Overlay"]
    imgs   = [orig_rgb / 255.0, cam, overlay]
    cmaps  = [None, "jet", None]

    for ax, title, im, cmap in zip(axes, titles, imgs, cmaps):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    pred  = result["prediction"]
    conf  = result["confidence"]
    color = CLASS_COLORS.get(pred, "black")
    fig.suptitle(f"Prediction: {pred}  ({conf:.1%})", fontsize=13,
                 color=color, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input",      required=True, help="Image file or folder")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--output_dir", default="results/predictions")
    p.add_argument("--gradcam",    action="store_true")
    p.add_argument("--save_json",  action="store_true")
    p.add_argument("--topk",       type=int, default=3,
                   help="Show top-k class probabilities")
    return p.parse_args()


def main():
    args   = parse_args()
    logger = setup_logger("predict")

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_val_transform(cfg["data"]["image_size"])
    out_dir   = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(ROOT / args.checkpoint, cfg, device)

    gradcam = None
    if args.gradcam:
        try:
            layer   = get_gradcam_layer(model)
            gradcam = GradCAM(model, layer)
            logger.info("Grad-CAM enabled")
        except RuntimeError as e:
            logger.warning(f"Grad-CAM disabled: {e}")

    input_path = Path(args.input)
    exts       = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    paths      = ([input_path] if input_path.is_file()
                  else [p for p in input_path.rglob("*") if p.suffix.lower() in exts])

    all_results = []
    print(f"\n{'IMAGE':<40} {'PREDICTION':<20} {'CONF':>6}  TOP-{args.topk}")
    print("-" * 80)

    for img_path in sorted(paths):
        result, cam, orig_rgb = predict_image(model, img_path, transform, device, gradcam)
        all_results.append(result)

        # Top-k display
        sorted_probs = sorted(result["probabilities"].items(), key=lambda x: -x[1])
        topk_str     = "  ".join(f"{c}:{p:.2f}" for c, p in sorted_probs[:args.topk])
        print(f"{img_path.name:<40} {result['prediction']:<20} {result['confidence']:>6.1%}  {topk_str}")

        if gradcam and cam is not None:
            save_gradcam_figure(
                orig_rgb, cam, result,
                out_dir / f"gradcam_{img_path.stem}.png",
            )

    print("-" * 80)
    print(f"Total: {len(all_results)} images")

    if args.save_json:
        p = out_dir / "predictions.json"
        p.write_text(json.dumps(all_results, indent=2))
        print(f"\nSaved JSON → {p}")

    if gradcam:
        print(f"Grad-CAM images → {out_dir}/")


if __name__ == "__main__":
    main()
