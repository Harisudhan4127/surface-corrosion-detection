"""
predict.py — Run inference on a single image or a folder of images
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# Grad-CAM implementation
# ---------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:],
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

CLASS_NAMES = ["no_corrosion", "corrosion"]


def load_model(ckpt_path: str, model_type: str = "resnet", num_classes: int = 2):
    if model_type == "resnet":
        from models.resnet_model import CorrosionResNet
        model = CorrosionResNet(num_classes=num_classes, pretrained=False)
    else:
        from models.cnn_model import CorrosionCNN
        model = CorrosionCNN(num_classes=num_classes)

    state = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def predict_single(model, img_path: str, device, gradcam=None):
    img = Image.open(img_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx  = int(probs.argmax())
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    result = {
        "image":      str(img_path),
        "prediction": pred_label,
        "confidence": round(confidence, 4),
        "probabilities": {
            CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)
        },
    }

    cam_overlay = None
    if gradcam is not None:
        cam = gradcam.generate(tensor, class_idx=pred_idx)
        rgb = np.array(img.resize((224, 224))) / 255.0
        heatmap = cm.jet(cam)[:, :, :3]
        cam_overlay = (0.5 * rgb + 0.5 * heatmap)
        cam_overlay = np.clip(cam_overlay, 0, 1)

    return result, cam_overlay


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Corrosion Detection Inference")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p.add_argument("--input",      required=True, help="Image file or folder")
    p.add_argument("--model_type", default="resnet", choices=["resnet", "custom_cnn"])
    p.add_argument("--output_dir", default="results/predictions")
    p.add_argument("--gradcam",    action="store_true", help="Generate Grad-CAM visualisations")
    p.add_argument("--save_json",  action="store_true", help="Save results as JSON")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, args.model_type).to(device)

    gradcam = None
    if args.gradcam:
        # Attach to last conv layer of backbone
        if hasattr(model, "backbone"):
            target = list(model.backbone.children())[-1]
        else:
            target = list(model.features.children())[-1]
        gradcam = GradCAM(model, target)

    input_path = Path(args.input)
    paths = (
        [input_path]
        if input_path.is_file()
        else list(input_path.glob("**/*.jpg")) + list(input_path.glob("**/*.png"))
    )

    all_results = []
    for img_path in paths:
        result, overlay = predict_single(model, str(img_path), device, gradcam)
        all_results.append(result)
        print(f"[{result['prediction'].upper():>12}] {result['confidence']:.1%}  {img_path.name}")

        if overlay is not None:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            orig = Image.open(img_path).resize((224, 224))
            axes[0].imshow(orig); axes[0].set_title("Original"); axes[0].axis("off")
            axes[1].imshow(overlay); axes[1].set_title(f"Grad-CAM ({result['prediction']})"); axes[1].axis("off")
            plt.tight_layout()
            plt.savefig(out_dir / f"gradcam_{img_path.stem}.png", dpi=150)
            plt.close()

    if args.save_json:
        json_path = out_dir / "predictions.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
