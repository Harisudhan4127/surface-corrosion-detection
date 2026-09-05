#!/usr/bin/env python3
"""
setup_project.py
================
One-shot setup script:
  1. Creates all required directories
  2. Downloads / verifies pretrained ResNet-50 ImageNet weights via torchvision
  3. Generates a synthetic demo dataset (no real images needed to run smoke test)
  4. Exports a baseline ONNX model
  5. Prints a readiness summary

Run:
    python setup_project.py
    python setup_project.py --skip_demo   # skip synthetic data generation
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


# ── 1. Ensure project root is on path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip_demo", action="store_true",
                   help="Skip synthetic demo dataset generation")
    p.add_argument("--export_onnx", action="store_true",
                   help="Export model to ONNX format after setup")
    return p.parse_args()


# ── 2. Directory scaffold ────────────────────────────────────────────────────
DIRS = [
    "data/raw",
    "data/processed",
    "data/augmented/train/corrosion",
    "data/augmented/train/no_corrosion",
    "data/augmented/val/corrosion",
    "data/augmented/val/no_corrosion",
    "data/augmented/test/corrosion",
    "data/augmented/test/no_corrosion",
    "models/saved",
    "models/checkpoints",
    "logs/tensorboard",
    "results/predictions",
    "results/reports",
    "notebooks",
    "tests",
    "docs",
]

def create_dirs():
    print("\n📁  Creating directory structure...")
    for d in DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    print("    ✓ All directories ready.")


# ── 3. Download pretrained ResNet-50 weights ─────────────────────────────────
def download_pretrained():
    print("\n⬇️   Loading pretrained ResNet-50 (ImageNet) weights via torchvision...")
    try:
        import torch
        from torchvision import models
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        print(f"    ✓ ResNet-50 loaded  — params: {sum(p.numel() for p in model.parameters()):,}")

        # Save backbone state dict for reference
        save_path = ROOT / "models" / "saved" / "resnet50_imagenet_backbone.pth"
        torch.save(model.state_dict(), save_path)
        print(f"    ✓ Backbone weights saved → {save_path}")
        return True
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return False


# ── 4. Synthetic demo dataset ────────────────────────────────────────────────
def create_demo_data(n_per_class: int = 30, img_size: int = 224):
    print(f"\n🖼️   Generating synthetic demo dataset ({n_per_class} images / class / split)...")
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFilter
        import random

        random.seed(42)
        np.random.seed(42)

        splits = {"train": n_per_class, "val": max(8, n_per_class // 4), "test": max(8, n_per_class // 4)}

        for split, count in splits.items():
            for cls in ["corrosion", "no_corrosion"]:
                folder = ROOT / "data" / "augmented" / split / cls
                folder.mkdir(parents=True, exist_ok=True)

                for i in range(count):
                    arr = np.random.randint(80, 200, (img_size, img_size, 3), dtype=np.uint8)

                    if cls == "corrosion":
                        # Simulate rust patches: orange-brown blotches
                        for _ in range(random.randint(3, 8)):
                            cx = random.randint(20, img_size - 20)
                            cy = random.randint(20, img_size - 20)
                            r  = random.randint(10, 40)
                            for dy in range(-r, r):
                                for dx in range(-r, r):
                                    if dx*dx + dy*dy < r*r:
                                        px, py = cx+dx, cy+dy
                                        if 0 <= px < img_size and 0 <= py < img_size:
                                            arr[py, px] = [
                                                random.randint(160, 200),
                                                random.randint(60,  100),
                                                random.randint(10,   40),
                                            ]
                    img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(1))
                    img.save(folder / f"{cls}_{i:04d}.jpg", quality=90)

        print(f"    ✓ Demo dataset created in data/augmented/")
        # Count
        total = sum(
            len(list((ROOT / "data" / "augmented" / s / c).glob("*.jpg")))
            for s in splits for c in ["corrosion", "no_corrosion"]
        )
        print(f"    ✓ Total images: {total}")
        return True
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return False


# ── 5. ONNX export ───────────────────────────────────────────────────────────
def export_onnx():
    print("\n📦  Exporting model to ONNX...")
    try:
        import torch
        from models.resnet_model import CorrosionResNet

        model = CorrosionResNet(num_classes=2, pretrained=False).eval()
        dummy = torch.randn(1, 3, 224, 224)
        out_path = str(ROOT / "models" / "saved" / "corrosion_detector.onnx")

        torch.onnx.export(
            model, dummy, out_path,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"    ✓ ONNX model saved → {out_path}")
        return True
    except Exception as e:
        print(f"    ✗ ONNX export failed: {e}")
        return False


# ── 6. Smoke-test the training pipeline ──────────────────────────────────────
def smoke_test():
    print("\n🔥  Running smoke test (2 epochs, tiny batch)...")
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        from torchvision import transforms

        from data.dataset import CorrosionDataset
        from models.resnet_model import CorrosionResNet

        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        ds = CorrosionDataset(
            root_dir=str(ROOT / "data" / "augmented"),
            split="train",
            transform=tf,
        )
        if len(ds) == 0:
            print("    ✗ No training images found — run with demo data first.")
            return False

        loader = DataLoader(ds, batch_size=4, shuffle=True)
        device = torch.device("cpu")
        model  = CorrosionResNet(num_classes=2, pretrained=False).to(device)
        opt    = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(2):
            imgs, labels = next(iter(loader))
            opt.zero_grad()
            loss = criterion(model(imgs.to(device)), labels.to(device))
            loss.backward()
            opt.step()
            print(f"    Epoch {epoch+1}/2  loss={loss.item():.4f}")

        print("    ✓ Smoke test passed!")
        return True
    except Exception as e:
        print(f"    ✗ Smoke test failed: {e}")
        return False


# ── 7. Summary ───────────────────────────────────────────────────────────────
def print_summary(results: dict):
    print("\n" + "="*60)
    print("  SETUP SUMMARY")
    print("="*60)
    for step, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {step}")
    print("="*60)
    if all(results.values()):
        print("\n🚀  Project is ready! Next steps:")
        print("    1. Add real images to  data/raw/  (corrosion / no_corrosion)")
        print("    2. python src/data/preprocess.py   # resize + split")
        print("    3. python src/train.py --model resnet")
        print("    4. python src/evaluate.py --checkpoint models/checkpoints/best_model.pth")
        print("    5. python src/predict.py  --checkpoint models/checkpoints/best_model.pth \\")
        print("                              --input path/to/image.jpg --gradcam")
    else:
        print("\n⚠️   Some steps failed. Check errors above and re-run.")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    print("="*60)
    print("  Corrosion Detection CNN — Project Setup")
    print("="*60)

    results = {}
    create_dirs()
    results["Directories created"]         = True
    results["Pretrained weights download"] = download_pretrained()

    if not args.skip_demo:
        results["Demo dataset generation"]    = create_demo_data()
        results["Smoke test (2-epoch train)"] = smoke_test()

    if args.export_onnx:
        results["ONNX export"] = export_onnx()

    print_summary(results)
