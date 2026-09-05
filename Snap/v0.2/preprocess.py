"""
preprocess.py — Data preprocessing, augmentation helpers, and split utilities
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np


# ---------------------------------------------------------------------------
# Resize and normalise a raw dataset folder
# ---------------------------------------------------------------------------
def preprocess_raw_images(
    raw_dir: str,
    out_dir: str,
    size: int = 224,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
):
    """
    Walk raw_dir, resize every image to (size × size), convert to RGB,
    and save to out_dir preserving the sub-folder class structure.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    processed = 0
    skipped = 0

    for img_path in raw_dir.rglob("*"):
        if img_path.suffix.lower() not in extensions:
            continue
        rel = img_path.relative_to(raw_dir)
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((size, size), Image.LANCZOS)
            img.save(dest)
            processed += 1
        except Exception as e:
            print(f"[WARN] Skipping {img_path}: {e}")
            skipped += 1

    print(f"Preprocessed {processed} images ({skipped} skipped) → {out_dir}")


# ---------------------------------------------------------------------------
# Train / Val / Test split
# ---------------------------------------------------------------------------
def split_dataset(
    src_dir: str,
    dest_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Split a flat class-folder dataset into train/val/test sub-splits.
    src_dir layout:   src_dir/<class_name>/<images>
    dest_dir layout:  dest_dir/{train,val,test}/<class_name>/<images>
    """
    random.seed(seed)
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)

    for cls_dir in src_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        images = [p for p in cls_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:],
        }

        for split, files in splits.items():
            out = dest_dir / split / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, out / f.name)

        print(f"  {cls_dir.name}: {n_train} train | {n_val} val | {n - n_train - n_val} test")

    print(f"Split complete → {dest_dir}")


# ---------------------------------------------------------------------------
# Basic augmentation helpers (offline augmentation)
# ---------------------------------------------------------------------------
def augment_class_folder(
    cls_folder: str,
    target_count: int,
    seed: int = 42,
):
    """
    Offline augmentation: duplicates + augments images in cls_folder
    until target_count images exist. Useful for class imbalance.
    """
    random.seed(seed)
    folder = Path(cls_folder)
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

    if len(images) >= target_count:
        print(f"[{folder.name}] Already has {len(images)} ≥ {target_count}, skipping.")
        return

    aug_idx = 0
    while len(list(folder.glob("*.jpg"))) + len(list(folder.glob("*.png"))) < target_count:
        src = random.choice(images)
        img = Image.open(src).convert("RGB")

        # Random augmentation pipeline
        ops = random.sample(["flip_h", "flip_v", "rotate", "blur", "noise"], k=random.randint(1, 3))
        for op in ops:
            if op == "flip_h":
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif op == "flip_v":
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            elif op == "rotate":
                img = img.rotate(random.choice([90, 180, 270]))
            elif op == "blur":
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
            elif op == "noise":
                arr = np.array(img, dtype=np.float32)
                arr += np.random.normal(0, 10, arr.shape)
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)

        out_name = folder / f"aug_{aug_idx:06d}.jpg"
        img.save(out_name)
        aug_idx += 1

    print(f"[{folder.name}] Augmented to {aug_idx} new images.")


if __name__ == "__main__":
    # Quick demo
    preprocess_raw_images("data/raw", "data/processed", size=224)
    split_dataset("data/processed", "data/augmented")
