"""
dataset.py — CorrosionDataset with support for classification + segmentation
"""

import os
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class CorrosionDataset(Dataset):
    """
    Directory structure expected:
        data/
          train/
            corrosion/   *.jpg | *.png
            no_corrosion/
          val/
            corrosion/
            no_corrosion/
          test/
            corrosion/
            no_corrosion/

    Labels are derived automatically from subfolder names.
    Pass a custom label_map to override.
    """

    DEFAULT_CLASSES = ["no_corrosion", "corrosion"]

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        label_map: dict = None,
    ):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.classes = label_map or {c: i for i, c in enumerate(self.DEFAULT_CLASSES)}
        self.samples = self._load_samples()

    # ------------------------------------------------------------------
    def _load_samples(self):
        samples = []
        for cls_name, label in self.classes.items():
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                    samples.append((str(img_path), label))
        return samples

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    # ------------------------------------------------------------------
    @property
    def class_names(self):
        return list(self.classes.keys())

    def class_counts(self):
        from collections import Counter
        counter = Counter(label for _, label in self.samples)
        return {name: counter[idx] for name, idx in self.classes.items()}


# ---------------------------------------------------------------------------
# Utility: generate a synthetic demo dataset (random noise images)
# for quick smoke-testing without real data.
# ---------------------------------------------------------------------------
def create_demo_dataset(root: str, n_per_class: int = 20, img_size: int = 224):
    """Creates a tiny fake dataset so the pipeline runs without real images."""
    import numpy as np
    from PIL import Image

    root = Path(root)
    for split in ["train", "val", "test"]:
        for cls in ["corrosion", "no_corrosion"]:
            folder = root / split / cls
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                arr = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                img.save(folder / f"sample_{i:04d}.jpg")
    print(f"Demo dataset created at {root}")


if __name__ == "__main__":
    create_demo_dataset("data", n_per_class=10)
