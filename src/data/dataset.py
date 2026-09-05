"""
src/data/dataset.py
====================
NEU Surface Defect Dataset — PyTorch Dataset with Albumentations augmentation.

Classes (6):
  0 - Crazing          (Cr)
  1 - Inclusion        (In)
  2 - Patches          (Pa)
  3 - Pitted_surface   (Ps)
  4 - Rolled-in_scale  (Rs)
  5 - Scratches        (Sc)

Directory layout expected:
  data/splits/
    train/
      Crazing/         *.jpg | *.png (grayscale 200×200, stored as RGB)
      Inclusion/
      Patches/
      Pitted_surface/
      Rolled-in_scale/
      Scratches/
    val/  ...
    test/ ...
"""

import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


# ── Official NEU class definitions ─────────────────────────────────────────
NEU_CLASSES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted_surface",
    "Rolled-in_scale",
    "Scratches",
]
NEU_CLASS_TO_IDX = {c: i for i, c in enumerate(NEU_CLASSES)}

# Short abbreviation lookup (for display)
NEU_ABBREV = {
    "Crazing": "Cr",
    "Inclusion": "In",
    "Patches": "Pa",
    "Pitted_surface": "Ps",
    "Rolled-in_scale": "Rs",
    "Scratches": "Sc",
}


class NEUSurfaceDataset(Dataset):
    """
    Parameters
    ----------
    root_dir  : str   Path to data/splits/ (contains train/val/test subfolders)
    split     : str   'train' | 'val' | 'test'
    transform :       torchvision transforms or albumentations Compose
    use_albumentations : bool
                      If True, expects transform to be albumentations.Compose
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        use_albumentations: bool = False,
    ):
        self.root = Path(root_dir) / split
        self.transform = transform
        self.use_albumentations = use_albumentations
        self.classes = NEU_CLASSES
        self.class_to_idx = NEU_CLASS_TO_IDX
        self.samples = self._scan()

    # ── scan ─────────────────────────────────────────────────────
    def _scan(self):
        samples = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                # try uppercase / lowercase fallback
                for d in self.root.iterdir():
                    if d.name.lower() == cls_name.lower() and d.is_dir():
                        cls_dir = d
                        break
            if not cls_dir.exists():
                continue
            label = self.class_to_idx[cls_name]
            for p in cls_dir.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    samples.append((str(p), label))
        return samples

    # ── dunder ───────────────────────────────────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # NEU images are grayscale — convert to RGB for pretrained CNNs
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            if self.use_albumentations:
                arr = np.array(image)
                augmented = self.transform(image=arr)
                image = augmented["image"]  # already a tensor
            else:
                image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    # ── helpers ──────────────────────────────────────────────────
    def class_counts(self) -> dict:
        from collections import Counter
        cnt = Counter(lbl for _, lbl in self.samples)
        return {self.classes[i]: cnt[i] for i in range(len(self.classes))}

    def compute_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for CrossEntropyLoss."""
        counts = self.class_counts()
        total = sum(counts.values())
        weights = [total / (len(self.classes) * counts[c]) for c in self.classes]
        return torch.tensor(weights, dtype=torch.float32)

    @property
    def class_names(self):
        return self.classes
