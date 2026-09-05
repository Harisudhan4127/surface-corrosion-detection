#!/usr/bin/env python3
"""
scripts/download_dataset.py
============================
Downloads the NEU Surface Defect Database from Kaggle and
splits it into train / val / test sets.

Prerequisites
-------------
1. Install kaggle API:  pip install kaggle
2. Get your API token from https://www.kaggle.com/settings
   → "Create New Token" → downloads kaggle.json
3. Place kaggle.json at:
       Linux/macOS : ~/.kaggle/kaggle.json
       Windows     : C:\\Users\\<user>\\.kaggle\\kaggle.json
4. chmod 600 ~/.kaggle/kaggle.json  (Linux/macOS only)

Usage
-----
    python scripts/download_dataset.py
    python scripts/download_dataset.py --skip_download   # if zip already exists
"""

import os
import sys
import shutil
import random
import argparse
import zipfile
from pathlib import Path

# ── NEU dataset label map ──────────────────────────────────────────────────
# The Kaggle zip folder names → our class names
FOLDER_TO_CLASS = {
    "Cr": "Crazing",
    "In": "Inclusion",
    "Pa": "Patches",
    "Ps": "Pitted_surface",
    "Rs": "Rolled-in_scale",
    "Sc": "Scratches",
}

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="Download & prepare NEU dataset")
    p.add_argument("--skip_download", action="store_true",
                   help="Skip Kaggle download (use existing zip)")
    p.add_argument("--zip_path", type=str, default=None,
                   help="Path to manually downloaded zip (skip Kaggle API)")
    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio",   type=float, default=0.15)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


# ── Step 1: Download ────────────────────────────────────────────────────────
def download_from_kaggle(dest_dir: Path):
    print("\n📥  Downloading NEU Surface Defect Database from Kaggle...")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "kaustubhdikshit/neu-surface-defect-database",
            path=str(dest_dir),
            unzip=False,
        )
        zips = list(dest_dir.glob("*.zip"))
        if not zips:
            raise FileNotFoundError("Kaggle download produced no zip file.")
        print(f"    ✓ Downloaded: {zips[0]}")
        return zips[0]
    except Exception as e:
        print(f"\n❌  Kaggle download failed: {e}")
        print(
            "\nManual download instructions:\n"
            "  1. Visit https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database\n"
            "  2. Click 'Download' and save the zip\n"
            "  3. Re-run:  python scripts/download_dataset.py --zip_path /path/to/file.zip\n"
        )
        sys.exit(1)


# ── Step 2: Extract ─────────────────────────────────────────────────────────
def extract_zip(zip_path: Path, dest: Path):
    print(f"\n📦  Extracting {zip_path.name} → {dest} ...")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    print("    ✓ Extraction complete")

    # Show what was extracted
    for item in sorted(dest.iterdir()):
        print(f"       {item.name}/")


# ── Step 3: Discover class folders ──────────────────────────────────────────
def discover_class_folders(raw_dir: Path):
    """
    The Kaggle zip may nest the images differently depending on version.
    Searches recursively for folders matching known class abbreviations.
    Returns dict: class_name → folder Path
    """
    found = {}
    for folder in raw_dir.rglob("*"):
        if folder.is_dir() and folder.name in FOLDER_TO_CLASS:
            cls = FOLDER_TO_CLASS[folder.name]
            found[cls] = folder
    # Also handle if class names are already full (e.g. "Crazing")
    inv = {v: k for k, v in FOLDER_TO_CLASS.items()}
    for folder in raw_dir.rglob("*"):
        if folder.is_dir() and folder.name in inv:
            found[folder.name] = folder
    return found


# ── Step 4: Split ───────────────────────────────────────────────────────────
def split_dataset(class_folders: dict, split_dir: Path,
                  train_ratio: float, val_ratio: float, seed: int):
    print(f"\n✂️   Splitting dataset ({train_ratio:.0%} train / "
          f"{val_ratio:.0%} val / {1-train_ratio-val_ratio:.0%} test) ...")
    random.seed(seed)

    total_copied = 0
    summary = []

    for cls_name, src_folder in class_folders.items():
        images = sorted([
            p for p in src_folder.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ])
        random.shuffle(images)

        n       = len(images)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        n_test  = n - n_train - n_val

        splits = {
            "train": images[:n_train],
            "val":   images[n_train : n_train + n_val],
            "test":  images[n_train + n_val :],
        }

        for split_name, files in splits.items():
            out = split_dir / split_name / cls_name
            out.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, out / f.name)

        total_copied += n
        summary.append(f"    {cls_name:<20} total={n:>3}  "
                        f"train={n_train}  val={n_val}  test={n_test}")

    print("\n".join(summary))
    print(f"\n    ✓ Total images split: {total_copied}")
    print(f"    ✓ Split dataset ready at: {split_dir}")


# ── Step 5: Verify ──────────────────────────────────────────────────────────
def verify_split(split_dir: Path):
    print("\n🔍  Verifying split directory...")
    for split in ["train", "val", "test"]:
        split_path = split_dir / split
        if not split_path.exists():
            print(f"    ✗ Missing: {split_path}")
            continue
        classes = sorted(p.name for p in split_path.iterdir() if p.is_dir())
        counts = {c: len(list((split_path / c).glob("*.*"))) for c in classes}
        total = sum(counts.values())
        print(f"    {split:<6}: {total:>4} images | "
              + " | ".join(f"{c}={n}" for c, n in counts.items()))
    print("    ✓ Verification complete")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    raw_dir   = ROOT / "data" / "raw"
    split_dir = ROOT / "data" / "splits"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download or use provided zip
    if args.zip_path:
        zip_path = Path(args.zip_path)
        extract_zip(zip_path, raw_dir)
    elif not args.skip_download:
        zip_path = download_from_kaggle(raw_dir)
        extract_zip(zip_path, raw_dir)
    else:
        print("⏭️   Skipping download — using existing data/raw/ contents")

    # Find class folders
    class_folders = discover_class_folders(raw_dir)
    if not class_folders:
        print(
            "\n❌  No class folders found in data/raw/\n"
            "    Expected folders named: Cr, In, Pa, Ps, Rs, Sc\n"
            "    Please check the extracted contents of the zip."
        )
        sys.exit(1)

    print(f"\n    Found {len(class_folders)} class folders: "
          + ", ".join(class_folders.keys()))

    # Split
    split_dataset(class_folders, split_dir, args.train_ratio, args.val_ratio, args.seed)
    verify_split(split_dir)

    print("\n🎉  Dataset ready! Run training with:")
    print("        python src/train.py")


if __name__ == "__main__":
    main()
