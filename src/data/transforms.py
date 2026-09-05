"""
src/data/transforms.py
=======================
Torchvision + Albumentations transforms for the NEU steel defect dataset.
NEU images are 200×200 grayscale; we convert to RGB and resize to 224×224.
"""

import torch
from torchvision import transforms

# ImageNet normalisation (used even for grayscale-to-RGB images)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 224):
    """Heavy augmentation for training — critical for 300 images/class."""
    return transforms.Compose([
        transforms.Resize((image_size + 16, image_size + 16)),   # slight oversize
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.35,
            contrast=0.35,
            saturation=0.20,
            hue=0.05,
        ),
        transforms.RandomGrayscale(p=0.10),         # keep some grayscale contrast
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.10)),   # occlusion sim
    ])


def get_val_transform(image_size: int = 224):
    """Deterministic transform for val / test."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_tta_transforms(image_size: int = 224, n_augments: int = 5):
    """
    Test-Time Augmentation: returns a list of N val-like transforms
    with random flips/rotations for ensemble prediction.
    """
    tfs = [get_val_transform(image_size)]
    for _ in range(n_augments - 1):
        tfs.append(transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]))
    return tfs


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation for visualisation."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)
