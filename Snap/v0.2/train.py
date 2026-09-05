"""
train.py — Main training script for corrosion detection CNN
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

from data.dataset import CorrosionDataset
from models.cnn_model import CorrosionCNN
from models.resnet_model import CorrosionResNet
from utils.trainer import Trainer
from utils.metrics import MetricsTracker
from utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Train Corrosion Detection CNN")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--model", type=str, default="resnet",
                        choices=["custom_cnn", "resnet"],
                        help="Model architecture to use")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from config")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate from config")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_transforms(cfg):
    train_tf = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.epochs:    cfg["training"]["epochs"] = args.epochs
    if args.batch_size: cfg["training"]["batch_size"] = args.batch_size
    if args.lr:        cfg["training"]["lr"] = args.lr

    logger = setup_logger("train", cfg["logging"]["log_dir"])
    logger.info(f"Config: {cfg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Transforms
    train_tf, val_tf = build_transforms(cfg["data"])

    # Datasets & loaders
    train_ds = CorrosionDataset(
        root_dir=cfg["data"]["data_dir"],
        split="train",
        transform=train_tf,
    )
    val_ds = CorrosionDataset(
        root_dir=cfg["data"]["data_dir"],
        split="val",
        transform=val_tf,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Model
    num_classes = cfg["data"]["num_classes"]
    if args.model == "custom_cnn":
        model = CorrosionCNN(num_classes=num_classes)
    else:
        model = CorrosionResNet(
            num_classes=num_classes,
            pretrained=cfg["model"]["pretrained"],
        )
    model = model.to(device)
    logger.info(f"Model: {model.__class__.__name__} | Params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg["training"].get("class_weights", [1.0] * num_classes),
                            dtype=torch.float32).to(device)
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=1e-6,
    )

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    # Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        cfg=cfg,
        logger=logger,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["training"]["epochs"],
        start_epoch=start_epoch,
    )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
