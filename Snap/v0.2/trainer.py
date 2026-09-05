"""
trainer.py — Trainer with early stopping, checkpointing, and TensorBoard logging
"""

import os
import time
import torch
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.counter = 0
        self.stop = False

    def step(self, value):
        if self.best is None:
            self.best = value
            return
        improved = (value < self.best - self.min_delta) if self.mode == "min" \
                   else (value > self.best + self.min_delta)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, cfg, logger):
        self.model     = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device    = device
        self.cfg       = cfg
        self.logger    = logger

        self.save_dir  = Path(cfg["training"]["checkpoint_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=cfg["logging"]["tensorboard_dir"])
        self.early_stop = EarlyStopping(
            patience=cfg["training"].get("patience", 15),
            mode="min",
        )

    # ------------------------------------------------------------------
    def _run_epoch(self, loader, training: bool):
        self.model.train() if training else self.model.eval()

        total_loss = 0.0
        all_preds, all_labels = [], []

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(images)
                loss   = self.criterion(logits, labels)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        return avg_loss, acc, all_preds, all_labels

    # ------------------------------------------------------------------
    def fit(self, train_loader, val_loader, epochs, start_epoch=0):
        best_val_loss = float("inf")

        for epoch in range(start_epoch, epochs):
            t0 = time.time()

            train_loss, train_acc, _, _ = self._run_epoch(train_loader, training=True)
            val_loss,   val_acc,  preds, labels = self._run_epoch(val_loader, training=False)

            self.scheduler.step()
            self.early_stop.step(val_loss)

            elapsed = time.time() - t0
            self.logger.info(
                f"Epoch [{epoch+1:>3}/{epochs}] "
                f"TrainLoss={train_loss:.4f} TrainAcc={train_acc:.4f} "
                f"ValLoss={val_loss:.4f} ValAcc={val_acc:.4f} "
                f"({elapsed:.1f}s)"
            )

            # TensorBoard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val",   val_loss,   epoch)
            self.writer.add_scalar("Acc/train",  train_acc,  epoch)
            self.writer.add_scalar("Acc/val",    val_acc,    epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            # Checkpoint (best model)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, "best_model.pth")
                self.logger.info(f"  ↳ New best model saved (val_loss={val_loss:.4f})")

            # Periodic checkpoint
            if (epoch + 1) % self.cfg["training"].get("save_every", 5) == 0:
                self._save_checkpoint(epoch, val_loss, f"ckpt_epoch_{epoch+1:03d}.pth")

            if self.early_stop.stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Final classification report
        self.logger.info("\n" + classification_report(labels, preds))
        self.writer.close()

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch, val_loss, filename):
        path = self.save_dir / filename
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss":             val_loss,
        }, path)
