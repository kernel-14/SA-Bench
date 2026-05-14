"""
Training and evaluation utilities for MoE-POT.

Implements:
- Pre-training with auto-regressive denoising objective
- Fine-tuning with frozen router-gating network
- L2 relative error (L2RE) evaluation metric
- One-cycle learning rate schedule
- Multi-GPU training support via DataParallel/DistributedDataParallel
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from .model import MoEPOT


def l2_relative_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 relative error (L2RE).

    L2RE = ||pred - target||_2 / ||target||_2

    Args:
        pred: Predicted tensor (B, C, H, W).
        target: Ground truth tensor (B, C, H, W).

    Returns:
        Mean L2RE over the batch.
    """
    # Flatten spatial and channel dimensions
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    numerator = torch.norm(pred_flat - target_flat, dim=-1)
    denominator = torch.norm(target_flat, dim=-1) + 1e-8

    return (numerator / denominator).mean()


class MoEPOTTrainer:
    """
    Trainer for MoE-POT pre-training and fine-tuning.

    Args:
        model: MoEPOT model instance.
        device: Training device.
        learning_rate: Initial learning rate (default 1e-3).
        weight_decay: Adam weight decay (default 1e-6).
        betas: Adam momentum parameters (default (0.9, 0.9)).
        noise_scale: Noise injection scale for pre-training (default 0.01).
        use_amp: Whether to use automatic mixed precision.
    """

    def __init__(
        self,
        model: MoEPOT,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        betas: Tuple[float, float] = (0.9, 0.9),
        noise_scale: float = 0.01,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.noise_scale = noise_scale
        self.use_amp = use_amp and device.type == "cuda"

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    def setup_scheduler(
        self,
        num_epochs: int,
        steps_per_epoch: int,
        warmup_epochs: int = 200,
    ):
        """
        Set up One-cycle learning rate schedule.

        Args:
            num_epochs: Total number of training epochs.
            steps_per_epoch: Number of steps per epoch.
            warmup_epochs: Number of warmup epochs.
        """
        total_steps = num_epochs * steps_per_epoch
        pct_start = warmup_epochs / num_epochs

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.param_groups[0]["lr"],
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
        )

    def freeze_router(self):
        """
        Freeze router-gating network parameters for fine-tuning.

        During fine-tuning, the router is frozen to preserve the expert
        assignment strategy learned during pre-training.
        """
        for block in self.model.blocks:
            for param in block.moe_layer.router.parameters():
                param.requires_grad = False
        print("Router-gating network frozen for fine-tuning.")

    def unfreeze_router(self):
        """Unfreeze router-gating network parameters."""
        for block in self.model.blocks:
            for param in block.moe_layer.router.parameters():
                param.requires_grad = True

    def train_epoch(
        self,
        dataloader: DataLoader,
        inject_noise: bool = True,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader.
            inject_noise: Whether to inject noise (True for pre-training).

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_balance_loss = 0.0
        total_l2re = 0.0
        num_batches = 0

        for batch in dataloader:
            u_input = batch["input"].to(self.device)   # (B, T, C, H, W)
            u_target = batch["target"].to(self.device)  # (B, C, H, W)

            self.optimizer.zero_grad()

            noise_scale = self.noise_scale if inject_noise else 0.0

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    total_loss_batch, pred_loss, balance_loss = self.model.compute_loss(
                        u_input, u_target, noise_scale=noise_scale
                    )
                self.scaler.scale(total_loss_batch).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss_batch, pred_loss, balance_loss = self.model.compute_loss(
                    u_input, u_target, noise_scale=noise_scale
                )
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            if hasattr(self, "scheduler"):
                self.scheduler.step()

            # Compute L2RE for monitoring
            with torch.no_grad():
                pred, _ = self.model(u_input, noise_scale=0.0)
                l2re = l2_relative_error(pred, u_target)

            total_loss += total_loss_batch.item()
            total_pred_loss += pred_loss.item()
            total_balance_loss += balance_loss.item()
            total_l2re += l2re.item()
            num_batches += 1
            self.global_step += 1

        return {
            "loss": total_loss / num_batches,
            "pred_loss": total_pred_loss / num_batches,
            "balance_loss": total_balance_loss / num_batches,
            "l2re": total_l2re / num_batches,
        }

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        rollout_steps: int = 1,
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.

        Args:
            dataloader: Evaluation data loader.
            rollout_steps: Number of auto-regressive rollout steps.

        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.eval()
        total_l2re = 0.0
        num_batches = 0

        for batch in dataloader:
            u_input = batch["input"].to(self.device)   # (B, T, C, H, W)
            u_target = batch["target"].to(self.device)  # (B, C, H, W)

            if rollout_steps == 1:
                pred, _ = self.model(u_input, noise_scale=0.0)
                l2re = l2_relative_error(pred, u_target)
            else:
                # Multi-step rollout
                current_input = u_input.clone()
                for step in range(rollout_steps):
                    pred, _ = self.model(current_input, noise_scale=0.0)
                    # Shift input window
                    current_input = torch.cat([
                        current_input[:, 1:],
                        pred.unsqueeze(1)
                    ], dim=1)
                l2re = l2_relative_error(pred, u_target)

            total_l2re += l2re.item()
            num_batches += 1

        return {"l2re": total_l2re / num_batches}

    def pretrain(
        self,
        train_loader: DataLoader,
        val_loaders: Optional[Dict[str, DataLoader]] = None,
        num_epochs: int = 1000,
        warmup_epochs: int = 200,
        save_dir: str = "checkpoints",
        save_every: int = 100,
        log_every: int = 10,
    ):
        """
        Pre-train the model on mixed PDE datasets.

        Args:
            train_loader: Mixed training data loader.
            val_loaders: Optional dict of validation data loaders per dataset.
            num_epochs: Total number of pre-training epochs.
            warmup_epochs: Number of warmup epochs.
            save_dir: Directory to save checkpoints.
            save_every: Save checkpoint every N epochs.
            log_every: Log metrics every N epochs.
        """
        os.makedirs(save_dir, exist_ok=True)

        steps_per_epoch = len(train_loader)
        self.setup_scheduler(num_epochs, steps_per_epoch, warmup_epochs)

        print(f"Starting pre-training for {num_epochs} epochs...")
        print(f"Steps per epoch: {steps_per_epoch}")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader, inject_noise=True)

            if (epoch + 1) % log_every == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Loss: {train_metrics['loss']:.4f} | "
                    f"Pred Loss: {train_metrics['pred_loss']:.4f} | "
                    f"Balance Loss: {train_metrics['balance_loss']:.4f} | "
                    f"L2RE: {train_metrics['l2re']:.4f}"
                )

                # Validate
                if val_loaders is not None:
                    for name, val_loader in val_loaders.items():
                        val_metrics = self.evaluate(val_loader)
                        print(f"  Val [{name}] L2RE: {val_metrics['l2re']:.4f}")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f"pretrain_epoch_{epoch+1}.pt")
                )

        # Save final checkpoint
        self.save_checkpoint(os.path.join(save_dir, "pretrain_final.pt"))
        print("Pre-training complete.")

    def finetune(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 200,
        warmup_epochs: int = 40,
        save_dir: str = "checkpoints",
        freeze_router: bool = True,
    ):
        """
        Fine-tune the pre-trained model on a specific dataset.

        Args:
            train_loader: Fine-tuning data loader.
            val_loader: Optional validation data loader.
            num_epochs: Number of fine-tuning epochs.
            warmup_epochs: Number of warmup epochs.
            save_dir: Directory to save checkpoints.
            freeze_router: Whether to freeze the router-gating network.
        """
        os.makedirs(save_dir, exist_ok=True)

        if freeze_router:
            self.freeze_router()

        steps_per_epoch = len(train_loader)
        self.setup_scheduler(num_epochs, steps_per_epoch, warmup_epochs)

        print(f"Starting fine-tuning for {num_epochs} epochs...")

        best_l2re = float("inf")
        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train (no noise injection during fine-tuning)
            train_metrics = self.train_epoch(train_loader, inject_noise=False)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Loss: {train_metrics['loss']:.4f} | "
                    f"L2RE: {train_metrics['l2re']:.4f}"
                )

                if val_loader is not None:
                    val_metrics = self.evaluate(val_loader)
                    print(f"  Val L2RE: {val_metrics['l2re']:.4f}")

                    if val_metrics["l2re"] < best_l2re:
                        best_l2re = val_metrics["l2re"]
                        self.save_checkpoint(
                            os.path.join(save_dir, "finetune_best.pt")
                        )

        self.save_checkpoint(os.path.join(save_dir, "finetune_final.pt"))
        print(f"Fine-tuning complete. Best L2RE: {best_l2re:.4f}")

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str, strict: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Checkpoint loaded from {path} (epoch {self.epoch})")
