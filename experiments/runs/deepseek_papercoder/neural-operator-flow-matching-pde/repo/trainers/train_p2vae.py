## trainers/train_p2vae.py
"""
Lightning module for training the P2VAE (Pretrained Physics Variational Autoencoder).

This module manages the training loop, validation, and optimizer/scheduler
configuration for the P2VAE model as described in the paper.  It expects
data batches of shape [B, seq_len=4, channels=3, H=128, W=128] and treats
each frame independently for reconstruction and KL loss computation.

All hyperparameters are read from the provided `config` dictionary (which
matches the structure of config.yaml).  The learning rate is scaled linearly
with the global batch size relative to the reference size of 256.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl

from models.p2vae import P2VAE


class P2VAETrainer(pl.LightningModule):
    """
    Lightning module for P2VAE training.

    Args:
        model: Instance of P2VAE (encoder + decoder).
        config: Full configuration dictionary (as loaded from config.yaml).
                The ``p2vae`` section is used for training hyperparameters.
    """

    def __init__(self, model: P2VAE, config: Dict[str, Any]) -> None:
        super().__init__()
        self.model = model

        # Extract P2VAE-specific configuration
        vae_cfg = config["p2vae"]
        self.kl_beta: float = vae_cfg["kl_beta"]
        self.lr_base: float = vae_cfg["optimizer"]["lr"]
        self.batch_size: int = vae_cfg["batch_size"]
        self.total_steps: int = vae_cfg["total_steps"]
        self.warmup_ratio: float = vae_cfg["scheduler"]["warmup_ratio"]
        self.betas: Tuple[float, float] = tuple(vae_cfg["optimizer"]["betas"])
        self.weight_decay: float = vae_cfg["optimizer"]["weight_decay"]

        # Placeholders for tracking
        self.save_hyperparameters(ignore=["model"])

    # ------------------------------------------------------------------
    # Training & Validation logic
    # ------------------------------------------------------------------

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Process one training batch.

        The dataloader returns (data, dataset_id) where data is a tensor
        of shape [B, 4, 3, 128, 128].  We flatten the time dimension and
        treat all frames independently.
        """
        frames, _ = batch                              # ignore dataset_id
        frames = frames.reshape(-1, 3, 128, 128)       # [B*4, C, H, W]

        recon, mu, logvar = self.model(frames)

        # Reconstruction loss (MSE)
        loss_recon = F.mse_loss(recon, frames, reduction="mean")

        # KL divergence loss: average over latent dims, mean over batch
        # For each sample in the batch: kl_i = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_per_sample = -0.5 * torch.sum(
            1.0 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]
        )
        loss_kl = kl_per_sample.mean() * self.kl_beta

        total_loss = loss_recon + loss_kl

        # Logging
        self.log("train/loss", total_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/loss_recon", loss_recon, on_step=True, on_epoch=False)
        self.log("train/loss_kl", loss_kl, on_step=True, on_epoch=False)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"],
                 on_step=True, on_epoch=False)

        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Process one validation batch.  Computes the same metrics as training.
        """
        frames, _ = batch
        frames = frames.reshape(-1, 3, 128, 128)

        recon, mu, logvar = self.model(frames)
        loss_recon = F.mse_loss(recon, frames, reduction="mean")
        kl_per_sample = -0.5 * torch.sum(
            1.0 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]
        )
        loss_kl = kl_per_sample.mean() * self.kl_beta
        total_loss = loss_recon + loss_kl

        # Log on epoch level for validation
        self.log("val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_recon", loss_recon, on_step=False, on_epoch=True)
        self.log("val/loss_kl", loss_kl, on_step=False, on_epoch=True)

    # ------------------------------------------------------------------
    # Optimizer & Learning rate schedule
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """
        AdamW with linear warmup then cosine decay, as described in the paper.

        Learning rate is scaled linearly with respect to the global batch size:
            lr = lr_base * (batch_size / 256)
        """
        # Scale learning rate
        scaled_lr = self.lr_base * (self.batch_size / 256.0)

        optimizer = AdamW(
            self.model.parameters(),
            lr=scaled_lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

        # Warmup and cosine schedule (updates every step)
        warmup_steps = int(self.total_steps * self.warmup_ratio)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            else:
                progress = (current_step - warmup_steps) / max(
                    1, (self.total_steps - warmup_steps)
                )
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "name": "cosine_warmup",
        }

        return [optimizer], [scheduler_config]

    # ------------------------------------------------------------------
    # Optional helper to make external calls easier (matches design sketch)
    # ------------------------------------------------------------------

    def train(self) -> None:
        """
        Compatibility wrapper – actual training is handled by PyTorch Lightning's
        Trainer, so this method does nothing.
        """
        pass

    def save_checkpoint(self, path: str) -> None:
        """
        Convenience method to save the full model state dict.
        Usually Lightning checkpoints are preferred, but this matches the
        design's interface.
        """
        torch.save({"state_dict": self.model.state_dict()}, path)

