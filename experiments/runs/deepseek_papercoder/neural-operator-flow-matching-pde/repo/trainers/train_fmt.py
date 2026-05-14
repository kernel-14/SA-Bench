## trainers/train_fmt.py
"""
Lightning module for training the Flow Marching Transformer (FMT).

This module manages the training loop, validation, and optimizer/scheduler
configuration for the FMT model as described in the paper.  It expects
data batches of shape [B, seq_len=4, channels=3, H=128, W=128] and uses
a frozen P2VAE encoder to convert them into latent representations.
During training, noisy latent states are constructed via the location‑scale
interpolation kernel, diffusion forcing is unrolled, and the conditional
flow matching loss (Eq. 13) is minimised.

Validation computes the same loss on a held‑out set; separate evaluation
scripts handle rollout prediction and ensemble generation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl

from models.fmt import FMT
from models.p2vae import P2VAE
from utils.data_utils import generate_noisy_latent


class FMTTrainer(pl.LightningModule):
    """
    Lightning module wrapper around FMT and a frozen P2VAE.

    Args:
        fmt: The Flow Marching Transformer model (already instantiated).
        vae: The pretrained P2VAE model (its encoder and decoder are frozen).
        config: Full configuration dictionary (as loaded from config.yaml).
                Expected keys: 'fmt' section with all training hyperparameters.
    """

    def __init__(self, fmt: FMT, vae: P2VAE, config: Dict[str, Any]) -> None:
        super().__init__()

        # -- Models --
        self.fmt = fmt
        self.vae = vae

        # Freeze the VAE completely
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

        # -- Configuration extraction --
        fmt_cfg = config.get("fmt", {})
        self.fmt_config = fmt_cfg
        self.latent_dim = fmt_cfg.get("latent_dim", 16)
        self.pyramid_factors: List[int] = fmt_cfg.get("pyramid_factors", [8, 4, 2, 1])

        # Optimizer / scheduler settings
        self.lr_base: float = fmt_cfg["optimizer"]["lr"]
        self.betas: Tuple[float, float] = tuple(fmt_cfg["optimizer"]["betas"])
        self.weight_decay: float = fmt_cfg["optimizer"]["weight_decay"]
        self.warmup_ratio: float = fmt_cfg["scheduler"]["warmup_ratio"]
        self.total_steps: int = fmt_cfg["total_steps"]
        self.grad_accum: int = fmt_cfg.get("gradient_accumulation", 1)
        self.batch_size: int = fmt_cfg["batch_size"]

        # Mixed precision flag (handled by Trainer, but we use for noise generation)
        self.use_amp = fmt_cfg.get("mixed_precision", True)

        # Optional: store global variance for VRMSE (if needed later)
        self.global_var: Optional[torch.Tensor] = None

        self.save_hyperparameters(ignore=["fmt", "vae"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _latents_from_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of 4‑frame physical trajectories into latent space.

        Args:
            frames: (B, 4, C, H, W) physical fields.
        Returns:
            latents: (B, 4, latent_dim, 16, 16) deterministic latent representations
                     (mean from the VAE encoder).
        """
        B = frames.size(0)
        # Flatten to (B*4, C, H, W)
        frames_flat = frames.reshape(-1, *frames.shape[2:])
        with torch.no_grad():
            mu, _ = self.vae.encode(frames_flat)
        # Reshape back to (B, 4, latent_dim, 16, 16)
        latents = mu.reshape(B, 4, self.latent_dim, 16, 16)
        return latents

    def _construct_noisy_latents(
        self,
        latents: torch.Tensor,
        ts: List[torch.Tensor],
        ks: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Build the list of four noisy latent states using the location‑scale
        interpolation kernel (Eq. 1).  The last frame is kept clean
        (k=1, t=0) as there is no target beyond it during training.

        Args:
            latents: (B, 4, C, 16, 16) clean latent trajectory.
            ts: list of 4 per‑sample time tensors (B,).
            ks: list of 4 per‑sample bridge tensors (B,).
        Returns:
            noisy_list: list of 4 tensors, each (B, C, 16, 16).
        """
        noisy_list = []
        for s in range(3):
            y_s = latents[:, s]
            y_next = latents[:, s + 1]
            # Use the utility from data_utils (handles broadcasting and noise)
            # We generate noise in float32 to avoid underflow in fp16,
            # then cast to the original type.
            with torch.autocast(device_type="cuda", enabled=False):
                y_noisy = generate_noisy_latent(
                    y_s.float(), y_next.float(), ts[s].float(), ks[s].float()
                ).to(dtype=latents.dtype)
            noisy_list.append(y_noisy)
        # Last frame: keep clean
        noisy_list.append(latents[:, 3])
        return noisy_list

    def _compute_loss(
        self,
        latents: torch.Tensor,
        noisy_list: List[torch.Tensor],
        ts: List[torch.Tensor],
        ks: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through FMT and computation of the conditional
        flow matching loss (Eq. 13).

        Args:
            latents: (B, 4, C, 16, 16) clean latents.
            noisy_list: list of 4 tensors (B, C, 16, 16).
            ts, ks: per‑frame tensors (B,).
        Returns:
            total_loss, dict of per‑transition losses for logging.
        """
        # Predict velocity tokens
        velocities = self.fmt.compute_velocity(noisy_list, ts, ks)

        total_loss = 0.0
        loss_dict = {}

        for s in range(3):
            # Full residual (y_{s+1} - y_{s,ts}^{ks})
            res_full = latents[:, s + 1] - noisy_list[s]  # (B, C, 16, 16)

            # Target token resolution for this frame
            factor = self.pyramid_factors[s]
            target_h = 16 // factor
            target_w = 16 // factor

            # Reshape predicted velocity tokens to spatial grid (B, C, H, W)
            N = velocities[s].shape[1]  # should equal target_h * target_w
            vel_grid = velocities[s].reshape(-1, target_h, target_w, self.latent_dim)
            vel_grid = vel_grid.permute(0, 3, 1, 2)  # (B, C, H, W)

            # Downsample the full residual to the same resolution
            res_down = F.adaptive_avg_pool2d(res_full, (target_h, target_w))

            # Scale velocity by (1 - t_s)
            t_s = ts[s]  # shape (B,)
            scaling = (1.0 - t_s).to(vel_grid.dtype)  # (B,)
            scaled_vel = vel_grid * scaling.view(-1, 1, 1, 1)

            # MSE loss
            loss_s = 0.5 * F.mse_loss(scaled_vel, res_down)
            total_loss += loss_s
            loss_dict[f"loss_s{s}"] = loss_s.detach()

        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # Training & Validation steps
    # ------------------------------------------------------------------

    def _shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], prefix: str
    ) -> torch.Tensor:
        """
        Common logic for training and validation: encodes, constructs
        noisy latents, runs FMT, computes the flow matching loss, and logs.

        Args:
            batch: tuple (frames, dataset_id) from dataset, frames shape (B,4,C,H,W).
            prefix: "train" or "val" for metric naming.
        Returns:
            total loss (scalar).
        """
        frames, _ = batch
        B = frames.size(0)
        device = frames.device

        # 1. Encode to latents
        latents = self._latents_from_batch(frames)  # (B,4,C,16,16)

        # 2. Sample flow times and bridge parameters
        ts = [torch.rand(B, device=device) for _ in range(4)]
        ks = [torch.rand(B, device=device) for _ in range(4)]
        # Override last frame to clean (k=1, t=0)
        ts[3] = torch.zeros(B, device=device)
        ks[3] = torch.ones(B, device=device)

        # 3. Create noisy latents
        noisy_list = self._construct_noisy_latents(latents, ts, ks)

        # 4. Compute loss
        loss, loss_details = self._compute_loss(latents, noisy_list, ts, ks)

        # 5. Logging
        self.log(f"{prefix}/loss", loss, on_step=(prefix == "train"), on_epoch=(prefix == "val"),
                 prog_bar=True)
        self.log(f"{prefix}/lr", self.trainer.optimizers[0].param_groups[0]["lr"],
                 on_step=True, on_epoch=False)
        for name, val in loss_details.items():
            self.log(f"{prefix}/{name}", val, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        # Validation uses the same loss; Lightning automatically applies no_grad
        self._shared_step(batch, "val")

    # ------------------------------------------------------------------
    # Optimizer & LR schedule
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """
        AdamW with linear warmup and cosine decay, as specified in the paper.
        The learning rate scales linearly with the effective batch size relative
        to the reference size of 256.
        """
        # Effective batch size (global)
        effective_bs = self.batch_size * self.trainer.world_size * self.grad_accum
        lr = self.lr_base * (effective_bs / 256.0)

        optimizer = AdamW(
            self.fmt.parameters(),
            lr=lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

        warmup_steps = max(1, int(self.total_steps * self.warmup_ratio))
        total_steps = self.total_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(warmup_steps)
            else:
                progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    # ------------------------------------------------------------------
    # Optional methods to match the design interface (used by main)
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> FMTTrainer:
        """
        Override to ensure VAE stays in eval mode.
        """
        super().train(mode)
        self.vae.eval()
        return self

    def save_checkpoint(self, path: str) -> None:
        """
        Convenience method to save the full FMT model state dict.
        Usually Lightning checkpoints are preferred, but this matches the
        design's interface.
        """
        torch.save({"state_dict": self.fmt.state_dict()}, path)

