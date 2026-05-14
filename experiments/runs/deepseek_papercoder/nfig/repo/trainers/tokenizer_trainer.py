"""
trainers/tokenizer_trainer.py

TokenizerTrainer for the Frequency-guided Residual-quantized VAE (FR-VAE).

Implements adversarial training with a DINO discriminator following the NFIG paper.
Handles pixel reconstruction, frequency reconstruction, perceptual (LPIPS),
VQ commitment/codebook losses, and GAN adversarial loss with a hinge objective.

All hyperparameters are read from the project configuration dictionary (from config.yaml).
"""

import os
from typing import Any, Dict, Tuple

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.discriminator import DinoDiscriminator
from models.fr_vae import FRVAE


class TokenizerTrainer:
    """Orchestrates training of the FR‑VAE tokenizer with a DINO discriminator."""

    def __init__(
        self,
        model: FRVAE,
        disc: DinoDiscriminator,
        config: Dict[str, Any],
    ) -> None:
        """
        Args:
            model: FR‑VAE tokenizer to be trained.
            disc: DINO discriminator for adversarial training.
            config: Full project configuration dictionary loaded from config.yaml.
        """
        self.fr_vae = model
        self.discriminator = disc
        self.config = config

        # ---- Unpack hyperparameters ----
        tokenizer_cfg = config["tokenizer"]
        self.epochs = tokenizer_cfg.get("tokenizer_epochs", 200)
        self.lr = tokenizer_cfg.get("tokenizer_lr", 1.0e-4)
        self.commitment_cost = tokenizer_cfg.get("commitment_cost", 0.25)

        loss_weights = tokenizer_cfg.get("loss_weights", {})
        self.w_pixel = loss_weights.get("pixel_recon", 1.0)
        self.w_freq = loss_weights.get("freq_recon", 1.0)
        self.w_perceptual = loss_weights.get("perceptual", 1.0)
        self.w_gan = loss_weights.get("gan", 0.5)

        # ---- LPIPS perceptual loss ----
        self.lpips_fn = lpips.LPIPS(net="alex").eval()

        # ---- Optimisers (Adam with GAN‑friendly betas) ----
        self.optimizer_G = optim.Adam(
            self.fr_vae.parameters(),
            lr=self.lr,
            betas=(0.5, 0.9),
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
            betas=(0.5, 0.9),
        )

        # ---- Mixed precision (optional) ----
        self.use_amp = config.get("training", {}).get("use_amp", False)
        self.scaler_G = GradScaler(enabled=self.use_amp)
        self.scaler_D = GradScaler(enabled=self.use_amp)

        # ---- Logging & checkpointing ----
        log_cfg = config.get("logging", {})
        self.checkpoints_dir = log_cfg.get("checkpoints_dir", "./checkpoints")
        self.log_interval = log_cfg.get("log_interval", 100)
        self.eval_interval = log_cfg.get("eval_interval", 5)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Move everything to the appropriate device
        self.device = next(self.fr_vae.parameters()).device
        self.lpips_fn.to(self.device)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step on a batch of images.

        Args:
            batch: Tuple (images, labels); labels are ignored for tokenizer training.

        Returns:
            Dictionary containing all scalar loss values for logging.
        """
        images, _ = batch
        images = images.to(self.device, non_blocking=True)

        # ------------------------------------------------------------------
        # 1. Forward pass through FR‑VAE (manual reconstruction to obtain
        #    all quantisation intermediates for VQ loss computation)
        # ------------------------------------------------------------------
        f = self.fr_vae.encode(images)                      # (B, C, 16, 16)
        f_hat_list = self.fr_vae.frequency_decompose(f)     # list of (B, C, 16, 16)

        R_prev = torch.zeros_like(f)
        f_tilde = torch.zeros_like(f)
        vq_loss_total = 0.0
        # We iterate over levels using the residual quantizer directly
        for i, f_hat_i in enumerate(f_hat_list):
            target = f_hat_i if i == 0 else R_prev + f_hat_i
            u_i, _, v_i, v_i_q = self.fr_vae.quantizer(target, level_idx=i)

            # Standard VQ loss (codebook + commitment)
            codebook_loss = F.mse_loss(v_i_q, v_i.detach())  # ||e - sg[z_e]||^2
            commit_loss = F.mse_loss(v_i_q.detach(), v_i)     # ||sg[e] - z_e||^2
            vq_loss_total = vq_loss_total + codebook_loss + self.commitment_cost * commit_loss

            f_tilde = f_tilde + u_i
            R_prev = R_prev + (f_hat_i - u_i)

        # ------------------------------------------------------------------
        # 2. Decode to obtain reconstruction image
        # ------------------------------------------------------------------
        recon_images = self.fr_vae.decode(f_tilde)          # (B, 3, 256, 256)

        # ------------------------------------------------------------------
        # 3. Reconstruction losses
        # ------------------------------------------------------------------
        loss_pixel = F.mse_loss(recon_images, images)
        loss_freq  = F.mse_loss(f_tilde, f)

        # ------------------------------------------------------------------
        # 4. Perceptual loss (LPIPS)
        # ------------------------------------------------------------------
        loss_perceptual = self.lpips_fn(recon_images, images, normalize=True).mean()

        # ------------------------------------------------------------------
        # 5. Adversarial loss (generator side) – hinge‑based
        # ------------------------------------------------------------------
        # Discriminator expects (real, reconst) to produce a score map.
        fake_logits = self.discriminator(images, recon_images)   # (B, 1, Hp, Wp)
        loss_gan_G = -fake_logits.mean()

        # ------------------------------------------------------------------
        # 6. Total generator loss
        # ------------------------------------------------------------------
        loss_G = (
            self.w_pixel * loss_pixel
            + self.w_freq * loss_freq
            + self.w_perceptual * loss_perceptual
            + self.w_gan * loss_gan_G
            + vq_loss_total
        )

        # ------------------------------------------------------------------
        # 7. Update generator (FR‑VAE)
        # ------------------------------------------------------------------
        self.optimizer_G.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler_G.scale(loss_G).backward()
            self.scaler_G.unscale_(self.optimizer_G)
            torch.nn.utils.clip_grad_norm_(self.fr_vae.parameters(), 1.0)
            self.scaler_G.step(self.optimizer_G)
            self.scaler_G.update()
        else:
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(self.fr_vae.parameters(), 1.0)
            self.optimizer_G.step()

        # ------------------------------------------------------------------
        # 8. Discriminator update
        # ------------------------------------------------------------------
        # real pair: (real, real) should be considered positive
        real_logits = self.discriminator(images, images)
        # fake pair: (real, recon) is considered negative
        # (already computed above but we need a fresh graph – usually we recompute
        #  to avoid stale graph issues; however we can reuse if we do it before
        #  generator backward, but standard practice is to recompute for D)
        fake_logits_d = self.discriminator(images, recon_images.detach())  # detach from G

        loss_D_real = F.relu(1.0 - real_logits).mean()
        loss_D_fake = F.relu(1.0 + fake_logits_d).mean()
        loss_D = loss_D_real + loss_D_fake

        self.optimizer_D.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler_D.scale(loss_D).backward()
            self.scaler_D.unscale_(self.optimizer_D)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.scaler_D.step(self.optimizer_D)
            self.scaler_D.update()
        else:
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.optimizer_D.step()

        # ------------------------------------------------------------------
        # 9. Collect scalar loss values for logging
        # ------------------------------------------------------------------
        return {
            "loss_G": loss_G.item(),
            "pixel": loss_pixel.item(),
            "freq": loss_freq.item(),
            "perceptual": loss_perceptual.item(),
            "gan_G": loss_gan_G.item(),
            "vq": vq_loss_total.item(),
            "loss_D": loss_D.item(),
        }

    # ------------------------------------------------------------------
    # Validation step (identical to training step but without gradients)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Evaluate the current models on a validation batch.

        Args:
            batch: Tuple (images, labels).

        Returns:
            Dictionary of evaluation losses.
        """
        images, _ = batch
        images = images.to(self.device, non_blocking=True)

        f = self.fr_vae.encode(images)
        f_hat_list = self.fr_vae.frequency_decompose(f)

        R_prev = torch.zeros_like(f)
        f_tilde = torch.zeros_like(f)
        vq_loss_total = 0.0

        for i, f_hat_i in enumerate(f_hat_list):
            target = f_hat_i if i == 0 else R_prev + f_hat_i
            u_i, _, v_i, v_i_q = self.fr_vae.quantizer(target, level_idx=i)

            codebook_loss = F.mse_loss(v_i_q, v_i.detach())
            commit_loss = F.mse_loss(v_i_q.detach(), v_i)
            vq_loss_total = vq_loss_total + codebook_loss + self.commitment_cost * commit_loss

            f_tilde = f_tilde + u_i
            R_prev = R_prev + (f_hat_i - u_i)

        recon_images = self.fr_vae.decode(f_tilde)

        loss_pixel = F.mse_loss(recon_images, images)
        loss_freq  = F.mse_loss(f_tilde, f)
        loss_perceptual = self.lpips_fn(recon_images, images, normalize=True).mean()

        # Discriminator (only for monitoring, not for validation loss)
        real_logits = self.discriminator(images, images)
        fake_logits = self.discriminator(images, recon_images)
        loss_D_real = F.relu(1.0 - real_logits).mean()
        loss_D_fake = F.relu(1.0 + fake_logits).mean()
        loss_D = loss_D_real + loss_D_fake

        return {
            "loss_G": (loss_pixel + loss_freq + loss_perceptual + vq_loss_total).item(),
            "pixel": loss_pixel.item(),
            "freq": loss_freq.item(),
            "perceptual": loss_perceptual.item(),
            "gan_G": 0.0,  # not computed for validation
            "vq": vq_loss_total.item(),
            "loss_D": loss_D.item(),
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------
    def train(
        self,
        dataloader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """
        Run the complete tokenizer training procedure.

        Args:
            dataloader: DataLoader providing training batches (image, label).
            val_loader: DataLoader providing validation batches.
        """
        print(f"[TokenizerTrainer] Starting training for {self.epochs} epochs.")
        best_val_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            self.fr_vae.train()
            self.discriminator.train()

            # --- Training ---
            losses_avg = {}
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.epochs} [Train]", leave=False)
            for batch_idx, batch in enumerate(pbar):
                losses = self.training_step(batch)

                # Accumulate for logging
                for k, v in losses.items():
                    losses_avg[k] = losses_avg.get(k, 0.0) + v

                if (batch_idx + 1) % self.log_interval == 0:
                    avg = {k: v / (batch_idx + 1) for k, v in losses_avg.items()}
                    pbar.set_postfix(avg)

            # Average training losses over the epoch
            for k in losses_avg:
                losses_avg[k] /= len(dataloader)
            print(f"Epoch {epoch:3d}  Train loss_G={losses_avg['loss_G']:.4f}  "
                  f"pixel={losses_avg['pixel']:.4f}  freq={losses_avg['freq']:.4f}  "
                  f"lpips={losses_avg['perceptual']:.4f}  "
                  f"vq={losses_avg['vq']:.4f}  loss_D={losses_avg['loss_D']:.4f}")

            # --- Validation ---
            if epoch % self.eval_interval == 0 or epoch == self.epochs:
                self.fr_vae.eval()
                self.discriminator.eval()
                val_losses = {}
                n_val = 0
                # Process a limited number of batches to keep validation fast
                max_val_batches = min(50, len(val_loader))
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= max_val_batches:
                        break
                    losses = self.validation_step(batch)
                    for k, v in losses.items():
                        val_losses[k] = val_losses.get(k, 0.0) + v
                    n_val += 1

                for k in val_losses:
                    val_losses[k] /= n_val
                print(f"Epoch {epoch:3d}  Val   loss_G={val_losses['loss_G']:.4f}  "
                      f"lpips={val_losses['perceptual']:.4f}  "
                      f"vq={val_losses['vq']:.4f}  loss_D={val_losses['loss_D']:.4f}")

                # Save best model based on validation pixel+lpips loss
                current_val = val_losses.get("loss_G", float("inf"))
                if current_val < best_val_loss:
                    best_val_loss = current_val
                    self._save_checkpoint(epoch, is_best=True)

            # Regular checkpoint at the end of each epoch
            if epoch % self.eval_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

        # Save final model
        final_path = os.path.join(self.checkpoints_dir, "fr_vae_final.pth")
        torch.save(self.fr_vae.state_dict(), final_path)
        print(f"[TokenizerTrainer] Training finished. Final model saved to {final_path}")

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Write model checkpoints to disk."""
        state = {
            "epoch": epoch,
            "fr_vae_state_dict": self.fr_vae.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
        }
        if is_best:
            path = os.path.join(self.checkpoints_dir, "best.pt")
        else:
            path = os.path.join(self.checkpoints_dir, f"epoch_{epoch:03d}.pt")
        torch.save(state, path)
        if is_best:
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")
