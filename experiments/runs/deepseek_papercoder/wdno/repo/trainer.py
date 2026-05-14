## trainer.py
"""
Trainer for the Wavelet Diffusion Neural Operator (WDNO).

Implements training loops for the Base‑Resolution Model (BRM) and the
Super‑Resolution Model (SRM) using the DDPM noise prediction loss in the
wavelet domain.  Uses configuration from ``config.yaml`` and conforms to the
design specified in the project's class diagram.
"""

import os
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from utils import get_device
from dataset import MultiResolutionDataset
from models.wdno import WDNO


logger = logging.getLogger(__name__)


class Trainer:
    """
    Manages training of the WDNO components.

    Args:
        wdno: WDNO instance containing the BRM and optionally the SRM.
        dataset_brm: Multi‑resolution dataset configured for base‑resolution
                     training (mode ``'brm'``).
        config: Full experiment configuration.
        dataset_srm: Optional multi‑resolution dataset configured for
                     super‑resolution training (mode ``'srm'``). Required
                     only if ``train_srm()`` is called.
    """

    def __init__(
        self,
        wdno: WDNO,
        dataset_brm: MultiResolutionDataset,
        config: Config,
        dataset_srm: Optional[MultiResolutionDataset] = None,
    ) -> None:
        self.wdno = wdno
        self.dataset_brm = dataset_brm
        self.dataset_srm = dataset_srm
        self.config = config

        # Device handling
        self.device = get_device(config.get_device())
        self.wdno.brm.to(self.device)
        if self.wdno.srm is not None:
            self.wdno.srm.to(self.device)

        # Training hyperparameters
        train_cfg = config.get_training_config()
        self.batch_size = train_cfg["batch_size"]
        self.lr = train_cfg["learning_rate"]
        self.grad_clip = train_cfg.get("grad_clip", 0.0)
        self.steps_brm = train_cfg["steps_brm"]
        self.steps_srm = train_cfg["steps_srm"]
        self.drop_cond_prob = config.get_diffusion_config().get("drop_cond_prob", 0.1)

        # Checkpoint directory
        exp_name = config.get_experiment_name()
        self.ckpt_dir = os.path.join("checkpoints", exp_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Optimizer / scheduler state (initialized during training)
        self.optimizer_brm: Optional[optim.Adam] = None
        self.scheduler_brm: Optional[optim.lr_scheduler.LRScheduler] = None
        self.optimizer_srm: Optional[optim.Adam] = None
        self.scheduler_srm: Optional[optim.lr_scheduler.LRScheduler] = None
        self.global_step_brm = 0
        self.global_step_srm = 0

    # ------------------------------------------------------------------
    # Collate functions
    # ------------------------------------------------------------------

    @staticmethod
    def _collate_brm(batch: list) -> tuple:
        """
        Collate a batch from the BRM dataset.
        Each element is a dict with keys ``'W_state'`` and ``'W_cond'``.
        Returns (W_state, W_cond) stacked tensors.
        """
        W_states = torch.stack([item["W_state"] for item in batch])
        W_conds = torch.stack([item["W_cond"] for item in batch])
        return W_states, W_conds

    @staticmethod
    def _collate_srm(batch: list) -> tuple:
        """
        Collate a batch from the SRM dataset.
        Each element is a dict with keys ``'W_high'``, ``'W_low_up'``,
        ``'W_cond_high'``.
        Returns (W_high, W_low_up, W_cond_high) stacked tensors.
        """
        W_highs = torch.stack([item["W_high"] for item in batch])
        W_low_ups = torch.stack([item["W_low_up"] for item in batch])
        W_cond_highs = torch.stack([item["W_cond_high"] for item in batch])
        return W_highs, W_low_ups, W_cond_highs

    # ------------------------------------------------------------------
    # Checkpointing helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, model_type: str, step: int) -> None:
        """Save model, optimizer, and scheduler state."""
        if model_type == "brm":
            model = self.wdno.brm.denoiser
            optimizer = self.optimizer_brm
            scheduler = self.scheduler_brm
            fname = f"brm_step{step}.pth"
        elif model_type == "srm":
            if self.wdno.srm is None:
                return
            model = self.wdno.srm.denoiser
            optimizer = self.optimizer_srm
            scheduler = self.scheduler_srm
            fname = f"srm_step{step}.pth"
        else:
            raise ValueError("model_type must be 'brm' or 'srm'")

        ckpt = {
            "denoiser": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "step": step,
        }
        path = os.path.join(self.ckpt_dir, fname)
        torch.save(ckpt, path)
        logger.info(f"Saved {model_type} checkpoint at step {step} → {path}")

    def _load_checkpoint(self, model_type: str, path: str) -> None:
        """Restore model, optimizer, and scheduler state."""
        ckpt = torch.load(path, map_location=self.device)
        if model_type == "brm":
            self.wdno.brm.denoiser.load_state_dict(ckpt["denoiser"])
            if self.optimizer_brm and ckpt.get("optimizer"):
                self.optimizer_brm.load_state_dict(ckpt["optimizer"])
            if self.scheduler_brm and ckpt.get("scheduler"):
                self.scheduler_brm.load_state_dict(ckpt["scheduler"])
            self.global_step_brm = ckpt["step"]
        elif model_type == "srm":
            if self.wdno.srm is None:
                return
            self.wdno.srm.denoiser.load_state_dict(ckpt["denoiser"])
            if self.optimizer_srm and ckpt.get("optimizer"):
                self.optimizer_srm.load_state_dict(ckpt["optimizer"])
            if self.scheduler_srm and ckpt.get("scheduler"):
                self.scheduler_srm.load_state_dict(ckpt["scheduler"])
            self.global_step_srm = ckpt["step"]
        else:
            raise ValueError("model_type must be 'brm' or 'srm'")
        logger.info(f"Loaded {model_type} checkpoint from {path}")

    # ------------------------------------------------------------------
    # Training loops
    # ------------------------------------------------------------------

    def train_brm(self) -> None:
        """
        Train the Base‑Resolution Model using the DDPM noise prediction loss.
        The denoiser learns the conditional distribution p(W_u | W_a)
        (or p(W_f | W_a) for control tasks).
        """
        denoiser = self.wdno.brm.denoiser
        denoiser.train()

        dataloader = DataLoader(
            self.dataset_brm,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=self._collate_brm,
            drop_last=True,
        )

        optimizer = optim.Adam(denoiser.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.steps_brm, eta_min=0
        )
        self.optimizer_brm = optimizer
        self.scheduler_brm = scheduler

        logger.info("Starting BRM training ...")
        step = 0
        pbar = tqdm(total=self.steps_brm, desc="BRM training")

        for W_state, W_cond in dataloader:
            W_state = W_state.to(self.device)
            W_cond = W_cond.to(self.device)
            batch_size = W_state.shape[0]

            # Classifier‑free dropout: zero out condition channels for a fraction of samples
            if self.drop_cond_prob > 0.0 and W_cond.shape[1] > 0:
                # Build a per‑sample mask that zeros all condition channels
                mask_shape = (batch_size, 1) + (1,) * (W_cond.dim() - 2)
                mask = (
                    torch.rand(mask_shape, device=self.device) > self.drop_cond_prob
                )
                W_cond = W_cond * mask

            # Sample random diffusion timestep
            t = torch.randint(
                0, self.wdno.brm.n_timesteps, (batch_size,), device=self.device
            ).long()

            # Forward diffusion step
            noise = torch.randn_like(W_state)
            x_t = self.wdno.brm.add_noise(W_state, noise, t)

            # Predict the added noise
            noise_pred = denoiser(x_t, t, W_cond)

            # Loss
            loss = nn.functional.mse_loss(noise_pred, noise)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(denoiser.parameters(), self.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())

            if step % 5000 == 0:
                self._save_checkpoint("brm", step)

            if step >= self.steps_brm:
                break

        self._save_checkpoint("brm", step)
        logger.info("BRM training finished.")

    def train_srm(self) -> None:
        """
        Train the Super‑Resolution Model using multi‑resolution wavelet pairs.
        The denoiser learns the mapping from low‑resolution (upsampled) wavelet
        coefficients and high‑resolution conditions to high‑resolution wavelet
        coefficients.
        """
        if self.wdno.srm is None:
            raise RuntimeError("SRM is not present in WDNO instance.")
        if self.dataset_srm is None:
            raise RuntimeError("dataset_srm is required for SRM training but is None.")

        denoiser = self.wdno.srm.denoiser
        denoiser.train()

        dataloader = DataLoader(
            self.dataset_srm,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=self._collate_srm,
            drop_last=True,
        )

        optimizer = optim.Adam(denoiser.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.steps_srm, eta_min=0
        )
        self.optimizer_srm = optimizer
        self.scheduler_srm = scheduler

        logger.info("Starting SRM training ...")
        step = 0
        pbar = tqdm(total=self.steps_srm, desc="SRM training")

        for W_high, W_low_up, W_cond_high in dataloader:
            W_high = W_high.to(self.device)
            W_low_up = W_low_up.to(self.device)
            W_cond_high = W_cond_high.to(self.device)
            batch_size = W_high.shape[0]

            # Full condition: concatenate upsampled low‑res and high‑res condition
            if W_cond_high.shape[1] > 0:
                full_cond = torch.cat([W_low_up, W_cond_high], dim=1)
            else:
                full_cond = W_low_up

            # Classifier‑free dropout
            if self.drop_cond_prob > 0.0:
                mask_shape = (batch_size, 1) + (1,) * (full_cond.dim() - 2)
                mask = (
                    torch.rand(mask_shape, device=self.device) > self.drop_cond_prob
                )
                full_cond = full_cond * mask

            # Sample diffusion timestep
            t = torch.randint(
                0, self.wdno.srm.n_timesteps, (batch_size,), device=self.device
            ).long()

            # Forward diffusion
            noise = torch.randn_like(W_high)
            x_t = self.wdno.srm.add_noise(W_high, noise, t)

            # Predict noise
            noise_pred = denoiser(x_t, t, full_cond)

            loss = nn.functional.mse_loss(noise_pred, noise)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(denoiser.parameters(), self.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())

            if step % 5000 == 0:
                self._save_checkpoint("srm", step)

            if step >= self.steps_srm:
                break

        self._save_checkpoint("srm", step)
        logger.info("SRM training finished.")

