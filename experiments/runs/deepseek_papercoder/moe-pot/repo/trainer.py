# trainer.py
"""Training and evaluation orchestration for MoE‑POT.

This module implements the ``Trainer`` class, which manages all
training phases (pre‑training, fine‑tuning, downstream adaptation) and
autoregressive validation.  It follows the hyperparameters defined in
``Config`` and uses the utility functions from ``utils.py`` for
mask‑aware loss and L2 relative error computation.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from model import Model
from utils import Utils


class Trainer:
    """Manages training and evaluation of the MoE‑POT model.

    The Trainer is initialised with pre‑built data loaders for a
    specific phase (pre‑training, fine‑tuning, or downstream) and
    handles the optimisation loop, noise injection, load‑balance loss
    aggregation, and autoregressive validation.

    Parameters
    ----------
    model : Model
        The MoE‑POT neural operator (already on device).
    config : Config
        Global configuration object.
    train_loader : DataLoader
        DataLoader for training.
    val_loaders : Dict[str, DataLoader]
        Per‑dataset validation/test loaders (keys are dataset names).
    """

    def __init__(
        self,
        model: Model,
        config: Config,
        train_loader: DataLoader,
        val_loaders: Dict[str, DataLoader],
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loaders = val_loaders
        self.device = next(model.parameters()).device

        # Optimizer will be re‑created per training phase.
        self.optimizer: Optional[Adam] = None
        self.scheduler: Optional[LRScheduler] = None

    # ------------------------------------------------------------------
    # Public interface for different training modes
    # ------------------------------------------------------------------

    def pretrain(self) -> None:
        """Run the full pre‑training loop on the mixed‑dataset loader.

        Uses the hyperparameters from ``config.pretrain_*`` fields.
        Noise injection is enabled and the balanced training loader is
        used.  The model is saved to ``config.output_dir`` after
        pre‑training.
        """
        self._reset_optimizer(
            lr=self.config.pretrain_learning_rate,
            weight_decay=self.config.pretrain_weight_decay,
            betas=(self.config.pretrain_beta1, self.config.pretrain_beta2),
        )
        total_steps = self.config.pretrain_epochs * len(self.train_loader)
        self._create_scheduler(
            total_steps=total_steps,
            warmup_epochs=self.config.pretrain_warmup_epochs,
            max_lr=self.config.pretrain_learning_rate,
            epochs=self.config.pretrain_epochs,
        )

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = output_dir / "pretrain_best.pt"

        best_metric = float("inf")
        for epoch in range(1, self.config.pretrain_epochs + 1):
            train_loss = self._train_epoch(
                noise_enabled=self.config.pretrain_noise_enabled,
            )
            self.scheduler.step()

            # Validate every 50 epochs, and always at the final epoch
            if epoch % 50 == 0 or epoch == self.config.pretrain_epochs:
                val_metrics = self._validate()
                print(
                    f"Epoch {epoch:4d}/{self.config.pretrain_epochs} | "
                    f"Train loss: {train_loss:.6f} | Val L2RE: "
                    + ", ".join(f"{k}={v:.5f}" for k, v in val_metrics.items())
                )
                # Use the average L2RE across datasets as checkpoint criterion
                avg_l2re = float(np.mean(list(val_metrics.values())))
                if avg_l2re < best_metric:
                    best_metric = avg_l2re
                    self._save_checkpoint(best_ckpt)
                    print(f"  -> new best model (avg L2RE = {best_metric:.5f})")

        # Final save
        final_ckpt = output_dir / "pretrain_final.pt"
        self._save_checkpoint(final_ckpt)
        print(f"Pre‑training finished. Final model saved to {final_ckpt}")

    def finetune(self, task_name: str) -> None:
        """Fine‑tune the pre‑trained model on a single PDE dataset.

        The router‑gating network is frozen.  The training and
        validation loaders are expected to have been updated via
        ``set_data_loaders`` before calling this method.

        Parameters
        ----------
        task_name : str
            Name of the dataset being fine‑tuned (for logging).
        """
        # Freeze router
        self.model.freeze_router()

        self._reset_optimizer(
            lr=self.config.finetune_learning_rate,
            weight_decay=self.config.pretrain_weight_decay,  # same as pre‑train
            betas=(self.config.pretrain_beta1, self.config.pretrain_beta2),
        )
        total_steps = self.config.finetune_epochs * len(self.train_loader)
        self._create_scheduler(
            total_steps=total_steps,
            warmup_epochs=self.config.finetune_warmup_epochs,
            max_lr=self.config.finetune_learning_rate,
            epochs=self.config.finetune_epochs,
        )

        output_dir = Path(self.config.output_dir) / f"finetune_{task_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = output_dir / "best.pt"

        best_metric = float("inf")
        for epoch in range(1, self.config.finetune_epochs + 1):
            train_loss = self._train_epoch(
                noise_enabled=False,  # no noise during fine‑tuning
            )
            self.scheduler.step()

            if epoch % 20 == 0 or epoch == self.config.finetune_epochs:
                val_metrics = self._validate()
                l2re = list(val_metrics.values())[0]  # only one dataset
                print(
                    f"[Fine‑tune {task_name}] Epoch {epoch:3d}/"
                    f"{self.config.finetune_epochs} | "
                    f"Train loss: {train_loss:.6f} | Val L2RE: {l2re:.5f}"
                )
                if l2re < best_metric:
                    best_metric = l2re
                    self._save_checkpoint(best_ckpt)
                    print(f"  -> new best model (L2RE = {best_metric:.5f})")

        final_ckpt = output_dir / "final.pt"
        self._save_checkpoint(final_ckpt)
        print(f"Fine‑tuning finished. Model saved to {final_ckpt}")

    def downstream(self, task_name: str) -> None:
        """Adapt the pre‑trained model to a downstream PDE task.

        Similar to ``finetune`` but uses downstream‑specific
        hyperparameters (more epochs, different warmup).  As before,
        the router is frozen.

        Parameters
        ----------
        task_name : str
            Name of the downstream dataset.
        """
        self.model.freeze_router()

        self._reset_optimizer(
            lr=self.config.downstream_learning_rate,
            weight_decay=self.config.pretrain_weight_decay,
            betas=(self.config.pretrain_beta1, self.config.pretrain_beta2),
        )
        total_steps = self.config.downstream_epochs * len(self.train_loader)
        self._create_scheduler(
            total_steps=total_steps,
            warmup_epochs=self.config.downstream_warmup_epochs,
            max_lr=self.config.downstream_learning_rate,
            epochs=self.config.downstream_epochs,
        )

        output_dir = Path(self.config.output_dir) / f"downstream_{task_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = output_dir / "best.pt"

        best_metric = float("inf")
        for epoch in range(1, self.config.downstream_epochs + 1):
            train_loss = self._train_epoch(noise_enabled=False)
            self.scheduler.step()

            if epoch % 50 == 0 or epoch == self.config.downstream_epochs:
                val_metrics = self._validate()
                l2re = list(val_metrics.values())[0]
                print(
                    f"[Downstream {task_name}] Epoch {epoch:3d}/"
                    f"{self.config.downstream_epochs} | "
                    f"Train loss: {train_loss:.6f} | Val L2RE: {l2re:.5f}"
                )
                if l2re < best_metric:
                    best_metric = l2re
                    self._save_checkpoint(best_ckpt)
                    print(f"  -> new best model (L2RE = {best_metric:.5f})")

        final_ckpt = output_dir / "final.pt"
        self._save_checkpoint(final_ckpt)
        print(f"Downstream training finished. Model saved to {final_ckpt}")

    # ------------------------------------------------------------------
    # Data loader swapping (used by main after DatasetLoader provides new loaders)
    # ------------------------------------------------------------------

    def set_data_loaders(
        self,
        train_loader: DataLoader,
        val_loaders: Dict[str, DataLoader],
    ) -> None:
        """Replace the current training and validation loaders.

        Useful when switching from pre‑training to fine‑tuning /
        downstream without re‑instantiating the Trainer.
        """
        self.train_loader = train_loader
        self.val_loaders = val_loaders

    # ------------------------------------------------------------------
    # Internal: one training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, noise_enabled: bool) -> float:
        """Run a single training epoch.

        Parameters
        ----------
        noise_enabled : bool
            If ``True``, add Gaussian noise scaled by per‑sample norm
            to the input (only during pre‑training).

        Returns
        -------
        float
            Average training loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_samples = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            # batch: (input_seq, target_frame, mask, channels_per_task)
            # shapes: input_seq (B, T, H, W, C), target_frame (B, H, W, C),
            #          mask (B, H, W, C) or (B, C, H, W) – we standardise to (B, H, W, C)
            input_seq, target_frame, mask, _ = batch
            input_seq = input_seq.to(self.device, non_blocking=True)
            target_frame = target_frame.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            # Ensure mask has shape (B, H, W, C) – the dataset always returns (B, H, W, C)
            # If it comes as (B, C, H, W) we permute, but we assume correct shape.
            if mask.dim() == 4 and mask.shape[1] == self.config.max_channels:
                # Dataset returns (B, C, H, W) – permute to (B, H, W, C)
                mask = mask.permute(0, 2, 3, 1)
            # else assume already (B, H, W, C) – no action needed

            B = input_seq.shape[0]

            # Noise injection
            if noise_enabled:
                # Compute per‑sample L2 norm of the whole spatiotemporal input
                norm = input_seq.reshape(B, -1).norm(dim=1)  # (B,)
                # Add epsilon to avoid zero norm
                norm = norm + 1e-8
                noise = torch.randn_like(input_seq) * (
                    self.config.noise_std * norm.view(B, 1, 1, 1, 1)
                )
                input_seq = input_seq + noise

            # Forward pass
            output, balance_loss = self.model(input_seq)  # output (B, H, W, C)

            # Primary loss: masked MSE (average over valid elements)
            mse_loss = Utils.masked_mse(
                output.permute(0, 3, 1, 2),       # (B, C, H, W)
                target_frame.permute(0, 3, 1, 2),
                mask.permute(0, 3, 1, 2),
            )
            # Total loss = primary + balance
            loss = mse_loss + balance_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item() * B
            num_samples += B

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "bal": f"{balance_loss.item():.6f}"})

        return total_loss / num_samples

    # ------------------------------------------------------------------
    # Internal: validation across all test loaders
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Perform autoregressive rollout and compute L2RE on all
        validation datasets.

        Returns
        -------
        Dict[str, float]
            Mapping from dataset name to its average L2RE (lower is better).
        """
        self.model.eval()
        results: Dict[str, float] = {}
        num_rollout = self.config.rollout_steps

        for name, val_loader in self.val_loaders.items():
            l2re_list: list[float] = []
            for batch in val_loader:
                # batch: (context, target_rollout, mask_rollout)
                # shapes: context (B, T, H, W, C), target_rollout (B, rollout, H, W, C),
                #          mask_rollout (B, rollout, H, W, C) or (B, rollout, C, H, W)
                context, target_seq, mask_seq = batch
                context = context.to(self.device, non_blocking=True)
                target_seq = target_seq.to(self.device, non_blocking=True)
                mask_seq = mask_seq.to(self.device, non_blocking=True)

                # Ensure mask has shape (B, rollout, H, W, C) – dataset may return (B, C, H, W) per step
                if mask_seq.dim() == 5 and mask_seq.shape[2] == self.config.max_channels:
                    # Assumed (B, rollout, C, H, W) -> (B, rollout, H, W, C)
                    mask_seq = mask_seq.permute(0, 1, 3, 4, 2)
                # else keep as is (B, rollout, H, W, C)

                current_input = context  # (B, T, H, W, C)
                preds = []
                for _ in range(num_rollout):
                    output, _ = self.model(current_input)  # (B, H, W, C)
                    preds.append(output)
                    # Shift input: drop oldest, append prediction
                    current_input = torch.cat(
                        [
                            current_input[:, 1:, ...],
                            output.unsqueeze(1),
                        ],
                        dim=1,
                    )

                # Stack predictions: (B, rollout, H, W, C)
                pred_seq = torch.stack(preds, dim=1)

                # Compute masked L2 relative error over the rollout
                l2re_val = Utils.l2_relative_error(
                    pred_seq.permute(0, 1, 4, 2, 3),   # (B, rollout, C, H, W)
                    target_seq.permute(0, 1, 4, 2, 3),
                    mask_seq.permute(0, 1, 4, 2, 3),
                )
                l2re_list.append(l2re_val.item())

            results[name] = float(np.mean(l2re_list)) if l2re_list else float("inf")

        return results

    # ------------------------------------------------------------------
    # Internal helpers (optimizer, scheduler, checkpointing)
    # ------------------------------------------------------------------

    def _reset_optimizer(
        self,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.9),
    ) -> None:
        """Create a fresh Adam optimizer for the current model parameters."""
        self.optimizer = Adam(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    def _create_scheduler(
        self,
        total_steps: int,
        warmup_epochs: int,
        max_lr: float,
        epochs: int,
    ) -> None:
        """Instantiate a 1‑cycle learning rate scheduler.

        The warmup phase is specified in epochs and converted to a
        fraction of total steps.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be created before scheduler.")
        pct_start = warmup_epochs / epochs  # fraction of total steps for warmup
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="linear",
        )

    def _save_checkpoint(self, path: Path) -> None:
        """Save the current model weights to disk."""
        torch.save(self.model.state_dict(), path)

