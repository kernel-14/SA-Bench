## mdm_trainer.py

"""
MDM Trainer for masked diffusion models.

Implements the training loop with score‑entropy (ELBO) loss, optional
gradient accumulation, cosine warmup scheduler, and checkpointing.
The trainer expects a time‑embedding‑free denoising network (``MDMTransformer``).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs import ExperimentConfig
from model import MDMTransformer
from utils import MASK_TOKEN_ID, mask_tokens


class MDMTrainer:
    """
    Trainer for Masked Diffusion Models (MDMs).

    Args:
        model: A time‑embedding‑free ``MDMTransformer`` instance.
        config: Full experiment configuration.
        train_loader: DataLoader yielding batches of clean token sequences.
    """

    def __init__(
        self,
        model: MDMTransformer,
        config: ExperimentConfig,
        train_loader: DataLoader,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader

        # Device setup
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Training hyperparameters
        self._batch_size = config.training.batch_size
        self._grad_accum_steps = config.training.gradient_accumulation_steps
        self._log_interval = config.training.log_interval
        self._save_interval = config.training.save_interval
        self._checkpoint_dir = Path(config.training.checkpoint_dir)

        # Total number of parameter updates
        self.total_steps = self._compute_total_steps()

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            weight_decay=config.training.weight_decay,
        )

        # Learning‑rate scheduler: linear warmup followed by cosine decay
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = GradScaler(enabled=(config.device == "cuda"))
        self._mixed_precision = config.device == "cuda"

        # Logging
        self.use_wandb = config.use_wandb
        if self.use_wandb:
            import wandb

            wandb.init(project=config.wandb_project, config=config.__dict__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """
        Run the full training loop for ``self.total_steps`` updates.
        The data loader is cycled continuously (epochs are implicit).
        """
        self.model.train()
        global_step = 0
        progress = tqdm(total=self.total_steps, desc="Training")

        # We use an infinite iterator over the dataloader
        loader_iter = iter(self.train_loader)

        while global_step < self.total_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                # Reinitialize the iterator (new epoch)
                loader_iter = iter(self.train_loader)
                batch = next(loader_iter)

            x0 = batch["input_ids"].to(self.device)
            # The MDM denoiser predicts the original token values; labels = x0.
            loss = self.compute_loss(x0)
            loss = loss / self._grad_accum_steps

            # Backward with mixed precision
            if self._mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step if accumulation boundary reached
            if (global_step + 1) % self._grad_accum_steps == 0:
                if self._mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

            global_step += 1
            progress.update(1)

            # Logging
            if global_step % self._log_interval == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                if self.use_wandb:
                    wandb.log({"train/loss": loss.item(), "train/lr": current_lr, "step": global_step})
                else:
                    print(f"Step {global_step}/{self.total_steps} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}")

            # Checkpointing
            if global_step % self._save_interval == 0:
                self.save_checkpoint(global_step)

        progress.close()
        # Final checkpoint
        self.save_checkpoint(self.total_steps, final=True)

    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Compute the continuous‑time score‑entropy loss for a batch of clean sequences.

        For each sequence:
        - Sample t ~ U(0,1).
        - Compute α_t = cos(πt/2), α'_t = -π/2 sin(πt/2).
        - Mask each token independently with probability 1−α_t to obtain x_t.
        - Evaluate log‑likelihood on masked positions, weighted by
          α'_t / (1 - α_t).

        The final loss is the mean over the batch of the per‑sequence
        weighted sum of negative log‑probabilities.

        Args:
            x0: Clean token ids, shape ``(B, L)``.  Token values are assumed
                to be in {1..m} (the mask token is 0).

        Returns:
            Scalar loss value.
        """
        B, L = x0.shape

        # Sample diffusion times per element
        t = torch.rand(B, device=x0.device)                     # (B,)

        # Compute noise schedule values
        alpha_t = torch.cos(t * math.pi / 2.0)                  # (B,)
        alpha_prime = -0.5 * math.pi * torch.sin(t * math.pi / 2.0)  # (B,)  derivative

        # Corrupt inputs: independent masking with probability 1 - alpha_t
        mask_prob = 1.0 - alpha_t                               # (B,)
        x_t, masked_mask = mask_tokens(
            x0, mask_prob
        )  # x_t has MASK_TOKEN_ID (0) at masked positions

        # Forward pass through the denoiser (time‑embedding‑free)
        logits = self.model.get_logits(x_t)                     # (B, L, num_real_tokens)

        # Convert ground‑truth tokens to 0‑based indices for cross‑entropy.
        # The model outputs logits only over real tokens (1..m), so we
        # shift the target indices by -1.
        targets = x0 - 1                                        # {0..m-1}, valid only for non‑mask
        targets = targets.clamp(min=0)  # safety clamp; masked positions are not used

        # Cross‑entropy per position, reduction='none' preserves shape (B, L)
        ce_loss = F.cross_entropy(
            logits.permute(0, 2, 1), targets, reduction="none"
        )                                                       # (B, L)

        # Sum loss over masked positions within each sequence
        mask_float = masked_mask.float()                        # (B, L)
        loss_masked = (ce_loss * mask_float).sum(dim=1)         # (B,)

        # Weighting factor (clamp denominator to avoid division by zero)
        eps = 1e-4
        weight = alpha_prime / (1.0 - alpha_t + eps)            # (B,)

        # Per‑sequence weighted loss; mean over batch approximates the integral
        weighted_loss = (weight * loss_masked).mean()
        return weighted_loss

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _compute_total_steps(self) -> int:
        """Determine the total number of optimizer steps from the config."""
        if self.config.training.num_iterations is not None:
            return self.config.training.num_iterations
        elif self.config.training.epochs is not None:
            # Number of batches per epoch (approximate)
            steps_per_epoch = len(self.train_loader)
            return self.config.training.epochs * steps_per_epoch // self._grad_accum_steps
        else:
            raise ValueError("Either num_iterations or epochs must be set in the training config.")

    def _create_scheduler(self) -> LambdaLR:
        """
        Create a learning‑rate scheduler with linear warmup followed by
        cosine decay to ``min_learning_rate``.
        """
        warmup_steps = self.config.training.warmup_steps
        total_steps = self.total_steps
        initial_lr = self.config.training.learning_rate
        min_lr = self.config.training.min_learning_rate

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay from initial to min
                progress = float(current_step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                # Interpolate between initial and min
                return (min_lr + (initial_lr - min_lr) * cosine_decay) / initial_lr

        return LambdaLR(self.optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int, final: bool = False) -> None:
        """
        Save model, optimizer, and scheduler states.

        Args:
            step: The current training step.
            final: If ``True``, save as ``final.pt``; otherwise as
                   ``checkpoint_{step}.pt``.
        """
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": step,
            "config": self.config,
        }
        if final:
            path = self._checkpoint_dir / "final.pt"
        else:
            path = self._checkpoint_dir / f"checkpoint_{step}.pt"
        torch.save(checkpoint, path)
        if self.use_wandb:
            wandb.save(str(path))

    def load_checkpoint(self, path: Union[str, Path]) -> int:
        """
        Restore training state from a checkpoint.

        Args:
            path: Path to the checkpoint file.

        Returns:
            The step at which the checkpoint was saved.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["step"]
