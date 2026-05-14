# trainer.py

"""
Trainer module for the Universal Neural Operators reproduction pipeline.

The Trainer orchestrates training and validation loops for multiphysics
neural operator models. It can be used in three modes:

- **pretrain**: train all parameters (body + multiple adapters) on a mix of
  PDE problems provided by a MultiPhysicsLoader.
- **finetune**: freeze the shared body and train only a new adapter on a
  single target problem. The optimizer will automatically ignore frozen
  parameters.
- **scratch**: train a randomly initialised model (body + single adapter)
  on a single problem; equivalent to the “from scratch” baseline.

The class expects that the model's parameters have already been frozen
or unfrozen by the caller (e.g., via `model.freeze_body()`) before
construction. The optimizer is created on the model's *current* trainable
parameters.

Configuration:
  All hyperparameters are read from the corresponding phase section of the
  global Config object (e.g., config.training.pretrain for phase 'pretrain').
  See config.yaml for the default values.

Logging:
  Training progress and validation metrics are printed to stdout every
  `print_freq` batches and written to TensorBoard if enabled.

Checkpointing:
  The best model (by validation loss) is saved to config.logging.checkpoint_dir.
"""

import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

# Intra-package imports (assuming these modules exist in the same directory)
from models import ModelBase
from data_utils import MultiPhysicsLoader
from config import Config


class Trainer:
    """
    Training harness for pre‑training, fine‑tuning and scratch experiments.

    Attributes:
        model (ModelBase): The neural operator model with per‑problem adapters.
        train_loader (MultiPhysicsLoader): Yields (problem_name, x, y) batches for training.
        val_loader (Optional[MultiPhysicsLoader]): If provided, yields validation batches.
        phase (str): One of {'pretrain', 'finetune', 'scratch'}.
        config (Config): Global configuration.
        epochs (int): Number of training epochs.
        lr (float): Base learning rate.
        optimizer (torch.optim.Optimizer): AdamW optimizer on trainable parameters.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
        criterion (nn.Module): Loss function (MSE).
        grad_clip (float): Gradient clipping value (0 means no clipping).
        writer (Optional[SummaryWriter]): TensorBoard writer.
        best_val_loss (float): Best observed validation loss.
        current_epoch (int): Last completed epoch.
        epoch_times (list): Per‑epoch wall‑clock times (in seconds).
        avg_epoch_time (float): Average epoch time computed after training.
    """

    def __init__(
        self,
        model: ModelBase,
        train_loader: MultiPhysicsLoader,
        config: Config,
        val_loader: Optional[MultiPhysicsLoader] = None,
        phase: str = "pretrain",
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            model: A ModelBase instance with pre-configured adapters; the body
                   may be frozen depending on the caller's setup.
            train_loader: Iterable that yields (problem, x, y) tuples.
            config: Full configuration object.
            val_loader: Optional validation data loader.
            phase: Which training phase to use ('pretrain', 'finetune', 'scratch').
                   Determines hyperparameter retrieval from config.training[phase].
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.phase = phase
        self.config = config

        # Validate phase and retrieve training parameters
        allowed_phases = {"pretrain", "finetune", "scratch"}
        if phase not in allowed_phases:
            raise ValueError(
                f"Invalid phase '{phase}'. Must be one of {allowed_phases}."
            )
        train_cfg = config.training_params.get(phase)
        if train_cfg is None:
            raise ValueError(
                f"Configuration for training phase '{phase}' is missing in config.training."
            )

        self.epochs = train_cfg.get("epochs", 500)                # default 500
        self.lr = train_cfg.get("learning_rate", 1e-3)            # default 1e-3
        weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.grad_clip = train_cfg.get("grad_clip", 1.0)          # 0 means off
        optimizer_type = train_cfg.get("optimizer", "adamw")
        scheduler_type = train_cfg.get("scheduler", "cosine")

        # Build optimizer (only parameters with requires_grad=True)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        if optimizer_type == "adamw":
            self.optimizer = AdamW(trainable_params, lr=self.lr, weight_decay=weight_decay)
        else:
            # Extendable to other optimizers; default to AdamW
            self.optimizer = AdamW(trainable_params, lr=self.lr, weight_decay=weight_decay)

        # Build scheduler
        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        elif scheduler_type is None or scheduler_type == "none":
            self.scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        # Loss function – paper uses MSE implicitly.
        self.criterion = nn.MSELoss()

        # Logging
        log_cfg = config.log_params.get("logging", {})
        self.print_freq = log_cfg.get("print_freq", 10)
        self.log_dir = log_cfg.get("log_dir", "./logs")
        self.checkpoint_dir = log_cfg.get("checkpoint_dir", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.tensorboard = log_cfg.get("tensorboard", False)
        self.writer = None
        if self.tensorboard:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        # Tracking
        self.best_val_loss = float("inf")
        self.current_epoch = 0
        self.epoch_times = []
        self.avg_epoch_time = 0.0

    def train(self) -> None:
        """
        Run the full training loop for the configured number of epochs.
        At the end, the best model checkpoint is loaded back.
        """
        device = next(self.model.parameters()).device

        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Training phase
            self.model.train()
            train_loss = self._train_epoch(device)

            epoch_duration = time.time() - epoch_start
            self.epoch_times.append(epoch_duration)

            # Step scheduler before validation (cosine typically after each epoch)
            if self.scheduler is not None:
                self.scheduler.step()

            # Validation & checkpointing
            if self.val_loader is not None:
                val_loss = self.validate(self.val_loader, device)
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, val_loss)
                val_str = f"val_loss: {val_loss:.6f}"
            else:
                # Without validation, save based on training loss (best or last)
                is_best = train_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = train_loss
                self._save_checkpoint(epoch, train_loss)
                val_str = "val_loss: N/A"

            # Print progress
            if self.print_freq > 0 and (epoch % self.print_freq == 0 or epoch == self.epochs):
                lr_current = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch:4d}/{self.epochs:4d}] "
                    f"train_loss: {train_loss:.6f} | {val_str} | "
                    f"time: {epoch_duration:.2f}s | lr: {lr_current:.2e}"
                )

            # TensorBoard
            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                if self.val_loader is not None:
                    self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Time/epoch", epoch_duration, epoch)
                self.writer.add_scalar("lr", lr_current, epoch)

        # After training, compute average epoch time
        self.avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)

        # Load the best checkpoint (if any) back into the model
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"best_model_phase_{self.phase}.pt"
        )
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"Loaded best model (val loss {checkpoint['val_loss']:.6f}) "
                f"from epoch {checkpoint['epoch']}."
            )

        if self.writer:
            self.writer.close()

    def _train_epoch(self, device: torch.device) -> float:
        """
        Iterate over the training loader for one full epoch and return
        the average loss over all batches.

        Args:
            device: The torch device to place tensors on.

        Returns:
            Average training loss for the epoch.
        """
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (problem_name, x, y) in enumerate(self.train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            self.optimizer.zero_grad()
            pred = self.model(problem_name, x)
            loss = self.criterion(pred, y)
            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Optional: log per batch (if very frequent)
            if self.print_freq > 0 and batch_idx % (self.print_freq * 10) == 0:
                print(
                    f"   Batch [{batch_idx:5d}] loss: {loss.item():.6f}"
                )

        return total_loss / n_batches

    @torch.no_grad()
    def validate(
        self,
        val_loader: MultiPhysicsLoader,
        device: torch.device,
    ) -> float:
        """
        Evaluate the model on the entire validation loader.

        Args:
            val_loader: MultiPhysicsLoader yielding (problem, x, y).
            device: Torch device for tensors.

        Returns:
            Average validation loss over all batches.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for problem_name, x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = self.model(problem_name, x)
            loss = self.criterion(pred, y)
            total_loss += loss.item()
            n_batches += 1

        # Return to training mode isn't strictly necessary because train()
        # sets model.train() at the beginning of each epoch, but safe.
        self.model.train()
        return total_loss / n_batches

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """
        Save a checkpoint containing model, optimizer, and current state.

        Args:
            epoch: The epoch number.
            val_loss: The validation (or training) loss at this point.
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"best_model_phase_{self.phase}.pt"
        )
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }
        torch.save(state, checkpoint_path)

    def time_epoch(self) -> float:
        """
        Return the average epoch time computed during training.
        If training has not been performed, raise an error.

        Returns:
            Average wall‑clock time per epoch (seconds).
        """
        if not self.epoch_times:
            raise RuntimeError(
                "Training has not been run yet; cannot compute average epoch time."
            )
        return self.avg_epoch_time
