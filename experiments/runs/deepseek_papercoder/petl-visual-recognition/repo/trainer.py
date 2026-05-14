## trainer.py

"""
Training loop implementation for PEFT reproducibility study.

Supports both hyperparameter‑tuning mode (with validation and best‑accuracy
tracking) and final‑training mode (full‑dataset training without validation).
Uses AdamW optimizer, cosine‑decay learning‑rate scheduling, and stochastic
depth (drop path) regularization.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config import Config
from model_builder import PEFTModel   # expected to expose set_drop_path_rate and get_trainable_params

logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles model training for a single experiment.

    Attributes:
        model: the PEFT‑adapted model (already built).
        train_loader: DataLoader providing training batches.
        val_loader: optional DataLoader for validation.
        config: resolved Config with explicit training hyperparameters.
        device: torch device on which to run training.
        criterion: loss function (cross‑entropy).
        optimizer: AdamW optimizer (created in `train()`).
        scheduler: cosine‑annealing LR scheduler (created in `train()`).
    """

    def __init__(self,
                 model: PEFTModel,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader],
                 config: Config) -> None:
        """
        Args:
            model:          PEFT model with frozen backbone and trainable head.
            train_loader:   DataLoader for the training split.
            val_loader:     DataLoader for validation, or None for final‑training mode.
            config:         Config object containing the resolved hyperparameters.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.misc["device"])

        # Ensure model is on the correct device
        self.model.to(self.device)

        # Loss function (standard for classification)
        self.criterion = nn.CrossEntropyLoss()

        # Placeholders – created when training starts
        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[CosineAnnealingLR] = None

    # ------------------------------------------------------------------ #
    #  Internal helpers (called from `train()`)
    # ------------------------------------------------------------------ #
    def _configure_optimizer(self) -> AdamW:
        """
        Create AdamW optimizer over all trainable parameters of the model.
        Uses the learning rate and weight decay from the config.
        """
        trainable_params = self.model.get_trainable_params()
        if not trainable_params:
            raise RuntimeError("No trainable parameters found in the model.")
        lr = self._get_training_param("learning_rate")
        wd = self._get_training_param("weight_decay")
        return AdamW(trainable_params, lr=lr, weight_decay=wd)

    def _configure_scheduler(self, optimizer: AdamW) -> CosineAnnealingLR:
        """
        Create a cosine‑decay learning‑rate scheduler.
        Total number of epochs is taken from the config.
        """
        epochs = self._get_training_param("epochs")
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)

    def _train_epoch(self) -> float:
        """
        Perform one full training epoch.

        Returns:
            Average training loss over all batches.
        """
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            loss.backward()
            self.optimizer.step()

            # Accumulate weighted loss
            batch_size = batch_x.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

        avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
        return avg_loss

    # ------------------------------------------------------------------ #
    #  Public interface
    # ------------------------------------------------------------------ #
    def validate(self) -> float:
        """
        Evaluate the model on the validation set and return top‑1 accuracy.

        Returns:
            Validation accuracy as a float between 0 and 1.
        """
        if self.val_loader is None:
            raise RuntimeError("validate() called but no validation loader was provided.")

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                logits = self.model(batch_x)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        return correct / total if total > 0 else 0.0

    def train(self) -> Optional[float]:
        """
        Run the full training loop.

        - Applies drop‑path stochastic depth (from config).
        - Creates optimizer and scheduler.
        - Executes `epochs` training epochs.
        - If a validation loader is available, validates each epoch and
          tracks the best validation accuracy.  Returns this best accuracy;
          otherwise returns None.

        Returns:
            Best validation accuracy (float) if validation was enabled,
            else None.
        """
        # 1. Set stochastic depth rate (drop path)
        drop_path_rate = self._get_training_param("drop_path_rate")
        self.model.set_drop_path_rate(drop_path_rate)

        # 2. Build optimizer and scheduler
        self.optimizer = self._configure_optimizer()
        self.scheduler = self._configure_scheduler(self.optimizer)

        # 3. Training loop
        num_epochs = self._get_training_param("epochs")
        best_val_acc = 0.0 if self.val_loader is not None else None

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch()

            # Log progress
            if self.val_loader is not None:
                val_acc = self.validate()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                logger.info(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"train_loss: {train_loss:.4f} | val_acc: {val_acc:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"train_loss: {train_loss:.4f}"
                )

            # Step the scheduler once per epoch
            self.scheduler.step()

        return best_val_acc   # None if no validation

    # ------------------------------------------------------------------ #
    #  Small helper to retrieve training parameters
    # ------------------------------------------------------------------ #
    def _get_training_param(self, key: str):
        """
        Retrieve a training hyperparameter from the config’s training section.
        For VTAB and many‑shot, the config.training dict contains resolved
        single values (e.g., float for learning_rate). For robustness, the
        same structure holds.
        """
        val = self.config.training.get(key)
        if val is None:
            raise KeyError(f"Training parameter '{key}' not found in config.")
        return val
