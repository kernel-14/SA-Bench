"""
trainer.py – Training loop for LoRA‑SB.

The Trainer class handles:
- Optimiser configuration (AdamW only).
- Learning rate scheduling (linear or cosine with warmup).
- Gradient accumulation as specified in the config.
- Per‑batch training step (forward pass).
- Logging of training loss (via tqdm and optional WandB).

All hyperparameters are read from the ExperimentConfig object.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

# Local imports (assume all other modules are in the same package)
from config import ExperimentConfig
from dataset import DatasetLoader
from modeling import ModelWrapper


class Trainer:
    """
    Orchestrates low‑rank fine‑tuning of a LoRA‑SB model.

    Only the trainable parameters (R matrices and, for GLUE, the classification head)
    are updated. The trainer supports arbitrary gradient accumulation and
    per‑epoch learning rate scheduling.

    Args:
        model: ModelWrapper containing the pre‑trained model with injected LoRA‑SB layers.
        data: DatasetLoader that provides the training DataLoader.
        config: ExperimentConfig with all training hyperparameters.
    """

    def __init__(
        self,
        model: ModelWrapper,
        data: DatasetLoader,
        config: ExperimentConfig,
    ) -> None:
        self.model = model
        self.config = config
        self.data = data

        # ---- Device handling ----
        self.device = config.device
        self.model.model.to(self.device)

        # ---- Trainable parameters ----
        trainable_params = model.get_trainable_parameters()
        if not trainable_params:
            raise RuntimeError(
                "No trainable parameters found. Ensure apply_lora_sb() has been called."
            )

        # ---- Optimiser ----
        self.optimizer = AdamW(
            trainable_params,
            lr=config.lr,
            weight_decay=0.01,   # sensible default; can be made configurable later
        )

        # ---- Learning rate scheduler ----
        self._configure_scheduler()

        # ---- Training state ----
        self.accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.epochs = config.epochs
        self._train_loader: Optional[DataLoader] = None

        # ---- Logging ----
        self._progress_bar: Optional[tqdm] = None
        self._global_step = 0  # used for logging

    def _configure_scheduler(self) -> None:
        """
        Compute total training steps and instantiate the scheduler.
        The scheduler is stepped after each optimizer update (i.e., after accumulation).
        """
        # Determine total number of optimizer steps
        num_batches = len(self.data.get_train_dataloader())
        total_steps = (num_batches * self.config.epochs) // self.accumulation_steps
        # Ensure at least one step to avoid scheduler issues
        if total_steps == 0:
            total_steps = 1

        warmup_steps = int(self.config.warmup_ratio * total_steps)

        if self.config.lr_scheduler.lower() == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif self.config.lr_scheduler.lower() == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            # Fallback to no scheduler (constant LR)
            self.scheduler = None

    def train(self) -> None:
        """
        Main training loop. Supports gradient accumulation and per‑epoch progress bars.
        """
        self.model.model.train()

        for epoch in range(self.epochs):
            # Re‑fetch dataloader each epoch to ensure proper shuffling
            train_loader = self.data.get_train_dataloader()
            self._progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                unit="batch",
                leave=True,
            )

            accumulated_loss = 0.0
            step_in_acc = 0

            for batch in self._progress_bar:
                loss = self._training_step(batch)
                # Scale the loss for gradient accumulation
                (loss / self.accumulation_steps).backward()

                accumulated_loss += loss.item()
                step_in_acc += 1

                if step_in_acc % self.accumulation_steps == 0:
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                    # Post‑update logging
                    avg_loss = accumulated_loss / self.accumulation_steps
                    self._progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

                    self._global_step += 1
                    accumulated_loss = 0.0

            # End of epoch: handle any remaining accumulated gradients
            if step_in_acc % self.accumulation_steps != 0:
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                avg_loss = accumulated_loss / (step_in_acc % self.accumulation_steps)
                # Update progress bar post‑fix for the last mini‑step
                self._progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

        # Cleanup
        self._progress_bar = None
        # Return the model to evaluation mode (caller may evaluate)
        self.model.model.eval()

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Single forward pass. Moves the batch to the correct device and returns the loss.

        Args:
            batch: dictionary containing inputs and labels (already tokenized).

        Returns:
            A scalar loss tensor for backpropagation.
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model.model(**batch)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError(
                "Model did not return a loss. Ensure labels are present in the batch."
            )
        return loss

