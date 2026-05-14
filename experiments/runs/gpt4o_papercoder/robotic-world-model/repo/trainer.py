# trainer.py

import torch
import time
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
from rwm_model import RWMModel


class Trainer:
    """
    Trainer class to train the Robotic World Model (RWM) using an autoregressive training
    approach. Handles data preprocessing, model optimization, logging, and checkpointing.
    """

    def __init__(self, model: RWMModel, data: Dict[str, DataLoader], config: dict):
        """
        Initialize the Trainer with the model, dataset, and configuration.

        Args:
            model (RWMModel): The Robotic World Model to be trained.
            data (Dict[str, DataLoader]): Dictionary containing train, validation, 
                                          and/or test DataLoaders.
            config (dict): Configuration settings from `config.yaml`.
        """
        self.model = model
        self.data = data  # Train, validation DataLoaders
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Training parameters from config
        self.learning_rate = self.config["training"]["learning_rate"]
        self.weight_decay = self.config["training"]["weight_decay"]
        self.epochs = self.config["training"]["epochs"]
        self.gradient_clip = self.config["training"]["gradient_clip_val"]
        self.max_training_hours = self.config["training"].get("max_training_hours", 1)  # Default: 1 hour

        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Loss decay factor for forecast horizon
        self.forecast_decay = self.config["training"]["forecast_decay"]
        self.forecast_horizon = self.config["training"]["forecast_horizon"]  # `N`
        self.history_horizon = self.config["training"]["history_horizon"]  # `M`

        # Logging
        self.checkpoint_path = "checkpoints/"
        self.log_interval = 50  # Log every `log_interval` iterations

    def train(self) -> None:
        """
        Execute the training process for the Robotic World Model (RWM).
        """
        start_time = time.time()
        best_val_loss = float("inf")

        print("Starting training...")

        for epoch in range(self.epochs):
            print(f"Epoch [{epoch + 1}/{self.epochs}]")

            # Training phase
            train_loss = self._run_epoch(self.data["train"], training=True)

            # Validation phase - compute validation loss
            if "val" in self.data:
                val_loss = self._run_epoch(self.data["val"], training=False)
                print(f"Validation Loss: {val_loss:.4f}")
                # Save checkpoint if validation improves
                if val_loss < best_val_loss:
                    self._save_checkpoint(epoch)
                    best_val_loss = val_loss

            # Monitor training time and terminate if exceeds max hours
            elapsed_hours = (time.time() - start_time) / 3600
            if elapsed_hours > self.max_training_hours:
                print(f"Training terminated after {elapsed_hours:.2f} hours.")
                break

    def _run_epoch(self, dataloader: DataLoader, training: bool) -> float:
        """
        Run a single epoch, either for training or validation.

        Args:
            dataloader (DataLoader): DataLoader for the current phase (train/val).
            training (bool): Whether the model is in training mode.

        Returns:
            float: Average loss for the epoch.
        """
        # Select appropriate mode
        self.model.train() if training else self.model.eval()

        # Batches and loss accumulation
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (history, actions, targets_obs, targets_priv) in enumerate(dataloader):
            history = history.to(self.device)  # (B, M, input_dim)
            actions = actions.to(self.device)  # (B, M, action_dim)
            targets_obs = targets_obs.to(self.device)  # (B, N, output_dim)
            targets_priv = targets_priv.to(self.device)  # (B, N, privileged_dim)

            # Start forward pass
            with torch.set_grad_enabled(training):
                predictions = self.model(history, actions)  # forward pass
                loss = self.model.compute_loss(predictions, (targets_obs, targets_priv))

            # Backward pass and optimization if in training mode
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
                self.optimizer.step()

            # Accumulate loss for logging
            total_loss += loss.item()
            num_batches += 1

            # Log progress periodically
            if training and batch_idx % self.log_interval == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Compute average loss for the epoch
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def _save_checkpoint(self, epoch: int) -> None:
        """
        Save the model's state as a checkpoint during training.

        Args:
            epoch (int): The current training epoch.
        """
        save_file = f"{self.checkpoint_path}/rwm_epoch_{epoch}.pt"
        torch.save(self.model.state_dict(), save_file)
        print(f"Model checkpoint saved at: {save_file}")

