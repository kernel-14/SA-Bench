## trainer.py

"""
This module implements the `Trainer` class, which orchestrates the pretraining and fine-tuning of neural operator models.
It includes utilities for checkpoint management, logging, and leveraging datasets for the training pipeline.

Classes:
    - Trainer: Defines methods for pretraining, fine-tuning, and checkpoint saving/loading.
"""

import os
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import set_seed, log_metrics
import numpy as np


class Trainer:
    """
    The Trainer class is responsible for pretraining and fine-tuning neural operator models based on the methodology outlined in the paper.

    Attributes:
        model (nn.Module): The neural operator model being trained or fine-tuned.
        train_data (Dataset): Dataset used for training.
        val_data (Dataset): Dataset used for validation.
        config (dict): Configuration dictionary containing hyperparameters and file paths.
        optimizer (optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device to run the training (CPU/GPU).
    """

    def __init__(self, model: nn.Module, train_data: torch.utils.data.Dataset, val_data: torch.utils.data.Dataset, config: dict):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): Neural operator model (e.g., FNO, Mamba-SSM, Perceiver IO).
            train_data (Dataset): Training dataset.
            val_data (Dataset): Validation dataset.
            config (dict): Configuration dictionary.
        """
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config

        # Set device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Hyperparameters
        self.learning_rate = config["training"]["learning_rate"]
        self.fine_tuning_lr = config["fine_tuning"]["learning_rate"]
        self.batch_size = config["training"]["batch_size"]
        self.epochs = config["training"]["epochs"]
        self.fine_tuning_epochs = config["fine_tuning"]["epochs"]

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        # Logging and checkpoint directories
        self.checkpoint_dir = config["logging"]["checkpoint_dir"]
        self.log_dir = config["logging"]["log_dir"]

        # Deterministic behavior
        set_seed(config["random_seed"])

    def pretrain(self) -> None:
        """
        Pretrains the neural operator model on the training dataset.
        The model trains all components (lifting, kernel operator, and projection layers).
        """
        print("Starting pretraining...")

        # Ensure training mode on the model
        self.model.train()

        # DataLoader for batching
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                predictions = self.model(inputs)
                loss = self.loss_fn(predictions, targets)

                # Backward propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Logging metrics for the epoch
            epoch_loss /= len(train_loader)
            log_metrics({"epoch": epoch, "pretraining_loss": epoch_loss}, os.path.join(self.log_dir, "pretraining_log.yaml"))

            print(f"[Pretraining] Epoch {epoch}/{self.epochs}, Loss: {epoch_loss:.6f}")

            # Save checkpoint
            if epoch % 10 == 0:  # Save every 10 epochs
                checkpoint_path = os.path.join(self.checkpoint_dir, f"pretrain_epoch_{epoch}.pth")
                self.save_checkpoint(checkpoint_path)

            # Validate the model
            val_loss = self._validate(val_loader)
            print(f"[Pretraining] Validation Loss: {val_loss:.6f}")

    def finetune(self) -> None:
        """
        Fine-tunes the pretrained model on a downstream dataset.
        Only adapter-specific parameters (lifting and projection layers) are updated.
        """
        print("Starting fine-tuning...")

        # Freeze core operator layers
        self.model.freeze_layers(except_layers=["lifting", "projection"])

        # Update optimizer to focus on adapter-specific parameters
        adapter_params = [param for param in self.model.parameters() if param.requires_grad]
        adapter_optimizer = optim.Adam(adapter_params, lr=self.fine_tuning_lr)

        # DataLoader for batching
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

        for epoch in range(1, self.fine_tuning_epochs + 1):
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                predictions = self.model(inputs)
                loss = self.loss_fn(predictions, targets)

                # Backpropagation
                adapter_optimizer.zero_grad()
                loss.backward()
                adapter_optimizer.step()

                epoch_loss += loss.item()

            # Logging metrics for the epoch
            epoch_loss /= len(train_loader)
            log_metrics({"epoch": epoch, "fine_tuning_loss": epoch_loss}, os.path.join(self.log_dir, "fine_tuning_log.yaml"))

            print(f"[Fine-tuning] Epoch {epoch}/{self.fine_tuning_epochs}, Loss: {epoch_loss:.6f}")

            # Save checkpoint
            if epoch % 10 == 0:  # Save every 10 epochs
                checkpoint_path = os.path.join(self.checkpoint_dir, f"finetune_epoch_{epoch}.pth")
                self.save_checkpoint(checkpoint_path)

            # Validate the model
            val_loss = self._validate(val_loader)
            print(f"[Fine-tuning] Validation Loss: {val_loss:.6f}")

    def save_checkpoint(self, file_path: str) -> None:
        """
        Saves the current model and optimizer state to a checkpoint file.

        Args:
            file_path (str): Path to save the checkpoint.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, file_path)
        print(f"Checkpoint saved at {file_path}")

    def load_checkpoint(self, file_path: str) -> None:
        """
        Loads model and optimizer states from a checkpoint file.

        Args:
            file_path (str): Path to the checkpoint file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Checkpoint file '{file_path}' does not exist.")
        
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from {file_path}")

    def _validate(self, val_loader: DataLoader) -> float:
        """
        Validates the model on the validation dataset.

        Args:
            val_loader (DataLoader): DataLoader for the validation dataset.

        Returns:
            float: Validation loss.
        """
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                predictions = self.model(inputs)
                loss = self.loss_fn(predictions, targets)
                val_loss += loss.item()

        return val_loss / len(val_loader)
