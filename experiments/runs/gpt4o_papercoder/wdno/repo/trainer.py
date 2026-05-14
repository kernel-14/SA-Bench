# trainer.py
"""
Trainer module for training the Wavelet Diffusion Neural Operator (WDNO).
This class manages the training and multi-resolution dataset handling for simulation and control tasks,
as well as checkpoint management for saving and restoring progress.
"""

import os
from typing import Any, Dict, Tuple, List
from tqdm import tqdm
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from utils import save_checkpoint, load_checkpoint, calculate_metrics
import math


class Trainer:
    """
    Trainer class to handle the training process for WDNO.

    Attributes:
        model (Model): The WDNO diffusion model.
        dataset (DatasetLoader): DatasetLoader instance for loading multi-resolution wavelet datasets.
        config (Dict[str, Any]): Configuration for training parameters such as learning rate, batch size, etc.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scheduler (Any): Learning rate scheduler (cosine annealing).
        checkpoint_path (str): Path to save and load model checkpoints.
    """

    def __init__(self, model: nn.Module, dataset: Any, config: Dict[str, Any]) -> None:
        """
        Initializes the Trainer class.

        Args:
            model (nn.Module): The WDNO model to train.
            dataset (Any): DatasetLoader instance with training and validation data.
            config (Dict[str, Any]): Configuration with training settings.
        """
        self.model = model
        self.dataset = dataset
        self.config = config

        # Extract training hyperparameters
        self.learning_rate = config["training"]["learning_rate"]
        self.batch_size = config["training"]["batch_size"]
        self.epochs = config["training"]["epochs"]
        self.checkpoint_path = config["training"]["model_checkpoint_path"]
        self.ddim_steps = config["training"]["ddim_steps"]
        self.control_guidance_intensity = config["training"]["control_guidance_intensity"]

        # Initialize optimizer and learning rate scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        # Define loss function
        self.loss_function = nn.MSELoss()

    def train(self) -> None:
        """
        Main training loop for the WDNO model. Handles multi-resolution training and checkpointing.
        """
        train_loader = self._get_train_dataloader()  # Wrap dataset with DataLoader
        self.model.train()

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            print(f"Epoch {epoch}/{self.epochs}")

            # Iterate over batches
            for batch_idx, batch_data in enumerate(tqdm(train_loader, desc="Training Batches")):
                # Handle multi-resolution dataset preparation
                low_res_data, high_res_data = self.handle_multi_resolution(batch_data)

                # Generate noisy data for training diffusion steps
                noise, time_step_schedule, clean_data = self._generate_training_noise(high_res_data)

                # Model forward pass
                self.optimizer.zero_grad()
                predictions = self.model.forward(noise, {
                    "initial_state": low_res_data,
                    "parameters": batch_data["parameters"]
                })

                # Compute loss
                loss = self.loss_function(predictions, clean_data)
                loss.backward()

                # Update model parameters and record loss
                self.optimizer.step()
                epoch_loss += loss.item()

            # Scheduler step
            self.scheduler.step()

            # Log epoch performance
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

            # Save checkpoints periodically
            if epoch % 1000 == 0 or epoch == self.epochs:
                self.save_checkpoint(epoch)

            # Perform periodic validation
            if epoch % 100 == 0:
                validation_metrics = self.validate()
                print(f"Validation Results (Epoch {epoch}): {validation_metrics}")

    def validate(self) -> Dict[str, float]:
        """
        Validates the model on the validation dataset.

        Returns:
            Dict[str, float]: Validation metrics such as MSE for evaluation.
        """
        val_loader = self._get_validation_dataloader()
        self.model.eval()
        validation_metrics = {"mse": 0.0}
        evaluation_count = 0

        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="Validation Batches"):
                low_res_data, high_res_data = self.handle_multi_resolution(batch_data)

                # Model inference
                predictions = self.model.sample(
                    time_steps=self.ddim_steps,
                    noise=torch.randn_like(low_res_data),
                    conditions={
                        "initial_state": low_res_data,
                        "parameters": batch_data["parameters"]
                    }
                )
                # Compute metrics
                mse = calculate_metrics(predictions, high_res_data)["mse"]
                validation_metrics["mse"] += mse
                evaluation_count += 1

        # Average metrics over all evaluation batches
        for key in validation_metrics:
            validation_metrics[key] /= evaluation_count

        return validation_metrics

    def save_checkpoint(self, epoch: int) -> None:
        """
        Saves the model's state and optimizer state to a checkpoint file.

        Args:
            epoch (int): The current training epoch.
        """
        checkpoint_data = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        save_checkpoint(checkpoint_data, self.checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch} to {self.checkpoint_path}")

    def load_checkpoint(self, file_path: str) -> None:
        """
        Loads model and optimizer state from a checkpoint file.

        Args:
            file_path (str): Path to the checkpoint file.
        """
        checkpoint_data = load_checkpoint(file_path)
        if checkpoint_data is None:
            raise ValueError(f"Checkpoint file not found at {file_path}")

        self.model.load_state_dict(checkpoint_data["model_state"])
        self.optimizer.load_state_dict(checkpoint_data["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint_data["scheduler_state"])
        print(f"Checkpoint loaded from {file_path}")

    def handle_multi_resolution(self, batch_data: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Prepares multi-resolution dataset pairs for super-resolution training tasks.

        Args:
            batch_data (Tensor): Input batch data tensor.

        Returns:
            Tuple[List[Tensor], List[Tensor]]: Pairs of low-resolution and high-resolution data tensors.
        """
        # Downsample data for low-resolution
        high_res_data = batch_data["high_resolution"]
        low_res_data = batch_data["low_resolution"]

        return low_res_data, high_res_data

    def _get_train_dataloader(self):
        """
        Wraps the training dataset into a DataLoader for batch processing.

        Returns:
            DataLoader: PyTorch DataLoader for training dataset.
        """
        return DataLoader(
            dataset=self.dataset.load_data()[0]["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def _get_validation_dataloader(self):
        """
        Wraps the validation dataset into a DataLoader for batch processing.

        Returns:
            DataLoader: PyTorch DataLoader for validation dataset.
        """
        return DataLoader(
            dataset=self.dataset.load_data()[0]["validation"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

    def _generate_training_noise(self, clean_data: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generates noisy data for diffusion model training.

        Args:
            clean_data (Tensor): Ground truth data.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (Noise, time step schedule, ground truth data).
        """
        batch_size = clean_data.size(0)
        time_steps = torch.randint(0, self.ddim_steps, size=(batch_size,), device=clean_data.device)
        alphas = torch.gather(self.model.alpha_bar, 0, time_steps)
        noise = torch.randn_like(clean_data)

        noisy_data = torch.sqrt(alphas).unsqueeze(-1) * clean_data + torch.sqrt(1 - alphas).unsqueeze(-1) * noise

        return noisy_data, time_steps, clean_data
