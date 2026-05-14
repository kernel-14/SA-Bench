# trainer.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from typing import Dict, Tuple, Any
import numpy as np
from model import DiffusionModel
from utils import Utils


class Trainer:
    """
    Trainer class handles the training of diffusion models, including score function pretraining
    using score-matching techniques and validating forward and reverse processes during training.
    """

    def __init__(self, model: DiffusionModel, dataset: Tuple[torch.Tensor, torch.Tensor], config: Dict[str, Any]) -> None:
        """
        Initialize the Trainer class.

        Args:
            model (DiffusionModel): The diffusion model to train.
            dataset (Tuple[torch.Tensor, torch.Tensor]): Training and testing datasets.
            config (Dict[str, Any]): Configuration parameters.
        """
        self.model = model
        self.train_data, self.test_data = dataset
        self.config = config
        self.learning_rate = config["training"]["learning_rate"]
        self.batch_size = config["training"]["batch_size"]
        self.epochs = config["training"]["epochs"]
        self.gradient_clip = config["training"].get("gradient_clip", 1.0)
        
        # Set up optimizer
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        # Set up DataLoader
        self.train_loader = DataLoader(
            TensorDataset(self.train_data),
            batch_size=self.batch_size,
            shuffle=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Logging utilities
        self.utils = Utils()

    def pretrain_score_function(self) -> None:
        """
        Pretrain the score function using score-matching techniques.
        The score function estimates ∇ log p_(X_t)(x) and is learned by minimizing the error
        against true gradient values using noisy training samples.
        """
        print("Starting score function pretraining...")

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for batch in self.train_loader:
                # Retrieve the batch and send it to the device
                x_batch = batch[0].to(self.device)

                # Add noise to the data (simulate X_t for forward process)
                t = torch.rand(x_batch.size(0), device=self.device)  # Random t ∈ [0, 1]
                alpha_t = torch.sqrt(1 - t)
                noise = torch.randn_like(x_batch, device=self.device)  # Gaussian noise
                xt = alpha_t.view(-1, 1) * x_batch + torch.sqrt(t).view(-1, 1) * noise

                # Compute true score (numerically approximated gradient)
                true_score = (-noise / t.view(-1, 1)).detach()  # True score ∇ log p(X_t)

                # Predict the score using the model
                predicted_score = self.model.score_function(xt, t)

                # Calculate score-matching loss
                loss = torch.mean((predicted_score - true_score) ** 2)

                # Update model weights
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.optimizer.step()

                epoch_loss += loss.item()

            # Save and log metrics
            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")
            self.utils.save_checkpoint(self.model, epoch)
            self.utils.log_training_metrics(epoch, avg_loss)

        print("Score function pretraining completed.")

    def train(self) -> None:
        """
        Execute forward and reverse processes to validate the trained score function
        and ensure proper data transitions between noise and data distributions.
        """
        print("Starting forward and reverse process training...")

        # Generate noise samples for reverse process
        gaussian_noise = torch.randn(self.test_data.size(), device=self.device)
        y_pred = self.model.reverse_process(gaussian_noise)

        # Validate reverse process output
        forward_noisy_data = self.model.forward_process(self.test_data.to(self.device))

        # Ensure forward process outputs approach Gaussian distribution
        forward_error = torch.mean((forward_noisy_data - gaussian_noise) ** 2).item()
        print(f"Forward process validation error: {forward_error:.6f}")

        # Validate reverse process by reconstructing data
        reverse_error = torch.mean((y_pred - self.test_data.to(self.device)) ** 2).item()
        print(f"Reverse process reconstruction error: {reverse_error:.6f}")

        print("Training process completed.")

    def save_checkpoint(self, epoch: int) -> None:
        """
        Save a model checkpoint for a particular epoch.

        Args:
            epoch (int): The current epoch of training.
        """
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
        self.utils.save_checkpoint(self.model, epoch)
        print(f"Checkpoint saved: {checkpoint_name}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load the checkpoint to resume training or evaluation.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        self.model = self.utils.load_checkpoint(checkpoint_path)
        print(f"Checkpoint loaded: {checkpoint_path}")
