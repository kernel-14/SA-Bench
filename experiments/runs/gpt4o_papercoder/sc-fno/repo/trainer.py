## trainer.py

# Import necessary libraries
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from typing import Dict, Optional, Tuple
from model import Model
from losses import LossFunctions
import os

class Trainer:
    """
    Trainer class responsible for managing the training loop, validation, 
    and checkpointing for the SC-FNO model.
    """

    def __init__(
        self,
        model: Model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict
    ):
        """
        Initialize the Trainer with model, dataloaders, and configuration.

        Args:
            model (Model): The SC-FNO model instance.
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            config (Dict): Configuration dictionary loaded from config.yaml.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to configured device
        self.model.to(self.device)

        # Optimizer setup
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config["training"].get("learning_rate", 0.001)
        )

        # Loss function instance
        self.loss_functions = LossFunctions(config_path="config/config.yaml")

        # Logging containers
        self.train_metrics = []
        self.val_metrics = []

        # Initialize optional learning rate scheduler
        self.scheduler = None
        self.epochs = config["training"].get("epochs", 500)
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self) -> None:
        """
        Main training loop across epochs.
        """
        print("Starting training...")

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            # Iterate through batches in training loader
            for batch in self.train_loader:
                inputs, targets, gradients = batch
                inputs, targets, gradients = (
                    inputs.to(self.device),
                    targets.to(self.device),
                    gradients.to(self.device),
                )

                # Forward pass
                predicted_outputs = self.model.forward(inputs)

                # Compute sensitivities (Jacobian)
                predicted_sensitivities = self.model.compute_sensitivities(inputs)

                # Calculate losses
                primary_loss = self.loss_functions.compute_primary_loss(predicted_outputs, targets)
                sensitivity_loss = self.loss_functions.compute_sensitivity_loss(
                    predicted_sensitivities, gradients
                )

                total_loss = self.loss_functions.compute_total_loss(primary_loss, sensitivity_loss)

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Accumulate training loss
                train_loss += total_loss.item()

            # Average loss over batches
            train_loss /= len(self.train_loader)
            self.train_metrics.append(train_loss)
            print(f"Epoch {epoch + 1}/{self.epochs}, Training Loss: {train_loss:.6f}")

            # Run validation
            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.epochs:
                val_loss = self.validate(epoch)
                self.val_metrics.append(val_loss)

            # Save checkpoint
            self.save_checkpoint(epoch)

    def validate(self, epoch: int) -> float:
        """
        Run validation on the model.

        Args:
            epoch (int): The current epoch number.

        Returns:
            float: Validation loss for the epoch.
        """
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets, gradients = batch
                inputs, targets, gradients = (
                    inputs.to(self.device),
                    targets.to(self.device),
                    gradients.to(self.device),
                )

                # Forward pass
                predicted_outputs = self.model.forward(inputs)

                # Compute sensitivities (Jacobian)
                predicted_sensitivities = self.model.compute_sensitivities(inputs)

                # Calculate losses
                primary_loss = self.loss_functions.compute_primary_loss(predicted_outputs, targets)
                sensitivity_loss = self.loss_functions.compute_sensitivity_loss(
                    predicted_sensitivities, gradients
                )

                total_loss = self.loss_functions.compute_total_loss(primary_loss, sensitivity_loss)

                # Accumulate validation loss
                val_loss += total_loss.item()

        # Average loss over batches
        val_loss /= len(self.val_loader)
        print(f"Validation Loss after Epoch {epoch + 1}: {val_loss:.6f}")
        return val_loss

    def save_checkpoint(self, epoch: int, path: Optional[str] = None) -> None:
        """
        Save model and optimizer state to a checkpoint file.

        Args:
            epoch (int): Current epoch number.
            path (Optional[str]): Path to save the checkpoint. Defaults to ./checkpoints/.
        """
        checkpoint_path = path or os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch + 1
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, path: str) -> int:
        """
        Load model and optimizer states from a checkpoint file.

        Args:
            path (str): Path to the checkpoint file.

        Returns:
            int: Epoch to resume training from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file '{path}' does not exist!")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {path}, resuming from Epoch {epoch + 1}")
        return epoch
