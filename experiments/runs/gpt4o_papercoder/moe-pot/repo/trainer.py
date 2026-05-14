## trainer.py

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict
from utilities import set_random_seeds, log_metrics
from moe_pot_model import MoEPOTModel
from torch.utils.data import DataLoader
import os


class Trainer:
    """
    Trainer class for handling model training and fine-tuning
    for the Mixture-of-Experts Operator Transformer (MoE-POT).
    """

    def __init__(
        self,
        model: MoEPOTModel,
        train_data: torch.utils.data.Dataset,
        val_data: torch.utils.data.Dataset,
        config: Dict,
    ):
        """
        Initialize the Trainer with model, datasets, and configurations.

        Args:
            model (MoEPOTModel): The neural network model for training.
            train_data (Dataset): Training dataset.
            val_data (Dataset): Validation dataset.
            config (Dict): Training configuration loaded from config.yaml.
        """
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config

        # Hyperparameters from the configuration
        self.epochs = config["training"].get("epochs_pretraining", 1000)
        self.warmup_epochs = config["training"].get("warmup_epochs_pretraining", 200)
        self.batch_size = config["training"].get("batch_size", 20)
        self.learning_rate = config["training"].get("learning_rate", 0.001)
        self.weight_decay = config["training"].get("weight_decay", 1e-6)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

        # Put model on the correct device
        self.model.to(self.device)

        # Optimizer and scheduler
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = self._get_lr_scheduler()

        # Set random seeds for reproducibility
        set_random_seeds()

        # Initialize DataLoaders
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        # Track training and validation metrics
        self.log = {"train_loss": [], "val_loss": [], "learning_rate": []}

    def _get_lr_scheduler(self):
        """
        Define a learning rate scheduler with a warm-up phase.
        """
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # Linear warmup
                return (epoch + 1) / self.warmup_epochs
            return 1.0  # Constant learning rate after warmup
        
        return LambdaLR(self.optimizer, lr_lambda)

    def train(self) -> None:
        """
        Main training loop: iterates over epochs and batches,
        evaluates on validation data, and saves checkpoints.
        """
        best_val_loss = float("inf")
        checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                predictions = self.model(inputs)

                # Compute loss
                losses = self.calculate_losses(predictions, targets)
                total_loss = losses["total_loss"]

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Accumulate batch loss
                train_loss += total_loss.item()

            # Scheduler adjustment
            self.scheduler.step()

            # Log average training loss
            avg_train_loss = train_loss / len(self.train_loader)
            self.log["train_loss"].append(avg_train_loss)

            # Validation phase
            val_loss = self._validate()
            self.log["val_loss"].append(val_loss)

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model_checkpoint(os.path.join(checkpoint_dir, "best_model.pth"))

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.log["learning_rate"].append(current_lr)

            # Print metrics for epoch
            print(f"Epoch [{epoch + 1}/{self.epochs}] "
                  f"Train Loss: {avg_train_loss:.6f} "
                  f"Val Loss: {val_loss:.6f} "
                  f"LR: {current_lr:.6f}")

            # Save metrics to file
            metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            }
            log_metrics(metrics, os.path.join(checkpoint_dir, "training_log.json"))

    def calculate_losses(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the losses for a batch of data.

        Args:
            predictions (Tensor): Model predictions.
            targets (Tensor): Ground truth targets.

        Returns:
            Dict[str, Tensor]: Loss components and the total loss.
        """
        # Prediction loss (L2 error)
        prediction_loss = torch.mean((predictions - targets) ** 2)

        # Load balancing loss from the Router-Gating Network
        load_balancing_loss = 0.0
        if hasattr(self.model, "router_gating_network"):
            router = self.model.router_gating_network
            expert_importance = torch.sum(router(input_tensor=predictions), dim=0)  # Sum over batches
            expert_variance = torch.var(expert_importance)
            load_balancing_loss = self.config["architecture"].get("w_balance", 0.1) * expert_variance

        # Total loss
        total_loss = prediction_loss + load_balancing_loss
        return {"prediction_loss": prediction_loss, "load_balancing_loss": load_balancing_loss, "total_loss": total_loss}

    def _validate(self) -> float:
        """
        Validate the model on the validation dataset.

        Returns:
            float: The average validation L2 Relative Error (L2RE).
        """
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                predictions = self.model(inputs)

                # Compute validation loss (L2 Relative Error)
                val_loss += torch.mean((predictions - targets) ** 2).item()

        return val_loss / len(self.val_loader)

    def save_model_checkpoint(self, path: str) -> None:
        """
        Save the model checkpoint to the specified path.

        Args:
            path (str): File path where the model checkpoint will be saved.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "log": self.log,
        }
        torch.save(checkpoint, path)
