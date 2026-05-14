## trainer.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Dict, Any
from utils import compute_grad_norm, log_message
from memoryless_noise_schedule import NoiseSchedule
from model import BaseModel


class Trainer:
    """
    Trainer class for managing the training and fine-tuning loops for Flow Matching and Diffusion models.
    This handles data loading, model optimization, loss computation, logging metrics, and model checkpointing.
    """
    def __init__(self, model: BaseModel, config: Dict[str, Any], train_loader: DataLoader, val_loader: DataLoader):
        """
        Initialize the Trainer with the model, dataset loaders, and configuration.

        Args:
            model (BaseModel): Model instance of FlowMatching or Diffusion type.
            config (Dict[str, Any]): Configuration dictionary loaded from `config.yaml`.
            train_loader (DataLoader): DataLoader for training dataset.
            val_loader (DataLoader): DataLoader for validation dataset.
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer settings
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training'].get('learning_rate', 2e-5),
            betas=tuple(self.config['training'].get('adam_betas', [0.95, 0.999])),
            weight_decay=self.config['training'].get('weight_decay', 1e-2)
        )
        self.gradient_clipping = self.config['training'].get('gradient_clipping', 1.0)

        # Learning rate scheduler (optional)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training'].get('epochs', 50)
        )

        # Set noise schedule
        self.noise_schedule = NoiseSchedule(self.config)

        # Logging and checkpointing settings
        self.log_dir = self.config['logging'].get('log_dir', "./logs")
        self.checkpoint_path = self.config['checkpoint'].get('checkpoint_path', "./checkpoints")
        self.save_model_every = self.config['logging'].get('save_model_every', 1)
        self.print_frequency = self.config['logging'].get('print_frequency', 100)

        # Reproducibility seed
        torch.manual_seed(self.config['general'].get('seed', 42))

        # Device configuration
        self.device = self.config['general'].get('device', "cuda")
        self.model.to(self.device)

    def train(self) -> None:
        """
        Execute the primary training loop. Iterate over epochs and batches, compute losses, perform
        backpropagation, clip gradients, and save checkpoints periodically.
        """
        epochs = self.config['training'].get('epochs', 50)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for step, (inputs, targets) in enumerate(self.train_loader):
                # Move inputs and targets to the specified device
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass and compute loss
                self.optimizer.zero_grad()
                noise_schedule = self.noise_schedule.get_schedule()
                loss = self.compute_losses(inputs, targets, noise_schedule)

                # Backpropagation
                loss.backward()

                # Gradient clipping
                if self.gradient_clipping > 0:
                    grad_norm = compute_grad_norm(self.model.parameters())
                    if grad_norm > self.gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

                # Optimization step
                self.optimizer.step()
                running_loss += loss.item()

                # Logging (per step)
                if step % self.print_frequency == 0:
                    log_message(f"[INFO] Epoch: {epoch+1}, Step: {step}, Loss: {loss.item():.4f}")

            # Update learning rate scheduler
            if self.scheduler:
                self.scheduler.step()

            # Epoch-level logging
            log_message(
                f"[INFO] Epoch {epoch+1}/{epochs} | Average Loss: {running_loss / len(self.train_loader):.4f}"
            )

            # Save model checkpoint periodically
            if (epoch + 1) % self.save_model_every == 0:
                self.save_model(epoch)

            # Validate model at epoch-level
            self._validate(epoch)

    def compute_losses(self, inputs: torch.Tensor, targets: torch.Tensor, noise_schedule: list) -> torch.Tensor:
        """
        Compute loss values based on the training task (pre-training or fine-tuning).

        Args:
            inputs (torch.Tensor): Training data inputs (e.g., images or latent representations).
            targets (torch.Tensor): Ground truth targets (e.g., velocity or noise vector fields).
            noise_schedule (list): Noise schedule values for timesteps.

        Returns:
            torch.Tensor: Computed loss value.
        """
        if self.config.get("mode", "fine-tuning") == "pre-training":
            # Velocity matching for Flow Matching
            predictions = self.model(inputs, noise_schedule)
            loss = torch.mean((predictions - targets) ** 2)
        else:
            # Fine-tuning using Adjoint Matching
            from adjoint_matching import AdjointMatching
            adjoint_matching = AdjointMatching(self.model, self.config, self.train_loader)
            loss = adjoint_matching.compute_adjoint_loss()

        return loss

    def save_model(self, epoch: int) -> None:
        """
        Save the model's state along with optimizer and training epoch info.

        Args:
            epoch (int): Current training epoch.
        """
        save_path = f"{self.checkpoint_path}/epoch_{epoch}.pth"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch
        }, save_path)
        log_message(f"[INFO] Model checkpoint saved to {save_path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load the model's state from a checkpoint to resume training.

        Args:
            path (str): Path to the checkpoint file.
        """
        if not path or not os.path.exists(path):
            log_message(f"[ERROR] Checkpoint not found at {path}", verbose=True)
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        log_message(f"[INFO] Resumed training from checkpoint at epoch {checkpoint['epoch']}")
        
    def _validate(self, epoch: int) -> None:
        """
        Validate the model performance on the validation set and log metrics.

        Args:
            epoch (int): Current training epoch.
        """
        self.model.eval()
        validation_loss = 0.0
        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                noise_schedule = self.noise_schedule.get_schedule()
                loss = self.compute_losses(inputs, targets, noise_schedule)
                validation_loss += loss.item()

        avg_val_loss = validation_loss / len(self.val_loader)
        log_message(f"[INFO] Validation Loss at Epoch {epoch+1}: {avg_val_loss:.4f}")
