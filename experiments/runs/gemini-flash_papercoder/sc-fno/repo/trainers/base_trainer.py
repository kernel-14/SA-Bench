## trainers/base_trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple

# Import project-specific modules
from config import Config
from losses import Losses
from utils import save_checkpoint

# Third-party library for progress bar
from tqdm import tqdm


class BaseTrainer(ABC):
    """
    Abstract base class for trainers. Provides a standardized framework for
    training and validating neural network models across different configurations
    (FNO, SC-FNO, SC-FNO-PINN).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any],
        config: Config,
        losses: Losses,
        device: str
    ) -> None:
        """
        Initializes the BaseTrainer with all necessary components for training.

        Args:
            model (nn.Module): The PyTorch model to be trained.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            optimizer (optim.Optimizer): PyTorch optimizer instance.
            scheduler (Optional[Any]): Optional learning rate scheduler. Can be None.
            config (Config): Configuration object containing experiment settings.
            losses (Losses): Instance of the Losses class to compute various loss components.
            device (str): The computational device ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.losses = losses
        self.device = device

        self.best_val_loss: float = float('inf')
        self.train_history: list[Dict[str, float]] = []
        self.val_history: list[Dict[str, float]] = []

        # Ensure checkpoint directory exists
        self.checkpoint_dir = self.config.get("experiment.checkpoint_dir", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.experiment_name = self.config.get("experiment.name", "default_experiment")

        print(f"Trainer initialized. Model on {self.device}.")

    def train(self) -> None:
        """
        Orchestrates the main training loop across multiple epochs.
        Handles training, validation, learning rate scheduling, and checkpointing.
        """
        num_epochs = self.config.get("training.epochs", 500)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")

            # Training Phase
            train_metrics = self._train_epoch(epoch)
            self.train_history.append(train_metrics)
            print(f"Train metrics: {train_metrics}")

            # Validation Phase
            val_metrics = self._validate_epoch(epoch)
            self.val_history.append(val_metrics)
            print(f"Validation metrics: {val_metrics}")

            # Learning Rate Scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Current Learning Rate: {current_lr:.6f}")

            # Checkpointing
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_best.pth")
                state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'train_history': self.train_history,
                    'val_history': self.val_history,
                    'config_params': self.config.params,
                }
                save_checkpoint(state, checkpoint_path)
                print(f"New best model saved with validation loss: {self.best_val_loss:.6f}")
            
            # Optional: Save checkpoint every N epochs regardless of performance
            checkpoint_interval = self.config.get("training.checkpoint_interval", 0)
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                periodic_checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_epoch_{epoch}.pth")
                state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'current_val_loss': val_metrics['total_loss'],
                    'train_history': self.train_history,
                    'val_history': self.val_history,
                    'config_params': self.config.params,
                }
                save_checkpoint(state, periodic_checkpoint_path)
                print(f"Periodic checkpoint saved at epoch {epoch}.")

        print("\nTraining complete.")

    @abstractmethod
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Abstract method to define the training logic for a single epoch.
        Must be implemented by subclasses.

        Args:
            epoch (int): The current epoch number.

        Returns:
            Dict[str, float]: A dictionary containing averaged training metrics for the epoch.
        """
        raise NotImplementedError

    @abstractmethod
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Abstract method to define the validation logic for a single epoch.
        Must be implemented by subclasses.

        Args:
            epoch (int): The current epoch number.

        Returns:
            Dict[str, float]: A dictionary containing averaged validation metrics for the epoch.
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_batch_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Abstract method to compute the total loss and individual loss components for a given batch.
        This method will vary significantly between FNO, SC-FNO, and SC-FNO-PINN due to
        differences in forward pass and loss components.

        Args:
            batch (Dict[str, torch.Tensor]): A dictionary containing batch data (e.g., u_true,
                                             du_true_dp, fno_input_encoder_data, fno_params_for_ad).

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: A tuple containing:
                - total_loss (torch.Tensor): The scalar total weighted loss for the batch.
                - loss_details (Dict[str, float]): A dictionary of individual loss values (e.g., 'u_loss', 's_loss').
        """
        raise NotImplementedError

