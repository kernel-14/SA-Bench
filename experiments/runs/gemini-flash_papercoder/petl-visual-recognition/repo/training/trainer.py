## training/trainer.py

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Tuple, Iterator, List
from copy import deepcopy

# Import necessary components from other modules
from models.peft_model_wrapper import PEFTModelWrapper
from utils.logger import Logger
from training.optimizer_scheduler import create_optimizer_scheduler
# Evaluator needs to be imported here to avoid circular dependency if it also imports Trainer,
# but it's okay for Trainer to import Evaluator since validation is a separate step.
from evaluation.evaluator import Evaluator


class Trainer:
    """
    Orchestrates the training process for a PEFTModelWrapper instance.
    Manages epochs, optimization steps, validation, and saving of the best model.
    """

    def __init__(self, model: PEFTModelWrapper, config: Dict[str, Any], logger: Logger) -> None:
        """
        Initializes the Trainer instance.

        Args:
            model (PEFTModelWrapper): An initialized PEFTModelWrapper instance.
            config (Dict[str, Any]): The training configuration for the current experiment run.
            logger (Logger): An instance of the Logger utility.
        """
        self.model: PEFTModelWrapper = model
        self.config: Dict[str, Any] = config
        self.logger: Logger = logger

        # Determine the computational device
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize the loss function
        self.criterion: nn.Module = nn.CrossEntropyLoss()

        # Optimizer and scheduler will be initialized later by create_optimizer_scheduler
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[LRScheduler] = None

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[PEFTModelWrapper, Dict[str, Any]]:
        """
        Executes the main training loop across multiple epochs.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.

        Returns:
            Tuple[PEFTModelWrapper, Dict[str, Any]]:
                - The trained model instance, loaded with the weights corresponding to the best validation accuracy.
                - A dictionary containing the training history (epoch-wise metrics).
        """
        epochs: int = self.config.get('epochs', 10)  # Default epochs to 10 if not specified
        learning_rate: float = self.config.get('learning_rate') # Specific LR for current run
        weight_decay: float = self.config.get('weight_decay') # Specific WD for current run
        eval_frequency_epochs: int = self.config['training'].get('eval_frequency_epochs', 1)

        if learning_rate is None or weight_decay is None:
             raise ValueError("Learning rate and weight decay must be specified in the training config for the current run.")

        # Calculate total steps for the scheduler
        total_steps: int = len(train_loader) * epochs

        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = create_optimizer_scheduler(
            model_parameters=self.model.get_trainable_parameters(),
            total_steps=total_steps,
            lr=learning_rate,
            weight_decay=weight_decay,
            config=self.config
        )

        best_val_accuracy: float = -1.0
        best_model_state_dict: Optional[Dict[str, Any]] = None
        training_history: List[Dict[str, Any]] = []

        self.logger.info(f"Starting training for {epochs} epochs on device: {self.device}")
        self.logger.info(f"Total trainable parameters: {sum(p.numel() for p in self.model.get_trainable_parameters())}")

        for epoch in range(epochs):
            self.logger.info(f"--- Epoch {epoch + 1}/{epochs} ---")
            
            # Train for one epoch
            train_loss: float = self._run_epoch(epoch, train_loader)
            self.logger.log_metrics({"train_loss": train_loss}, epoch=epoch + 1, prefix="Train")

            # Validate periodically or at the last epoch
            if (epoch + 1) % eval_frequency_epochs == 0 or (epoch + 1) == epochs:
                val_metrics: Dict[str, Any] = self._validate(epoch, val_loader)
                self.logger.log_metrics(val_metrics, epoch=epoch + 1, prefix="Validation")
                
                current_val_accuracy: float = val_metrics.get('val_top1_accuracy', -1.0)
                
                # Update best model if current validation accuracy is higher
                if current_val_accuracy > best_val_accuracy:
                    best_val_accuracy = current_val_accuracy
                    best_model_state_dict = deepcopy(self.model.state_dict())
                    self.logger.info(f"New best model found at Epoch {epoch + 1} with validation accuracy: {best_val_accuracy:.4f}")
                
                # Record history
                epoch_history = {"epoch": epoch + 1, "train_loss": train_loss, **val_metrics}
                training_history.append(epoch_history)
            else:
                # Record history without validation metrics if not evaluated
                epoch_history = {"epoch": epoch + 1, "train_loss": train_loss}
                training_history.append(epoch_history)

        # After training loop, load the best model weights
        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
            self.logger.info(f"Loaded best model (validation accuracy: {best_val_accuracy:.4f}) for final evaluation.")
        else:
            self.logger.warning("No best model state dict saved. Returning model from last epoch.")

        return self.model, {"training_history": training_history, "best_val_accuracy": best_val_accuracy}

    def _run_epoch(self, epoch: int, train_loader: DataLoader) -> float:
        """
        Conducts a single pass over the training data, performing forward and
        backward propagation and parameter updates.

        Args:
            epoch (int): The current epoch number (0-indexed).
            train_loader (DataLoader): DataLoader for the training data.

        Returns:
            float: The average loss recorded across all batches in the epoch.
        """
        self.model.train()  # Set model to training mode
        total_loss: float = 0.0
        
        max_grad_norm: Optional[float] = self.config['training'].get('max_grad_norm')

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", leave=False)
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()  # Zero the gradients

            outputs: torch.Tensor = self.model(inputs)
            loss: torch.Tensor = self.criterion(outputs, targets)

            loss.backward()  # Backpropagation

            # Apply gradient clipping if configured
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            self.optimizer.step()  # Update model parameters
            self.scheduler.step()  # Update learning rate

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss: float = total_loss / len(train_loader)
        return avg_loss

    @torch.no_grad() # Disable gradient calculations for validation
    def _validate(self, epoch: int, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluates the current state of the model on the validation dataset.

        Args:
            epoch (int): The current epoch number (0-indexed).
            val_loader (DataLoader): DataLoader for the validation data.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics for the validation set.
        """
        self.model.eval()  # Set model to evaluation mode

        # Instantiate Evaluator to calculate metrics.
        # It needs the current model state.
        evaluator = Evaluator(self.model, self.config, self.logger)
        
        # Evaluate accuracy on the validation set
        val_metrics: Dict[str, Any] = evaluator.evaluate_accuracy(val_loader, prefix='val')
        
        return val_metrics

    # This method is removed as `create_optimizer_scheduler` is now a standalone function
    # in `training/optimizer_scheduler.py` as per design, and called once from `train`.
    # The `Trainer` class now accepts `learning_rate` and `weight_decay` as part of its
    # `config` passed during `__init__`, and these specific values are used to set up the optimizer.
    # The current design explicitly states `create_optimizer_scheduler` is a function.
    # Its logic was moved to `training/optimizer_scheduler.py`.
    # I need to ensure `train` method calls the imported function `create_optimizer_scheduler`.
    # (Self-correction: I already did this in `train` method for the initialization part.)

