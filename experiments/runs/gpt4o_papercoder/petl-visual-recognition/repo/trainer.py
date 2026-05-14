## trainer.py

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Optional
from utils import set_random_seed, save_results

class Trainer:
    """
    Trainer Class: Manages the training process for models using PEFT methods. It handles training, validation,
    and saving results in accordance with configurations provided.
    """

    def __init__(self, 
                 model: nn.Module, 
                 optimizer: str = "AdamW", 
                 scheduler: str = "cosine_decay", 
                 train_data: torch.utils.data.Dataset, 
                 val_data: Optional[torch.utils.data.Dataset] = None, 
                 config: dict = None):
        """
        Initializes the Trainer class.

        Args:
            model (nn.Module): The model to be trained with integrated PEFT methods.
            optimizer (str): The optimizer name (default is "AdamW").
            scheduler (str): The learning rate scheduler name (default is "cosine_decay").
            train_data (torch.utils.data.Dataset): The training dataset.
            val_data (Optional[torch.utils.data.Dataset]): The validation dataset (default is None).
            config (dict): The configuration dictionary loaded from 'config.yaml'.
        """
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config or {}
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        
        # Load training config from the provided dictionary
        training_config = self.config.get("training", {})
        self.learning_rate = training_config.get("learning_rate", [0.001])[0]
        self.weight_decay = training_config.get("weight_decay", [0.0001])[0]
        self.batch_size = training_config.get("batch_size", 64)
        self.drop_path_rate = training_config.get("drop_path_rate", 0.1)
        self.epochs = training_config.get("epochs", 100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Seed for reproducibility
        seed = training_config.get("seed", 42)
        set_random_seed(seed)

        # Move the model to the appropriate device
        self.model = self.model.to(self.device)

        # Initialize progress tracking
        self.metrics = {
            "train_loss": [],
            "val_accuracy": [],
            "learning_rate": []
        }

        # Setup optimizer and scheduler
        self.optimizer = self._initialize_optimizer()
        self.scheduler = self._initialize_scheduler()

    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        """
        Initializes the optimizer with the model's tunable parameters.

        Returns:
            torch.optim.Optimizer: The optimizer instance.
        """
        tunable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.optimizer_name == "AdamW":
            return AdamW(tunable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer '{self.optimizer_name}' is not implemented.")

    def _initialize_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Initializes the learning rate scheduler.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The scheduler instance.
        """
        if self.scheduler_name == "cosine_decay":
            return CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        else:
            raise NotImplementedError(f"Scheduler '{self.scheduler_name}' is not implemented.")

    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for the predictions and targets.

        Args:
            predictions (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed loss.
        """
        criterion = nn.CrossEntropyLoss()  # Assuming classification tasks
        return criterion(predictions, targets)

    def _evaluate(self) -> float:
        """
        Evaluates the model on the validation dataset and computes Top-1 Accuracy.

        Returns:
            float: Validation Top-1 Accuracy.
        """
        self.model.eval()
        correct = 0
        total = 0

        if self.val_data is None:
            return 0.0

        val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total * 100.0
        return val_accuracy

    def train(self, epochs: int = None, save_model: bool = True) -> None:
        """
        Conducts the training process for a given number of epochs.

        Args:
            epochs (int): Number of epochs for training (default is None, using config).
            save_model (bool): Whether to save the trained model (default is True).
        """
        epochs = epochs or self.epochs
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                predictions = self.model(inputs)
                loss = self._compute_loss(predictions, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track the loss
                epoch_loss += loss.item()

            # Scheduler step
            self.scheduler.step()
            learning_rate = self.scheduler.get_last_lr()[0]
            self.metrics["learning_rate"].append(learning_rate)

            # Validation
            val_accuracy = self._evaluate()

            # Track metrics
            self.metrics["train_loss"].append(epoch_loss / len(train_loader))
            self.metrics["val_accuracy"].append(val_accuracy)

            # Logging the metrics
            print(f"Epoch [{epoch + 1}/{epochs}]: Train Loss: {epoch_loss / len(train_loader):.4f} "
                  f"| Val Accuracy: {val_accuracy:.2f}% | LR: {learning_rate:.6f}")

        # Save the final model and metrics if required
        if save_model:
            model_save_path = "trained_model.pth"
            torch.save(self.model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}.")

        # Save metrics
        metrics_save_path = "training_metrics.json"
        save_results(self.metrics, metrics_save_path)
        print(f"Training metrics saved to {metrics_save_path}.")
