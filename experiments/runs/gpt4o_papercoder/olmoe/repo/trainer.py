## trainer.py
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, Any
from datasets import Dataset
import wandb

from model import Model
from utils import set_random_seed, generate_optimizer, generate_scheduler, log_metrics


class Trainer:
    """
    Handles training workflows, including pretraining and adaptation for the Mixture-of-Experts model.
    """

    def __init__(self, model: Model, train_data: Dataset, config: Dict[str, Any]) -> None:
        """
        Initialize Trainer with model, training data, and configuration.

        Args:
            model (Model): MoE-based transformer model.
            train_data (Dataset): Training dataset (pretraining or adaptation).
            config (Dict[str, Any]): Configuration dictionary containing hyperparameters and settings.
        """
        self.model = model
        self.train_data = train_data
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        set_random_seed(self.config.get("global_seed", 42))
        self.optimizer = None
        self.scheduler = None

    def _initialize_training_components(self, phase: str) -> None:
        """
        Initialize optimizer and learning rate scheduler for the specified training phase.

        Args:
            phase (str): Training phase, either 'pretraining', 'sft', or 'dpo'.
        """
        phase_config = self.config["training"].get(phase, {})
        total_steps = len(self.train_data) * phase_config.get("epochs", 1)

        self.optimizer = generate_optimizer(self.model, phase_config)
        if phase == "pretraining":
            self.scheduler = generate_scheduler(self.optimizer, self.config, total_steps)
        else:
            self.scheduler = None  # Constant learning rate for SFT and DPO.

    def train_pretraining(self) -> None:
        """
        Train the model on the pretraining dataset using Mixture-of-Experts architecture.

        Implements token-choice routing and auxiliary loss calculations during training.
        """
        self._initialize_training_components("pretraining")

        self.model.train()
        self.optimizer.zero_grad()
        
        for epoch in range(int(self.config["training"]["pretraining"]["epochs"])):
            for step, batch in enumerate(self.train_data):
                inputs = batch["input_ids"].to(self.device)
                targets = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets)
                
                # Auxiliary losses (Load Balancing and Router Z-loss)
                auxiliary_losses = self._compute_auxiliary_losses(inputs)
                total_loss = loss + auxiliary_losses["load_balancing_loss"] + auxiliary_losses["router_z_loss"]

                # Backward pass and optimizer step
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["pretraining"]["global_max_grad_norm"])
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Logging
                if step % 100 == 0:
                    log_metrics({"total_loss": total_loss.item(), "main_loss": loss.item()}, step, log_to_wandb=True)

                # Save checkpoint
                if step % 5000 == 0:
                    self.model.save_checkpoint(f"checkpoint_epoch_{epoch}_step_{step}.pt")

    def train_adaptation(self, adaptation_type: str = "sft") -> None:
        """
        Fine-tune the pretrained model for instruction tuning (SFT) or preference optimization (DPO).

        Args:
            adaptation_type (str): Type of adaptation, either 'sft' or 'dpo'.
        """
        if adaptation_type not in ["sft", "dpo"]:
            raise ValueError(f"Invalid adaptation type '{adaptation_type}'. Supported types: 'sft', 'dpo'.")

        self._initialize_training_components(adaptation_type)
        
        self.model.train()
        self.optimizer.zero_grad()

        adaptation_config = self.config["training"]["adaptation"][adaptation_type]
        for epoch in range(int(adaptation_config["epochs"])):
            for step, batch in enumerate(self.train_data):
                inputs = batch["input_ids"].to(self.device)
                targets = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets)

                # For DPO, apply preference tuning with beta weighting
                if adaptation_type == "dpo":
                    preference_loss = self._compute_preference_loss(outputs, batch["preferences"])
                    loss = loss * adaptation_config["beta"] + preference_loss

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["pretraining"]["global_max_grad_norm"])
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Logging
                if step % 100 == 0:
                    log_metrics({"adaptation_loss": loss.item()}, step, log_to_wandb=True)

                # Save checkpoint
                if step % 1000 == 0:
                    checkpoint_path = f"{adaptation_type}_checkpoint_epoch_{epoch}_step_{step}.pt"
                    self.model.save_checkpoint(checkpoint_path)

    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the main loss (e.g., cross-entropy) for training.

        Args:
            outputs (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground-truth labels.

        Returns:
            torch.Tensor: Computed loss.
        """
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

    def _compute_auxiliary_losses(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary losses for Mixture-of-Experts layers during pretraining.

        Args:
            inputs (torch.Tensor): Input tensor for calculating auxiliary losses.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing auxiliary losses.
        """
        router_logits = self.model.compute_router_logits(inputs)
        routing_probs = torch.softmax(router_logits, dim=-1)
        return self.model.compute_auxiliary_losses(router_logits, routing_probs)

    def _compute_preference_loss(self, outputs: torch.Tensor, preferences: torch.Tensor) -> torch.Tensor:
        """
        Compute preference loss for Direct Preference Optimization (DPO).

        Args:
            outputs (torch.Tensor): Model predictions.
            preferences (torch.Tensor): User preferences for tuning.

        Returns:
            torch.Tensor: Computed preference loss.
        """
        return torch.mean((outputs - preferences) ** 2)  # Simplified placeholder
