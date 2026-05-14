## trainer.py

import os
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict
from utils import save_model, load_model

class Trainer:
    """
    Manages the training process for Masked Diffusion Models (MDMs), including dataset batching,
    loss computation, optimization steps, and checkpoint saving.
    """
    def __init__(self, model, dataset, config: Dict):
        """
        Initializes Trainer with the model, dataset, and configuration.

        Args:
            model: The Masked Diffusion Model (MDM) to train.
            dataset: Preprocessed dataset provided by DatasetLoader.
            config (Dict): Training configuration with hyperparameters, paths, and hardware settings.
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config["hardware"]["use_gpu"] else "cpu")

        # Transfer model to the appropriate device
        self.model = self.model.to(self.device)

        # Training parameters from config
        self.learning_rate = self.config["training"].get("learning_rate", 0.001)
        self.batch_size = self.config["training"].get("batch_size", 128)
        self.epochs = self.config["training"].get("epochs", 300)
        self.optimizer_type = self.config["training"].get("optimizer", "AdamW")
        self.weight_decay = self.config["training"].get("weight_decay", 0.1)
        self.beta1 = self.config["training"].get("beta1", 0.9)
        self.beta2 = self.config["training"].get("beta2", 0.95)
        self.checkpoint_path = "./checkpoints"

        # Initialize optimizer
        self.optimizer = self._setup_optimizer()

        # Noise schedule for masking (alpha values)
        self.alpha_start = self.config["dataset"]["masking_schedule"]["alpha_start"]
        self.alpha_end = self.config["dataset"]["masking_schedule"]["alpha_end"]

    def _setup_optimizer(self):
        """
        Initializes the optimizer defined in the configuration file.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        if self.optimizer_type.lower() == "adamw":
            return AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(self.beta1, self.beta2),
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

    def train(self):
        """
        Handles the end-to-end training loop for the specified number of epochs.
        """
        # Prepare DataLoader for batching
        train_loader = DataLoader(
            self.dataset["masked_sequences"],
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_path, exist_ok=True)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_idx, masked_batch in enumerate(train_loader):
                masked_batch["input_sequences"] = masked_batch["input_sequences"].to(self.device)
                masked_batch["targets"] = masked_batch["targets"].to(self.device)

                # Compute loss for current batch
                batch_loss = self._compute_loss(masked_batch["input_sequences"], masked_batch["targets"], masked_batch["metadata"])
                epoch_loss += batch_loss.item()

                # Backward pass and optimization
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            # Report loss after each epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{self.epochs}] - Training Loss: {avg_epoch_loss:.4f}")

            # Save checkpoints periodically
            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.epochs:
                self._save_checkpoint(epoch)

    def _compute_loss(self, masked_sequences, targets, metadata):
        """
        Computes the ELBO-based loss function for MDMs.

        Args:
            masked_sequences (torch.Tensor): Input masked sequences.
            targets (torch.Tensor): Ground-truth tokens.
            metadata (torch.Tensor): Mask positions metadata.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        batch_size, seq_length = masked_sequences.shape
        model_logits = self.model.forward(masked_sequences)  # [batch_size, seq_length, vocab_size]

        # Mask metadata indicates the positions that must be considered
        masked_positions = (metadata == 1)  # Extract masked regions based on metadata
        loss = 0.0

        # Iterate over timestep noise levels to compute ELBO contributions
        for timestep in np.linspace(self.alpha_start, self.alpha_end, num=10):
            noise_scaling_factor = self._get_noise_scaling(timestep)

            # Filter logits for masked positions only
            masked_logits = model_logits[masked_positions]
            masked_targets = targets[masked_positions]

            # Calculate categorical cross-entropy loss for masked tokens
            cross_entropy_loss = torch.nn.functional.cross_entropy(
                masked_logits, masked_targets
            )
            loss += noise_scaling_factor * cross_entropy_loss

        # Average loss across the batch
        return loss / batch_size

    def _get_noise_scaling(self, alpha_t):
        """
        Compute noise scaling factor ∂α(t)/∂t / (1 - α(t)).

        Args:
            alpha_t (float): Current timestep noise level.

        Returns:
            float: Noise scaling factor.
        """
        alpha_prime = self.alpha_end - self.alpha_start  # Constant derivative across timesteps
        return alpha_prime / (1 - alpha_t)

    def _save_checkpoint(self, epoch):
        """
        Saves the current model state and optimizer state at the specified checkpoint path.

        Args:
            epoch (int): Current epoch number to label checkpoint.
        """
        checkpoint_model_path = os.path.join(self.checkpoint_path, f"model_epoch_{epoch + 1}.pt")
        checkpoint_optimizer_path = os.path.join(self.checkpoint_path, f"optimizer_epoch_{epoch + 1}.pt")

        save_model(self.model, checkpoint_model_path)
        torch.save(self.optimizer.state_dict(), checkpoint_optimizer_path)

        print(f"Checkpoint saved: Epoch {epoch + 1} -> {checkpoint_model_path}")

    def validate(self, validation_data):
        """
        Validates the model on validation data after each epoch.

        Args:
            validation_data (torch.utils.data.Dataset): The validation dataset.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        validation_loader = DataLoader(validation_data, batch_size=self.batch_size)

        validation_loss = 0.0

        with torch.no_grad():
            for batch in validation_loader:
                batch["input_sequences"] = batch["input_sequences"].to(self.device)
                batch["targets"] = batch["targets"].to(self.device)

                batch_loss = self._compute_loss(batch["input_sequences"], batch["targets"], batch["metadata"])
                validation_loss += batch_loss.item()

        avg_validation_loss = validation_loss / len(validation_loader)
        print(f"Validation Loss: {avg_validation_loss:.4f}")
        return avg_validation_loss
