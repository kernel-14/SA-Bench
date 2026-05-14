## trainer.py
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Optimizer
from typing import Dict, Any
from utils import save_checkpoint, load_checkpoint
from model import ConsistencyModel
from dataset_loader import DatasetLoader


class Trainer:
    """
    Trainer class responsible for training and managing the consistency model
    using both independent coupling (IC) and generator-augmented flows (GC).
    """

    def __init__(self, model: ConsistencyModel, dataloaders: Dict[str, DataLoader], config: Dict[str, Any]):
        """
        Initializes the Trainer class.

        Args:
            model (ConsistencyModel): The consistency model to train.
            dataloaders (Dict[str, DataLoader]): A dictionary containing "train", "val", and "test" DataLoaders.
            config (Dict[str, Any]): Loaded configuration parameters from config.yaml.
        """
        self.model = model
        self.train_loader = dataloaders["train"]
        self.val_loader = dataloaders["val"]
        self.config = config

        # Noise schedule parameters
        self.sigma_0 = config["training"]["noise_schedule"]["sigma_0"]
        self.sigma_t = config["training"]["noise_schedule"]["sigma_t"]
        self.rho = config["training"]["noise_schedule"]["rho"]
        self.num_timesteps = len(self._generate_noise_schedule())

        # Training parameters
        self.learning_rate = config["training"].get("learning_rate", 0.00008)
        self.batch_size = config["training"].get("batch_size", 128)
        self.epochs = config["training"].get("epochs", 1)
        self.mu = config["shared_settings"].get("mu", 0.5)
        self.weight_function = lambda sigma: 1 / (self.sigma_t - self.sigma_0)

        # Hardware settings
        self.device = torch.device("cuda" if torch.cuda.is_available() and config["hardware"]["use_gpu"] else "cpu")
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # Logging
        self.save_checkpoints = config["logging"]["save_checkpoints"]
        self.checkpoint_path = config["logging"].get("checkpoint_path", "checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.log_frequency = config["logging"].get("log_frequency_steps", 100)

    def _generate_noise_schedule(self):
        """
        Generates the noise schedule (sigma_t) based on config parameters.

        Returns:
            List[float]: Noise schedule values.
        """
        sigmas = []
        for i in range(self.num_timesteps):
            sigma = (self.sigma_0 ** (1 / self.rho) + i / (self.num_timesteps - 1) * (self.sigma_t ** (1 / self.rho) - self.sigma_0 ** (1 / self.rho))) ** self.rho
            sigmas.append(sigma)
        return sigmas

    def _compute_loss(self, x_t, sigma_t, x_target, loss_type="L2"):
        """
        Computes the consistency loss for given inputs.

        Args:
            x_t (Tensor): Input noisy sample.
            sigma_t (float): Current noise level.
            x_target (Tensor): Target data sample.
            loss_type (str): Type of loss function ("L2" or other).

        Returns:
            Tensor: Computed loss value.
        """
        predictions = self.model(x_t, sigma_t)
        if loss_type == "L2":
            return nn.functional.mse_loss(predictions, x_target)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def train(self):
        """
        Main training loop for the model.
        """
        noise_schedule = self._generate_noise_schedule()
        for epoch in range(self.epochs):
            self.model.train()
            for step, (x_star, _) in enumerate(self.train_loader):
                # Move data to the GPU/CPU
                x_star = x_star.to(self.device)

                # Sample z from the noise distribution
                z = torch.randn_like(x_star)

                # Randomly sample a timestep
                t_idx = torch.randint(0, len(noise_schedule), (x_star.size(0),), device=self.device)
                sigma_t = torch.tensor([noise_schedule[i] for i in t_idx], device=self.device).view(-1, 1, 1, 1)

                # Generate IC noisy points
                x_t_ic = x_star + sigma_t * z

                # Construct GC Noisy Points (predictions as endpoint predictors)
                pred_endpoints = self.model(x_t_ic, sigma_t)
                x_t_gc = pred_endpoints + sigma_t * z

                # Compute Loss for IC and GC trajectories
                loss_ic = self._compute_loss(x_t_ic, sigma_t, x_star)
                loss_gc = self._compute_loss(x_t_gc, sigma_t, x_star)

                # Weighted Combination of IC and GC Losses
                joint_loss = self.mu * loss_gc + (1 - self.mu) * loss_ic

                # Backpropagation
                self.optimizer.zero_grad()
                joint_loss.backward()
                self.optimizer.step()

                # Logging
                if step % self.log_frequency == 0:
                    print(f"Epoch [{epoch + 1}/{self.epochs}], Step [{step}], Loss: {joint_loss.item():.4f}")

            # Save model checkpoints
            if self.save_checkpoints:
                checkpoint_file = os.path.join(self.checkpoint_path, f"model_epoch_{epoch + 1}.pth")
                save_checkpoint(self.model, checkpoint_file)

            # Optional Validation (if val_loader is provided)
            if self.val_loader is not None:
                self.validate()

    def validate(self):
        """
        Runs validation to evaluate the model on a held-out dataset.
        """
        self.model.eval()
        noise_schedule = self._generate_noise_schedule()
        val_loss = 0.0

        with torch.no_grad():
            for x_star, _ in self.val_loader:
                x_star = x_star.to(self.device)
                z = torch.randn_like(x_star)
                t_idx = torch.randint(0, len(noise_schedule), (x_star.size(0),), device=self.device)
                sigma_t = torch.tensor([noise_schedule[i] for i in t_idx], device=self.device).view(-1, 1, 1, 1)
                x_t = x_star + sigma_t * z
                loss = self._compute_loss(x_t, sigma_t, x_star)
                val_loss += loss.item()

        avg_loss = val_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")

