## trainer.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Dict, Optional, Any
from time import time
from utils.kvcache import KVCache
from model import SpatialTemporalTransformer
from config import Config
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


class Trainer:
    """
    The Trainer class orchestrates the training, validation, and checkpointing of the Ca2-VDM model.
    It integrates train/validation pipelines with autoregression and KV-caching mechanisms.
    """

    def __init__(
        self,
        model: SpatialTemporalTransformer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Config,
    ) -> None:
        """
        Initializes the Trainer with model, dataset loaders, and configuration.

        Args:
            model (SpatialTemporalTransformer): The Ca2-VDM model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (Optional[DataLoader]): DataLoader for validation data.
            config (Config): Configuration object holding hyperparameters.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Optimizer
        lr = self.config.get("training.learning_rate", 2e-5)
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        # Scheduler
        total_steps = (
            len(train_loader) 
            * self.config.get("training.epochs_t2v_stage1", 32)  # Default to T2V stage 1 settings.
        )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, total_iters=total_steps)

        # Gradient scaler for mixed precision training
        self.scaler = GradScaler()

        # Training settings
        self.epochs = self.config.get("training.epochs_t2v_stage1", 32)  # Default T2V stage I epochs
        self.clip_grad_norm = 1.0  # Gradient clipping limit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Loss weights
        self.simple_loss_weight = self.config.get("model.simple_loss_weight", 1.0)
        self.vlb_loss_weight = self.config.get("model.vlb_loss_weight", 0.1)

        # KV-Cache
        self.kv_cache = KVCache(
            max_length=self.config.get("model.kv_cache.max_length", 49),
            spatial_size=(self.model.latent_resolution, self.model.latent_resolution),
            channels=self.model.latent_channels,
        )

        # Logging and checkpoint paths
        self.checkpoint_dir = "./checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self) -> None:
        """
        Train the model over multiple epochs with periodic validation and checkpointing.
        """
        print(f"Starting training for {self.epochs} epochs...")
        self.model.train_mode()
        self.model.to(self.device)

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time()
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(self.train_loader):
                # Forward pass and loss computation
                batch_loss = self._train_step(batch)
                epoch_loss += batch_loss

                if (batch_idx + 1) % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx + 1}, Loss: {batch_loss:.4f}")

            epoch_duration = time() - epoch_start_time
            print(f"Epoch {epoch}/{self.epochs} completed in {epoch_duration:.2f}s, Loss: {epoch_loss:.4f}")

            # Validation
            if self.val_loader is not None and epoch % 5 == 0:
                self.validate()

            # Save checkpoint
            self.save_checkpoint(epoch)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Execute a single training step.

        Args:
            batch (Dict[str, torch.Tensor]): Batch dictionary containing video data and labels.

        Returns:
            float: Computed loss for the batch.
        """
        self.optimizer.zero_grad()

        # Prepare inputs and move to device
        clean_prefix, noisy_target, t, mask = (
            batch["clean_prefix"].to(self.device),
            batch["noisy_target"].to(self.device),
            batch["t"].to(self.device),
            batch["mask"].to(self.device),
        )

        # Use autocast for mixed precision training
        with autocast():
            # Forward pass
            predictions = self.model.forward(
                latent=torch.cat([clean_prefix, noisy_target], dim=1), t=t, cache=self.kv_cache
            )

            # Loss computation
            simple_loss = F.mse_loss(predictions * mask, noisy_target * mask)
            vlb_loss = self._compute_vlb_loss(predictions, clean_prefix)

            total_loss = (
                self.simple_loss_weight * simple_loss
                + self.vlb_loss_weight * vlb_loss
            )

        # Backward pass and update
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Scheduler step
        self.scheduler.step()

        return total_loss.item()

    def _compute_vlb_loss(self, predictions: torch.Tensor, clean_prefix: torch.Tensor) -> torch.Tensor:
        """
        Compute the variational lower bound (VLB) loss for the model.

        Args:
            predictions (Tensor): Model predictions for noisy latents.
            clean_prefix (Tensor): Ground truth clean prefix frames.

        Returns:
            torch.Tensor: Computed VLB loss.
        """
        # Placeholder for actual VLB loss computation
        vlb_loss = F.mse_loss(predictions, clean_prefix)
        return vlb_loss

    def validate(self) -> None:
        """
        Validate the model on the validation dataset using autoregressive inference.
        """
        print("Validating the model...")
        self.model.inference_mode()
        self.model.to(self.device)
        total_val_loss = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Prepare inputs and move to device
                clean_prefix, noisy_target, t, mask = (
                    batch["clean_prefix"].to(self.device),
                    batch["noisy_target"].to(self.device),
                    batch["t"].to(self.device),
                    batch["mask"].to(self.device),
                )

                predictions = self.model.forward(
                    latent=torch.cat([clean_prefix, noisy_target], dim=1), t=t, cache=self.kv_cache
                )

                simple_loss = F.mse_loss(predictions * mask, noisy_target * mask)
                total_val_loss += simple_loss.item()

        print(f"Validation Loss: {total_val_loss:.4f}")

    def save_checkpoint(self, epoch: int) -> None:
        """
        Save the model, optimizer, and scheduler state to a checkpoint file.

        Args:
            epoch (int): Current epoch number to include in the checkpoint filename.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load a model checkpoint for resuming training or inference.

        Args:
            path (str): Path to the checkpoint file.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Checkpoint loaded from {path}")
