"""
trainer.py

This module implements the Trainer class, which orchestrates the training process for the SAM 2 model.
It handles batching, forward passes, training loop, prompt simulation, memory updates, and evaluation.
"""

import os
from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from utils import generate_prompts, compute_metrics
from model import Model


class Trainer:
    """
    The Trainer class coordinates the training and evaluation processes for the SAM 2 model.
    """

    def __init__(self, model: Model, train_loader: DataLoader, val_loader: DataLoader, config: Dict):
        """
        Initialize the Trainer class with model, data loaders, and configuration.

        Args:
            model (Model): The SAM 2 segmentation model.
            train_loader (DataLoader): Dataloader for training dataset.
            val_loader (DataLoader): Dataloader for validation dataset.
            config (Dict): Configuration dictionary loaded from config.yaml.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Optimizer configuration
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )

        # Learning rate scheduler
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: 1 / (1 + step / self.config["training"]["warmup_steps"])
        )

        # Gradient clipping
        self.max_grad_norm = self.config["training"]["gradient_clipping"]

        # Logging and checkpointing
        self.log_dir = self.config["logging"]["log_dir"]
        self.checkpoint_dir = self.config["logging"]["checkpoint_dir"]
        self.checkpoint_frequency = self.config["logging"]["checkpoint_frequency"]

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def train(self):
        """
        Main training loop.
        Handles epoch-level iterations and validation steps.
        """
        best_val_metric = float('-inf')  # Save the best metric during validation
        num_epochs = self.config["training"]["epochs"]

        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            self.model.train()
            epoch_loss = 0.0

            # Iterate through training batches
            for batch in tqdm(self.train_loader, desc="Training", leave=False):
                batch_loss = self.train_step(batch)
                epoch_loss += batch_loss

            # Log training loss
            avg_train_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch {epoch}, Training Loss: {avg_train_loss:.4f}")

            # Run validation
            val_metric = self.validation_step()
            print(f"Epoch {epoch}, Validation Metric (T&F): {val_metric:.4f}")

            # Save checkpoint if validation improves
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                self.save_checkpoint(epoch, is_best=True)

            # Save regular checkpoint
            if epoch % self.checkpoint_frequency == 0:
                self.save_checkpoint(epoch)

        print("Training complete.")

    def train_step(self, batch: Dict) -> float:
        """
        Single training step logic. Handles forward and backward passes, loss computation, and memory updates.

        Args:
            batch (Dict): A dictionary with video frames and annotations.

        Returns:
            float: Training loss for the current step.
        """
        self.optimizer.zero_grad()

        # Load data to device
        video_frames = batch["frames"].to(self.device)  # [B, T, C, H, W]
        annotations = batch["annotations"].to(self.device)  # Ground truth masks
        memory_states = None  # Initialize memory states as None for the batch

        total_loss = 0.0

        # Loop through each frame sequence
        for t in range(video_frames.size(1)):  # Iterate over time steps in the video
            frame = video_frames[:, t]
            annotation = annotations[:, t]

            # Generate prompts (only on some frames)
            prompts = generate_prompts(annotation, strategy="clicks") if t == 0 else None

            # Forward pass
            mask_logits, iou_logits, occlusion_logits = self.model(frame, prompts, memory_states)

            # Compute segmentation loss (Focal + Dice)
            segmentation_loss = self._compute_segmentation_loss(mask_logits, annotation)

            # Compute IoU loss
            iou_loss = self._compute_iou_loss(iou_logits, annotation)

            # Compute occlusion classification loss
            occlusion_loss = self._compute_occlusion_loss(occlusion_logits, annotation)

            # Total loss for this frame
            frame_loss = segmentation_loss + iou_loss + occlusion_loss
            total_loss += frame_loss.item()

            # Backpropagation
            frame_loss.backward()

            # Update memory for the next frame
            self.model.update_memory(mask_logits, frame)

        # Clip gradients and optimize
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        return total_loss

    def validation_step(self) -> float:
        """
        Run validation on the validation dataset.

        Returns:
            float: Validation metric (e.g., T&F accuracy).
        """
        self.model.eval()
        total_metrics = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                video_frames = batch["frames"].to(self.device)
                annotations = batch["annotations"].to(self.device)
                memory_states = None

                for t in range(video_frames.size(1)):  # Validate over time
                    frame = video_frames[:, t]
                    annotation = annotations[:, t]

                    # Generate prompts only as corrections based on predictions
                    prompts = generate_prompts(
                        annotation, strategy="error_based"
                    ) if t > 0 else None

                    mask_logits, iou_logits, occlusion_logits = self.model(frame, prompts, memory_states)

                    # Evaluate metrics on predictions
                    metrics = compute_metrics(mask_logits, annotation)
                    total_metrics += metrics["T&F"]

        avg_metric = total_metrics / len(self.val_loader)
        return avg_metric

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save a checkpoint of the model.

        Args:
            epoch (int): Current epoch number.
            is_best (bool): Whether this epoch had the best validation metric.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filename = f"checkpoint_epoch_{epoch}.pt"
        if is_best:
            filename = "checkpoint_best.pt"

        save_path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
        }
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint: {save_path}")

    @staticmethod
    def _compute_segmentation_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the segmentation loss (Focal + Dice) for mask predictions.

        Args:
            predictions (torch.Tensor): Mask predictions from the model.
            targets (torch.Tensor): Ground truth masks.

        Returns:
            torch.Tensor: Computed segmentation loss.
        """
        # Placeholder for actual focal + dice loss
        loss = F.binary_cross_entropy_with_logits(predictions, targets)
        return loss

    @staticmethod
    def _compute_iou_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU loss between predictions and targets.

        Args:
            predictions (torch.Tensor): IoU scores predicted by the model.
            targets (torch.Tensor): Ground truth IoU scores.

        Returns:
            torch.Tensor: IoU loss.
        """
        loss = F.l1_loss(predictions, targets)
        return loss

    @staticmethod
    def _compute_occlusion_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute occlusion classification loss.

        Args:
            predictions (torch.Tensor): Occlusion predictions as logits.
            targets (torch.Tensor): Binary ground truth occlusion labels.

        Returns:
            torch.Tensor: Classification loss.
        """
        loss = F.binary_cross_entropy_with_logits(predictions, targets)
        return loss
