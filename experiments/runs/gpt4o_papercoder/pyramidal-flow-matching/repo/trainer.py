# trainer.py

import os
import time
from typing import Dict, Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from dataset_loader import DatasetLoader
from vae_model import VAEModel
from flow_matching_model import FlowMatchingModel
from pyramid_utils import PyramidUtils
from config import Config


class Trainer:
    """
    Orchestrates the training of the VAE and Flow Matching models across three progressive stages
    (Image training, Low-resolution video training, High-resolution video refinement).
    """

    def __init__(self, model_vae: VAEModel, model_flow: FlowMatchingModel, datasets: DatasetLoader, config: Dict[str, Any]) -> None:
        """
        Initialize the Trainer object.

        Args:
            model_vae (VAEModel): Pretrained VAE for latent video compression.
            model_flow (FlowMatchingModel): Flow Matching model for pyramid video generation.
            datasets (DatasetLoader): DatasetLoader instance for providing data batches.
            config (dict): Training and model hyperparameters.
        """
        self.model_vae = model_vae
        self.model_flow = model_flow
        self.datasets = datasets
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transfer models to GPU, if available
        self.model_vae.to(self.device)
        self.model_flow.to(self.device)

        # Configurations
        self.training_config = config["training"]
        self.model_config = config["model"]
        self.stage_epochs = [50_000, 200_000, 50_000]
        self.optimizer_vae = AdamW(self.model_vae.parameters(), lr=self.training_config["learning_rate"])
        self.optimizer_flow = AdamW(self.model_flow.parameters(), lr=self.training_config["learning_rate"])
        self.loss_log = {"vae_loss": [], "flow_loss": []}

        # Create checkpoint directory
        self.output_dir = "./checkpoints"
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self) -> None:
        """
        Main method to conduct training across all three stages:
        Stage 1: Image training for the VAE.
        Stage 2: Low-resolution video training for the Flow Matching Model.
        Stage 3: High-resolution video refinement.
        """
        print("Starting training...")
        self._train_stage_1()  # Image Training for VAE
        self._train_stage_2()  # Low-Resolution Video Training
        self._train_stage_3()  # High-Resolution Video Training

    def _train_stage_1(self) -> None:
        """
        Stage 1: Train the Variational Autoencoder (VAE) on image datasets.
        Objective: Learn latent space compression for efficient video representation.
        """
        print("=== Stage 1: Training VAE on Image Dataset ===")
        data_loader = self.datasets.load_images()
        scheduler = self._create_scheduler(self.optimizer_vae, self.stage_epochs[0])

        self.model_vae.train()
        for step in range(self.stage_epochs[0]):
            for batch in data_loader:
                batch = batch.to(self.device)
                loss = self.model_vae.train_step(batch, self.optimizer_vae)
                self.loss_log["vae_loss"].append(loss)

                # Step the LR scheduler
                scheduler.step()

            if step % 1_000 == 0:
                print(f"[Stage 1 - Step {step}] VAE Training Loss: {loss:.4f}")

            # Save checkpoints periodically
            if step % 10_000 == 0:
                self._save_checkpoint(stage=1, step=step)

    def _train_stage_2(self) -> None:
        """
        Stage 2: Train the Flow Matching model on 2-second low-resolution videos.
        Objective: Train the multi-stage spatial pyramid flow matching for video generation.
        """
        print("=== Stage 2: Training Flow Matching on Low-Resolution Videos ===")
        data_loader = self.datasets.load_videos(duration="short")
        scheduler = self._create_scheduler(self.optimizer_flow, self.stage_epochs[1])

        self.model_flow.train()
        for step in range(self.stage_epochs[1]):
            for batch in data_loader:
                batch = batch.to(self.device)

                # Compress video latents using the VAE
                with torch.no_grad():
                    latent, _, _ = self.model_vae(batch)

                # Add noise and compute flow matching loss
                noise = PyramidUtils.add_noise(latent, self.model_config["flow_matching"]["noise_level"][1])
                loss = self.model_flow.compute_loss(latent, noise, timestep=step)
                self.optimizer_flow.zero_grad()
                loss.backward()
                self.optimizer_flow.step()

                self.loss_log["flow_loss"].append(loss.item())

                # Step the LR scheduler
                scheduler.step()

            if step % 1_000 == 0:
                print(f"[Stage 2 - Step {step}] Flow Matching Training Loss: {loss:.4f}")

            # Save checkpoints periodically
            if step % 10_000 == 0:
                self._save_checkpoint(stage=2, step=step)

    def _train_stage_3(self) -> None:
        """
        Stage 3: Fine-tune the Flow Matching Model on high-resolution videos (5–10 seconds).
        Objective: Refine temporal pyramid consistency.
        """
        print("=== Stage 3: Fine-tuning Flow Matching on High-Resolution Videos ===")
        data_loader = self.datasets.load_videos(duration="long")
        scheduler = self._create_scheduler(self.optimizer_flow, self.stage_epochs[2])

        self.model_flow.train()
        for step in range(self.stage_epochs[2]):
            for batch in data_loader:
                batch = batch.to(self.device)

                # Compress video latents using the VAE
                with torch.no_grad():
                    latent, _, _ = self.model_vae(batch)

                # Temporal conditioning: progressively upsample history latents
                latent_compressed = PyramidUtils.compress_frames(latent, levels=1)
                noisy_latent = PyramidUtils.add_noise(latent_compressed, self.model_config["flow_matching"]["noise_level"][1])

                # Compute loss for temporal pyramid predictions
                loss = self.model_flow.compute_loss(latent, noisy_latent, timestep=step)
                self.optimizer_flow.zero_grad()
                loss.backward()
                self.optimizer_flow.step()

                self.loss_log["flow_loss"].append(loss.item())

                # Step the LR scheduler
                scheduler.step()

            if step % 1_000 == 0:
                print(f"[Stage 3 - Step {step}] Flow Matching Fine-Tuning Loss: {loss:.4f}")

            # Save checkpoints periodically
            if step % 10_000 == 0:
                self._save_checkpoint(stage=3, step=step)

    def _create_scheduler(self, optimizer: AdamW, total_steps: int):
        """
        Create a learning rate scheduler for gradual warm-up.

        Args:
            optimizer (AdamW): Optimizer instance.
            total_steps (int): Total training steps for the stage.

        Returns:
            LambdaLR: Learning rate scheduler instance.
        """
        warmup_steps = self.training_config["warmup_steps"]

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0

        return LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step))

    def _save_checkpoint(self, stage: int, step: int) -> None:
        """
        Save model and optimizer checkpoints.

        Args:
            stage (int): Current training stage.
            step (int): Current step within the stage.
        """
        checkpoint_dir = os.path.join(self.output_dir, f"stage_{stage}_step_{step}.pth")
        torch.save(
            {
                "vae_state_dict": self.model_vae.state_dict(),
                "flow_state_dict": self.model_flow.state_dict(),
                "optimizer_vae_state_dict": self.optimizer_vae.state_dict(),
                "optimizer_flow_state_dict": self.optimizer_flow.state_dict(),
                "loss_log": self.loss_log,
            },
            checkpoint_dir,
        )
        print(f"Checkpoint saved: {checkpoint_dir}")
