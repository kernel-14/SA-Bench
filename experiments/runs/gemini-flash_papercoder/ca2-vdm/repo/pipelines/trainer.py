import os
import logging
import random
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from diffusers.models import AutoencoderKL
from transformers import CLIPTextModel, AutoTokenizer

from config import Config
from models.ca2_vdm_model import Ca2VDM
from utils.diffusion_schedulers import DiffusionScheduler
# from data.video_dataset import VideoDataset # Avoiding direct import for type hinting only to prevent circular dependency if Trainer were used in VideoDataset during more complex init


logger = logging.getLogger(__name__)

class Trainer:
    """
    Orchestrates the training loop for the Ca2-VDM model.
    Handles data loading, model forward/backward passes, loss computation,
    validation, logging, and checkpointing.
    """

    def __init__(
        self,
        model: Ca2VDM,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: DiffusionScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        config: Config
    ):
        """
        Initializes the Trainer with all necessary components and configurations.

        Args:
            model (Ca2VDM): The core Ca2-VDM model.
            train_dataloader (DataLoader): DataLoader for the training dataset.
            val_dataloader (DataLoader): DataLoader for the validation dataset.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters (AdamW).
            scheduler (DiffusionScheduler): Diffusion scheduler for noise handling.
            vae (AutoencoderKL): Pre-initialized VAE model.
            text_encoder (CLIPTextModel): Pre-initialized text encoder model.
            config (Config): Global configuration object.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.vae = vae
        self.text_encoder = text_encoder
        self.config = config

        # Determine computing device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Move models to the specified device
        self.model.to(self.device)
        # VAE and text_encoder are typically moved to device in VideoDataset or main.py
        # but ensure they are on device for any potential direct use here.
        self.vae.to(self.device)
        self.text_encoder.to(self.device)

        # Initialize Tensorboard writer
        log_dir = os.path.join(self.config.save_path, "runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        logger.info(f"Tensorboard logs will be saved to: {log_dir}")

        self.global_step: int = 0
        self.total_training_steps: int = self.config.training_steps

        # Default intervals if not explicitly in config (as per design instruction to set defaults)
        # Note: config.yaml doesn't specify these, so we set reasonable defaults.
        self.val_intervals: int = getattr(self.config, 'val_intervals', max(1, self.total_training_steps // 10)) # Validate 10 times during training
        self.save_intervals: int = getattr(self.config, 'save_intervals', max(1, self.total_training_steps // 5)) # Save 5 checkpoints during training
        
        logger.info(f"Training for {self.total_training_steps} steps.")
        logger.info(f"Validation interval: {self.val_intervals} steps.")
        logger.info(f"Checkpoint save interval: {self.save_intervals} steps.")


    def _compute_loss(
        self,
        model_output: torch.Tensor,
        noise_target: torch.Tensor,
        latents: torch.Tensor, # This is the partially_noised_latents (z_t)
        timesteps: torch.Tensor,
        clean_prefix_mask: torch.Tensor # This is the loss mask 'm'
    ) -> torch.Tensor:
        """
        Calculates the combined training loss: simple objective + VLB loss.

        Args:
            model_output (torch.Tensor): The output of the Ca2-VDM model.
                                         Expected to contain predicted noise and possibly predicted log-variance.
            noise_target (torch.Tensor): The ground-truth noise (epsilon) used for the noisy frames.
            latents (torch.Tensor): The noisy latent frames (z_t).
            timesteps (torch.Tensor): The timestep vector for each frame.
            clean_prefix_mask (torch.Tensor): A mask to exclude clean prefix frames from loss calculation.

        Returns:
            torch.Tensor: The total computed loss.
        """
        # Split model_output into predicted_noise and predicted_log_variance
        # It's assumed that the model outputs 2*C channels, where first C are noise and second C are log-variance.
        # The number of channels C for latents is `latents.shape[1]`.
        num_latent_channels = latents.shape[1]
        
        # Ensure model_output has enough channels for noise and log-variance
        if model_output.shape[1] == 2 * num_latent_channels:
            predicted_noise = model_output[:, :num_latent_channels]
            predicted_log_variance = model_output[:, num_latent_channels:]
            has_learnable_variance = True
        elif model_output.shape[1] == num_latent_channels:
            predicted_noise = model_output
            predicted_log_variance = None
            has_learnable_variance = False
        else:
            raise ValueError(
                f"Model output channels ({model_output.shape[1]}) must be "
                f"equal to latent channels ({num_latent_channels}) or "
                f"twice the latent channels ({2 * num_latent_channels})."
            )

        # Reshape clean_prefix_mask to broadcast correctly across latent dimensions
        # (batch, frames, H, W) -> (batch, 1, frames, H, W) for element-wise multiplication
        # Original mask is (L_clip,), need to expand to match (B, C, L, H, W)
        mask_expanded = clean_prefix_mask.view(latents.shape[0], 1, latents.shape[2], 1, 1).to(self.device)


        # --- Simplified Objective (L_simple) ---
        # L_simple = E[ || (epsilon_theta - epsilon) * m ||^2 ]
        # Only compute loss for frames indicated by the mask (denoising target)
        squared_diff = torch.square(predicted_noise - noise_target)
        # Apply mask to squared differences
        masked_squared_diff = squared_diff * mask_expanded
        # Mean over all dimensions (batch, channels, frames, height, width) for masked parts
        loss_simple = masked_squared_diff.mean()

        # --- Variational Lower Bound Loss (L_vlb) ---
        # The paper mentions optimizing L_vlb with learnable covariance following Nichol & Dhariwal (2021).
        # However, the DiffusionScheduler design does not include methods for computing D_KL.
        # As per the design instructions, we will use a placeholder for L_vlb.
        # In a full implementation, this would involve detailed calculations for the posterior
        # mean and variance, and then computing KL divergence.
        loss_vlb = torch.tensor(0.0, device=self.device)

        if has_learnable_variance:
            # If learnable variance is enabled, the actual implementation would compute
            # the full D_KL term which uses predicted_log_variance.
            # Placeholder for now, as DiffusionScheduler doesn't expose necessary components.
            # In a real scenario, this would be crucial:
            # - Get q_posterior parameters (mean, log_variance) from z_0, z_t, t
            # - Get p_theta parameters (mean, log_variance) from z_t, predicted_noise, predicted_log_variance
            # - Compute KL(q_posterior || p_theta)
            # The mask should also be applied to L_vlb.

            # We add a small regularization to log_variance to prevent it from collapsing
            # This is a common practice when learnable variance is used without full VLB.
            # If full VLB is implemented, this regularization might not be needed.
            loss_vlb_reg = torch.mean(predicted_log_variance * 0.001) # Small regularization
            loss_vlb += loss_vlb_reg

        total_loss = loss_simple + loss_vlb
        return total_loss

    def train(self) -> None:
        """
        Executes the main training loop for the Ca2-VDM model.
        """
        self.model.train()
        train_iter = iter(self.train_dataloader)

        progress_bar = tqdm(
            initial=self.global_step,
            total=self.total_training_steps,
            desc="Training"
        )

        while self.global_step < self.total_training_steps:
            try:
                # Assuming VideoDataset returns:
                # (partially_noised_latents, text_embeddings, timesteps_vector, condition_mask, tpe_indices, noise_epsilon)
                # NOTE: As discussed in thought process, the current VideoDataset.__getitem__ interface
                # in the design (tuple of 5) does NOT include `noise_epsilon`.
                # For this `Trainer` to function correctly with `_compute_loss`,
                # `noise_epsilon` MUST be available. This implementation ASSUMES
                # `VideoDataset.__getitem__` actually returns `noise_epsilon` as the 6th element.
                # This is a necessary deviation to make the provided `_compute_loss` signature viable.
                batch = next(train_iter)
                (partially_noised_latents_input, text_embeddings_input, timesteps_vector_input,
                 condition_mask_input, tpe_indices_input, noise_epsilon_input) = batch

            except StopIteration:
                # Re-initialize iterator if dataset is exhausted
                train_iter = iter(self.train_dataloader)
                (partially_noised_latents_input, text_embeddings_input, timesteps_vector_input,
                 condition_mask_input, tpe_indices_input, noise_epsilon_input) = next(train_iter)


            # Move batch components to the device
            partially_noised_latents_input = partially_noised_latents_input.to(self.device)
            text_embeddings_input = text_embeddings_input.to(self.device)
            timesteps_vector_input = timesteps_vector_input.to(self.device)
            condition_mask_input = condition_mask_input.to(self.device)
            tpe_indices_input = tpe_indices_input.to(self.device)
            noise_epsilon_input = noise_epsilon_input.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            model_output = self.model(
                latents=partially_noised_latents_input,
                timesteps=timesteps_vector_input,
                text_embeddings=text_embeddings_input,
                condition_mask=condition_mask_input, # This condition_mask is likely for internal model logic, not the loss mask
                tpe_indices=tpe_indices_input
            )

            # Compute loss
            loss = self._compute_loss(
                model_output=model_output,
                noise_target=noise_epsilon_input,
                latents=partially_noised_latents_input,
                timesteps=timesteps_vector_input,
                clean_prefix_mask=condition_mask_input # This is the loss mask 'm'
            )

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            self.global_step += 1

            # Log training loss
            self.writer.add_scalar("Loss/train", loss.item(), self.global_step)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", step=self.global_step)
            progress_bar.update(1)

            # Validation
            if self.global_step % self.val_intervals == 0:
                logger.info(f"Step {self.global_step}: Running validation...")
                val_metrics = self._validate()
                self.writer.add_scalar("Loss/val", val_metrics["avg_val_loss"], self.global_step)
                logger.info(f"Validation Loss at step {self.global_step}: {val_metrics['avg_val_loss']:.4f}")
                self.model.train() # Set back to train mode

            # Checkpointing
            if self.global_step % self.save_intervals == 0:
                self._save_checkpoint(f"step_{self.global_step}")

        progress_bar.close()
        self._save_checkpoint("final")
        self.writer.close()
        logger.info("Training completed.")

    def _validate(self) -> Dict[str, float]:
        """
        Conducts a validation pass on the validation dataset.

        Returns:
            Dict[str, float]: A dictionary containing validation metrics.
        """
        self.model.eval()
        total_val_loss = 0.0
        num_batches = 0

        val_progress_bar = tqdm(self.val_dataloader, desc="Validation", leave=False)

        with torch.no_grad():
            for batch in val_progress_bar:
                (partially_noised_latents_input, text_embeddings_input, timesteps_vector_input,
                 condition_mask_input, tpe_indices_input, noise_epsilon_input) = batch

                # Move batch components to the device
                partially_noised_latents_input = partially_noised_latents_input.to(self.device)
                text_embeddings_input = text_embeddings_input.to(self.device)
                timesteps_vector_input = timesteps_vector_input.to(self.device)
                condition_mask_input = condition_mask_input.to(self.device)
                tpe_indices_input = tpe_indices_input.to(self.device)
                noise_epsilon_input = noise_epsilon_input.to(self.device)

                # Forward pass
                model_output = self.model(
                    latents=partially_noised_latents_input,
                    timesteps=timesteps_vector_input,
                    text_embeddings=text_embeddings_input,
                    condition_mask=condition_mask_input,
                    tpe_indices=tpe_indices_input
                )

                # Compute loss
                loss = self._compute_loss(
                    model_output=model_output,
                    noise_target=noise_epsilon_input,
                    latents=partially_noised_latents_input,
                    timesteps=timesteps_vector_input,
                    clean_prefix_mask=condition_mask_input
                )
                total_val_loss += loss.item()
                num_batches += 1
                val_progress_bar.set_postfix(val_loss=f"{loss.item():.4f}")

        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0
        return {"avg_val_loss": avg_val_loss}

    def _save_checkpoint(self, name: str) -> None:
        """
        Saves the current state of the model and optimizer.

        Args:
            name (str): A string to identify the checkpoint (e.g., "step_1000", "final").
        """
        checkpoint_dir = os.path.join(self.config.save_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model without DDP wrapper if it exists
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        checkpoint_path = os.path.join(checkpoint_dir, f"ca2_vdm_checkpoint_{name}.pt")
        torch.save(
            {
                "global_step": self.global_step,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config # Save config for reproducibility
            },
            checkpoint_path
        )
        logger.info(f"Checkpoint saved to: {checkpoint_path}")

