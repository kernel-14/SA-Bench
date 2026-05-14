# trainer.py

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Any, Dict
from model import HiMARTransformer, VAE, DiffusionHead
from utils import mask_tokens
from torchvision.transforms.functional import to_pil_image


class Trainer:
    """
    Trainer class for managing the training and validation of Hi-MAR models.
    Includes hierarchical masked autoregressive modeling with two phases:
    - Phase 1: Low-resolution global structure generation
    - Phase 2: High-resolution refinement guided by Phase 1 outputs.
    """

    def __init__(
        self,
        model: HiMARTransformer,
        vae: VAE,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        config: Dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Trainer.
        
        Args:
            model (HiMARTransformer): The Hi-MAR Transformer model.
            vae (VAE): Variational Autoencoder for token encoding/decoding.
            optimizer (torch.optim.Optimizer): Optimizer for model training.
            scheduler (Any): Learning rate scheduler.
            config (Dict[str, Any]): Training configuration dictionary.
            device (str): Device for training ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.vae = vae
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # Configuration options
        self.epochs = config["training"]["epochs"]
        self.batch_size = config["training"]["batch_size"]
        self.steps_phase1 = config["inference"]["steps_phase1"]
        self.steps_phase2 = config["inference"]["steps_phase2"]
        self.masking_phase1 = config["masking"]["phase1"]
        self.masking_phase2 = config["masking"]["phase2"]

    def train_epoch(self, data_loader: DataLoader) -> None:
        """
        Perform one training epoch.

        Args:
            data_loader (DataLoader): DataLoader containing the training data.
        """
        self.model.train()  # Set model to training mode
        total_loss = 0.0
        
        for batch_idx, (images, context_tokens) in enumerate(data_loader):
            # Move data to appropriate device
            images = images.to(self.device)
            context_tokens = context_tokens.to(self.device)

            # Phase 1: Low-resolution token prediction
            low_res_latent = self.vae.encode(images, resolution="low")
            masked_low_res = mask_tokens(low_res_latent, ratio=self.masking_phase1["ratio_range"][1], strategy=self.masking_phase1["strategy"])
            
            self.optimizer.zero_grad()
            conditional_tokens = self.model(masked_low_res, context_tokens)  # Phase 1 output
            phase1_loss = nn.functional.mse_loss(conditional_tokens, low_res_latent)

            # Phase 2: High-resolution refinement
            high_res_latent = self.vae.encode(images, resolution="high")
            masked_high_res = mask_tokens(high_res_latent, ratio=self.masking_phase2["ratio_range"][1], strategy=self.masking_phase2["strategy"])
            refined_tokens = self.model(torch.cat([masked_high_res, conditional_tokens], dim=1), context_tokens)
            phase2_loss = nn.functional.mse_loss(refined_tokens, high_res_latent)

            # Combine losses
            total_loss = phase1_loss + phase2_loss
            total_loss.backward()

            # Optimization step
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()  # Adjust learning rate if scheduler provided

            # Logging for debugging
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Phase 1 Loss: {phase1_loss.item()}, Phase 2 Loss: {phase2_loss.item()}")

    def validate(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        Perform validation on the held-out data.

        Args:
            data_loader (DataLoader): DataLoader for validation split.
        
        Returns:
            Dict[str, Any]: Validation metrics including FID and reconstruction accuracy.
        """
        self.model.eval()  # Set model to evaluation mode
        metrics = {"FID": None, "IS": None, "Precision": None, "Recall": None}

        with torch.no_grad():
            for images, context_tokens in data_loader:
                # Move data to appropriate device
                images = images.to(self.device)
                context_tokens = context_tokens.to(self.device)

                # Encode images into latent representations
                low_res_latent = self.vae.encode(images, resolution="low")
                high_res_latent = self.vae.encode(images, resolution="high")

                # Phase 1: Low-resolution prediction
                masked_low_res = mask_tokens(low_res_latent, ratio=self.masking_phase1["ratio_range"][1], strategy=self.masking_phase1["strategy"])
                conditional_tokens = self.model(masked_low_res, context_tokens)

                # Phase 2: High-resolution refinement
                masked_high_res = mask_tokens(high_res_latent, ratio=self.masking_phase2["ratio_range"][1], strategy=self.masking_phase2["strategy"])
                refined_tokens = self.model(torch.cat([masked_high_res, conditional_tokens], dim=1), context_tokens)

                # Decode high-resolution refined tokens
                reconstructed_images = self.vae.decode(refined_tokens)

                # Compute and store validation metrics (placeholder code for metrics)
                # metrics["FID"] = calculate_fid(...)
                # metrics["Precision"], metrics["Recall"] = calculate_precision_and_recall(...)
                pass  # Validation metric computation

        print(f"Validation complete. Metrics: {metrics}")
        return metrics

    def save_checkpoint(self, path: str) -> None:
        """
        Save a checkpoint of the model, optimizer, and scheduler states.

        Args:
            path (str): Path to save the checkpoint file.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "vae_state_dict": self.vae.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load a previously saved training checkpoint.

        Args:
            path (str): Path to the checkpoint file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.vae.load_state_dict(checkpoint["vae_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Checkpoint loaded from {path}")
