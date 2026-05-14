## trainer.py

import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional, Dict, Any, Union

from fr_vae import FRVAE
from transformer import TransformerGenerator
from dataset_loader import DatasetLoader
import utils


class Trainer:
    """
    Trainer class for orchestrating the training pipeline of FR-VAE and Transformer models.
    """

    def __init__(
        self,
        fr_vae: FRVAE,
        generator: TransformerGenerator,
        dataset_loader: DatasetLoader,
        config: Dict,
    ):
        """
        Initializes the Trainer with models, datasets, and training configurations.

        Args:
            fr_vae (FRVAE): Instance of the Frequency-guided Residual-Quantized VAE model.
            generator (TransformerGenerator): Instance of the autoregressive token prediction model.
            dataset_loader (DatasetLoader): Handles ImageNet dataset loading and processing.
            config (Dict): Configuration dictionary from config.yaml.
        """
        self.fr_vae = fr_vae
        self.generator = generator
        self.dataset_loader = dataset_loader
        self.config = config

        # Initialize optimizer with configurations from config.yaml
        self.optimizer = Adam(
            list(self.fr_vae.encoder.parameters()) +
            list(self.fr_vae.decoder.parameters()) +
            list(self.generator.parameters()),
            lr=self.config["training"]["learning_rate"]
        )
        self.epochs = self.config["training"]["epochs"]

        # Load data loaders from DatasetLoader
        self.train_loader = dataset_loader.load_train_data()
        self.val_loader = dataset_loader.load_val_data()
        self.test_loader = dataset_loader.load_test_data()

        # Device setup (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fr_vae.to(self.device)
        self.generator.to(self.device)

    def train(self) -> None:
        """
        Main training loop for the specified number of epochs. This function performs:
        - Training model on each batch of the dataset.
        - Periodically validates the model.
        - Monitors and saves checkpoints for resuming experiments.
        """
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch {epoch}/{self.epochs}")

            # Training Loop
            self.fr_vae.train()
            self.generator.train()
            training_loss = 0.0
            for i, (images, _) in enumerate(self.train_loader):
                images = images.to(self.device)

                # Step 1: Encode and Quantize
                frequency_bands = self.fr_vae.encode(images)
                quantized_tokens = [self.fr_vae.quantize(band) for band in frequency_bands]

                # Step 2: Supervised Training - Token Predictions
                quantized_token_seq = torch.cat([t.view(t.size(0), -1) for t in quantized_tokens], dim=1)  # Flatten tokens
                pred_logits = self.generator(quantized_token_seq)
                loss = self.compute_loss(pred_logits, quantized_token_seq)  # Cross-entropy loss

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                training_loss += loss.item()
                if i % 10 == 0:
                    print(f"Batch {i}, Loss: {loss.item():.6f}")

            # Calculate average training loss
            avg_training_loss = training_loss / len(self.train_loader)
            print(f"Epoch {epoch}, Average Training Loss: {avg_training_loss:.6f}")

            # Validate model periodically
            if epoch % 5 == 0:
                self.validate(epoch)

            # Save checkpoint
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(
                    "checkpoints", f"model_epoch_{epoch}.pth"
                )
                self.save_checkpoint(checkpoint_path, epoch)

    def validate(self, epoch: int) -> None:
        """
        Validates the model by generating images and evaluating them using FID and IS metrics.

        Args:
            epoch (int): Current training epoch.
        """
        self.fr_vae.eval()
        self.generator.eval()

        val_loss = 0.0
        generated_images = []
        with torch.no_grad():
            for i, (images, _) in enumerate(self.val_loader):
                images = images.to(self.device)

                # Step 1: Encode and Quantize
                frequency_bands = self.fr_vae.encode(images)
                quantized_tokens = [self.fr_vae.quantize(band) for band in frequency_bands]

                # Step 2: Generate tokens autoregressively
                quantized_token_seq = torch.cat(
                    [t.view(t.size(0), -1) for t in quantized_tokens], dim=1
                )  # Flatten tokens
                generated_tokens = self.generator.generate(
                    quantized_token_seq, num_steps=len(quantized_tokens)
                )

                # Step 3: Decode tokens to images
                generated_imgs = self.fr_vae.decode(generated_tokens)
                generated_images.append(generated_imgs.cpu())

                # Calculate loss for validation
                val_loss += self.compute_loss(generated_tokens, quantized_token_seq).item()

        avg_val_loss = val_loss / len(self.val_loader)
        print(f"Epoch {epoch}, Validation Loss: {avg_val_loss:.6f}")

        # Calculate evaluation metrics
        metrics = self.compute_metrics(generated_images)
        print(f"Epoch {epoch}, FID: {metrics['fid']:.4f}, IS: {metrics['is']:0.4f}")

    def compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Computes the supervised loss for token predictions.

        Args:
            predictions (Tensor): Predicted logits for frequency tokens.
            targets (Tensor): Ground truth token indices.

        Returns:
            Tensor: Cross-entropy loss value.
        """
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(predictions.view(-1, predictions.size(-1)), targets.view(-1))

    def save_checkpoint(self, filepath: str, epoch: int) -> None:
        """
        Saves the training state and model information to a checkpoint file.

        Args:
            filepath (str): Path to save the checkpoint file.
            epoch (int): Current epoch number.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "fr_vae_state": self.fr_vae.state_dict(),
            "generator_state": self.generator.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """
        Loads a saved checkpoint to resume training.

        Args:
            filepath (str): Path to the checkpoint file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file '{filepath}' not found!")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.fr_vae.load_state_dict(checkpoint["fr_vae_state"])
        self.generator.load_state_dict(checkpoint["generator_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"Checkpoint loaded from {filepath}")

    def compute_metrics(self, generated_images: list) -> Dict[str, float]:
        """
        Computes evaluation metrics like FID and IS for generated images.

        Args:
            generated_images (list): List of generated images as tensors.

        Returns:
            Dict[str, float]: Dictionary containing computed metrics (FID and IS).
        """
        # Placeholder for FID and IS metric computation
        # In a real implementation this would compare generated_images to ground truth dataset
        return {"fid": 0.0, "is": 0.0}  # Dummy placeholders
