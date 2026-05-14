# trainer.py

from typing import Optional
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from p2vae import P2VAE
from fmt_model import FMTModel
from utils import seed_everything, initialize_logger

class Trainer:
    """Handles the training processes for P2VAE and FMT models, including checkpoint management."""

    def __init__(self, model: torch.nn.Module, dataset: dict, config: dict):
        """
        Initializes the Trainer with model, dataset, and training configurations.

        Args:
            model (torch.nn.Module): Instance of the model to be trained (P2VAE/FMTModel).
            dataset (dict): Dictionary containing train, validation, and test splits.
            config (dict): Dictionary containing training and model configurations.
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config["hardware"]["devices"] == "gpu" else "cpu")
        self.logger = initialize_logger(config["logging"]["save_dir"])
        self.checkpoint_dir = config["logging"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        seed_everything(config["logging"]["seed"])

    def _create_dataloader(self, split: str, batch_size: int) -> DataLoader:
        """Creates a PyTorch DataLoader for the specified dataset split.
        
        Args:
            split (str): One of ['train', 'val', 'test'].
            batch_size (int): Batch size for the DataLoader.
        
        Returns:
            DataLoader: DataLoader instance for the specified split.
        """
        return DataLoader(self.dataset[split]['inputs'], batch_size=batch_size, shuffle=split == 'train')

    def train_p2vae(self) -> None:
        """Trains the Pretrained Physics Variational Autoencoder (P2VAE)."""
        p2vae_config = self.config["training"]["p2vae"]
        optimizer = AdamW(self.model.parameters(), lr=p2vae_config["learning_rate"], weight_decay=p2vae_config["weight_decay"])
        scheduler = CosineAnnealingLR(optimizer, T_max=p2vae_config["epochs"], eta_min=0)
        train_loader = self._create_dataloader('train', p2vae_config["batch_size"])

        self.model.to(self.device)
        self.model.train()
        self.logger.info("Starting P2VAE training.")

        for epoch in tqdm(range(p2vae_config["epochs"]), desc="P2VAE Training"):
            total_loss = 0.0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                loss = self.model.compute_loss(batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            self.logger.info(f"Epoch {epoch + 1}/{p2vae_config['epochs']}, Loss: {avg_loss:.6f}")

            # Save checkpoints intermittently
            if (epoch + 1) % 1000 == 0 or (epoch + 1) == p2vae_config["epochs"]:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"p2vae_checkpoint_epoch_{epoch + 1}.pt")
                self.save_checkpoint(checkpoint_path)
                self.logger.info(f"Checkpoint saved at {checkpoint_path}.")

    def train_fmt(self) -> None:
        """Trains the Flow Marching Transformer (FMT) using flow matching objectives."""
        fmt_config = self.config["training"]["fmt"]
        optimizer = AdamW(self.model.parameters(), lr=fmt_config["learning_rate"], weight_decay=fmt_config["weight_decay"])
        scheduler = CosineAnnealingLR(optimizer, T_max=fmt_config["epochs"], eta_min=0)
        train_loader = self._create_dataloader('train', fmt_config["batch_size"])

        self.model.to(self.device)
        self.model.train()
        self.logger.info("Starting FMT training.")

        for epoch in tqdm(range(fmt_config["epochs"]), desc="FMT Training"):
            total_loss = 0.0
            for batch in train_loader:
                batch = batch.to(self.device)
                batch_size = batch.size(0)

                # Prepare multiple frames for flow matching objectives
                frames = torch.split(batch, batch_size // 4, dim=0)  # Assuming 4 frames per trajectory
                x_prev, x_next = frames[:-1], frames[1:]

                # Compute autoregressive initial latent condition state (h)
                latent_state = torch.zeros(batch_size, fmt_config["fmt_model"]["rnn_dim"]).to(self.device)

                for i in range(len(x_prev)):
                    optimizer.zero_grad()
                    loss = self.model.compute_loss(x_prev[i], x_next[i], latent_state)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            self.logger.info(f"Epoch {epoch + 1}/{fmt_config['epochs']}, Loss: {avg_loss:.6f}")

            # Save checkpoints intermittently
            if (epoch + 1) % 1000 == 0 or (epoch + 1) == fmt_config["epochs"]:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"fmt_checkpoint_epoch_{epoch + 1}.pt")
                self.save_checkpoint(checkpoint_path)
                self.logger.info(f"Checkpoint saved at {checkpoint_path}.")

    def save_checkpoint(self, path: str) -> None:
        """Saves training checkpoint for resuming training later.
        
        Args:
            path (str): Path to save the checkpoint file.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Loads a saved checkpoint to resume training.
        
        Args:
            path (str): Path to the saved checkpoint file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Checkpoint loaded: {path}")
