# vae_model.py
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from pyramid_utils import PyramidUtils


class VAEModel(nn.Module):
    """
    3D Variational Autoencoder (VAE) for compressing video frames into low-dimensional latent representations.
    """

    def __init__(self, latent_dim: int = 512, downsample_ratio: Tuple[int, int, int] = (8, 8, 8)) -> None:
        """
        Initialize the 3D VAE model.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            downsample_ratio (Tuple[int, int, int]): Compression factors for spatial and temporal dimensions.
        """
        super(VAEModel, self).__init__()
        self.latent_dim = latent_dim
        self.downsample_ratio = downsample_ratio

        # Encoder: Stack of 3D convolutional layers with causal convolutions for temporal consistency.
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=4, stride=2, padding=1),  # B x 64 x H/2 x W/2 x T/2
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),  # B x 128 x H/4 x W/4 x T/4
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),  # B x 256 x H/8 x W/8 x T/8
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, latent_dim, kernel_size=1, stride=1),  # B x latent_dim x H/8 x W/8 x T/8
        )

        # Decoder: Reverse the encodings using upsampling and 3D convolution layers.
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, 256, kernel_size=4, stride=2, padding=1),  # B x 256 x H/4 x W/4 x T/4
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # B x 128 x H/2 x W/2 x T/2
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),  # B x 64 x H x W x T
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 3, kernel_size=1, stride=1),  # B x 3 x H x W x T
        )

        # Latent space for KL divergence
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input video frames into latent space and decode them for reconstruction.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, T), normalized to [0, 1].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - z: Latent representation tensor of shape (B, latent_dim, H/8, W/8, T/8).
                - x_reconstructed: Reconstructed tensor matching the input shape (B, C, H, W, T).
                - kl_loss: KL divergence loss for latent space regularization.
        """
        # Encoder forward pass
        encoded = self.encoder(x)  # Shape: (B, latent_dim, H/8, W/8, T/8)
        batch, latent_dim, h, w, t = encoded.size()

        # Flatten spatial and temporal dims for latent space
        encoded_flat = encoded.view(batch, latent_dim, -1).mean(dim=-1)  # Global avg across H/W/T

        # Latent variables
        mu = self.fc_mu(encoded_flat)  # Mean of latents
        logvar = self.fc_logvar(encoded_flat)  # Log variance of latents

        # Reparameterization trick to sample z ~ N(mu, sigma^2)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # Latent tensor of size (B, latent_dim)

        # Decoder forward pass: project z back to original spatial/temporal dimensions
        z_expanded = z.view(batch, latent_dim, 1, 1, 1).expand(-1, -1, h, w, t)
        x_reconstructed = self.decoder(z_expanded)  # Reconstructed output in original shape

        # KL divergence regularization: D_KL(q(z|x) || p(z)) where p(z) ~ N(0, I)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch

        return z, x_reconstructed, kl_loss

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct video frames from latent representations.

        Args:
            z (torch.Tensor): Latent tensor of shape (B, latent_dim).

        Returns:
            torch.Tensor: Reconstructed video frames of shape (B, C, H, W, T).
        """
        batch, latent_dim = z.size()
        # Re-expand latent representation for decoding
        z_expanded = z.view(batch, latent_dim, 1, 1, 1)
        return self.decoder(z_expanded)

    def train_step(self, batch: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """
        Perform a single training step: calculate loss and backpropagate.

        Args:
            batch (torch.Tensor): Input tensor of shape (B, C, H, W, T).
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.

        Returns:
            float: Combined total loss value for the step (reconstruction + KL divergence).
        """
        optimizer.zero_grad()

        # Forward pass
        _, x_reconstructed, kl_loss = self.forward(batch)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_reconstructed, batch, reduction="mean")

        # Total loss
        loss = recon_loss + kl_loss

        # Backpropagation
        loss.backward()
        optimizer.step()

        return loss.item()


# Example usage
if __name__ == "__main__":
    # Load configuration
    config = Config()
    model_config = config.get_model_config()

    # Initialize VAE model
    vae = VAEModel(
        latent_dim=model_config["vae"]["latent_dim"],
        downsample_ratio=tuple(model_config["vae"]["downsampling_ratio"]),
    )

    # Dummy Input (B, C, H, W, T)
    dummy_input = torch.rand(2, 3, 256, 256, 16)  # Example batch
    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-4)

    # Forward and training test
    latent, reconstructed, kl_div = vae(dummy_input)
    print("Latent Shape:", latent.shape)
    loss = vae.train_step(dummy_input, optimizer)
    print("Step Loss:", loss)
