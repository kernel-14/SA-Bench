## p2vae.py

import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

class P2VAE(nn.Module):
    """Pretrained Physics Variational Autoencoder (P2VAE) for compressing high-dimensional
    physical fields into latent representations and reconstructing them."""

    def __init__(self, latent_dim: int = 16, enc_params: dict = None, dec_params: dict = None):
        """
        Initializes the P2VAE model with encoder, decoder, and latent transformation layers.

        Args:
            latent_dim (int): Dimension of the latent representation. Default is 16.
            enc_params (dict): Parameters for the encoder network (e.g., base dimensions).
                Includes base dimension for convolution layers.
            dec_params (dict): Parameters for the decoder network. Must mirror `enc_params`.
        """
        super(P2VAE, self).__init__()
        # Default configuration if none is provided
        enc_params = enc_params or {"base_dim": 64}
        dec_params = dec_params or {"base_dim": 64}

        self.latent_dim = latent_dim
        self.beta = 1e-3  # KL weight for regularization as per config.yaml

        # Encoder: Compresses input (128x128x3) to intermediate representation
        self.encoder = nn.Sequential(
            nn.Conv2d(3, enc_params["base_dim"], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(enc_params["base_dim"]),
            nn.ReLU(),
            nn.Conv2d(enc_params["base_dim"], enc_params["base_dim"] * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(enc_params["base_dim"] * 2),
            nn.ReLU(),
            nn.Conv2d(enc_params["base_dim"] * 2, enc_params["base_dim"] * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(enc_params["base_dim"] * 4),
            nn.ReLU(),
            nn.Conv2d(enc_params["base_dim"] * 4, enc_params["base_dim"] * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(enc_params["base_dim"] * 8),
            nn.ReLU()
        )

        # Latent space projection: Predict mean and log-variance for Gaussian latent space
        self.fc_mu = nn.Linear(enc_params["base_dim"] * 8 * (16 * 16 // (2 ** 4)), latent_dim)
        self.fc_logvar = nn.Linear(enc_params["base_dim"] * 8 * (16 * 16 // (2 ** 4)), latent_dim)

        # Decoder: Reconstructs original (128x128x3) from latent space
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, dec_params["base_dim"] * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dec_params["base_dim"] * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(dec_params["base_dim"] * 8, dec_params["base_dim"] * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dec_params["base_dim"] * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(dec_params["base_dim"] * 4, dec_params["base_dim"] * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dec_params["base_dim"] * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(dec_params["base_dim"] * 2, dec_params["base_dim"], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dec_params["base_dim"]),
            nn.ReLU(),
            nn.Conv2d(dec_params["base_dim"], 3, kernel_size=3, stride=1, padding=1)  # Final output with 3 channels
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input field snapshot into latent space parameters (mean and log variance).

        Args:
            x (torch.Tensor): Input tensor (batch_size, 3, 128, 128).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean (mu) and log variance (logvar) tensors 
            of the latent Gaussian distribution.
        """
        batch_size = x.size(0)
        encoded = self.encoder(x)
        encoded = encoded.view(batch_size, -1)  # Flatten for fully connected layers
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Applies the reparameterization trick to sample from the latent distribution.

        Args:
            mu (torch.Tensor): Mean of the latent space.
            logvar (torch.Tensor): Log variance of the latent space.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)   # Sample from standard normal
        return mu + eps * std         # Reparameterize

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes a latent vector back to the original spatial field.

        Args:
            z (torch.Tensor): Latent vector (batch_size, latent_dim).

        Returns:
            torch.Tensor: Reconstructed tensor (batch_size, 3, 128, 128).
        """
        z = z.view(z.size(0), self.latent_dim, 1, 1)  # Reshape latent vector for transposed convolution
        return self.decoder(z)

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the VAE loss (reconstruction + KL divergence).

        Args:
            x (torch.Tensor): Input tensor (batch_size, 3, 128, 128).

        Returns:
            torch.Tensor: Total loss value.
        """
        # Encode the input and sample from latent space
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Reconstruct the input from the latent samples
        x_reconstructed = self.decode(z)

        # Reconstruction Loss (L2 norm)
        recon_loss = F.mse_loss(x_reconstructed, x, reduction="sum") / x.size(0)

        # KL Divergence Loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # Total VAE Loss
        return recon_loss + self.beta * kl_loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE model.

        Args:
            x (torch.Tensor): Input tensor (batch_size, 3, 128, 128).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstructed output, latent mean, and latent log variance.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar
