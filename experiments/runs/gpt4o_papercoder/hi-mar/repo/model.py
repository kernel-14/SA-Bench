# model.py

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
from utils import generate_sinusoidal_embedding, build_scale_vector


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for encoding images into latent representations and decoding back to image space.
    """

    def __init__(self, pretrained_path: str, resolutions: Dict[str, Tuple[int, int]]):
        """
        Initialize the VAE model with pretrained parameters.

        Args:
            pretrained_path (str): Path to the pre-trained VAE model.
            resolutions (Dict[str, Tuple[int, int]]): Dictionary with low and high resolutions.
                                                      Example: {"low": (128, 128), "high": (256, 256)}.
        """
        super(VAE, self).__init__()
        self.pretrained_path = pretrained_path
        self.resolutions = resolutions

        # Load pretrained VAE (assuming it is using state_dict or torch model definitions)
        self.vae = torch.load(self.pretrained_path)
        self.vae.eval()

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode an image into its latent token representation.

        Args:
            image (torch.Tensor): Tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Latent representation, shape (batch_size, latent_dim).
        """
        return self.vae.encode(image)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent tokens back into image space.

        Args:
            latent (torch.Tensor): Tensor of latent tokens, shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Reconstructed images, shape (batch_size, 3, H, W).
        """
        return self.vae.decode(latent)


class HiMARTransformer(nn.Module):
    """
    Hierarchical Masked Autoregressive Transformer for image token prediction at different scales.
    """

    def __init__(self, layers: int, hidden_size: int, scale_embedding: bool):
        """
        Initialize the Hi-MAR Transformer.

        Args:
            layers (int): Number of transformer blocks.
            hidden_size (int): Size of the hidden layer in the transformer.
            scale_embedding (bool): Whether to use scale-aware embedding.
        """
        super(HiMARTransformer, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.scale_embedding = scale_embedding

        # Transformers blocks
        self.encoder_layers = nn.ModuleList([
            HiMARTransformerBlock(hidden_size) for _ in range(self.layers)
        ])

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for masked token prediction.

        Args:
            tokens (torch.Tensor): Tensor of input tokens, shape (batch_size, seq_len, hidden_size).
            context (torch.Tensor): Contextual embeddings, shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: Predicted tokens, shape (batch_size, seq_len, hidden_size).
        """

        # Propagating through Transformer Layers
        for layer in self.encoder_layers:
            tokens = layer(tokens, context)

        return tokens


class HiMARTransformerBlock(nn.Module):
    """
    Single Hierarchical Transformer block with scale-aware operations, self-attention, and feedforward layers.
    """

    def __init__(self, hidden_size: int):
        """
        Args:
            hidden_size (int): Dimensionality of the hidden representation.
        """
        super(HiMARTransformerBlock, self).__init__()
        self.hidden_size = hidden_size

        # Self-Attention and Feedforward Components
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Compute the output of a single transformer block.

        Args:
            tokens (torch.Tensor): Input token embeddings, shape (batch_size, seq_len, hidden_dim).
            context (torch.Tensor): Context embeddings, shape (batch_size, seq_len, hidden_dim).

        Returns:
            torch.Tensor: Updated token embeddings, shape (batch_size, seq_len, hidden_dim).
        """
        # Multi-head Attention
        attention_out, _ = self.attention(tokens, tokens, tokens)
        tokens = tokens + attention_out  # Residual connection
        tokens = self.layer_norm_1(tokens)

        # Feedforward Layer
        feedforward_out = self.feedforward(tokens)
        tokens = tokens + feedforward_out  # Residual connection
        tokens = self.layer_norm_2(tokens)

        return tokens


class DiffusionHead(nn.Module):
    """
    Diffusion Head for modeling masked token prediction.
    """

    def __init__(self, type: str, params: Dict[str, Any]):
        """
        Initialize DiffusionHead for Phase 1 (MLP-based) or Phase 2 (Transformer-based).

        Args:
            type (str): Type of the diffusion head, either 'MLP' or 'Transformer'.
            params (Dict[str, Any]): Configuration parameters for the diffusion head. For example:
                                     {'num_layers': 6, 'hidden_size': 512}.
        """
        super(DiffusionHead, self).__init__()

        self.type = type
        if self.type == "MLP":
            # MLP-based Diffusion Head
            self.layers = nn.Sequential(
                nn.Linear(params["hidden_size"], params["hidden_size"] * 4),
                nn.GELU(),
                nn.Linear(params["hidden_size"] * 4, params["hidden_size"])
            )
        elif self.type == "Transformer":
            # Transformer-based Diffusion Head
            self.layers = nn.ModuleList([
                HiMARTransformerBlock(params["hidden_size"]) for _ in range(params["num_layers"])
            ])
        else:
            raise ValueError("Unsupported type for DiffusionHead. Expected 'MLP' or 'Transformer'.")

    def forward(self, z: torch.Tensor, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        Forward pass for token prediction and diffusion modeling.

        Args:
            z (torch.Tensor): Conditional token embeddings, shape (batch_size, seq_len, hidden_size).
            x_t (torch.Tensor): Noise-corrupted visual tokens, shape (batch_size, seq_len, hidden_size).
            t (int): Timestep for diffusion process.

        Returns:
            torch.Tensor: Predicted token outputs, shape (batch_size, seq_len, hidden_size).
        """
        if self.type == "MLP":
            # Forward through MLP layers
            return self.layers(z)
        elif self.type == "Transformer":
            # Forward through Transformer layers
            for layer in self.layers:
                z = layer(z, context=x_t)
            return z

    def compute_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute denoising loss for reconstructing target tokens from predicted tokens.

        Args:
            predicted (torch.Tensor): Predicted tokens, shape (batch_size, seq_len, hidden_size).
            target (torch.Tensor): Ground truth target tokens, shape (batch_size, seq_len, hidden_size).

        Returns:
            float: Mean squared error loss.
        """
        return nn.functional.mse_loss(predicted, target)

