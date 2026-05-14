## fmt_model.py

import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

class FMTModel(nn.Module):
    """
    Flow Marching Transformer (FMT) implementation for predicting future states of PDE-based systems.
    Combines deterministic neural operators and stochastic flow matching capabilities.
    """

    def __init__(
        self,
        input_dim: int = 128,  # Dimension of latent input (from P2VAE output, e.g., c16p16)
        embed_dim: int = 256,  # Embedding dimension for FMT-S
        num_heads: int = 4,  # Number of attention heads
        num_layers: int = 6,  # Number of Transformer layers
        rnn_dim: int = 256  # Dimension of GRU-based latent state
    ):
        """
        Initializes the Flow Marching Transformer model with Transformer backbone and RNN-based diffusion forcing.

        Args:
            input_dim (int): Dimensionality of the input features.
            embed_dim (int): Embedding size for Transformer layers.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer layers.
            rnn_dim (int): GRU latent state dimension.
        """
        super(FMTModel, self).__init__()

        # Embedding layer to align input dimensions with Transformer input space
        self.embedding = nn.Linear(input_dim + 1, embed_dim)  # Include time step (t) in input

        # Transformer backbone for processing temporal dynamics
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="swiglu",  # SwiGLU activation per LLaMA-2 recommendations
            batch_first=True,  # Enable batch-first mode for easier integration with data pipelines
            norm_first=True  # RMSNorm normalization before attention/residual connections
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Diffusion-forcing mechanism with latent GRU
        self.gru = nn.GRU(input_dim, rnn_dim, batch_first=True)

        # Output layer to map latent features back to velocity predictions
        self.output_layer = nn.Linear(embed_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,  # Latent states (batch_size, seq_len, input_dim)
        h: torch.Tensor  # Latent condition state (batch_size, rnn_dim)
    ) -> torch.Tensor:
        """
        Forward pass for the Flow Marching Transformer.

        Args:
            x (torch.Tensor): Current latent representation of PDE states.
            h (torch.Tensor): Current latent condition state (e.g., autoregressive history).

        Returns:
            torch.Tensor: Predicted flow marching velocities.
        """
        batch_size, seq_len, _ = x.shape

        # Embed input tensor (includes time embedding as additional feature)
        t = torch.linspace(0, 1, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len, 1)
        concatenated_input = torch.cat((x, t), dim=-1)
        embedded_input = self.embedding(concatenated_input)

        # Process through Transformer
        transformer_output = self.transformer(embedded_input)

        # Update latent condition state via GRU
        _, h_next = self.gru(x, h.unsqueeze(0))  # GRU expects 3D tensor; h reshaped appropriately

        # Map Transformer output back to flow marching velocities
        velocity_predictions = self.output_layer(transformer_output)

        return velocity_predictions

    def compute_loss(
        self,
        x_prev: torch.Tensor,  # Previous latent state
        x_next: torch.Tensor,  # Target next latent state
        h: torch.Tensor  # Current latent condition state
    ) -> torch.Tensor:
        """
        Compute the conditional flow marching loss.

        Args:
            x_prev (torch.Tensor): Latent representation of the previous state.
            x_next (torch.Tensor): Latent representation of the target next state.
            h (torch.Tensor): Current latent condition state.

        Returns:
            torch.Tensor: Loss value.
        """
        # Forward propagation for predicted velocities
        predicted_velocities = self.forward(x_prev, h)

        # Compute residuals
        residuals = x_next - x_prev

        # Calculate time preconditioning factor
        batch_size, seq_len, _ = x_prev.shape
        t = torch.linspace(0, 1, seq_len, device=x_prev.device).unsqueeze(0).expand(batch_size, seq_len, 1)

        # Compute conditional flow marching loss
        loss = F.mse_loss((1 - t) * predicted_velocities, residuals, reduction="mean")
        return loss
