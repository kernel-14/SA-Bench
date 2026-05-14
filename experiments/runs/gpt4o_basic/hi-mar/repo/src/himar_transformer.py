import torch
import torch.nn as nn
import math

class ScaleAwareTransformerBlock(nn.Module):
    def __init__(self, hidden_size):
        """
        A scale-aware Transformer block as part of Hi-MAR's architecture.

        Args:
            hidden_size (int): Dimensionality of the hidden layer.
        """
        super(ScaleAwareTransformerBlock, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, scale_vector):
        """
        Forward pass for scale-aware transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, hidden_size).
            scale_vector (torch.Tensor): Scale vector for normalization.

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, hidden_size).
        """
        # Self-Attention with scaled inputs
        scale = scale_vector.unsqueeze(-1)
        x_scaled = x * scale
        x_norm = self.layernorm1(x_scaled)
        x_attended, _ = self.attention(x_norm, x_norm, x_norm)

        # Feedforward network
        x_ffn = self.layernorm2(x + x_attended)
        output = x_ffn + self.ffn(x_ffn)
        return output
