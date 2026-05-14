import torch
import torch.nn as nn

class MLPDiffusionHead(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        MLP-based Diffusion Head for processing masked tokens.

        Args:
            input_size (int): Size of the input layer.
            hidden_size (int): Size of the hidden layers.
        """
        super(MLPDiffusionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        """
        Forward pass for MLP-based diffusion head.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Processed tensor.
        """
        return self.mlp(x)

class DiffusionTransformerHead(nn.Module):
    def __init__(self, hidden_size):
        """
        Diffusion Transformer Head for modeling inter-token dependencies.

        Args:
            hidden_size (int): Size of the hidden layers.
        """
        super(DiffusionTransformerHead, self).__init__()
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=8,
            num_encoder_layers=6
        )

    def forward(self, x):
        """
        Forward pass for Diffusion Transformer Head.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Processed tensor accounting for inter-token dependencies.
        """
        return self.transformer(x)
