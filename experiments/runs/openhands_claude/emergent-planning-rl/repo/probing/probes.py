import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LinearProbe1x1(nn.Module):
    """
    1x1 linear probe for predicting square-level concepts.
    
    Takes as input the cell state activations at position (x, y) — i.e.,
    a vector of size hidden_channels — and predicts the concept class.
    
    Implemented as a 1x1 convolution to efficiently process all spatial
    positions simultaneously.
    
    Parameters: hidden_channels * num_classes = 32 * 5 = 160
    """

    def __init__(self, hidden_channels: int = 32, num_classes: int = 5):
        super().__init__()
        self.conv = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, cell_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cell_state: (B, C, H, W) cell state activations
        Returns:
            logits: (B, num_classes, H, W)
        """
        return self.conv(cell_state)

    def get_class_vectors(self) -> torch.Tensor:
        """
        Returns the weight vectors w_k for each class k.
        Shape: (num_classes, hidden_channels)
        These are the vectors used for interventions.
        """
        return self.conv.weight.squeeze(-1).squeeze(-1)


class LinearProbe3x3(nn.Module):
    """
    3x3 linear probe for predicting square-level concepts.
    
    Takes as input the 3x3 patch of cell state activations around (x, y)
    and predicts the concept class at (x, y).
    
    Parameters: (3*3*hidden_channels) * num_classes = 9*32*5 = 1440
    """

    def __init__(self, hidden_channels: int = 32, num_classes: int = 5):
        super().__init__()
        self.conv = nn.Conv2d(
            hidden_channels, num_classes, kernel_size=3, padding=1
        )

    def forward(self, cell_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cell_state: (B, C, H, W) cell state activations
        Returns:
            logits: (B, num_classes, H, W)
        """
        return self.conv(cell_state)


class LinearProbeNxN(nn.Module):
    """
    NxN linear probe for predicting square-level concepts.
    Generalizes 1x1 and 3x3 probes to arbitrary kernel sizes.
    """

    def __init__(self, hidden_channels: int = 32, num_classes: int = 5, kernel_size: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            hidden_channels, num_classes, kernel_size=kernel_size, padding=padding
        )
        self.kernel_size = kernel_size

    def forward(self, cell_state: torch.Tensor) -> torch.Tensor:
        return self.conv(cell_state)


class GlobalLinearProbe(nn.Module):
    """
    Global linear probe that receives the entire cell state as input.
    Used for predicting global concepts like 'Action To Take in N Steps'.
    
    Parameters: (H * W * hidden_channels) * num_classes = 8*8*32*5 = 10240
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        grid_size: int = 8,
        num_classes: int = 5,
    ):
        super().__init__()
        flat_size = hidden_channels * grid_size * grid_size
        self.linear = nn.Linear(flat_size, num_classes)

    def forward(self, cell_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cell_state: (B, C, H, W) cell state activations
        Returns:
            logits: (B, num_classes)
        """
        flat = cell_state.view(cell_state.shape[0], -1)
        return self.linear(flat)


def create_probe(
    probe_size: int,
    hidden_channels: int = 32,
    num_classes: int = 5,
) -> nn.Module:
    """Factory function for creating probes by kernel size."""
    if probe_size == 1:
        return LinearProbe1x1(hidden_channels, num_classes)
    elif probe_size == 3:
        return LinearProbe3x3(hidden_channels, num_classes)
    else:
        return LinearProbeNxN(hidden_channels, num_classes, kernel_size=probe_size)
