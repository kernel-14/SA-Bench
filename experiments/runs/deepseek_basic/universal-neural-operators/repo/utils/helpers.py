"""Helper utilities for the neural operators project."""
import torch
import numpy as np
import random


def count_parameters(model, trainable_only=True):
    """Count number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def create_grid(nx, ny, device='cpu'):
    """Create 2D coordinate grid.

    Args:
        nx, ny: Grid dimensions
        device: Torch device

    Returns:
        Tensor of shape (nx, ny, 2) with x,y coordinates in [0, 1]
    """
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid.to(device)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
