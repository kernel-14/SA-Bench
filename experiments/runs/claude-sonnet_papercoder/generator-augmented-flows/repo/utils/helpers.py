## utils/helpers.py
"""Utility functions and classes shared across the iCT-GC codebase.

This module is the foundational utility layer with zero circular dependencies.
It provides: reproducibility seeding, device management, YAML I/O, model
parameter counting, image normalization, grid visualization, and loss tracking.

All other modules import from this file; this file imports nothing from the
project itself.
"""

import os
import random
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import yaml


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds for full reproducibility across all RNGs.

    Sets seeds for Python's built-in random module, NumPy, PyTorch (CPU and
    CUDA), and configures cuDNN for deterministic behaviour. This is required
    to reproduce the confidence-interval evaluations described in Appendix D
    of the paper (5 evaluation runs per model).

    Args:
        seed: Non-negative integer seed value. The config default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU safety
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------


def get_device(device_str: str = "cuda") -> torch.device:
    """Parse a device string from config and return a torch.device.

    Gracefully falls back to CPU if CUDA is requested but unavailable,
    printing a warning rather than raising an error.

    Args:
        device_str: One of 'cuda', 'cpu', or 'cuda:<index>' (e.g. 'cuda:0').
            Matches the ``device`` field in config.yaml.

    Returns:
        A ``torch.device`` object ready for use with ``.to(device)``.

    Raises:
        ValueError: If ``device_str`` is not a recognised format.
    """
    if device_str == "cpu":
        return torch.device("cpu")

    if device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print(
            "[helpers] WARNING: 'cuda' requested but CUDA is not available. "
            "Falling back to CPU."
        )
        return torch.device("cpu")

    if device_str.startswith("cuda:"):
        try:
            index = int(device_str.split(":")[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Invalid device string '{device_str}'. "
                "Expected format: 'cuda:<int>'."
            ) from exc

        num_devices = torch.cuda.device_count()
        if not torch.cuda.is_available() or index >= num_devices:
            print(
                f"[helpers] WARNING: '{device_str}' requested but only "
                f"{num_devices} CUDA device(s) available. "
                "Falling back to CPU."
            )
            return torch.device("cpu")
        return torch.device(device_str)

    raise ValueError(
        f"Unrecognised device string '{device_str}'. "
        "Expected one of: 'cpu', 'cuda', 'cuda:<index>'."
    )


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------


def save_yaml(data: dict, path: str) -> None:
    """Serialise a dictionary to a YAML file.

    Creates parent directories as needed. Uses block-style YAML for human
    readability and preserves key insertion order.

    Args:
        data: Dictionary to serialise (typically a config dict).
        path: Destination file path, e.g. './checkpoints/cifar10/config.yaml'.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_yaml(path: str) -> dict:
    """Deserialise a YAML file into a Python dictionary.

    Uses ``yaml.safe_load`` to prevent arbitrary code execution from
    potentially malicious YAML tags.

    Args:
        path: Path to the YAML file to load.

    Returns:
        Dictionary of parsed YAML contents. Returns ``{}`` for empty files.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"YAML configuration file not found: '{path}'"
        )

    with open(path, "r", encoding="utf-8") as f:
        result = yaml.safe_load(f)

    return result if result is not None else {}


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a model.

    Args:
        model: Any ``torch.nn.Module`` instance.

    Returns:
        Integer count of parameters where ``requires_grad=True``.
        For the SongUNet with ``model_channels=128`` on CIFAR-10 this is
        typically in the range of 55–60 million parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def normalize_images(x: torch.Tensor) -> torch.Tensor:
    """Convert images from training range [-1, 1] to evaluation range [0, 1].

    TorchMetrics ``FrechetInceptionDistance``, ``KernelInceptionDistance``,
    and ``InceptionScore`` with ``normalize=True`` expect float tensors in
    ``[0, 1]``. Without this conversion FID/KID/IS values will be incorrect.

    Args:
        x: Float tensor of shape ``(B, C, H, W)`` with values in ``[-1, 1]``.

    Returns:
        Float tensor of the same shape with values clamped to ``[0, 1]``.
    """
    x_out = (x + 1.0) / 2.0
    return torch.clamp(x_out, 0.0, 1.0)


def make_grid_image(
    samples: torch.Tensor,
    nrow: int = 8,
    padding: int = 2,
) -> np.ndarray:
    """Arrange a batch of image tensors into a single grid image.

    Converts a batch of images in ``[-1, 1]`` to a ``uint8`` numpy array
    suitable for TensorBoard ``add_image`` or ``matplotlib.pyplot.imshow``.

    The grid is constructed manually to avoid a hard dependency on
    ``torchvision.utils.make_grid`` in this base utility module.

    Args:
        samples: Float tensor of shape ``(B, C, H, W)`` in ``[-1, 1]``.
        nrow: Number of images per row in the grid.
        padding: Number of pixels of padding between images.

    Returns:
        ``uint8`` numpy array of shape ``(H_grid, W_grid, C)`` in ``[0, 255]``.
    """
    # Normalise to [0, 1] then convert to uint8 numpy
    normalized: torch.Tensor = normalize_images(samples.detach().cpu())
    imgs: np.ndarray = (normalized.numpy() * 255).astype(np.uint8)

    batch_size, channels, img_h, img_w = imgs.shape

    # Compute grid dimensions
    ncol: int = (batch_size + nrow - 1) // nrow  # number of rows in grid
    grid_h: int = ncol * img_h + (ncol + 1) * padding
    grid_w: int = nrow * img_w + (nrow + 1) * padding

    # Allocate white canvas
    grid: np.ndarray = np.full(
        (grid_h, grid_w, channels), fill_value=255, dtype=np.uint8
    )

    for idx in range(batch_size):
        row_idx: int = idx // nrow
        col_idx: int = idx % nrow

        y_start: int = padding + row_idx * (img_h + padding)
        x_start: int = padding + col_idx * (img_w + padding)

        # imgs[idx] is (C, H, W); transpose to (H, W, C) for grid placement
        grid[y_start : y_start + img_h, x_start : x_start + img_w, :] = (
            imgs[idx].transpose(1, 2, 0)
        )

    return grid


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


class AverageMeter:
    """Track running averages of scalar values across training batches.

    Typically used to monitor training loss. The ``update`` method supports
    a count argument so that pre-averaged batch losses are weighted correctly.

    Example::

        meter = AverageMeter()
        for batch in loader:
            loss = compute_loss(batch)
            meter.update(loss.item(), n=batch_size)
        print(f"Average loss: {meter.avg:.4f}")
        meter.reset()
    """

    def __init__(self) -> None:
        """Initialise all accumulators to zero."""
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators to zero."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update the running average with a new observation.

        Args:
            val: The scalar value to record. When ``val`` is a batch mean and
                ``n`` is the batch size, the weighted average is computed
                correctly across batches of different sizes.
            n: Number of samples this value represents. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        """Return a human-readable summary of the meter state."""
        return (
            f"AverageMeter(val={self.val:.6f}, avg={self.avg:.6f}, "
            f"count={self.count})"
        )
