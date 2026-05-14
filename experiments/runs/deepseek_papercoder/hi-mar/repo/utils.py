"""
utils.py

Shared utility functions and lightweight PyTorch modules used across the Hi‑MAR
implementation.  This module provides:
- Scale vector generation (`ScaleVectorMLP`)
- AdaLN modulation helpers (sinusoidal/time embeddings, modulation, chunk)
- Image ↔ tensor conversion utilities
- Miscellaneous helpers (parameter counting, logger setup, deterministic seeds)
"""

from __future__ import annotations

import logging
import math
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


# ---------------------------------------------------------------------------
#  Sinusoidal & time embedding
# ---------------------------------------------------------------------------

def get_sinusoidal_embedding(
    ids: Tensor,
    dim: int,
    max_period: float = 10000.0,
) -> Tensor:
    """
    Create sinusoidal positional embeddings as in "Attention Is All You Need".
    
    Args:
        ids: 1‑D tensor of integer (or float) identifiers, shape (B,).
        dim: Total embedding dimension (must be even).
        max_period: Largest period used by the scaling factors.
    
    Returns:
        Tensor of shape (B, dim).
    """
    if dim % 2 != 0:
        raise ValueError(f"Embedding dimension must be even, got {dim}")

    half_dim = dim // 2
    # Compute frequencies
    exponent = torch.arange(0, half_dim, dtype=torch.float32, device=ids.device)
    frequency = torch.exp(exponent * (-math.log(max_period) / half_dim))
    # Scale ids to [0, 2π) per frequency
    scaled_ids = ids.float().unsqueeze(1) * frequency.unsqueeze(0)  # (B, half)
    embedding = torch.cat([torch.sin(scaled_ids), torch.cos(scaled_ids)], dim=1)
    return embedding


def timestep_embedding(
    t: Tensor,
    dim: int,
    max_period: float = 1000.0,
) -> Tensor:
    """
    Embed a continuous timestep (normalised to [0, 1]) using sinusoidal encoding.
    
    Args:
        t: 1‑D tensor of timesteps, shape (B,).
        dim: Embedding dimension (must be even).
        max_period: Largest period.
    
    Returns:
        Tensor of shape (B, dim).
    """
    # Scale timestep by a large constant so that the lowest frequencies
    # respond to small changes.  This multiplier follows common practice (DiT).
    return get_sinusoidal_embedding(t * 1000.0, dim, max_period)


# ---------------------------------------------------------------------------
#  AdaLN modulation helpers
# ---------------------------------------------------------------------------

def modulate(
    feature: Tensor,
    scale: Tensor,
    shift: Tensor,
    gate: Tensor,
) -> Tensor:
    """
    Apply adaptive modulation:  out = gate * (feature * scale + shift).
    
    Supports both per‑token (B, N, C) and per‑sample (B, C) feature shapes.
    The scale/shift/gate tensors are broadcast appropriately.
    
    Args:
        feature: Input tensor, shape (B, N, C) or (B, C).
        scale:   Scale tensor, shape (B, C) or (B, C) broadcastable.
        shift:   Shift tensor, shape (B, C).
        gate:    Gating tensor, shape (B, C).
    
    Returns:
        Modulated tensor of the same shape as `feature`.
    """
    # Ensure scale, shift, gate are broadcastable to feature shape
    # If feature has 3 dims (B,N,C) and scale is (B,C), add a dummy token dim
    if feature.ndim == 3 and scale.ndim == 2:
        scale = scale.unsqueeze(1)   # (B, 1, C)
        shift = shift.unsqueeze(1)
        gate  = gate.unsqueeze(1)

    return gate * (feature * scale + shift)


def chunk_adaLN_parameters(
    linear_output: Tensor,
    hidden_size: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Split the output of an adaLN conditioning linear layer into six tensors,
    each of shape (B, hidden_size).

    These correspond to (α₁, β₁, γ₁, α₂, β₂, γ₂) used before self‑attention
    and FFN sub‑layers.

    Args:
        linear_output: Tensor of shape (B, 6 * hidden_size).
        hidden_size:   Number of features per modulation vector.

    Returns:
        Tuple of six tensors, each (B, hidden_size).
    """
    if linear_output.shape[-1] != 6 * hidden_size:
        raise ValueError(
            f"Expected last dim {6 * hidden_size}, got {linear_output.shape[-1]}"
        )
    chunks = linear_output.chunk(6, dim=-1)
    # tiny safety – chunk returns a tuple
    assert len(chunks) == 6
    return (chunks[0], chunks[1], chunks[2], chunks[3], chunks[4], chunks[5])


# ---------------------------------------------------------------------------
#  Scale Vector MLP (used by Hi‑MAR Transformer)
# ---------------------------------------------------------------------------

class ScaleVectorMLP(nn.Module):
    """
    Convert a discrete scale identifier (0 = low‑res, 1 = high‑res) into a
    conditioning vector `v` that will later be linearly transformed to yield
    the adaLN modulation parameters for each block.

    Design follows Section 3.2 of the paper.
    """

    def __init__(
        self,
        sin_embed_dim: int = 256,
        hidden_dim: int = 512,
        out_dim: int = 256,
    ):
        super().__init__()
        self.sin_embed_dim = sin_embed_dim
        self.out_dim = out_dim

        self.mlp = nn.Sequential(
            nn.Linear(sin_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, scale_id: Tensor) -> Tensor:
        """
        Args:
            scale_id: Integer tensor of shape (B,) containing scale ids.
        Returns:
            Scale vector `v`, shape (B, out_dim).
        """
        sin_emb = get_sinusoidal_embedding(scale_id, self.sin_embed_dim)
        return self.mlp(sin_emb)


# ---------------------------------------------------------------------------
#  Image ↔ Tensor utilities
# ---------------------------------------------------------------------------

def image_to_tensor(
    pil_image: Image.Image,
    resolution: Optional[int] = None,
) -> Tensor:
    """
    Convert a PIL image to a normalised torch tensor (range [-1, 1]).

    Args:
        pil_image: Input PIL image (mode RGB or L).
        resolution: Optional integer for resizing (default `256` if not given).

    Returns:
        Tensor of shape (3, H, W) with values in [-1, 1].
    """
    if resolution is None:
        resolution = 256  # default, can be overridden
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    # Resize keeping aspect ratio?  For reproducibility, resize to square.
    pil_image = pil_image.resize((resolution, resolution), Image.BICUBIC)

    # Convert to float tensor in [0, 1]
    img_tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0  # (H,W,3)
    img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)

    # Normalise to [-1, 1] as commonly expected by VAEs
    img_tensor = 2.0 * img_tensor - 1.0
    return img_tensor


def tensor_to_image(
    tensor: Tensor,
    denorm: bool = True,
) -> Image.Image:
    """
    Convert a torch tensor back to a PIL image.

    Args:
        tensor: Tensor of shape (C, H, W) or (H, W, C) with values in [-1, 1] (if
                denorm=True) or [0, 1] otherwise.
        denorm: If True, applies ``(t + 1) / 2`` before conversion.

    Returns:
        PIL Image in RGB mode.
    """
    if denorm:
        tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0.0, 1.0)

    # Ensure CHW layout
    if tensor.ndim == 3 and tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)  # (H, W, C)
    elif tensor.ndim == 3 and tensor.shape[-1] == 3:
        pass  # already HWC
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

    arr = (tensor.cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------------------
#  Miscellaneous helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters of a module.

    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def setup_logger(
    name: str,
    output_dir: Optional[str] = None,
    level: int = logging.INFO,
    filename: str = "run.log",
) -> logging.Logger:
    """
    Create a logger that writes to both stdout and a file.

    Args:
        name:        Logger name (usually ``__name__``).
        output_dir:  Directory for the log file.  If None, file logging is skipped.
        level:       Logging level.
        filename:    Name of the log file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove any existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # Stream handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # File handler (optional)
    if output_dir is not None:
        import os

        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, filename)
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(console_fmt)
        logger.addHandler(file_handler)

    return logger


def seed_everything(seed: int) -> None:
    """
    Set global random seeds for Python, NumPy and PyTorch (CPU/GPU).

    Also tries to turn on deterministic algorithms in cuDNN for reproducibility,
    which may impact performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enforce determinism (may slow things down)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
