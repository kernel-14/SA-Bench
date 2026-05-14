## utils/data_utils.py
"""
Utility functions for data preprocessing and flow-marching noise generation.

All functions operate on PyTorch tensors and are stateless.
Expected tensor shapes:
    - field tensors: (B, C, H, W) or (C, H, W) – spatial dims last.
    - latent tensors: (B, C, H, W) – from P2VAE encoder.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union

# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------

def normalize_field(
    tensor: torch.Tensor,
    method: str = "minmax",
    stats: Optional[Dict[str, Union[torch.Tensor, list]]] = None,
) -> torch.Tensor:
    """
    Normalize a physical field tensor channel-wise.

    Args:
        tensor: (B, C, H, W) or (C, H, W) field.
        method: "minmax" or "zscore".
        stats: Dictionary with per-channel statistics.
               For "minmax": {'min': ..., 'max': ...}
               For "zscore": {'mean': ..., 'std': ...}
               Values can be list or tensor of length C.

    Returns:
        Normalized tensor of same shape, dtype float32.
    """
    if stats is None:
        raise ValueError("Normalization stats must be provided.")

    original_shape = tensor.shape
    # Ensure batch dimension exists for broadcasting
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
        was_3d = True
    else:
        was_3d = False

    # Work in float32
    tensor = tensor.float()

    C = tensor.size(1)

    # Convert stats to tensors if needed
    def _to_tensor(val, name):
        if isinstance(val, list):
            val = torch.tensor(val, device=tensor.device, dtype=torch.float32)
        elif isinstance(val, torch.Tensor):
            val = val.to(device=tensor.device, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported type for {name}: {type(val)}")
        if val.numel() != C:
            raise ValueError(
                f"{name} must have length C={C}, got {val.numel()}"
            )
        return val.view(1, C, 1, 1)

    if method == "minmax":
        min_val = _to_tensor(stats["min"], "min")
        max_val = _to_tensor(stats["max"], "max")
        denom = (max_val - min_val).clamp_min(1e-8)
        tensor = (tensor - min_val) / denom
        tensor = tensor.clamp(0.0, 1.0)
    elif method == "zscore":
        mean = _to_tensor(stats["mean"], "mean")
        std = _to_tensor(stats["std"], "std")
        tensor = (tensor - mean) / (std + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    if was_3d:
        tensor = tensor.squeeze(0)
    return tensor


# -----------------------------------------------------------------------------
# Spatial resizing
# -----------------------------------------------------------------------------

def resize_field(
    tensor: torch.Tensor,
    target_size: Union[int, Tuple[int, int]],
) -> torch.Tensor:
    """
    Resize spatial dimensions of a field using bilinear interpolation.

    Args:
        tensor: (..., H, W) or (..., C, H, W). Last two dims are spatial.
        target_size: int (square) or (H, W) tuple.

    Returns:
        Resized tensor with spatial dims target_size.
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    ndim = tensor.dim()
    if ndim == 3:
        # Interpret as (C, H, W) -> unsqueeze batch
        tensor = tensor.unsqueeze(0)
        single = True
    elif ndim == 4:
        single = False
    else:
        raise ValueError(
            f"Expected 3D (C,H,W) or 4D (B,C,H,W) tensor, got {ndim}D"
        )

    resized = F.interpolate(
        tensor.float(),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )

    if single:
        resized = resized.squeeze(0)
    return resized


# -----------------------------------------------------------------------------
# Channel padding to 3
# -----------------------------------------------------------------------------

def pad_channels(
    tensor: torch.Tensor,
    target_channels: int = 3,
) -> torch.Tensor:
    """
    Pad or truncate channels to exactly target_channels.
    Used to enforce c3p128 consistency across datasets.

    Args:
        tensor: (..., C, H, W). Channel dimension is second last.
        target_channels: int, default 3.

    Returns:
        Tensor with C = target_channels.
    """
    if tensor.size(-3) == target_channels:
        return tensor

    if tensor.size(-3) < target_channels:
        # Pad with zeros
        missing = target_channels - tensor.size(-3)
        shape = list(tensor.shape)
        shape[-3] = missing
        zeros = torch.zeros(shape, device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, zeros], dim=-3)
    else:
        # Truncate (should not happen normally)
        tensor = tensor[..., :target_channels, :, :]

    return tensor


# -----------------------------------------------------------------------------
# Latent pyramid downsampling (average pooling)
# -----------------------------------------------------------------------------

def downsample_token_grid(
    tensor: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """
    Downsample a latent grid to a target spatial resolution using average pooling.
    This implements the latent temporal pyramid described in the paper.

    Args:
        tensor: (B, C, H, W) latent tensor (from P2VAE encoder, e.g., 16x16).
        target_h, target_w: desired spatial size (e.g., 2,2 for first frame).

    Returns:
        Downsampled tensor of shape (B, C, target_h, target_w).
    """
    if tensor.dim() != 4:
        raise ValueError(
            f"Expect 4D tensor (B,C,H,W), got {tensor.dim()}D"
        )

    return F.adaptive_avg_pool2d(tensor, (target_h, target_w))


# -----------------------------------------------------------------------------
# Flow‑marching noise kernel (Eq. 1)
# -----------------------------------------------------------------------------

def generate_noisy_latent(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: Union[float, torch.Tensor],
    k: Union[float, torch.Tensor],
    z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Construct the intermediate state x_t^k using the location-scale interpolation kernel.

    x_t^k = μ_t + σ_t * z
    with:
        μ_t = t * x1 + k * (1 - t) * x0
        σ_t = (1 - t) * (1 - k)
        z ~ N(0, I)

    This bridges between clean states x0 and x1 under bridge parameter k
    and flow time t ∈ [0,1].

    Args:
        x0: Tensor (B, C, H, W) – current state.
        x1: Tensor (B, C, H, W) – next state.
        t: scalar or tensor broadcastable to x0 shape; flow time.
        k: scalar or tensor broadcastable to x0 shape; bridge parameter (0=stochastic, 1=deterministic).
        z: optional pre‑sampled noise, otherwise sampled internally.

    Returns:
        x_t_k: noisy state of same shape as x0.
    """
    # Convert t, k to tensors and broadcast
    if isinstance(t, (float, int)):
        t = torch.tensor(t, device=x0.device, dtype=x0.dtype)
    if isinstance(k, (float, int)):
        k = torch.tensor(k, device=x0.device, dtype=x0.dtype)

    # Reshape scalars to (1,1,1,1) for broadcasting
    if t.dim() == 0:
        t = t.view(1, 1, 1, 1)
    elif t.dim() == 1:
        t = t.view(-1, 1, 1, 1)
    # k similarly
    if k.dim() == 0:
        k = k.view(1, 1, 1, 1)
    elif k.dim() == 1:
        k = k.view(-1, 1, 1, 1)

    # Compute mean
    mu = t * x1 + k * (1.0 - t) * x0

    # Compute scale
    sigma = (1.0 - t) * (1.0 - k)
    # Add tiny epsilon to avoid zero in later computations (though not needed here)
    sigma = sigma.clamp_min(1e-8)

    if z is None:
        z = torch.randn_like(x0)

    x_t_k = mu + sigma * z
    return x_t_k
