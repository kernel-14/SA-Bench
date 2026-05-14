"""
Utility functions for SC-FNO experiments.

Also contains `rebuild_input_with_params`, shared by train.py and evaluate.py
to avoid circular imports.
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from config import ExperimentConfig, FNOConfig
from models.fno import FNO1d, FNO2d


def rebuild_input_with_params(
    fno_input: torch.Tensor,
    params_rg: torch.Tensor,
    equation_type: str,
) -> torch.Tensor:
    """
    Rebuild the FNO input tensor with params_rg (requires_grad=True) embedded.

    Parameters occupy the last n_params channels of fno_input. We replace them
    with params_rg expanded to match the spatial-temporal dimensions so that
    AD can flow through params_rg during sensitivity loss computation.

    Args:
        fno_input: original input tensor (detached)
        params_rg: (batch, n_params) with requires_grad=True
        equation_type: "ode", "pde1d", or "pde2d"

    Returns:
        new_input: same shape as fno_input but with params_rg embedded
    """
    n_params = params_rg.shape[1]

    if equation_type == "ode":
        # fno_input: (batch, T_in, 2+n_params)
        non_param = fno_input[:, :, :-n_params].detach()
        T_in = fno_input.shape[1]
        p_rep = params_rg.unsqueeze(1).expand(-1, T_in, -1)
        return torch.cat([non_param, p_rep], dim=-1)

    elif equation_type == "pde1d":
        # fno_input: (batch, Sx, T_in, 3+n_params)
        non_param = fno_input[:, :, :, :-n_params].detach()
        Sx, T_in = fno_input.shape[1], fno_input.shape[2]
        p_rep = params_rg.unsqueeze(1).unsqueeze(1).expand(-1, Sx, T_in, -1)
        return torch.cat([non_param, p_rep], dim=-1)

    elif equation_type == "pde2d":
        # fno_input: (batch, Sx, Sy, 3+n_params)
        non_param = fno_input[:, :, :, :-n_params].detach()
        Sx, Sy = fno_input.shape[1], fno_input.shape[2]
        p_rep = params_rg.unsqueeze(1).unsqueeze(1).expand(-1, Sx, Sy, -1)
        return torch.cat([non_param, p_rep], dim=-1)

    else:
        raise ValueError(f"Unknown equation_type: {equation_type}")


def build_fno_model(
    equation: str,
    cfg: FNOConfig,
    n_params: int,
    device: torch.device,
) -> nn.Module:
    """
    Build the appropriate FNO model for a given equation.

    Input channel counts:
      ODE:   1 (u) + 1 (t) + n_params = 2 + n_params
      PDE1D: 1 (u) + 2 (x,t) + n_params = 3 + n_params
      PDE2D: 1 (ω) + 2 (x,y) + n_params = 3 + n_params

    For PDE3 (Navier-Stokes), FNO2d is used with (Sx, Sy) as the two spatial
    dimensions (no explicit time axis since we map IC → final state directly).

    Output channels: always 1 (scalar field)
    """
    if equation in ("ode1", "ode2"):
        in_channels = 2 + n_params  # u, t, params
        model = FNO1d(
            modes=cfg.modes_t,
            width=cfg.width,
            in_channels=in_channels,
            out_channels=1,
            n_layers=cfg.n_layers,
        )
    elif equation in ("pde1", "pde2", "pde2_zoned", "pde4"):
        in_channels = 3 + n_params  # u, x, t, params
        model = FNO2d(
            modes1=cfg.modes_x,
            modes2=cfg.modes_t,
            width=cfg.width,
            in_channels=in_channels,
            out_channels=1,
            n_layers=cfg.n_layers,
        )
    elif equation == "pde3":
        # 2D spatial FNO: (batch, Sx, Sy, in_channels)
        # Treated as FNO2d with modes1=modes_x, modes2=modes_y
        in_channels = 3 + n_params  # ω, x, y, params
        model = FNO2d(
            modes1=cfg.modes_x,
            modes2=cfg.modes_y,
            width=cfg.width,
            in_channels=in_channels,
            out_channels=1,
            n_layers=cfg.n_layers,
        )
    else:
        raise ValueError(f"Unknown equation: {equation}")

    return model.to(device)


def get_equation_type(equation: str) -> str:
    """Map equation name to type string used in training/evaluation."""
    if equation in ("ode1", "ode2"):
        return "ode"
    elif equation in ("pde1", "pde2", "pde2_zoned", "pde4"):
        return "pde1d"
    elif equation == "pde3":
        return "pde2d"
    else:
        raise ValueError(f"Unknown equation: {equation}")


def count_parameters(model: nn.Module) -> int:
    """Count total number of learnable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_data(
    data: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize data to zero mean and unit variance.

    Returns:
        normalized data, mean, std
    """
    if mean is None:
        mean = data.mean()
    if std is None:
        std = data.std()
    return (data - mean) / (std + eps), mean, std


def denormalize_data(
    data: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Reverse normalization."""
    return data * (std + eps) + mean


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(use_gpu: bool = True) -> torch.device:
    """Get computation device."""
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_model_info(model: nn.Module, name: str = "Model") -> None:
    """Print model architecture summary."""
    n_params = count_parameters(model)
    print(f"{name}: {n_params:,} learnable parameters")


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dict as a readable string."""
    parts = []
    for k, v in metrics.items():
        parts.append(f"{k}={v:.4f}")
    return ", ".join(parts)


def save_results(results: Dict, path: str) -> None:
    """Save experiment results to disk."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(results, path)


def load_results(path: str) -> Dict:
    """Load experiment results from disk."""
    return torch.load(path, map_location="cpu")
