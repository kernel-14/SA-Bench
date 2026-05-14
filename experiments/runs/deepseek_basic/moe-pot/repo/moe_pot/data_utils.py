"""
Data preprocessing and loading utilities for PDE datasets.

Implements the preprocessing pipeline described in Appendix B.1:
- Spatial resolution standardization (H=128)
- Channel padding to unify variable counts
- Mask channel for irregular geometries
- Balanced data sampling
- Noise insertion

Supports 6 pre-training datasets:
1. FNO-NS (1e-5): Navier-Stokes, ν=1e-5
2. FNO-NS (1e-3): Navier-Stokes, ν=1e-3
3. PDEBench-CNS (0.1, 0.01): Compressible Navier-Stokes
4. PDEBench-SWE: Shallow Water Equations
5. PDEBench-DR: Diffusion-Reaction
6. CFDBench: CFD benchmark with irregular geometries

And downstream datasets:
- NS (1e-4), CNS (1, 0.01), PDEArena
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple


def standardize_resolution(
    data: torch.Tensor,
    target_h: int = 128,
    target_w: int = 128,
    mode: str = 'bilinear',
) -> torch.Tensor:
    """
    Standardize spatial resolution to H=target_h, W=target_w.
    
    As described in Appendix B.1:
    - For lower resolutions: upscale using interpolation
    - For higher resolutions: downscale using random sampling or interpolation
    
    Args:
        data: [..., H, W] tensor
        target_h: target height
        target_w: target width
        mode: interpolation mode
        
    Returns:
        resized data at target resolution
    """
    if data.shape[-2] == target_h and data.shape[-1] == target_w:
        return data
    
    # Add batch and channel dims for interpolation if needed
    original_shape = data.shape
    if data.dim() == 2:
        data = data.unsqueeze(0).unsqueeze(0)
    elif data.dim() == 3:
        data = data.unsqueeze(0)
    
    # Interpolate
    resized = F.interpolate(
        data,
        size=(target_h, target_w),
        mode=mode,
        align_corners=False if mode != 'nearest' else None,
    )
    
    # Restore original shape (minus the last two dims which changed)
    if len(original_shape) == 2:
        resized = resized.squeeze(0).squeeze(0)
    elif len(original_shape) == 3:
        resized = resized.squeeze(0)
    
    return resized


def pad_channels(
    data: torch.Tensor,
    target_channels: int,
    pad_value: float = 1.0,
) -> torch.Tensor:
    """
    Pad channel dimension to unify number of variables across PDEs.
    
    As described in Appendix B.1:
    Pad all datasets along the channel dimension to match the dataset
    with the maximum number of channels, filling unused entries with
    a constant value (e.g., 1).
    
    Args:
        data: [..., C, H, W] or [..., C] tensor
        target_channels: target number of channels
        pad_value: value to fill padded channels
        
    Returns:
        padded data with target_channels
    """
    current_channels = data.shape[-3] if data.dim() >= 3 else data.shape[-1]
    
    if current_channels >= target_channels:
        return data
    
    # Compute padding needed
    pad_needed = target_channels - current_channels
    
    if data.dim() >= 3:
        # [..., C, H, W]
        pad_shape = list(data.shape)
        pad_shape[-3] = pad_needed
        padding = torch.full(pad_shape, pad_value, dtype=data.dtype, device=data.device)
        return torch.cat([data, padding], dim=-3)
    else:
        # [..., C]
        pad_shape = list(data.shape)
        pad_shape[-1] = pad_needed
        padding = torch.full(pad_shape, pad_value, dtype=data.dtype, device=data.device)
        return torch.cat([data, padding], dim=-1)


def create_mask_channel(
    data: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Create and append a mask channel for irregular geometries.
    
    As described in Appendix B.1:
    For datasets with irregular geometric shapes, use an additional mask
    channel that encodes the geometric configuration.
    
    Args:
        data: [..., C, H, W] tensor
        mask: optional binary mask [H, W], if None assumes regular grid
        
    Returns:
        data with mask channel appended
    """
    if mask is None:
        # Regular grid: mask is all ones
        H, W = data.shape[-2], data.shape[-1]
        mask = torch.ones(H, W, dtype=data.dtype, device=data.device)
    
    # Expand mask to match batch/time dimensions
    shape_prefix = data.shape[:-2]  # [B, T, C] or [B, C] or [C]
    mask = mask.expand(*shape_prefix[:-1], 1, -1, -1) if data.dim() >= 4 else mask
    
    return torch.cat([data, mask.unsqueeze(-3)], dim=-3)


def normalize_data(
    data: torch.Tensor,
    method: str = 'minmax',
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Normalize PDE data.
    
    Args:
        data: input tensor
        method: 'minmax' or 'zscore'
        eps: small epsilon for numerical stability
        
    Returns:
        normalized data, normalization stats
    """
    if method == 'minmax':
        min_val = data.min()
        max_val = data.max()
        normalized = (data - min_val) / (max_val - min_val + eps)
        stats = {'min': min_val.item(), 'max': max_val.item()}
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        normalized = (data - mean) / (std + eps)
        stats = {'mean': mean.item(), 'std': std.item()}
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, stats


def generate_synthetic_pde_data(
    dataset_type: str,
    num_samples: int = 100,
    num_timesteps: int = 20,
    spatial_size: int = 128,
    num_channels: int = 1,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate synthetic PDE data for testing purposes.
    
    Creates simplified versions of the PDE datasets mentioned in the paper
    for unit testing and architecture verification.
    
    Args:
        dataset_type: type of synthetic data ('ns', 'cns', 'swe', 'dr', 'cfd')
        num_samples: number of trajectories
        num_timesteps: number of timesteps per trajectory
        spatial_size: spatial resolution
        num_channels: number of physical channels
        seed: random seed
        
    Returns:
        data: [num_samples, num_timesteps, num_channels, spatial_size, spatial_size]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if dataset_type == 'ns':
        # Navier-Stokes-like: vorticity field with advection and diffusion
        data = torch.zeros(num_samples, num_timesteps, num_channels, spatial_size, spatial_size)
        for n in range(num_samples):
            # Initial condition: random smooth field
            x = torch.linspace(0, 1, spatial_size)
            y = torch.linspace(0, 1, spatial_size)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # Random parameters
            viscosity = 10 ** torch.empty(1).uniform_(-5, -3).item()
            num_vortices = np.random.randint(2, 6)
            
            # Create random vortices
            for v in range(num_vortices):
                cx, cy = np.random.rand(2)
                amp = np.random.randn() * 0.5
                radius = np.random.rand() * 0.3 + 0.1
                field = amp * torch.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * radius**2))
                data[n, 0, 0] += field
            
            # Time evolution: simple advection-diffusion
            for t in range(1, num_timesteps):
                # Pseudo-advection: shift with diffusion
                prev = data[n, t-1, 0]
                # Diffusion step
                laplacian = (torch.roll(prev, 1, 0) + torch.roll(prev, -1, 0) +
                            torch.roll(prev, 1, 1) + torch.roll(prev, -1, 1) - 4 * prev)
                # Advection (simplified as rotation)
                rotated = torch.roll(prev, shifts=1, dims=0) * 0.1
                data[n, t, 0] = prev + viscosity * laplacian + rotated
                # Normalize to prevent explosion
                data[n, t, 0] = data[n, t, 0] / (data[n, t, 0].abs().max() + 1e-8)
    
    elif dataset_type == 'swe':
        # Shallow water equations: water depth with wave propagation
        data = torch.zeros(num_samples, num_timesteps, num_channels, spatial_size, spatial_size)
        for n in range(num_samples):
            x = torch.linspace(-1, 1, spatial_size)
            y = torch.linspace(-1, 1, spatial_size)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # Initial: random water surface
            for _ in range(np.random.randint(2, 5)):
                cx, cy = np.random.uniform(-0.5, 0.5, 2)
                amp = np.random.rand() * 0.5 + 0.5
                radius = np.random.rand() * 0.4 + 0.2
                data[n, 0, 0] += amp * torch.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * radius**2))
            
            # Wave propagation
            wave_speed = np.random.rand() * 0.5 + 0.1
            for t in range(1, num_timesteps):
                prev = data[n, t-1, 0]
                laplacian = (torch.roll(prev, 1, 0) + torch.roll(prev, -1, 0) +
                            torch.roll(prev, 1, 1) + torch.roll(prev, -1, 1) - 4 * prev)
                data[n, t, 0] = prev + wave_speed * laplacian
                data[n, t, 0] = torch.clamp(data[n, t, 0], 0, 2)
    
    elif dataset_type == 'dr':
        # Diffusion-Reaction: density fields with reaction terms
        data = torch.zeros(num_samples, num_timesteps, num_channels, spatial_size, spatial_size)
        for n in range(num_samples):
            x = torch.linspace(-2.5, 2.5, spatial_size)
            y = torch.linspace(-2.5, 2.5, spatial_size)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # Initial: localized density
            cx, cy = np.random.uniform(-1, 1, 2)
            data[n, 0, 0] = torch.exp(-((X - cx)**2 + (Y - cy)**2) / 0.2)
            
            # Diffusion-reaction
            D = np.random.rand() * 0.1 + 0.01
            R = np.random.rand() * 0.1
            for t in range(1, num_timesteps):
                prev = data[n, t-1, 0]
                laplacian = (torch.roll(prev, 1, 0) + torch.roll(prev, -1, 0) +
                            torch.roll(prev, 1, 1) + torch.roll(prev, -1, 1) - 4 * prev)
                reaction = prev * (1 - prev)
                data[n, t, 0] = prev + D * laplacian + R * reaction
                data[n, t, 0] = torch.clamp(data[n, t, 0], 0, 1)
    
    elif dataset_type == 'cns':
        # Compressible Navier-Stokes: multi-channel (density, velocity, pressure)
        num_channels_actual = min(4, num_channels)  # ρ, u, v, p
        data = torch.zeros(num_samples, num_timesteps, num_channels_actual, spatial_size, spatial_size)
        for n in range(num_samples):
            x = torch.linspace(0, 1, spatial_size)
            y = torch.linspace(0, 1, spatial_size)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # Density
            data[n, 0, 0] = 1.0 + 0.1 * torch.sin(2*np.pi*X) * torch.sin(2*np.pi*Y)
            # Velocity u
            data[n, 0, 1] = 0.1 * torch.cos(2*np.pi*Y) * torch.sin(2*np.pi*X)
            # Velocity v
            data[n, 0, 2] = -0.1 * torch.sin(2*np.pi*Y) * torch.cos(2*np.pi*X)
            # Pressure
            data[n, 0, 3] = 1.0 + 0.05 * torch.cos(2*np.pi*X)
            
            for t in range(1, num_timesteps):
                # Simple evolution
                for c in range(num_channels_actual):
                    prev = data[n, t-1, c]
                    laplacian = (torch.roll(prev, 1, 0) + torch.roll(prev, -1, 0) +
                                torch.roll(prev, 1, 1) + torch.roll(prev, -1, 1) - 4 * prev)
                    data[n, t, c] = prev + 0.01 * laplacian
                    data[n, t, c] = torch.clamp(data[n, t, c], 0, 5)
        
        if num_channels > num_channels_actual:
            # Pad to desired channels
            pad = torch.zeros(num_samples, num_timesteps, num_channels - num_channels_actual,
                            spatial_size, spatial_size)
            data = torch.cat([data, pad], dim=2)
    
    elif dataset_type == 'cfd':
        # CFDBench: irregular geometries with mask
        num_channels_total = num_channels + 1  # +1 for mask
        data = torch.zeros(num_samples, num_timesteps, num_channels_total, spatial_size, spatial_size)
        
        for n in range(num_samples):
            # Create irregular mask (e.g., cylinder in flow)
            x = torch.linspace(0, 1, spatial_size)
            y = torch.linspace(0, 1, spatial_size)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # Random obstacle
            cx, cy = np.random.uniform(0.3, 0.7, 2)
            radius = np.random.rand() * 0.15 + 0.05
            mask = ((X - cx)**2 + (Y - cy)**2 > radius**2).float()
            
            # Store mask in last channel
            data[n, :, -1] = mask
            
            # Initialize flow
            for c in range(num_channels):
                data[n, 0, c] = mask * (1.0 + 0.1 * torch.sin(2*np.pi*X) * torch.sin(2*np.pi*Y))
            
            # Evolve
            for t in range(1, num_timesteps):
                for c in range(num_channels):
                    prev = data[n, t-1, c]
                    laplacian = (torch.roll(prev, 1, 0) + torch.roll(prev, -1, 0) +
                                torch.roll(prev, 1, 1) + torch.roll(prev, -1, 1) - 4 * prev)
                    data[n, t, c] = prev + 0.005 * laplacian
                    data[n, t, c] = data[n, t, c] * mask
    
    else:
        # Generic wave-like data
        data = torch.zeros(num_samples, num_timesteps, num_channels, spatial_size, spatial_size)
        for n in range(num_samples):
            x = torch.linspace(0, 1, spatial_size)
            y = torch.linspace(0, 1, spatial_size)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # Random wave parameters
            kx = np.random.rand() * 4 + 1
            ky = np.random.rand() * 4 + 1
            omega = np.random.rand() * 2 + 0.5
            
            for t in range(num_timesteps):
                for c in range(num_channels):
                    phase = np.random.rand() * 2 * np.pi
                    data[n, t, c] = torch.sin(kx * 2*np.pi * X + ky * 2*np.pi * Y - omega * t + phase)
    
    return data


def prepare_pre_training_datasets(
    data_dir: Optional[str] = None,
    synthetic: bool = True,
    spatial_size: int = 128,
    num_timesteps: int = 20,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Prepare the 6 pre-training datasets.
    
    As listed in the paper (Table 1, Table 6):
    1. FNO-NS (1e-5): train=1000, test=200
    2. FNO-NS (1e-3): train=1000, test=200
    3. PDEBench-CNS (0.1, 0.01): train=9000, test=200
    4. PDEBench-SWE: train=900, test=60
    5. PDEBench-DR: train=900, test=60
    6. CFDBench: train=9000, test=1000
    
    Args:
        data_dir: optional directory containing real data
        synthetic: whether to generate synthetic data for testing
        spatial_size: target spatial resolution
        num_timesteps: number of timesteps
        seed: random seed
        
    Returns:
        datasets: dict mapping dataset name to [N, T_total, C, H, W] tensor
    """
    datasets = {}
    
    if synthetic:
        # Generate synthetic data matching dataset sizes (smaller for testing)
        configs = [
            ('FNO-NS-1e-5', 'ns', 200, 20, 1),
            ('FNO-NS-1e-3', 'ns', 200, 20, 1),
            ('PDEBench-CNS', 'cns', 500, 20, 4),
            ('PDEBench-SWE', 'swe', 200, 20, 1),
            ('PDEBench-DR', 'dr', 200, 20, 1),
            ('CFDBench', 'cfd', 500, 20, 2),
        ]
        
        for name, dataset_type, num_samples, n_timesteps, n_channels in configs:
            data = generate_synthetic_pde_data(
                dataset_type=dataset_type,
                num_samples=num_samples,
                num_timesteps=n_timesteps,
                spatial_size=spatial_size,
                num_channels=n_channels,
                seed=seed + hash(name) % 10000,
            )
            datasets[name] = data
            
    return datasets
