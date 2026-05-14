"""
Visualization utilities for Pyramidal Flow Matching.

Tools for visualizing pyramid stages, generated videos, and
comparisons between different configurations.
"""

import torch
from typing import List, Tuple, Optional


def visualize_pyramid_stages(
    stages: List[torch.Tensor],
    stage_names: Optional[List[str]] = None,
) -> None:
    """
    Visualize the outputs at different pyramid stages.
    
    Shows the progressive refinement from low-resolution noisy
    to high-resolution clean output.
    
    Args:
        stages: List of tensors from each pyramid stage
        stage_names: Optional names for each stage
    """
    if stage_names is None:
        stage_names = [f"Stage {i}" for i in range(len(stages))]
    
    print("Pyramid Stage Visualization:")
    print("-" * 40)
    for name, stage_output in zip(stage_names, stages):
        print(f"{name}: shape={stage_output.shape}, "
              f"min={stage_output.min().item():.4f}, "
              f"max={stage_output.max().item():.4f}, "
              f"mean={stage_output.mean().item():.4f}")


def create_sample_grid(
    samples: List[torch.Tensor],
    num_cols: int = 4,
) -> torch.Tensor:
    """
    Create a grid of sample images/frames for visualization.
    
    Args:
        samples: List of image tensors (C, H, W)
        num_cols: Number of columns in the grid
        
    Returns:
        Grid tensor (C, grid_H, grid_W)
    """
    if not samples:
        return torch.zeros(3, 256, 256)
    
    num_samples = len(samples)
    num_rows = (num_samples + num_cols - 1) // num_cols
    
    # Get sample dimensions
    C, H, W = samples[0].shape
    
    # Create grid
    grid = torch.zeros(C, num_rows * H, num_cols * W)
    
    for idx, sample in enumerate(samples):
        row = idx // num_cols
        col = idx % num_cols
        if row < num_rows and col < num_cols:
            y_start = row * H
            x_start = col * W
            grid[:, y_start:y_start + H, x_start:x_start + W] = sample
    
    return grid


def extract_keyframes(
    video: torch.Tensor,
    num_keyframes: int = 8,
) -> List[torch.Tensor]:
    """
    Extract uniformly-spaced keyframes from a video.
    
    Args:
        video: Video tensor (T, C, H, W)
        num_keyframes: Number of keyframes to extract
        
    Returns:
        List of keyframe tensors
    """
    T = video.shape[0]
    indices = torch.linspace(0, T - 1, num_keyframes, dtype=torch.long)
    return [video[i] for i in indices]


def normalize_for_display(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize latent tensor to [0, 1] range for display.
    
    Args:
        x: Tensor of any shape
        
    Returns:
        Normalized tensor
    """
    x_min = x.min()
    x_max = x.max()
    if x_max > x_min:
        return (x - x_min) / (x_max - x_min)
    return x
