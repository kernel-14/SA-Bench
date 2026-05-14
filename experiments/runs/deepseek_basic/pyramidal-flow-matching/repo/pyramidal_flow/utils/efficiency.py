"""
Efficiency metrics computation for Pyramidal Flow Matching.

Computes and compares the computational requirements of pyramidal flow
vs traditional full-sequence diffusion approaches.
"""

import math
from typing import Tuple, Dict


def compute_efficiency_metrics(
    video_frames: int = 241,
    spatial_resolution: Tuple[int, int] = (96, 96),
    num_spatial_stages: int = 3,
    num_temporal_levels: int = 3,
    history_frames: int = 12,
) -> Dict[str, float]:
    """
    Compute efficiency metrics comparing pyramidal flow to full-sequence diffusion.
    
    Full-sequence diffusion:
        Tokens: T * N
        Compute (attention): T^2 * N^2
    
    Pyramidal flow matching (spatial):
        Tokens: ~T * N / 4^K
        Compute: ~T^2 * N^2 / 16^K
    
    With temporal pyramid:
        History tokens reduced by 1/4^K
    
    Args:
        video_frames: T - number of latent frames
        spatial_resolution: (H, W) of latent
        num_spatial_stages: K for spatial pyramid
        num_temporal_levels: K' for temporal pyramid
        history_frames: number of history frames
        
    Returns:
        Dict with efficiency comparison metrics
    """
    H, W = spatial_resolution
    N = H * W  # tokens per frame
    T = video_frames
    K = num_spatial_stages
    
    # Full-sequence diffusion
    full_tokens = T * N
    full_compute = T * T * N * N
    
    # Spatial pyramid only
    spatial_tokens = full_tokens / (4 ** K)
    spatial_compute = full_compute / (16 ** K)
    
    # Temporal pyramid: history compression
    # Frames per level: newest 2 at full res, next 4 at half res, rest at quarter res
    history_tokens_full = history_frames * N
    history_tokens_pyramid = 0
    for l in range(num_temporal_levels):
        factor = 2 ** l
        frames_at_level = min(2 ** (l + 1), max(history_frames - sum(2 ** i for i in range(l)), 0))
        history_tokens_pyramid += frames_at_level * (N // (factor * factor))
    
    temporal_factor = history_tokens_full / max(history_tokens_pyramid, 1)
    
    # Combined savings
    combined_tokens = spatial_tokens / temporal_factor
    combined_compute = spatial_compute / temporal_factor
    
    return {
        # Tokens
        'full_sequence_tokens': full_tokens,
        'spatial_pyramid_tokens': spatial_tokens,
        'combined_pyramid_tokens': combined_tokens,
        'token_reduction_spatial': 4 ** K,
        'token_reduction_temporal': temporal_factor,
        'token_reduction_total': full_tokens / combined_tokens,
        
        # Compute
        'full_sequence_compute': full_compute,
        'spatial_pyramid_compute': spatial_compute,
        'combined_pyramid_compute': combined_compute,
        'compute_reduction_spatial': 16 ** K,
        'compute_reduction_total': full_compute / combined_compute,
        
        # Metadata
        'video_frames': T,
        'spatial_resolution': f"{H}x{W}",
        'num_spatial_stages': K,
        'num_temporal_levels': num_temporal_levels,
        'estimated_gpu_hours': 20700,  # from paper
    }


def format_efficiency_report(metrics: Dict[str, float]) -> str:
    """Format efficiency metrics as a readable report."""
    report = []
    report.append("=" * 50)
    report.append("PYRAMIDAL FLOW MATCHING EFFICIENCY REPORT")
    report.append("=" * 50)
    report.append(f"\nVideo configuration:")
    report.append(f"  Frames: {metrics['video_frames']}")
    report.append(f"  Resolution: {metrics['spatial_resolution']}")
    report.append(f"  Spatial stages: {metrics['num_spatial_stages']}")
    report.append(f"  Temporal levels: {metrics['num_temporal_levels']}")
    
    report.append(f"\nToken comparison:")
    report.append(f"  Full-sequence: {metrics['full_sequence_tokens']:,.0f}")
    report.append(f"  Spatial pyramid: {metrics['spatial_pyramid_tokens']:,.0f}")
    report.append(f"  Combined pyramid: {metrics['combined_pyramid_tokens']:,.0f}")
    report.append(f"  Total reduction: {metrics['token_reduction_total']:,.1f}x")
    
    report.append(f"\nCompute comparison:")
    report.append(f"  Full-sequence: {metrics['full_sequence_compute']:,.0f}")
    report.append(f"  Spatial pyramid: {metrics['spatial_pyramid_compute']:,.0f}")
    report.append(f"  Combined pyramid: {metrics['combined_pyramid_compute']:,.0f}")
    report.append(f"  Total reduction: {metrics['compute_reduction_total']:,.1f}x")
    
    report.append(f"\nTraining efficiency:")
    report.append(f"  GPU hours (10s, 768p, 24fps): {metrics['estimated_gpu_hours']:,}")
    report.append(f"  Comparison: Open-Sora 1.2 requires ~42.6k GPU hours for 97 frames")
    
    return "\n".join(report)
