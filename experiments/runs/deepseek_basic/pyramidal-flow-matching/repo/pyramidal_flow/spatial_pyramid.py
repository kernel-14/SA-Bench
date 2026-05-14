"""
Spatial Pyramid Implementation for Pyramidal Flow Matching.

Implements the piecewise flow across multiple spatial resolutions, coupled noise
sampling, and the renoising scheme at jump points between pyramid stages.

Key equations reference (from paper):
  - Eq. (7):  Piecewise flow within stage k: x_t = t' * Down(x_{e_k}, 2^k) + (1-t') * Up(Down(x_{s_k}, 2^{k+1}))
  - Eq. (9):  Endpoint:  x_{e_k} = e_k * Down(x_1, 2^k) + (1-e_k) * n
  - Eq. (10): Start point: x_{s_k} = s_k * Up(Down(x_1, 2^{k+1})) + (1-s_k) * n
  - Eq. (15): Renoising at jump: x_{s_k} = (1+s_k)/2 * Up(x_{e_{k+1}}) + sqrt(3)*(1-s_k)/2 * n'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math


def nearest_downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Downsample by factor using nearest-neighbor (spatial dimensions only)."""
    if factor == 1:
        return x
    if x.dim() == 4:
        return F.interpolate(x, scale_factor=1.0 / factor, mode='nearest')
    elif x.dim() == 5:
        B, C, T, H, W = x.shape
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x_down = F.interpolate(x_reshaped, scale_factor=1.0 / factor, mode='nearest')
        _, _, H_new, W_new = x_down.shape
        return x_down.reshape(B, T, C, H_new, W_new).permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Expected 4D or 5D input, got shape {x.shape}")


def bilinear_downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Downsample by factor using bilinear interpolation."""
    if factor == 1:
        return x
    if x.dim() == 4:
        return F.interpolate(x, scale_factor=1.0 / factor, mode='bilinear', align_corners=False)
    elif x.dim() == 5:
        B, C, T, H, W = x.shape
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x_down = F.interpolate(x_reshaped, scale_factor=1.0 / factor, mode='bilinear', align_corners=False)
        _, _, H_new, W_new = x_down.shape
        return x_down.reshape(B, T, C, H_new, W_new).permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Expected 4D or 5D input, got shape {x.shape}")


def nearest_upsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Upsample by factor using nearest-neighbor."""
    if factor == 1:
        return x
    if x.dim() == 4:
        return F.interpolate(x, scale_factor=float(factor), mode='nearest')
    elif x.dim() == 5:
        B, C, T, H, W = x.shape
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x_up = F.interpolate(x_reshaped, scale_factor=float(factor), mode='nearest')
        _, _, H_new, W_new = x_up.shape
        return x_up.reshape(B, T, C, H_new, W_new).permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Expected 4D or 5D input, got shape {x.shape}")


def bilinear_upsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Upsample by factor using bilinear interpolation."""
    if factor == 1:
        return x
    if x.dim() == 4:
        return F.interpolate(x, scale_factor=float(factor), mode='bilinear', align_corners=False)
    elif x.dim() == 5:
        B, C, T, H, W = x.shape
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x_up = F.interpolate(x_reshaped, scale_factor=float(factor), mode='bilinear', align_corners=False)
        _, _, H_new, W_new = x_up.shape
        return x_up.reshape(B, T, C, H_new, W_new).permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Expected 4D or 5D input, got shape {x.shape}")


class SpatialPyramid(nn.Module):
    """
    Spatial Pyramid for Pyramidal Flow Matching.
    
    Implements the piecewise flow across K spatial resolutions, where each stage
    operates at a different resolution (halving at each step). Only the final stage
    (k=0) operates at full resolution.
    
    Args:
        num_stages (K): Number of pyramid stages (default: 3)
        stage_timesteps: Custom time windows. If None, defaults to uniform partition.
        gamma: Decorrelation parameter for corrective noise (default: -1/3)
    """
    
    def __init__(
        self,
        num_stages: int = 3,
        stage_timesteps: Optional[List[Tuple[float, float]]] = None,
        gamma: float = -1.0 / 3.0,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.gamma = gamma
        
        if stage_timesteps is not None:
            assert len(stage_timesteps) == num_stages
            self.stage_timesteps = stage_timesteps
        else:
            # Uniform partition of [0, 1] into K stages
            # Stage 0 (finest, full res): [0, 1/K]
            # Stage 1: [1/K, 2/K]
            # Stage K-1 (coarsest): [(K-1)/K, 1]
            self.stage_timesteps = []
            for k in range(num_stages):
                s_k = float(k) / num_stages
                e_k = float(k + 1) / num_stages
                self.stage_timesteps.append((s_k, e_k))
        
        # Precompute renoising coefficients for jump points
        # Eq. (15): x_{s_k} = (1+s_k)/2 * Up(x_{e_{k+1}}) + sqrt(3)*(1-s_k)/2 * n'
        self.renoising_coeffs = []
        for k in range(num_stages - 1):
            # When jumping from stage k+1 (coarser) to stage k (finer)
            # s_k refers to stage k's start time
            s_k = self.stage_timesteps[k][0]
            rescale = (1.0 + s_k) / 2.0
            noise_std = math.sqrt(3.0) * (1.0 - s_k) / 2.0
            self.renoising_coeffs.append((rescale, noise_std))
    
    def get_stage_for_timestep(self, t: float) -> int:
        """Determine which pyramid stage a given timestep belongs to."""
        for k, (s_k, e_k) in enumerate(self.stage_timesteps):
            if s_k <= t <= e_k:
                return k
        if t < self.stage_timesteps[0][0]:
            return 0
        return self.num_stages - 1
    
    def get_rescaling_factor(self, stage_idx: int) -> int:
        """Get the resolution factor 2^k for stage index k (0 = full resolution)."""
        return 2 ** stage_idx
    
    def sample_endpoints(
        self,
        x1: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        stage_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample coupled endpoints for a pyramid stage (Eqs. 9 and 10).
        
        The key insight: both endpoints use the SAME noise n at the resolution
        of the ENDPOINT (higher resolution). The start point's noise is this
        same noise, so the start point and end point share the same noise
        direction.
        
        Args:
            x1: Clean data latent at full resolution
            noise: Base noise tensor (same direction used for both endpoints)
            stage_idx: Stage index (k). If None, samples uniformly random stage.
            
        Returns:
            (x_start, x_end, stage_idx): Start and end points for the selected stage
        """
        if stage_idx is None:
            stage_idx = torch.randint(0, self.num_stages, ()).item()
        
        s_k, e_k = self.stage_timesteps[stage_idx]
        k = stage_idx
        
        # Resolution factors
        factor_curr = 2 ** k       # resolution for end point
        factor_next = 2 ** (k + 1)  # resolution for start point (coarser)
        
        # Downsampled clean latents
        x1_down_curr = bilinear_downsample(x1, factor_curr)    # Down(x1, 2^k) - at stage resolution
        x1_up_from_next = bilinear_upsample(
            bilinear_downsample(x1, factor_next), 2
        )  # Up(Down(x1, 2^{k+1})) - upsampled to stage resolution
        
        # Generate shared noise at the ENDPOINT resolution
        if noise is None:
            noise_shape = x1_down_curr.shape
            noise = torch.randn(noise_shape, device=x1.device, dtype=x1.dtype)
        
        # End point: Eq. (9) - at resolution factor_curr
        x_end = e_k * x1_down_curr + (1.0 - e_k) * noise
        
        # Start point: Eq. (10) - same noise, at stage resolution
        # x1_up_from_next is already at the same resolution as x_end
        x_start = s_k * x1_up_from_next + (1.0 - s_k) * noise
        
        return x_start, x_end, stage_idx
    
    def get_conditional_vector_field(
        self,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the conditional vector field: u_t = x_end - x_start."""
        return x_end - x_start
    
    def interpolate(
        self,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        t: float,
        stage_idx: int,
    ) -> torch.Tensor:
        """
        Interpolate between start and end points at timestep t (Eq. 7).
        
        x_t = t' * x_end + (1-t') * x_start where t' = (t - s_k)/(e_k - s_k)
        """
        s_k, e_k = self.stage_timesteps[stage_idx]
        t_prime = (t - s_k) / max(e_k - s_k, 1e-8)
        t_prime = max(0.0, min(1.0, t_prime))
        return t_prime * x_end + (1.0 - t_prime) * x_start
    
    def renoise_jump(
        self,
        x_prev_end: torch.Tensor,
        target_stage_idx: int,
    ) -> torch.Tensor:
        """
        Apply renoising at jump point between stages (Eq. 15).
        
        Transforms the end point of stage (target+1, coarser) into 
        the start point of stage (target, finer).
        
        Args:
            x_prev_end: End point from stage target+1 (lower resolution)
            target_stage_idx: The stage we're jumping TO (finer resolution)
            
        Returns:
            x_start: Start point for target stage (higher resolution with corrective noise)
        """
        if target_stage_idx >= self.num_stages - 1:
            return x_prev_end
        
        # Upsample to target resolution
        x_up = nearest_upsample(x_prev_end, 2)
        
        # Apply renoising: Eq. (15)
        rescale, noise_std = self.renoising_coeffs[target_stage_idx]
        
        # Generate corrective noise with blockwise covariance structure
        noise = self._generate_corrective_noise(x_up.shape, x_up.device, x_up.dtype)
        
        return rescale * x_up + noise_std * noise
    
    def _generate_corrective_noise(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Generate corrective noise with blockwise covariance Σ'_block.
        
        For nearest-neighbor upsampling, the covariance has 4x4 blocks
        with diagonals = 1 and off-diagonals = gamma (=-1/3).
        We decorrelate by projecting out the 2x2 block mean.
        """
        noise = torch.randn(shape, device=device, dtype=dtype)
        
        if noise.dim() == 4:  # B, C, H, W
            B, C, H, W = shape
            noise_reshaped = noise.reshape(B, C, H // 2, 2, W // 2, 2)
            block_mean = noise_reshaped.mean(dim=(3, 5), keepdim=True)
            noise_centered = noise_reshaped - block_mean
            noise_decorrelated = noise_centered * math.sqrt(4.0 / 3.0)
            noise = noise_decorrelated.reshape(B, C, H, W)
        elif noise.dim() == 5:  # B, C, T, H, W
            B, C, T, H, W = shape
            noise_reshaped = noise.reshape(B, C, T, H // 2, 2, W // 2, 2)
            block_mean = noise_reshaped.mean(dim=(4, 6), keepdim=True)
            noise_centered = noise_reshaped - block_mean
            noise_decorrelated = noise_centered * math.sqrt(4.0 / 3.0)
            noise = noise_decorrelated.reshape(B, C, T, H, W)
        
        return noise
    
    def compute_flow_matching_loss(
        self,
        velocity_model: nn.Module,
        x1: torch.Tensor,
        stage_idx: Optional[int] = None,
        t: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute the unified flow matching loss (Eq. 11).
        
        L = E_{k, t, (x_{e_k}, x_{s_k})} || v_t(x_t) - (x_{e_k} - x_{s_k}) ||^2
        """
        B = x1.shape[0]
        device = x1.device
        dtype = x1.dtype
        
        if stage_idx is None:
            stage_idx = torch.randint(0, self.num_stages, (B,), device=device)
        elif isinstance(stage_idx, int):
            stage_idx = torch.full((B,), stage_idx, device=device, dtype=torch.long)
        
        if t is None:
            t = torch.rand(B, device=device, dtype=dtype)
        
        total_loss = 0.0
        
        for i in range(B):
            k = stage_idx[i].item()
            t_i = t[i].item()
            
            x_start, x_end, _ = self.sample_endpoints(x1[i:i+1], stage_idx=k)
            x_t = self.interpolate(x_start, x_end, t_i, k)
            u_t = self.get_conditional_vector_field(x_start, x_end)
            
            v_pred = velocity_model(x_t, t_i, stage_idx=k)
            total_loss = total_loss + F.mse_loss(v_pred, u_t)
        
        return total_loss / B
