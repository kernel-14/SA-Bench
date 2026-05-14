"""
Main WDNO (Wavelet Diffusion Neural Operator) model.

This module implements the complete WDNO framework combining:
1. Wavelet transform for domain conversion
2. Diffusion model for generation in wavelet space
3. Multi-resolution training with Base-Resolution Model (BRM) and Super-Resolution Model (SRM)

Key algorithms from the paper:
- Algorithm 1: Training and Sampling for WDNO
- Simulation: p(W_u | W_a) - conditional generation of wavelet coefficients
- Control: p(W_f | W_a) with guidance from objective I
- Super-resolution: p(W_h | W_l, W_a_h) - conditional generation at higher resolution
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Tuple

from .diffusion import GaussianDiffusion, DDIMSampler
from .unet_1d import UNet1D
from .unet_2d import UNet3D


class WDNO1D(nn.Module):
    """
    WDNO for 1D PDE experiments (Burgers, advection, compressible NS).
    
    Operates on 1D spatiotemporal data by:
    1. Applying 2D wavelet transform to (T, X) data
    2. Running diffusion in wavelet space
    3. Applying inverse wavelet transform to get predictions
    
    For simulation: conditions on W_a = wavelet(u0, f) to generate W_u
    For control: conditions on W_a = wavelet(u0, u_T) to generate W_f
    """
    
    def __init__(
        self,
        # U-Net parameters
        unet_dim=128,
        unet_dim_mults=(1, 2, 4, 8),
        unet_resnet_groups=8,
        unet_attn_dim_head=32,
        unet_attn_heads=4,
        # Diffusion parameters
        diffusion_timesteps=1000,
        beta_schedule="cosine",
        # Wavelet parameters
        wavelet="bior2.4",
        wavelet_mode="periodization",
        # Data parameters
        n_state_channels=4,   # 4 wavelet subbands for state
        n_cond_channels=4,    # 4 wavelet subbands for condition
        # DDIM parameters
        ddim_timesteps=50,
        ddim_eta=1.0,
        # Task
        task="simulation",  # "simulation" or "control"
    ):
        super().__init__()
        
        self.wavelet = wavelet
        self.wavelet_mode = wavelet_mode
        self.task = task
        self.n_state_channels = n_state_channels
        self.n_cond_channels = n_cond_channels
        self.ddim_timesteps = ddim_timesteps
        self.ddim_eta = ddim_eta
        
        # Build U-Net denoising model
        self.unet = UNet1D(
            dim=unet_dim,
            dim_mults=unet_dim_mults,
            channels=n_state_channels,
            cond_channels=n_cond_channels,
            resnet_block_groups=unet_resnet_groups,
            attn_dim_head=unet_attn_dim_head,
            attn_heads=unet_attn_heads,
        )
        
        # Build diffusion model
        self.diffusion = GaussianDiffusion(
            model=self.unet,
            timesteps=diffusion_timesteps,
            beta_schedule=beta_schedule,
        )
        
        # DDIM sampler
        self.sampler = DDIMSampler(
            diffusion_model=self.diffusion,
            ddim_timesteps=ddim_timesteps,
            ddim_eta=ddim_eta,
        )
    
    def get_wavelet_transforms(self, device):
        """Get wavelet transform objects."""
        try:
            import pytorch_wavelets as pw
            dwt = pw.DWTForward(wave=self.wavelet, J=1, mode=self.wavelet_mode).to(device)
            idwt = pw.DWTInverse(wave=self.wavelet, mode=self.wavelet_mode).to(device)
            return dwt, idwt
        except ImportError:
            raise ImportError("pytorch_wavelets is required")
    
    def apply_wavelet(self, x, dwt):
        """
        Apply 2D wavelet transform to 1D PDE data.
        
        Args:
            x: Input (B, T, X) or (B, C, T, X)
            dwt: DWT transform object
        
        Returns:
            Packed wavelet coefficients (B, 4*C, T//2, X//2)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, T, X)
        
        yl, yh = dwt(x)
        B, C, H, W = yl.shape
        yh_reshaped = yh[0].reshape(B, C * 3, H, W)
        return torch.cat([yl, yh_reshaped], dim=1)
    
    def apply_inverse_wavelet(self, coeffs, idwt, n_channels=1):
        """
        Apply inverse 2D wavelet transform.
        
        Args:
            coeffs: Packed wavelet coefficients (B, 4*C, H, W)
            idwt: IDWT transform object
            n_channels: Number of original channels
        
        Returns:
            Reconstructed data (B, C, T, X)
        """
        B, _, H, W = coeffs.shape
        yl = coeffs[:, :n_channels]
        yh_flat = coeffs[:, n_channels:]
        yh = [yh_flat.reshape(B, n_channels, 3, H, W)]
        return idwt((yl, yh))
    
    def forward(self, x_target, cond, cond_1d=None):
        """
        Training forward pass.
        
        Args:
            x_target: Target wavelet coefficients (B, 4, H, W)
            cond: Conditioning wavelet coefficients (B, 4, H, W)
            cond_1d: Optional 1D conditioning (initial condition or target state)
                     Shape: (B, 4, H, W) after wavelet transform and repeat
        
        Returns:
            Diffusion loss
        """
        if cond_1d is not None:
            full_cond = torch.cat([cond, cond_1d], dim=1)
        else:
            full_cond = cond
        
        return self.diffusion(x_target, cond=full_cond)
    
    @torch.no_grad()
    def simulate(self, cond, cond_1d=None, shape=None):
        """
        Simulation inference: generate state trajectory from conditions.
        
        Args:
            cond: Conditioning wavelet coefficients (B, 4, H, W)
            cond_1d: Optional 1D conditioning
            shape: Output shape (if None, inferred from cond)
        
        Returns:
            Generated wavelet coefficients (B, 4, H, W)
        """
        if shape is None:
            B = cond.shape[0]
            shape = (B, self.n_state_channels, cond.shape[-2], cond.shape[-1])
        
        if cond_1d is not None:
            full_cond = torch.cat([cond, cond_1d], dim=1)
        else:
            full_cond = cond
        
        return self.sampler.sample(shape, cond=full_cond)
    
    def control(self, cond, guidance_fn, guidance_scale, cond_1d=None, shape=None):
        """
        Control inference: generate control sequence with guidance.
        
        Implements the control update from the paper:
        W_f^{(k-1)} = W_f^{(k)} - eta * (eps_theta(W_f^{(k)}, W_a, k) 
                       + lambda * grad_W_f I(W_hat_f^{(k)})) + xi
        
        Args:
            cond: Conditioning wavelet coefficients (B, 4, H, W)
            guidance_fn: Function computing control objective I
            guidance_scale: Lambda in the paper
            cond_1d: Optional 1D conditioning
            shape: Output shape
        
        Returns:
            Generated control wavelet coefficients (B, 4, H, W)
        """
        if shape is None:
            B = cond.shape[0]
            shape = (B, self.n_state_channels, cond.shape[-2], cond.shape[-1])
        
        if cond_1d is not None:
            full_cond = torch.cat([cond, cond_1d], dim=1)
        else:
            full_cond = cond
        
        return self.sampler.sample_with_guidance(
            shape=shape,
            cond=full_cond,
            guidance_fn=guidance_fn,
            guidance_scale=guidance_scale,
        )


class WDNO2D(nn.Module):
    """
    WDNO for 2D PDE experiments (incompressible fluid, ERA5).
    
    Operates on 2D spatiotemporal data by:
    1. Applying 3D wavelet transform to (T, H, W) data
    2. Running diffusion in wavelet space
    3. Applying inverse wavelet transform to get predictions
    """
    
    def __init__(
        self,
        # U-Net parameters
        unet_dim=64,
        unet_dim_mults=(1, 2, 4, 8),
        unet_resnet_groups=8,
        unet_attn_dim_head=32,
        unet_attn_heads=4,
        # Diffusion parameters
        diffusion_timesteps=1000,
        beta_schedule="cosine",
        # Wavelet parameters
        wavelet="bior1.3",
        wavelet_mode="zero",
        # Data parameters
        n_state_channels=8,   # 8 wavelet subbands for state (1 low + 7 high)
        n_cond_channels=8,    # 8 wavelet subbands for condition
        # DDIM parameters
        ddim_timesteps=100,
        ddim_eta=1.0,
        # Task
        task="simulation",
    ):
        super().__init__()
        
        self.wavelet = wavelet
        self.wavelet_mode = wavelet_mode
        self.task = task
        self.n_state_channels = n_state_channels
        self.n_cond_channels = n_cond_channels
        self.ddim_timesteps = ddim_timesteps
        self.ddim_eta = ddim_eta
        
        # Build 3D U-Net denoising model
        self.unet = UNet3D(
            dim=unet_dim,
            dim_mults=unet_dim_mults,
            channels=n_state_channels,
            cond_channels=n_cond_channels,
            resnet_block_groups=unet_resnet_groups,
            attn_dim_head=unet_attn_dim_head,
            attn_heads=unet_attn_heads,
        )
        
        # Build diffusion model
        self.diffusion = GaussianDiffusion(
            model=self.unet,
            timesteps=diffusion_timesteps,
            beta_schedule=beta_schedule,
        )
        
        # DDIM sampler
        self.sampler = DDIMSampler(
            diffusion_model=self.diffusion,
            ddim_timesteps=ddim_timesteps,
            ddim_eta=ddim_eta,
        )
    
    def forward(self, x_target, cond):
        """Training forward pass."""
        return self.diffusion(x_target, cond=cond)
    
    @torch.no_grad()
    def simulate(self, cond, shape=None):
        """Simulation inference."""
        if shape is None:
            B = cond.shape[0]
            shape = (B, self.n_state_channels, *cond.shape[-3:])
        
        return self.sampler.sample(shape, cond=cond)
    
    def control(self, cond, guidance_fn, guidance_scale, shape=None):
        """Control inference with guidance."""
        if shape is None:
            B = cond.shape[0]
            shape = (B, self.n_state_channels, *cond.shape[-3:])
        
        return self.sampler.sample_with_guidance(
            shape=shape,
            cond=cond,
            guidance_fn=guidance_fn,
            guidance_scale=guidance_scale,
        )


class WDNOSuperResolution(nn.Module):
    """
    Super-Resolution Model (SRM) for WDNO.
    
    Implements the multi-resolution training framework from the paper.
    Learns p(W_h | W_l, W_a_h) to generate high-resolution wavelet coefficients
    from low-resolution ones.
    
    Training data: pairs of (high-res, low-res) obtained by downsampling.
    Inference: iteratively apply to achieve zero-shot super-resolution.
    """
    
    def __init__(
        self,
        base_model,  # Base-Resolution Model (BRM)
        sr_model,    # Super-Resolution Model (SRM)
        wavelet="bior2.4",
        wavelet_mode="periodization",
        dim="1d",    # "1d" or "2d"
    ):
        super().__init__()
        self.base_model = base_model
        self.sr_model = sr_model
        self.wavelet = wavelet
        self.wavelet_mode = wavelet_mode
        self.dim = dim
    
    @torch.no_grad()
    def super_resolve(self, cond_high_res, n_levels=1):
        """
        Perform zero-shot super-resolution.
        
        Algorithm:
        1. Downsample high-res condition to base resolution
        2. Generate base-resolution wavelet coefficients using BRM
        3. Iteratively apply SRM to generate higher-resolution coefficients
        
        Args:
            cond_high_res: High-resolution condition (e.g., initial condition)
            n_levels: Number of super-resolution levels
        
        Returns:
            High-resolution wavelet coefficients
        """
        # Step 1: Downsample to base resolution
        cond_base = cond_high_res
        for _ in range(n_levels):
            cond_base = F.avg_pool2d(cond_base, 2) if self.dim == "1d" else F.avg_pool3d(cond_base, 2)
        
        # Step 2: Generate base-resolution result
        current_result = self.base_model.simulate(cond_base)
        
        # Step 3: Iteratively apply SRM
        for level in range(n_levels):
            # Upsample current result to match next resolution
            if self.dim == "1d":
                low_res_upsampled = F.interpolate(current_result, scale_factor=2, mode="nearest")
            else:
                low_res_upsampled = F.interpolate(current_result, scale_factor=2, mode="nearest")
            
            # Get condition at this resolution level
            downsample_factor = n_levels - level - 1
            if downsample_factor > 0:
                cond_at_level = cond_high_res
                for _ in range(downsample_factor):
                    if self.dim == "1d":
                        cond_at_level = F.avg_pool2d(cond_at_level, 2)
                    else:
                        cond_at_level = F.avg_pool3d(cond_at_level, 2)
            else:
                cond_at_level = cond_high_res
            
            # Concatenate low-res result with high-res condition
            sr_cond = torch.cat([low_res_upsampled, cond_at_level], dim=1)
            
            # Generate high-resolution result
            current_result = self.sr_model.simulate(sr_cond)
        
        return current_result


def build_wdno_1d(config):
    """
    Build WDNO model for 1D experiments.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        WDNO1D model
    """
    return WDNO1D(
        unet_dim=config.get("unet_dim", 128),
        unet_dim_mults=tuple(config.get("unet_dim_mults", [1, 2, 4, 8])),
        unet_resnet_groups=config.get("unet_resnet_groups", 8),
        unet_attn_dim_head=config.get("unet_attn_dim_head", 32),
        unet_attn_heads=config.get("unet_attn_heads", 4),
        diffusion_timesteps=config.get("diffusion_timesteps", 1000),
        beta_schedule=config.get("beta_schedule", "cosine"),
        wavelet=config.get("wavelet", "bior2.4"),
        wavelet_mode=config.get("wavelet_mode", "periodization"),
        n_state_channels=config.get("n_state_channels", 4),
        n_cond_channels=config.get("n_cond_channels", 4),
        ddim_timesteps=config.get("ddim_timesteps", 50),
        ddim_eta=config.get("ddim_eta", 1.0),
        task=config.get("task", "simulation"),
    )


def build_wdno_2d(config):
    """
    Build WDNO model for 2D experiments.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        WDNO2D model
    """
    return WDNO2D(
        unet_dim=config.get("unet_dim", 64),
        unet_dim_mults=tuple(config.get("unet_dim_mults", [1, 2, 4, 8])),
        unet_resnet_groups=config.get("unet_resnet_groups", 8),
        unet_attn_dim_head=config.get("unet_attn_dim_head", 32),
        unet_attn_heads=config.get("unet_attn_heads", 4),
        diffusion_timesteps=config.get("diffusion_timesteps", 1000),
        beta_schedule=config.get("beta_schedule", "cosine"),
        wavelet=config.get("wavelet", "bior1.3"),
        wavelet_mode=config.get("wavelet_mode", "zero"),
        n_state_channels=config.get("n_state_channels", 8),
        n_cond_channels=config.get("n_cond_channels", 8),
        ddim_timesteps=config.get("ddim_timesteps", 100),
        ddim_eta=config.get("ddim_eta", 1.0),
        task=config.get("task", "simulation"),
    )
