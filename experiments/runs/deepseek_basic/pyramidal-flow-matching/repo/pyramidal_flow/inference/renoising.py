"""
Renoising Inference for Pyramidal Flow Matching.

Implements the inference procedure with corrective renoising at jump points
between pyramid stages (Algorithm 1 in the paper).

The key insight: when transitioning between pyramid stages of different resolutions,
we add corrective Gaussian noise to maintain continuity of the probability path.
The renoising rule (Eq. 15) is:

    x_{s_k} = (1+s_k)/2 * Up(x_{e_{k+1}}) + sqrt(3)*(1-s_k)/2 * n'

where the relationship e_{k+1} = 2*s_k/(1+s_k) is maintained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, List
import math


class RenoisingInference:
    """
    Inference with corrective renoising for pyramidal flow matching.
    
    Implements Algorithm 1 from the paper. The inference proceeds backward
    through the pyramid stages, starting from pure noise at the coarsest
    resolution and progressively refining to full resolution.
    
    At each jump point between stages, corrective renoising is applied to
    maintain continuity of the probability path across resolutions.
    
    Args:
        spatial_pyramid: The SpatialPyramid instance defining stage structure
        velocity_model: The velocity prediction model v_t(x_t)
        num_sampling_steps: Number of ODE solver steps per stage
        solver: ODE solver type ('euler', 'midpoint', 'rk4')
        guidance_scale: Classifier-free guidance scale
    """
    
    def __init__(
        self,
        spatial_pyramid,
        velocity_model: nn.Module,
        num_sampling_steps: int = 50,
        solver: str = 'euler',
        guidance_scale: float = 7.0,
    ):
        self.spatial_pyramid = spatial_pyramid
        self.velocity_model = velocity_model
        self.num_sampling_steps = num_sampling_steps
        self.solver = solver
        self.guidance_scale = guidance_scale
        
        # Pre-compute renoising coefficients (Eq. 15)
        self._compute_renoising_coeffs()
    
    def _compute_renoising_coeffs(self):
        """Pre-compute renoising coefficients for all jump points."""
        self.renoising_coeffs = []
        num_stages = self.spatial_pyramid.num_stages
        
        for k in range(num_stages - 1):
            s_k = self.spatial_pyramid.stage_timesteps[k][0]
            # Eq. (15): 
            #   rescale = (1+s_k)/2
            #   noise_std = sqrt(3)*(1-s_k)/2
            rescale = (1.0 + s_k) / 2.0
            noise_std = math.sqrt(3.0) * (1.0 - s_k) / 2.0
            self.renoising_coeffs.append((rescale, noise_std))
    
    @torch.no_grad()
    def generate(
        self,
        conditioning: Optional[torch.Tensor] = None,
        uncond_conditioning: Optional[torch.Tensor] = None,
        image_shape: Optional[Tuple[int, ...]] = None,
        history_conditioning: Optional[List[torch.Tensor]] = None,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Generate a sample using pyramidal flow matching inference.
        
        Algorithm 1: Start from pure noise at coarsest resolution (stage K-1),
        solve ODE within each stage, renoise at jump points, and progress to
        full resolution at stage 0.
        
        Args:
            conditioning: Text/image conditioning embeddings (guided)
            uncond_conditioning: Unconditional embeddings for CFG
            image_shape: Shape of the full-resolution latent (B, C, H, W) or (B, C, T, H, W)
            history_conditioning: Temporal pyramid history for autoregressive gen
            device: Device
            dtype: Data type
            
        Returns:
            Generated sample at full resolution
        """
        num_stages = self.spatial_pyramid.num_stages
        B = image_shape[0]
        
        # Determine spatial dimensions
        if len(image_shape) == 4:  # Image: (B, C, H, W)
            _, C, H, W = image_shape
            is_video = False
        elif len(image_shape) == 5:  # Video: (B, C, T, H, W)
            _, C, T, H, W = image_shape
            is_video = True
        else:
            raise ValueError(f"Expected 4D or 5D shape, got {image_shape}")
        
        # Stage K-1: Start from pure noise at coarsest resolution
        k = num_stages - 1
        s_k, e_k = self.spatial_pyramid.stage_timesteps[k]
        factor = 2 ** k
        
        # Noise at coarsest resolution
        if is_video:
            current_latent = torch.randn(B, C, T, H // factor, W // factor, 
                                        device=device, dtype=dtype)
        else:
            current_latent = torch.randn(B, C, H // factor, W // factor, 
                                        device=device, dtype=dtype)
        
        # Iterate backward through stages: K-1 -> 0
        for k in range(num_stages - 1, -1, -1):
            s_k, e_k = self.spatial_pyramid.stage_timesteps[k]
            factor_curr = 2 ** k
            
            # Solve ODE within this stage [s_k, e_k]
            current_latent = self._solve_stage(
                current_latent, k, s_k, e_k,
                conditioning, uncond_conditioning,
                history_conditioning,
            )
            
            # If not the final stage, apply renoising to jump to next stage
            if k > 0:
                current_latent = self._renoising_jump(current_latent, k - 1)
        
        return current_latent
    
    def _solve_stage(
        self,
        x_start: torch.Tensor,
        stage_idx: int,
        s_k: float,
        e_k: float,
        conditioning: Optional[torch.Tensor],
        uncond_conditioning: Optional[torch.Tensor],
        history_conditioning: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Solve the ODE within a single pyramid stage from s_k to e_k.
        
        Uses the velocity model to integrate from the noisy start point
        to the cleaner end point.
        
        Args:
            x_start: Noisy latent at start of stage (at time s_k)
            stage_idx: Which stage
            s_k, e_k: Stage time boundaries
            conditioning: Conditioning for guided generation
            uncond_conditioning: Unconditional for CFG
            history_conditioning: Temporal pyramid history
            
        Returns:
            Cleaner latent at end of stage (at time e_k)
        """
        dt = (e_k - s_k) / self.num_sampling_steps
        x = x_start.clone()
        t = s_k
        
        for step in range(self.num_sampling_steps):
            # Current timestep
            t_current = t + step * dt
            
            # Predict velocity
            v = self._predict_velocity(
                x, t_current, stage_idx,
                conditioning, uncond_conditioning,
                history_conditioning,
            )
            
            # Euler step (or higher-order)
            # Reshape velocity to match x if needed
            if v.dim() == 3 and x.dim() == 4:
                B, N, C = v.shape
                H = W = int(math.sqrt(N))
                v = v.permute(0, 2, 1).reshape(B, C, H, W)
            elif v.dim() == 3 and x.dim() == 5:
                B, N, C = v.shape
                # Need to infer T, H, W from original
                v = v.permute(0, 2, 1).reshape(x.shape)
            if self.solver == 'euler':
                x = x + v * dt
            elif self.solver == 'midpoint':
                x_mid = x + v * dt / 2
                t_mid = t_current + dt / 2
                v_mid = self._predict_velocity(
                    x_mid, t_mid, stage_idx,
                    conditioning, uncond_conditioning,
                    history_conditioning,
                )
                x = x + v_mid * dt
            elif self.solver == 'rk4':
                # RK4 integration
                k1 = v
                x2 = x + k1 * dt / 2
                k2 = self._predict_velocity(
                    x2, t_current + dt/2, stage_idx,
                    conditioning, uncond_conditioning, history_conditioning
                )
                x3 = x + k2 * dt / 2
                k3 = self._predict_velocity(
                    x3, t_current + dt/2, stage_idx,
                    conditioning, uncond_conditioning, history_conditioning
                )
                x4 = x + k3 * dt
                k4 = self._predict_velocity(
                    x4, t_current + dt, stage_idx,
                    conditioning, uncond_conditioning, history_conditioning
                )
                x = x + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
            else:
                raise ValueError(f"Unknown solver: {self.solver}")
        
        return x
    
    def _predict_velocity(
        self,
        x: torch.Tensor,
        t: float,
        stage_idx: int,
        conditioning: Optional[torch.Tensor],
        uncond_conditioning: Optional[torch.Tensor],
        history_conditioning: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Predict velocity with optional classifier-free guidance.
        
        v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)
        """
        if conditioning is not None and uncond_conditioning is not None:
            # Classifier-free guidance
            v_cond = self.velocity_model(x, t, stage_idx, 
                                         conditioning=conditioning,
                                         history=history_conditioning)
            v_uncond = self.velocity_model(x, t, stage_idx,
                                           conditioning=uncond_conditioning,
                                           history=history_conditioning)
            v = v_uncond + self.guidance_scale * (v_cond - v_uncond)
        elif conditioning is not None:
            v = self.velocity_model(x, t, stage_idx,
                                    conditioning=conditioning,
                                    history=history_conditioning)
        else:
            v = self.velocity_model(x, t, stage_idx,
                                    history=history_conditioning)
        return v
    
    def _renoising_jump(
        self,
        x_prev_end: torch.Tensor,
        target_stage_idx: int,
    ) -> torch.Tensor:
        """
        Apply renoising at jump point to transition to target stage.
        
        Eq. (15): x_{s_k} = (1+s_k)/2 * Up(x_{e_{k+1}}) + sqrt(3)*(1-s_k)/2 * n'
        
        Args:
            x_prev_end: End point of previous (higher-k) stage
            target_stage_idx: The stage index we're jumping to (k, not k+1)
            
        Returns:
            Start point for target stage
        """
        rescale, noise_std = self.renoising_coeffs[target_stage_idx]
        
        # Upsample to target resolution (factor of 2)
        if x_prev_end.dim() == 4:  # Image
            x_up = F.interpolate(x_prev_end, scale_factor=2.0, mode='nearest')
        elif x_prev_end.dim() == 5:  # Video
            B, C, T, H, W = x_prev_end.shape
            x_reshaped = x_prev_end.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            x_up = F.interpolate(x_reshaped, scale_factor=2.0, mode='nearest')
            _, _, H_new, W_new = x_up.shape
            x_up = x_up.reshape(B, T, C, H_new, W_new).permute(0, 2, 1, 3, 4)
        
        # Generate corrective noise with blockwise covariance
        noise = self._generate_corrective_noise(
            x_up.shape, x_up.device, x_up.dtype
        )
        
        # Apply renoising
        x_start = rescale * x_up + noise_std * noise
        
        return x_start
    
    def _generate_corrective_noise(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        gamma: float = -1.0/3.0,
    ) -> torch.Tensor:
        """
        Generate corrective noise with blockwise covariance Σ'_block.
        
        Same as in SpatialPyramid._generate_corrective_noise.
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
