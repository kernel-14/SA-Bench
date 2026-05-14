"""
Pyramidal Flow Matching: Unified training and inference.

Core algorithm that combines spatial pyramid flow matching with temporal
pyramid autoregressive video generation. This is the main entry point
that ties together all components.

Reference: Section 3 of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import math

from .spatial_pyramid import SpatialPyramid
from .temporal_pyramid import TemporalPyramidConditioning, TemporalPyramidHistory
from .inference.renoising import RenoisingInference
from .inference.sampler import PyramidalFlowSampler


class PyramidalFlowMatching(nn.Module):
    """
    Complete Pyramidal Flow Matching framework.
    
    Combines:
    1. Spatial Pyramid: Multi-resolution piecewise flow that reduces
       computation by operating at low resolutions in early timesteps.
    2. Temporal Pyramid: Compressed history conditioning for efficient
       autoregressive video generation.
    3. Unified Flow Matching Objective: Joint training of all stages
       in a single DiT model.
    
    Args:
        velocity_model: The DiT velocity prediction model v_t(x_t, c)
        num_spatial_stages: K in the spatial pyramid (default: 3)
        num_temporal_levels: K' in the temporal pyramid
        max_history_frames: Maximum history frames for temporal conditioning
        stage_timesteps: Custom time windows (uses uniform partition if None)
        gamma: Decorrelation parameter (default: -1/3)
    """
    
    def __init__(
        self,
        velocity_model: nn.Module,
        num_spatial_stages: int = 3,
        num_temporal_levels: int = 3,
        max_history_frames: int = 12,
        stage_timesteps: Optional[List[Tuple[float, float]]] = None,
        gamma: float = -1.0 / 3.0,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.num_spatial_stages = num_spatial_stages
        
        # Initialize spatial pyramid
        self.spatial_pyramid = SpatialPyramid(
            num_stages=num_spatial_stages,
            stage_timesteps=stage_timesteps,
            gamma=gamma,
        )
        
        # Initialize temporal pyramid conditioning
        self.temporal_conditioning = TemporalPyramidConditioning(
            num_pyramid_levels=num_temporal_levels,
            max_history_frames=max_history_frames,
        )
        
        # Training statistics
        self.register_buffer('training_steps', torch.zeros(1, dtype=torch.long))
    
    def compute_loss(
        self,
        x1: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        past_frames: Optional[List[torch.Tensor]] = None,
        noise_strength_range: Tuple[float, float] = (0.0, 1.0/3.0),
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the unified pyramidal flow matching loss.
        
        This implements the combined training objective that:
        1. Randomly samples a spatial pyramid stage
        2. Constructs the flow matching target for that stage
        3. Incorporates temporal pyramid history conditioning
        
        Args:
            x1: Clean data latent (B, C, H, W) for image or (B, C, T, H, W) for video
            conditioning: Text/prompt embeddings
            past_frames: Previous frames for autoregressive training
            noise_strength_range: (min, max) for history noise
            
        Returns:
            Dict with 'loss' and optional debug info
        """
        B = x1.shape[0]
        device = x1.device
        
        # Sample stage uniformly
        stage_idx = torch.randint(0, self.num_spatial_stages, (B,), device=device)
        
        # Sample timestep uniformly
        t = torch.rand(B, device=device)
        
        # Sample history noise strength for training
        noise_min, noise_max = noise_strength_range
        history_noise = torch.rand(1, device=device).item() * (noise_max - noise_min) + noise_min
        
        total_loss = 0.0
        
        for i in range(B):
            k = stage_idx[i].item()
            s_k, e_k = self.spatial_pyramid.stage_timesteps[k]
            t_i = t[i].item()
            
            # Sample coupled endpoints for this stage
            x_start, x_end, _ = self.spatial_pyramid.sample_endpoints(
                x1[i:i+1], stage_idx=k
            )
            
            # Interpolate to get x_t
            x_t = self.spatial_pyramid.interpolate(x_start, x_end, t_i, k)
            
            # Compute target velocity
            u_t = self.spatial_pyramid.get_conditional_vector_field(x_start, x_end)
            
            # Prepare temporal history conditioning
            history = None
            if past_frames is not None and len(past_frames) > 0:
                # Get past frames for this sample
                sample_past = [f[i:i+1] for f in past_frames] if isinstance(past_frames, list) else []
                if sample_past:
                    history = self.temporal_conditioning.prepare_conditioning(
                        current_noisy=x_t,
                        past_clean_frames=sample_past,
                        training=True,
                        noise_strength=history_noise,
                    )
            
            # Predict velocity
            v_pred = self.velocity_model(
                x_t, 
                t_i, 
                stage_idx=k,
                conditioning=conditioning[i:i+1] if conditioning is not None else None,
                history=history,
            )
            
            # MSE loss: ||v_t(x_t) - u_t||^2
            # Reshape u_t to match v_pred (tokens vs image format)
            if v_pred.dim() == 3 and u_t.dim() == 4:
                B, C, H, W = u_t.shape
                u_t_flat = u_t.permute(0, 2, 3, 1).reshape(B, H*W, C)
                loss = F.mse_loss(v_pred, u_t_flat)
            elif v_pred.dim() != u_t.dim():
                loss = F.mse_loss(v_pred.reshape(u_t.shape), u_t)
            else:
                loss = F.mse_loss(v_pred, u_t)
            total_loss = total_loss + loss
        
        self.training_steps += 1
        
        return {
            'loss': total_loss / B,
            'stage_idx': stage_idx.float().mean(),
            'timestep': t.mean(),
        }
    
    @torch.no_grad()
    def sample(
        self,
        conditioning: Optional[torch.Tensor] = None,
        uncond_conditioning: Optional[torch.Tensor] = None,
        image_shape: Tuple[int, ...] = (1, 16, 96, 96),
        past_frames: Optional[List[torch.Tensor]] = None,
        num_sampling_steps: int = 50,
        solver: str = 'euler',
        guidance_scale: float = 7.0,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Generate a sample using pyramidal flow matching.
        
        Args:
            conditioning: Conditioning embeddings
            uncond_conditioning: Unconditional embeddings for CFG
            image_shape: (B, C, H, W) for image or (B, C, T, H, W) for video
            past_frames: Previous frames for autoregressive generation
            num_sampling_steps: ODE solver steps per stage
            solver: ODE solver type
            guidance_scale: CFG scale
            
        Returns:
            Generated sample at full resolution
        """
        inference = RenoisingInference(
            spatial_pyramid=self.spatial_pyramid,
            velocity_model=self.velocity_model,
            num_sampling_steps=num_sampling_steps,
            solver=solver,
            guidance_scale=guidance_scale,
        )
        
        # Prepare history conditioning
        history = None
        if past_frames is not None and len(past_frames) > 0:
            history = self.temporal_conditioning.prepare_conditioning(
                current_noisy=torch.zeros_like(past_frames[-1]),
                past_clean_frames=past_frames,
                training=False,
            )
        
        return inference.generate(
            conditioning=conditioning,
            uncond_conditioning=uncond_conditioning,
            image_shape=image_shape,
            history_conditioning=history,
            device=device,
            dtype=dtype,
        )
    
    def get_sampler(
        self,
        num_sampling_steps: int = 50,
        solver: str = 'euler',
        guidance_scale: float = 7.0,
        fps: int = 24,
    ) -> PyramidalFlowSampler:
        """
        Create a PyramidalFlowSampler for high-level generation.
        
        Returns:
            Configured sampler for text-to-video and image-to-video generation
        """
        return PyramidalFlowSampler(
            velocity_model=self.velocity_model,
            spatial_pyramid=self.spatial_pyramid,
            temporal_pyramid=self.temporal_conditioning,
            num_sampling_steps=num_sampling_steps,
            solver=solver,
            guidance_scale=guidance_scale,
            fps=fps,
        )
    
    def get_efficiency_stats(
        self,
        video_frames: int = 241,
        frame_resolution: Tuple[int, int] = (96, 96),
    ) -> Dict[str, float]:
        """
        Compute efficiency statistics compared to full-sequence diffusion.
        
        For a video with T frame latents and N tokens per frame:
        - Full-sequence: T*N tokens, T^2*N^2 compute
        - Ours: ~T*N/4^K tokens, ~T^2*N^2/16^K compute
        
        Args:
            video_frames: T (number of frames)
            frame_resolution: (H, W) of latent
            
        Returns:
            Efficiency comparison statistics
        """
        H, W = frame_resolution
        N = H * W  # tokens per frame
        T = video_frames
        K = self.num_spatial_stages
        
        full_tokens = T * N
        full_compute = T * T * N * N
        
        # Spatial pyramid: factor ~1/K for tokens, ~1/4^K for compute
        pyramid_tokens = full_tokens / (4 ** K)
        pyramid_compute = full_compute / (16 ** K)
        
        # Add temporal pyramid savings
        temporal_pyramid = TemporalPyramidHistory(
            num_pyramid_levels=self.temporal_conditioning.num_pyramid_levels,
            history_length=self.temporal_conditioning.max_history_frames,
            base_resolution=frame_resolution,
        )
        temporal_factor = temporal_pyramid.compute_efficiency_gain()
        
        return {
            'full_sequence_tokens': full_tokens,
            'pyramidal_tokens': pyramid_tokens,
            'spatial_reduction_factor': 4 ** K,
            'temporal_reduction_factor': temporal_factor,
            'full_sequence_compute': full_compute,
            'pyramidal_compute': pyramid_compute,
            'compute_reduction_factor': 16 ** K,
            'estimated_gpu_hours_10s_video': 20700,  # from paper
        }
