"""
Pyramidal Flow Sampler for inference.

Provides a high-level interface for text-to-video and image-to-video generation
using the pyramidal flow matching framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import math

from ..spatial_pyramid import SpatialPyramid, nearest_upsample, bilinear_downsample
from ..temporal_pyramid import TemporalPyramidHistory, TemporalPyramidConditioning
from .renoising import RenoisingInference


class PyramidalFlowSampler(nn.Module):
    """
    High-level sampler for pyramidal flow matching video generation.
    
    Combines spatial pyramid inference with temporal pyramid autoregressive
    generation to produce videos of arbitrary length.
    
    Args:
        velocity_model: The DiT velocity prediction model
        spatial_pyramid: Spatial pyramid configuration
        temporal_pyramid: Temporal pyramid history configuration
        num_sampling_steps: ODE solver steps per pyramid stage
        solver: ODE solver ('euler', 'midpoint', 'rk4')
        guidance_scale: CFG scale
        fps: Frames per second for generated video
    """
    
    def __init__(
        self,
        velocity_model: nn.Module,
        spatial_pyramid: SpatialPyramid,
        temporal_pyramid: TemporalPyramidConditioning,
        num_sampling_steps: int = 50,
        solver: str = 'euler',
        guidance_scale: float = 7.0,
        fps: int = 24,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.spatial_pyramid = spatial_pyramid
        self.temporal_pyramid = temporal_pyramid
        self.fps = fps
        
        self.inference_engine = RenoisingInference(
            spatial_pyramid=spatial_pyramid,
            velocity_model=velocity_model,
            num_sampling_steps=num_sampling_steps,
            solver=solver,
            guidance_scale=guidance_scale,
        )
    
    @torch.no_grad()
    def text_to_video(
        self,
        prompt_embeddings: torch.Tensor,
        num_frames: int = 121,
        latent_shape: Tuple[int, int] = (96, 96),  # H, W at full res (768/8)
        num_latent_channels: int = 16,
        uncond_embeddings: Optional[torch.Tensor] = None,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Generate a video from text prompt.
        
        Uses autoregressive generation: first frame is generated as an image,
        then subsequent frames are generated conditioned on temporal pyramid history.
        
        Args:
            prompt_embeddings: Text embeddings (B, seq_len, dim)
            num_frames: Number of frames to generate
            latent_shape: Spatial shape of VAE latent (H, W)
            num_latent_channels: VAE latent channels
            uncond_embeddings: Unconditional embeddings for CFG
            
        Returns:
            Generated video latents (B, C, T, H, W)
        """
        B = prompt_embeddings.shape[0]
        H, W = latent_shape
        
        generated_frames = []
        
        for frame_idx in range(num_frames):
            # Prepare history conditioning from previous frames
            history = self.temporal_pyramid.prepare_conditioning(
                current_noisy=torch.zeros(B, num_latent_channels, H, W, device=device),
                past_clean_frames=generated_frames,
                training=False,
            ) if generated_frames else None
            
            # Generate single frame
            frame_latent = self.inference_engine.generate(
                conditioning=prompt_embeddings,
                uncond_conditioning=uncond_embeddings,
                image_shape=(B, num_latent_channels, H, W),
                history_conditioning=history,
                device=device,
                dtype=dtype,
            )
            
            generated_frames.append(frame_latent)
        
        # Stack frames into video
        video_latent = torch.stack(generated_frames, dim=2)  # B, C, T, H, W
        
        return video_latent
    
    @torch.no_grad()
    def image_to_video(
        self,
        first_frame: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        num_frames: int = 121,
        uncond_embeddings: Optional[torch.Tensor] = None,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Generate a video from an image and text prompt.
        
        The first frame is provided as the starting image, and subsequent
        frames are generated autoregressively.
        
        Args:
            first_frame: Initial image latent (B, C, H, W)
            prompt_embeddings: Text embeddings
            num_frames: Total number of frames (including first frame)
            
        Returns:
            Video latents (B, C, T, H, W)
        """
        generated_frames = [first_frame]
        _, C, H, W = first_frame.shape
        
        for frame_idx in range(1, num_frames):
            history = self.temporal_pyramid.prepare_conditioning(
                current_noisy=torch.zeros(1, C, H, W, device=device),
                past_clean_frames=generated_frames,
                training=False,
            )
            
            frame_latent = self.inference_engine.generate(
                conditioning=prompt_embeddings,
                uncond_conditioning=uncond_embeddings,
                image_shape=(1, C, H, W),
                history_conditioning=history,
                device=device,
                dtype=dtype,
            )
            
            generated_frames.append(frame_latent)
        
        video_latent = torch.stack(generated_frames, dim=2)
        
        return video_latent
    
    def compute_token_efficiency(
        self,
        num_history_frames: int,
        full_resolution: Tuple[int, int],
        num_pyramid_levels: int = 3,
    ) -> dict:
        """
        Compute training efficiency metrics for the temporal pyramid.
        
        Following the paper's analysis: the temporal pyramid reduces tokens
        by up to 1/4^K times, and training efficiency improves by 16^K/T.
        
        Args:
            num_history_frames: Number of history frames (T)
            full_resolution: (H, W) of full-res latent
            num_pyramid_levels: K pyramid levels
            
        Returns:
            Dict with efficiency metrics
        """
        H, W = full_resolution
        full_tokens = num_history_frames * H * W
        
        # Pyramid token count
        pyramid = TemporalPyramidHistory(
            num_pyramid_levels=num_pyramid_levels,
            history_length=num_history_frames,
            base_resolution=(H, W),
        )
        pyramid_tokens = pyramid.compute_token_count()
        
        # Spatial pyramid also reduces tokens: factor ~1/K
        # Combined token reduction: roughly 1/(K * 4^K) for spatial * temporal
        
        return {
            'full_resolution_tokens': full_tokens,
            'pyramid_tokens': pyramid_tokens,
            'token_reduction_ratio': full_tokens / pyramid_tokens,
            'training_speedup_factor': (16 ** num_pyramid_levels) / num_history_frames,
            'num_pyramid_levels': num_pyramid_levels,
        }
