"""Inference scheduler for pyramidal flow matching.

Implements Algorithm 1 from the paper:
1. Initialize from pure noise
2. For each pyramid stage (from lowest to highest resolution):
   a. Run ODE solver within the stage
   b. Apply renoising at jump points to transition to next stage
3. Return final full-resolution sample

The scheduler handles:
- Euler ODE solver within each stage
- Renoising at jump points (Eq. 15)
- Classifier-free guidance
- Autoregressive generation with temporal pyramid history
"""

import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from pyramid_flow.spatial_pyramid import (
    SpatialPyramidFlow,
    downsample_latent,
    upsample_latent,
)
from pyramid_flow.temporal_pyramid import TemporalPyramid


class PyramidFlowScheduler:
    """Scheduler for pyramidal flow matching inference.

    Implements the inference algorithm with:
    - Euler ODE integration within each pyramid stage
    - Corrective renoising at stage boundaries
    - Support for autoregressive video generation
    """

    def __init__(
        self,
        num_stages: int = 3,
        num_inference_steps: int = 20,
        stage_range: Optional[List[Tuple[float, float]]] = None,
        upsample_mode: str = "nearest",
        downsample_mode: str = "bilinear",
    ):
        self.num_stages = num_stages
        self.num_inference_steps = num_inference_steps
        self.upsample_mode = upsample_mode
        self.downsample_mode = downsample_mode

        if stage_range is None:
            # Default boundaries satisfying e_k = 2*s_{k+1}/(1+s_{k+1})
            stage_range = [(0.0, 0.5), (1/3, 0.8), (2/3, 1.0)]
        self.stage_range = stage_range

        # Steps per stage (uniform distribution)
        self.steps_per_stage = num_inference_steps // num_stages

        self.spatial_pyramid = SpatialPyramidFlow(
            num_stages=num_stages,
            stage_range=stage_range,
            upsample_mode=upsample_mode,
            downsample_mode=downsample_mode,
        )
        self.temporal_pyramid = TemporalPyramid(
            num_stages=num_stages,
            downsample_mode=downsample_mode,
            upsample_mode=upsample_mode,
        )

    def get_timesteps_for_stage(self, stage: int) -> torch.Tensor:
        """Get the ODE timesteps for a given pyramid stage.

        Returns timesteps from e_k down to s_k (denoising direction: high t -> low t).
        Note: in flow matching, t=1 is data and t=0 is noise, so we integrate
        from s_k (noisy start) to e_k (cleaner end) during generation.
        """
        s_k, e_k = self.stage_range[stage]
        steps = self.steps_per_stage
        # Timesteps from s_k to e_k (generation direction: noise -> data)
        timesteps = torch.linspace(s_k, e_k, steps + 1)
        return timesteps

    @torch.no_grad()
    def sample_image(
        self,
        model_fn: Callable,
        shape: Tuple[int, ...],
        t5_embeds: torch.Tensor,
        clip_pooled: torch.Tensor,
        null_t5_embeds: torch.Tensor,
        null_clip_pooled: torch.Tensor,
        cfg_scale: float = 7.5,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Generate a single image using pyramidal flow matching.

        Args:
            model_fn: callable that takes (x, t, t5, clip, num_frames, **kwargs) -> velocity
            shape: (B, C, H, W) output shape
            t5_embeds: (B, L, 4096) T5 text embeddings
            clip_pooled: (B, 2048) CLIP pooled embeddings
            null_t5_embeds: unconditional T5 embeddings
            null_clip_pooled: unconditional CLIP embeddings
            cfg_scale: classifier-free guidance scale
            device: target device
            dtype: computation dtype

        Returns:
            x1: (B, C, H, W) generated image latent
        """
        B, C, H, W = shape
        if device is None:
            device = t5_embeds.device

        # Initialize from pure noise at lowest resolution
        factor_0 = self.spatial_pyramid.get_resolution_factor(0)
        x = torch.randn(B, C, H // factor_0, W // factor_0, device=device, dtype=dtype)

        # Run through pyramid stages (from lowest to highest resolution)
        for stage in range(self.num_stages):
            s_k, e_k = self.stage_range[stage]
            factor_k = self.spatial_pyramid.get_resolution_factor(stage)
            timesteps = self.get_timesteps_for_stage(stage)

            # ODE integration within this stage (Euler method, s_k -> e_k)
            for i in range(len(timesteps) - 1):
                t_curr = timesteps[i]
                t_next = timesteps[i + 1]
                dt = t_next - t_curr  # positive (generation direction)

                t_batch = torch.full((B,), t_curr.item(), device=device, dtype=dtype)

                # Predict velocity with CFG
                velocity = self._predict_velocity_cfg(
                    model_fn, x, t_batch, t5_embeds, clip_pooled,
                    null_t5_embeds, null_clip_pooled, cfg_scale,
                    num_frames=1,
                )

                # Euler step: x_{t+dt} = x_t + dt * v_t
                x = x + dt * velocity

            # Apply renoising at jump point (except after last stage)
            if stage < self.num_stages - 1:
                s_next = self.stage_range[stage + 1][0]
                x = self.spatial_pyramid.renoise_at_jump_point(x, s_next)

        return x

    @torch.no_grad()
    def sample_video(
        self,
        model_fn: Callable,
        shape: Tuple[int, ...],
        t5_embeds: torch.Tensor,
        clip_pooled: torch.Tensor,
        null_t5_embeds: torch.Tensor,
        null_clip_pooled: torch.Tensor,
        cfg_scale: float = 7.5,
        num_frames: int = 121,
        frames_per_chunk: int = 8,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
        first_frame: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate a video autoregressively using pyramidal flow matching.

        Generates video in chunks, using the temporal pyramid to compress
        history frames as conditions for subsequent chunks.

        Args:
            model_fn: velocity prediction function
            shape: (B, C, H, W) per-frame latent shape
            t5_embeds: (B, L, 4096) T5 text embeddings
            clip_pooled: (B, 2048) CLIP pooled embeddings
            null_t5_embeds: unconditional T5 embeddings
            null_clip_pooled: unconditional CLIP embeddings
            cfg_scale: CFG scale
            num_frames: total number of frames to generate
            frames_per_chunk: frames generated per autoregressive step
            device: target device
            dtype: computation dtype
            first_frame: optional (B, C, H, W) first frame for image-to-video

        Returns:
            video: list of (B, C, H, W) frame latents
        """
        B, C, H, W = shape
        if device is None:
            device = t5_embeds.device

        generated_frames = []

        # If first frame is provided (image-to-video), use it as history
        if first_frame is not None:
            generated_frames.append(first_frame)
            start_frame = 1
        else:
            start_frame = 0

        # Generate frames autoregressively in chunks
        frame_idx = start_frame
        while frame_idx < num_frames:
            chunk_size = min(frames_per_chunk, num_frames - frame_idx)

            # Generate this chunk using pyramidal flow
            chunk_frames = self._generate_chunk(
                model_fn=model_fn,
                shape=(B, C, H, W),
                t5_embeds=t5_embeds,
                clip_pooled=clip_pooled,
                null_t5_embeds=null_t5_embeds,
                null_clip_pooled=null_clip_pooled,
                cfg_scale=cfg_scale,
                num_frames=chunk_size,
                history_frames=generated_frames,
                device=device,
                dtype=dtype,
            )

            generated_frames.extend(chunk_frames)
            frame_idx += chunk_size

        return generated_frames

    def _generate_chunk(
        self,
        model_fn: Callable,
        shape: Tuple[int, ...],
        t5_embeds: torch.Tensor,
        clip_pooled: torch.Tensor,
        null_t5_embeds: torch.Tensor,
        null_clip_pooled: torch.Tensor,
        cfg_scale: float,
        num_frames: int,
        history_frames: List[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[torch.Tensor]:
        """Generate a chunk of frames using pyramidal flow with history conditioning."""
        B, C, H, W = shape

        # Initialize from pure noise at lowest resolution
        factor_0 = self.spatial_pyramid.get_resolution_factor(0)
        h0, w0 = H // factor_0, W // factor_0
        # Shape: (B*T, C, H//factor, W//factor)
        x = torch.randn(B * num_frames, C, h0, w0, device=device, dtype=dtype)

        # Run through pyramid stages (lowest to highest resolution)
        for stage in range(self.num_stages):
            s_k, e_k = self.stage_range[stage]
            factor_k = self.spatial_pyramid.get_resolution_factor(stage)
            timesteps = self.get_timesteps_for_stage(stage)

            # Build temporal pyramid history for this stage
            history_compressed, hist_frame_indices = (
                self.temporal_pyramid.build_pyramid_history_sequence(
                    history_frames, current_stage=stage, training=False
                )
            )

            # ODE integration within this stage (s_k -> e_k)
            for i in range(len(timesteps) - 1):
                t_curr = timesteps[i]
                t_next = timesteps[i + 1]
                dt = t_next - t_curr  # positive

                t_batch = torch.full((B,), t_curr.item(), device=device, dtype=dtype)

                velocity = self._predict_velocity_cfg(
                    model_fn, x, t_batch, t5_embeds, clip_pooled,
                    null_t5_embeds, null_clip_pooled, cfg_scale,
                    num_frames=num_frames,
                    history_frames=history_compressed,
                    history_frame_indices=hist_frame_indices,
                )

                x = x + dt * velocity

            # Renoising at jump point (upsample to next stage resolution)
            if stage < self.num_stages - 1:
                s_next = self.stage_range[stage + 1][0]
                # Apply renoising to each frame independently
                h_k, w_k = H // factor_k, W // factor_k
                x_frames = x.reshape(B, num_frames, C, h_k, w_k)
                x_frames_up = []
                for f in range(num_frames):
                    x_f = x_frames[:, f]  # (B, C, H//factor_k, W//factor_k)
                    x_f_up = self.spatial_pyramid.renoise_at_jump_point(x_f, s_next)
                    x_frames_up.append(x_f_up)
                factor_next = self.spatial_pyramid.get_resolution_factor(stage + 1)
                h_next, w_next = H // factor_next, W // factor_next
                x = torch.stack(x_frames_up, dim=1).reshape(B * num_frames, C, h_next, w_next)

        # Return individual frames at full resolution
        x_frames = x.reshape(B, num_frames, C, H, W)
        return [x_frames[:, f] for f in range(num_frames)]

    def _predict_velocity_cfg(
        self,
        model_fn: Callable,
        x: torch.Tensor,
        t: torch.Tensor,
        t5_embeds: torch.Tensor,
        clip_pooled: torch.Tensor,
        null_t5_embeds: torch.Tensor,
        null_clip_pooled: torch.Tensor,
        cfg_scale: float,
        num_frames: int = 1,
        history_frames: Optional[List[torch.Tensor]] = None,
        history_frame_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Predict velocity with classifier-free guidance."""
        B = t5_embeds.shape[0]

        # Conditional prediction
        v_cond = model_fn(
            x, t, t5_embeds, clip_pooled,
            num_frames=num_frames,
            history_frames=history_frames,
            history_frame_indices=history_frame_indices,
        )

        # Unconditional prediction
        v_uncond = model_fn(
            x, t, null_t5_embeds, null_clip_pooled,
            num_frames=num_frames,
            history_frames=history_frames,
            history_frame_indices=history_frame_indices,
        )

        # CFG combination
        return v_uncond + cfg_scale * (v_cond - v_uncond)
