"""
Inference pipeline for Pyramidal Flow Matching video generation.

Implements:
1. Text-to-video generation
2. Image-to-video generation (via autoregressive conditioning)
3. Classifier-free guidance
4. Pyramidal inference with renoising at jump points

Algorithm 1 from the paper:
- Initialize x_0 ~ N(0, I)
- For k from K-1 to 0:
  - Compute endpoint x_ek from starting point x_sk using flow model
  - Compute next starting point by upsampling x_ek with renoising
- Return generated sample x_1
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import numpy as np

from ..models.pyramid_dit import PyramidDiT
from ..models.pyramidal_flow import PyramidalFlowMatching, TemporalPyramidCondition


class EulerSampler:
    """
    Euler ODE solver for flow matching inference.
    
    Integrates the velocity field from start to end of each pyramid stage.
    """
    
    def __init__(self, num_steps: int = 20):
        self.num_steps = num_steps
    
    def step(
        self,
        x: torch.Tensor,
        velocity: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Single Euler step: x_{t+dt} = x_t + dt * v_t(x_t)"""
        return x + dt * velocity
    
    def integrate(
        self,
        model_fn,
        x_start: torch.Tensor,
        t_start: float,
        t_end: float,
        **model_kwargs,
    ) -> torch.Tensor:
        """
        Integrate the ODE from t_start to t_end using Euler method.
        
        Args:
            model_fn: Function that takes (x, t, **kwargs) and returns velocity
            x_start: Starting point
            t_start: Starting timestep
            t_end: Ending timestep
            **model_kwargs: Additional arguments for model_fn
        
        Returns:
            x at t_end
        """
        x = x_start
        dt = (t_end - t_start) / self.num_steps
        
        for i in range(self.num_steps):
            t = t_start + i * dt
            t_tensor = torch.full((x.shape[0],), t, device=x.device)
            
            velocity = model_fn(x, t_tensor, **model_kwargs)
            x = self.step(x, velocity, dt)
        
        return x


class PyramidFlowPipeline:
    """
    Complete inference pipeline for Pyramidal Flow Matching.
    
    Supports:
    - Text-to-video generation
    - Image-to-video generation
    - Variable length video generation (autoregressive)
    - Classifier-free guidance
    """
    
    def __init__(
        self,
        model: PyramidDiT,
        vae: nn.Module,
        text_encoder_t5,
        text_encoder_clip,
        tokenizer_t5,
        tokenizer_clip,
        num_pyramid_stages: int = 3,
        num_inference_steps: int = 20,
        device: torch.device = None,
    ):
        """
        Args:
            model: Trained Pyramid DiT model
            vae: 3D VAE for encoding/decoding
            text_encoder_t5: T5 text encoder
            text_encoder_clip: CLIP text encoder
            tokenizer_t5: T5 tokenizer
            tokenizer_clip: CLIP tokenizer
            num_pyramid_stages: Number of pyramid stages K
            num_inference_steps: Number of ODE integration steps per stage
            device: Inference device
        """
        self.model = model
        self.vae = vae
        self.text_encoder_t5 = text_encoder_t5
        self.text_encoder_clip = text_encoder_clip
        self.tokenizer_t5 = tokenizer_t5
        self.tokenizer_clip = tokenizer_clip
        
        self.pyramid_flow = PyramidalFlowMatching(num_stages=num_pyramid_stages)
        self.temporal_pyramid = TemporalPyramidCondition()
        self.sampler = EulerSampler(num_steps=num_inference_steps)
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        self.model = self.model.to(self.device)
        self.vae = self.vae.to(self.device)
    
    @torch.no_grad()
    def encode_text(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode text prompts using T5 and CLIP encoders.
        
        Args:
            prompt: Text prompt(s)
            negative_prompt: Negative prompt(s) for CFG
        
        Returns:
            Tuple of (t5_embeds, clip_embeds, neg_t5_embeds, neg_clip_embeds)
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        
        # T5 encoding
        t5_inputs = self.tokenizer_t5(
            prompt,
            padding='max_length',
            max_length=256,
            truncation=True,
            return_tensors='pt',
        ).to(self.device)
        
        t5_embeds = self.text_encoder_t5(**t5_inputs).last_hidden_state
        
        # CLIP encoding
        clip_inputs = self.tokenizer_clip(
            prompt,
            padding='max_length',
            max_length=77,
            truncation=True,
            return_tensors='pt',
        ).to(self.device)
        
        clip_embeds = self.text_encoder_clip(**clip_inputs).pooler_output
        
        # Negative prompt encoding for CFG
        if negative_prompt is None:
            negative_prompt = [''] * len(prompt)
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * len(prompt)
        
        neg_t5_inputs = self.tokenizer_t5(
            negative_prompt,
            padding='max_length',
            max_length=256,
            truncation=True,
            return_tensors='pt',
        ).to(self.device)
        
        neg_t5_embeds = self.text_encoder_t5(**neg_t5_inputs).last_hidden_state
        
        neg_clip_inputs = self.tokenizer_clip(
            negative_prompt,
            padding='max_length',
            max_length=77,
            truncation=True,
            return_tensors='pt',
        ).to(self.device)
        
        neg_clip_embeds = self.text_encoder_clip(**neg_clip_inputs).pooler_output
        
        return t5_embeds, clip_embeds, neg_t5_embeds, neg_clip_embeds
    
    def apply_cfg(
        self,
        cond_velocity: torch.Tensor,
        uncond_velocity: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        Apply classifier-free guidance.
        
        v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
        
        Args:
            cond_velocity: Conditional velocity
            uncond_velocity: Unconditional velocity
            guidance_scale: CFG scale
        
        Returns:
            Guided velocity
        """
        return uncond_velocity + guidance_scale * (cond_velocity - uncond_velocity)
    
    def generate_single_segment(
        self,
        t5_embeds: torch.Tensor,
        clip_embeds: torch.Tensor,
        neg_t5_embeds: torch.Tensor,
        neg_clip_embeds: torch.Tensor,
        latent_shape: Tuple[int, ...],
        history_latents: Optional[List[torch.Tensor]] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
    ) -> torch.Tensor:
        """
        Generate a single video segment using pyramidal flow matching.
        
        Implements Algorithm 1 from the paper:
        1. Initialize from noise
        2. For each pyramid stage (from lowest to highest resolution):
           a. Run ODE integration within the stage
           b. Apply renoising at jump point to transition to next stage
        
        Args:
            t5_embeds: T5 text embeddings (B, L, D)
            clip_embeds: CLIP text embeddings (B, D)
            neg_t5_embeds: Negative T5 embeddings for CFG
            neg_clip_embeds: Negative CLIP embeddings for CFG
            latent_shape: Shape of the latent to generate (B, C, H, W) or (B, C, T, H, W)
            history_latents: Optional history latents for autoregressive generation
            guidance_scale: CFG guidance scale
            num_inference_steps: Number of ODE steps per stage
        
        Returns:
            Generated latent tensor
        """
        B = t5_embeds.shape[0]
        device = t5_embeds.device
        
        # Initialize from noise (Algorithm 1: x_0 ~ N(0, I))
        # Start at the lowest resolution
        K = self.pyramid_flow.num_stages
        
        # Determine initial resolution (lowest pyramid stage)
        # Stage 0 is lowest resolution: factor 2^(K-1)
        initial_down_factor = 2 ** (K - 1)
        
        if len(latent_shape) == 4:
            B, C, H, W = latent_shape
            initial_shape = (B, C, H // initial_down_factor, W // initial_down_factor)
        else:
            B, C, T, H, W = latent_shape
            initial_shape = (B, C, T, H // initial_down_factor, W // initial_down_factor)
        
        x = torch.randn(initial_shape, device=device)
        
        # Prepare history condition
        history_tokens = None
        if history_latents is not None:
            # Will be prepared per stage
            pass
        
        # Iterate through pyramid stages from lowest to highest resolution
        # Paper: "for k from K-1 to 0"
        for stage_idx in range(K):
            # In paper notation, k goes from K-1 (lowest res) to 0 (full res)
            # In our indexing, stage_idx goes from 0 (lowest) to K-1 (full)
            s_k, e_k = self.pyramid_flow.stage_time_windows[stage_idx]
            
            # Prepare history for this stage
            if history_latents is not None:
                compressed_history = self.temporal_pyramid.prepare_history_condition(
                    history_latents,
                    current_stage=stage_idx,
                    num_pyramid_stages=K,
                    training=False,
                )
                history_tokens = compressed_history
            
            # Define model function for this stage
            def model_fn(x_t, t, stage=stage_idx, hist=history_tokens):
                # Conditional prediction
                cond_vel = self.model(
                    x_t, t,
                    t5_embeds, clip_embeds,
                    pyramid_stage=stage,
                    history_tokens=hist,
                    use_causal_attention=(history_latents is not None),
                )
                
                if guidance_scale > 1.0:
                    # Unconditional prediction
                    uncond_vel = self.model(
                        x_t, t,
                        neg_t5_embeds, neg_clip_embeds,
                        pyramid_stage=stage,
                        history_tokens=hist,
                        use_causal_attention=(history_latents is not None),
                    )
                    return self.apply_cfg(cond_vel, uncond_vel, guidance_scale)
                
                return cond_vel
            
            # Run ODE integration within this stage
            x = self.sampler.integrate(
                model_fn,
                x_start=x,
                t_start=s_k,
                t_end=e_k,
            )
            
            # Apply renoising at jump point (if not the last stage)
            if stage_idx < K - 1:
                # Determine target size for next stage
                next_stage_idx = stage_idx + 1
                next_down_factor = 2 ** (K - 1 - next_stage_idx)
                
                if len(latent_shape) == 4:
                    B, C, H, W = latent_shape
                    target_size = (H // next_down_factor, W // next_down_factor)
                else:
                    B, C, T, H, W = latent_shape
                    target_size = (H // next_down_factor, W // next_down_factor)
                
                s_next = self.pyramid_flow.stage_time_windows[next_stage_idx][0]
                
                # Apply renoising rule (Eq. 15)
                x = self.pyramid_flow.renoise_at_jump_point(
                    x, s_next, target_size=target_size
                )
        
        return x
    
    @torch.no_grad()
    def generate_video(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_frames: int = 121,  # 5 seconds at 24fps (with 1 frame overlap)
        height: int = 768,
        width: int = 768,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        fps: int = 24,
        seed: Optional[int] = None,
        image_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate a video from a text prompt.
        
        Supports autoregressive generation for long videos by generating
        multiple segments and conditioning each on the previous.
        
        Args:
            prompt: Text prompt(s)
            negative_prompt: Negative prompt(s) for CFG
            num_frames: Total number of frames to generate
            height: Video height in pixels
            width: Video width in pixels
            guidance_scale: CFG guidance scale
            num_inference_steps: Number of ODE steps per pyramid stage
            fps: Frames per second
            seed: Random seed for reproducibility
            image_condition: Optional image for image-to-video generation
        
        Returns:
            Generated video tensor (B, C, T, H, W) in pixel space
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        if isinstance(prompt, str):
            prompt = [prompt]
        B = len(prompt)
        
        # Encode text
        t5_embeds, clip_embeds, neg_t5_embeds, neg_clip_embeds = self.encode_text(
            prompt, negative_prompt
        )
        
        # Compute latent dimensions (8x compression)
        latent_H = height // 8
        latent_W = width // 8
        latent_T = num_frames // 8  # Temporal compression
        latent_C = self.model.in_channels
        
        # Determine segment size for autoregressive generation
        # Paper generates 5-second segments (121 frames = ~15 latent frames)
        segment_frames = min(num_frames, 121)  # 5 seconds
        segment_latent_T = segment_frames // 8
        
        all_latents = []
        history_latents = []
        
        # Handle image conditioning (image-to-video)
        if image_condition is not None:
            # Encode the conditioning image
            image_latent = self.vae.encode_video(
                image_condition.unsqueeze(2).to(self.device)  # Add temporal dim
            )
            history_latents.append(image_latent)
        
        # Generate video autoregressively
        frames_generated = 0
        while frames_generated < num_frames:
            remaining = num_frames - frames_generated
            current_segment_frames = min(segment_frames, remaining)
            current_latent_T = max(current_segment_frames // 8, 1)
            
            latent_shape = (B, latent_C, current_latent_T, latent_H, latent_W)
            
            # Generate segment
            segment_latent = self.generate_single_segment(
                t5_embeds, clip_embeds,
                neg_t5_embeds, neg_clip_embeds,
                latent_shape=latent_shape,
                history_latents=history_latents if history_latents else None,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            
            all_latents.append(segment_latent)
            
            # Update history for next segment
            # Use temporal pyramid compression for history
            history_latents.append(segment_latent)
            
            # Keep only recent history (to manage memory)
            max_history = 4
            if len(history_latents) > max_history:
                history_latents = history_latents[-max_history:]
            
            frames_generated += current_segment_frames
        
        # Concatenate all segments
        all_latents_cat = torch.cat(all_latents, dim=2)  # Concatenate along temporal dim
        
        # Decode to pixel space
        video = self.vae.decode_video(all_latents_cat)
        
        # Normalize to [0, 1]
        video = (video + 1) / 2
        video = video.clamp(0, 1)
        
        return video
    
    @torch.no_grad()
    def generate_image(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 768,
        width: int = 768,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate an image from a text prompt.
        
        The model naturally supports image generation since the first frame
        of a video acts as an image.
        
        Args:
            prompt: Text prompt(s)
            negative_prompt: Negative prompt(s) for CFG
            height: Image height
            width: Image width
            guidance_scale: CFG guidance scale
            num_inference_steps: Number of ODE steps per pyramid stage
            seed: Random seed
        
        Returns:
            Generated image tensor (B, C, H, W) in pixel space
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        if isinstance(prompt, str):
            prompt = [prompt]
        B = len(prompt)
        
        # Encode text
        t5_embeds, clip_embeds, neg_t5_embeds, neg_clip_embeds = self.encode_text(
            prompt, negative_prompt
        )
        
        # Compute latent dimensions
        latent_H = height // 8
        latent_W = width // 8
        latent_C = self.model.in_channels
        
        latent_shape = (B, latent_C, latent_H, latent_W)
        
        # Generate image latent
        image_latent = self.generate_single_segment(
            t5_embeds, clip_embeds,
            neg_t5_embeds, neg_clip_embeds,
            latent_shape=latent_shape,
            history_latents=None,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        
        # Decode to pixel space
        # Add temporal dimension for VAE
        image_latent_3d = image_latent.unsqueeze(2)
        image = self.vae.decode_video(image_latent_3d)
        image = image.squeeze(2)  # Remove temporal dim
        
        # Normalize to [0, 1]
        image = (image + 1) / 2
        image = image.clamp(0, 1)
        
        return image
