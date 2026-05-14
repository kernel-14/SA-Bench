"""
Temporal Pyramid Implementation for Pyramidal Flow Matching.

Implements the autoregressive video generation with compressed history conditions.
Older frames are stored at progressively lower resolutions to reduce token count
while preserving semantic information (Eqs. 16 and 17).

Paper reference:
  Eq. (16) Training:  ... -> Down(x_{t'}^{i-2}, 2^{k+1}) -> Down(x_{t'}^{i-1}, 2^k) -> x_t^i
  Eq. (17) Inference: ... -> Down(x_1^{i-2}, 2^{k+1}) -> Down(x_1^{i-1}, 2^k) -> x_t^i
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class TemporalPyramidHistory(nn.Module):
    """
    Temporal Pyramid History Condition for Autoregressive Video Generation.
    
    Args:
        num_pyramid_levels: Number of temporal pyramid levels (K)
        history_length: Number of past frames to keep total
        base_resolution: Base spatial resolution (H, W) of full-res latents
    """
    
    def __init__(
        self,
        num_pyramid_levels: int = 3,
        history_length: int = 12,
        base_resolution: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.num_pyramid_levels = num_pyramid_levels
        self.history_length = history_length
        self.base_resolution = base_resolution
        self.frames_per_level = self._compute_frames_per_level()
    
    def _compute_frames_per_level(self) -> List[int]:
        """Compute how many frames to assign to each pyramid level."""
        frames_per_level = [0] * self.num_pyramid_levels
        remaining = self.history_length
        
        # Assign frames: newest to level 0 (full res), older to coarser levels
        # Level 0: most recent 2 frames
        # Level 1: next 4 frames  
        # Level 2: remaining frames
        for l in range(self.num_pyramid_levels):
            if remaining <= 0:
                break
            if l < self.num_pyramid_levels - 1:
                count = min(2 * (l + 1), remaining)
            else:
                count = remaining  # All remaining to coarsest
            frames_per_level[l] = count
            remaining -= count
        
        return frames_per_level
    
    def build_history_condition(
        self,
        past_frames: List[torch.Tensor],
        noise_strength: float = 0.0,
    ) -> List[torch.Tensor]:
        """
        Build the temporal pyramid history condition from past frames.
        
        Args:
            past_frames: List of past frame latents from oldest to newest.
            noise_strength: Corruptive noise strength from [0, 1/3] for training.
                           
        Returns:
            List of history condition tensors at various resolutions.
        """
        if not past_frames:
            return []
        
        history_tensors = []
        frame_idx = len(past_frames) - 1  # Start from newest
        
        for level in range(self.num_pyramid_levels):
            num_frames = self.frames_per_level[level]
            if num_frames <= 0:
                continue
            
            resolution_factor = 2 ** level
            
            level_frames = []
            for _ in range(num_frames):
                if frame_idx < 0:
                    break
                frame = past_frames[frame_idx].clone()
                
                # Downsample to this level's resolution
                if resolution_factor > 1:
                    if frame.dim() == 4:
                        frame = F.interpolate(
                            frame, scale_factor=1.0 / resolution_factor,
                            mode='bilinear', align_corners=False
                        )
                
                # Add corruptive noise during training
                if noise_strength > 0.0:
                    frame = frame + noise_strength * torch.randn_like(frame)
                
                level_frames.append(frame)
                frame_idx -= 1
            
            if level_frames:
                # Stack frames at this level: (B, C, T_level, H_level, W_level)
                level_tensor = torch.stack(list(reversed(level_frames)), dim=2)
                history_tensors.append(level_tensor)
        
        return history_tensors
    
    def build_inference_history(
        self,
        generated_frames: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Build temporal pyramid history for inference (Eq. 17, no noise)."""
        return self.build_history_condition(generated_frames, noise_strength=0.0)
    
    def compute_token_count(self) -> int:
        """Compute total number of tokens in the temporal pyramid history."""
        total_tokens = 0
        for level in range(self.num_pyramid_levels):
            resolution_factor = 2 ** level
            num_frames = self.frames_per_level[level]
            if self.base_resolution:
                H, W = self.base_resolution
                tokens_per_frame = (H // resolution_factor) * (W // resolution_factor)
            else:
                tokens_per_frame = 1
            total_tokens += num_frames * tokens_per_frame
        return total_tokens
    
    def compute_efficiency_gain(self) -> float:
        """Compute efficiency gain vs full-resolution history."""
        if not self.base_resolution:
            return float('inf')
        H, W = self.base_resolution
        full_res_tokens = self.history_length * H * W
        pyramid_tokens = self.compute_token_count()
        return full_res_tokens / max(pyramid_tokens, 1)


class TemporalPyramidConditioning(nn.Module):
    """
    Wraps temporal pyramid history as conditioning input to the velocity model.
    """
    
    def __init__(
        self,
        num_pyramid_levels: int = 3,
        max_history_frames: int = 12,
    ):
        super().__init__()
        self.num_pyramid_levels = num_pyramid_levels
        self.max_history_frames = max_history_frames
        self.history_pyramid = TemporalPyramidHistory(
            num_pyramid_levels=num_pyramid_levels,
            history_length=max_history_frames,
        )
    
    def prepare_conditioning(
        self,
        current_noisy: torch.Tensor,
        past_clean_frames: List[torch.Tensor],
        training: bool = True,
        noise_strength: float = 0.0,
    ) -> Optional[torch.Tensor]:
        """
        Prepare the full conditioning input by combining temporal pyramid
        history with the current noisy latent.
        
        Returns:
            Combined token tensor or None if no history
        """
        if not past_clean_frames:
            return None
        
        if training and noise_strength > 0:
            history = self.history_pyramid.build_history_condition(
                past_clean_frames, noise_strength=noise_strength
            )
        else:
            history = self.history_pyramid.build_inference_history(past_clean_frames)
        
        return self._combine_tokens(history, current_noisy)
    
    def _combine_tokens(
        self,
        history: List[torch.Tensor],
        current: torch.Tensor,
    ) -> torch.Tensor:
        """Combine history tokens at various resolutions with current noisy latent."""
        token_list = []
        
        # Flatten each history level
        for h in history:
            B, C, T, H, W = h.shape
            h_flat = h.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            token_list.append(h_flat)
        
        # Flatten current
        if current.dim() == 4:
            B, C, H, W = current.shape
            curr_flat = current.permute(0, 2, 3, 1).reshape(B, H * W, C)
        else:
            curr_flat = current.flatten(2).transpose(1, 2)
        
        token_list.append(curr_flat)
        
        return torch.cat(token_list, dim=1)
