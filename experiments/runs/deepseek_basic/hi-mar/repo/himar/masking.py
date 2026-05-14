"""
Masking strategies for Hi-MAR.
Phase 1: Uniform random masking with ratio in [0.7, 1.0] (as in MAR).
Phase 2: Cosine masking schedule (as in MaskGIT).
For text-to-image: Beta distribution masking with alpha=4, beta=1.
"""

import torch
import torch.nn as nn
import numpy as np


def cosine_schedule(t, T, s=0.008):
    """Cosine schedule for masking ratio."""
    return torch.cos((t / T + s) / (1 + s) * torch.pi * 0.5) ** 2


class RandomMasking(nn.Module):
    """
    Random masking strategy used in Phase 1.
    Masking ratio r ~ Uniform(low, high).
    """
    def __init__(self, mask_range=(0.7, 1.0)):
        super().__init__()
        self.mask_range = mask_range

    def forward(self, x, mask_token):
        """
        Args:
            x: token sequence [B, N, C]
            mask_token: learnable mask token [C] or [1, C]
        Returns:
            masked_x, mask_indices
        """
        B, N, C = x.shape
        r = torch.rand(B, device=x.device) * (self.mask_range[1] - self.mask_range[0]) + self.mask_range[0]
        num_mask = (r * N).long()
        
        # Randomly select positions to mask
        rand_idx = torch.rand(B, N, device=x.device).argsort(dim=-1)
        mask_pos = rand_idx < num_mask.unsqueeze(-1)
        
        masked_x = x.clone()
        if mask_token.dim() == 1:
            mask_token = mask_token.unsqueeze(0).unsqueeze(0).expand(B, N, -1)
        elif mask_token.dim() == 2:
            mask_token = mask_token.unsqueeze(1).expand(B, N, -1)
        masked_x[mask_pos] = mask_token[mask_pos]
        
        return masked_x, mask_pos


class CosineMasking(nn.Module):
    """
    Cosine masking strategy used in Phase 2 during inference.
    Follows MaskGIT-style schedule.
    During training, a random step t is sampled and mask ratio = cosine_schedule(t).
    """
    def __init__(self, total_steps=256, s=0.008):
        super().__init__()
        self.total_steps = total_steps
        self.s = s

    def get_mask_ratio(self, t):
        """Get mask ratio for step t."""
        return cosine_schedule(t, self.total_steps, self.s)

    def forward(self, x, mask_token, t=None):
        """
        Args:
            x: token sequence [B, N, C]
            mask_token: learnable mask token
            t: step (int or tensor) or None for training (random)
        Returns:
            masked_x, mask_pos
        """
        B, N, C = x.shape
        if t is None:
            # Training: random step
            t = torch.randint(0, self.total_steps, (B,), device=x.device)
        
        r = self.get_mask_ratio(t.float()).to(x.device)
        num_mask = (r * N).long().clamp(min=1, max=N)
        
        rand_idx = torch.rand(B, N, device=x.device).argsort(dim=-1)
        mask_pos = rand_idx < num_mask.unsqueeze(-1)
        
        masked_x = x.clone()
        if mask_token.dim() == 1:
            mask_token = mask_token.unsqueeze(0).unsqueeze(0).expand(B, N, -1)
        elif mask_token.dim() == 2:
            mask_token = mask_token.unsqueeze(1).expand(B, N, -1)
        masked_x[mask_pos] = mask_token[mask_pos]
        
        return masked_x, mask_pos


class BetaMasking(nn.Module):
    """
    Beta distribution masking for text-to-image generation.
    r ~ Beta(alpha, beta).
    """
    def __init__(self, alpha=4.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta_param = beta

    def forward(self, x, mask_token):
        """
        Args:
            x: token sequence [B, N, C]
            mask_token: learnable mask token
        Returns:
            masked_x, mask_pos
        """
        B, N, C = x.shape
        r = torch.distributions.Beta(self.alpha, self.beta_param).sample((B,)).to(x.device)
        num_mask = (r * N).long().clamp(min=1, max=N)
        
        rand_idx = torch.rand(B, N, device=x.device).argsort(dim=-1)
        mask_pos = rand_idx < num_mask.unsqueeze(-1)
        
        masked_x = x.clone()
        if mask_token.dim() == 1:
            mask_token = mask_token.unsqueeze(0).unsqueeze(0).expand(B, N, -1)
        elif mask_token.dim() == 2:
            mask_token = mask_token.unsqueeze(1).expand(B, N, -1)
        masked_x[mask_pos] = mask_token[mask_pos]
        
        return masked_x, mask_pos
