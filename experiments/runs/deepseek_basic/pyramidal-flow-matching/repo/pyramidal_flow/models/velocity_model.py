"""
Velocity Model wrapper for Pyramidal Flow Matching.

Wraps the DiT model to provide a clean interface for velocity prediction
with appropriate conditioning handling.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .dit import PyramidalDiT


class VelocityModel(nn.Module):
    """
    Velocity prediction model for flow matching.
    
    Wraps the DiT and handles:
    - Input/output reshaping
    - Timestep conditioning
    - Stage-dependent processing
    - Classifier-free guidance support
    
    Args:
        dit_model: The PyramidalDiT model
    """
    
    def __init__(self, dit_model: PyramidalDiT):
        super().__init__()
        self.dit = dit_model
    
    def forward(
        self,
        x: torch.Tensor,
        t: float,
        stage_idx: int = 0,
        conditioning: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict velocity v_t(x) at timestep t.
        
        Args:
            x: Noisy latent at timestep t
            t: Current timestep in [0, 1]
            stage_idx: Which spatial pyramid stage
            conditioning: Text/context embeddings
            history: Temporal pyramid history tokens
            attention_mask: Causal attention mask
            
        Returns:
            Predicted velocity v_t(x)
        """
        return self.dit(
            x, t, stage_idx,
            conditioning=conditioning,
            history=history,
            attention_mask=attention_mask,
        )
    
    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: float,
        stage_idx: int,
        conditioning: torch.Tensor,
        uncond_conditioning: torch.Tensor,
        guidance_scale: float = 7.0,
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with Classifier-Free Guidance.
        
        v_cfg = v_uncond + s * (v_cond - v_uncond)
        
        Args:
            x: Noisy latent
            t: Timestep
            stage_idx: Pyramid stage
            conditioning: Conditional embeddings
            uncond_conditioning: Unconditional embeddings
            guidance_scale: CFG scale s
            history: History conditioning
            
        Returns:
            CFG-guided velocity prediction
        """
        v_cond = self.forward(x, t, stage_idx, conditioning, history)
        v_uncond = self.forward(x, t, stage_idx, uncond_conditioning, history)
        return v_uncond + guidance_scale * (v_cond - v_uncond)
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_gb(self, dtype_size: int = 2) -> float:
        """Estimate model size in GB."""
        return self.get_num_params() * dtype_size / (1024 ** 3)
