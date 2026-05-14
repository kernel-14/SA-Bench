"""
Simple neural network models for testing and demonstration.

Includes:
1. Simple MLP velocity model for toy experiments
2. Time-conditioned U-Net-like architecture for image generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for conditioning on time."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time tensor [batch]
        
        Returns:
            Time embedding [batch, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class MLPVelocityModel(nn.Module):
    """
    Simple MLP velocity model for toy experiments.
    
    Takes (x, t) as input and outputs velocity v(x, t).
    """
    
    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        
        layers = []
        in_dim = data_dim + time_embed_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else data_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
            in_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: State [batch, data_dim]
            t: Time [batch]
        
        Returns:
            Velocity [batch, data_dim]
        """
        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)


class ConditionalMLPVelocityModel(nn.Module):
    """
    Conditional MLP velocity model.
    
    Takes (x, t, condition) as input and outputs velocity v(x, t | condition).
    """
    
    def __init__(
        self,
        data_dim: int,
        condition_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.condition_dim = condition_dim
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        
        layers = []
        in_dim = data_dim + time_embed_dim + condition_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else data_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
            in_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: State [batch, data_dim]
            t: Time [batch]
            condition: Conditioning [batch, condition_dim]
        
        Returns:
            Velocity [batch, data_dim]
        """
        t_emb = self.time_embed(t)
        
        if condition is not None:
            h = torch.cat([x, t_emb, condition], dim=-1)
        else:
            # Use zeros for unconditional
            zeros = torch.zeros(x.shape[0], self.condition_dim, device=x.device)
            h = torch.cat([x, t_emb, zeros], dim=-1)
        
        return self.net(h)


class ResBlock(nn.Module):
    """Residual block with time conditioning."""
    
    def __init__(self, dim: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, dim * 4)
        self.ff2 = nn.Linear(dim * 4, dim)
        self.time_proj = nn.Linear(time_dim, dim)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.ff1(h)
        h = h + self.time_proj(t_emb).unsqueeze(1) if h.dim() == 3 else h + self.time_proj(t_emb)
        h = self.act(h)
        h = self.ff2(h)
        h = self.norm2(h)
        return x + h


class LatentVelocityModel(nn.Module):
    """
    Velocity model for latent space (e.g., for image generation).
    
    Operates on flattened latent vectors with time and text conditioning.
    """
    
    def __init__(
        self,
        latent_dim: int,
        text_embed_dim: int = 512,
        hidden_dim: int = 1024,
        num_layers: int = 8,
        time_embed_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.text_proj = nn.Linear(text_embed_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, time_embed_dim)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Latent state [batch, latent_dim]
            t: Time [batch]
            text_embed: Text embedding [batch, text_embed_dim]
        
        Returns:
            Velocity [batch, latent_dim]
        """
        t_emb = self.time_embed(t)
        h = self.input_proj(x)
        
        if text_embed is not None:
            h = h + self.text_proj(text_embed)
        
        for block in self.blocks:
            h = block(h, t_emb)
        
        return self.output_proj(h)
