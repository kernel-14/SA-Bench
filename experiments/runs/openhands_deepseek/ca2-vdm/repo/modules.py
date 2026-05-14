import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


def get_sinusoidal_positional_encoding(max_len: int, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding."""
    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TimestepEmbedding(nn.Module):
    """Timestep embedding module that maps scalar t to a vector."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_emb.to(self.mlp[0].weight.dtype))


class JointTimestepEmbedding(nn.Module):
    """Joint timestep embedding for partial noising setting.
    
    Produces different embeddings for clean prefix (t=0) and denoising target (t=t).
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.t_embed = TimestepEmbedding(hidden_size, frequency_embedding_size)

    def forward(self, t: torch.Tensor, L: int, P: int) -> torch.Tensor:
        """
        Args:
            t: scalar timestep (B,) or (B, 1)
            L: total number of frames
            P: number of clean prefix frames
        
        Returns:
            timestep_embeds: (B, L, hidden_size) with tEmb(0) for prefix and tEmb(t) for target
        """
        B = t.shape[0]
        device = t.device
        t_clean = torch.zeros_like(t)  # t=0 for clean prefix
        t_noisy = t

        emb_clean = self.t_embed(t_clean)  # (B, hidden_size)
        emb_noisy = self.t_embed(t_noisy)  # (B, hidden_size)

        # Build per-frame embedding
        prefix_emb = emb_clean.unsqueeze(1).expand(B, P, self.hidden_size)  # (B, P, C)
        target_emb = emb_noisy.unsqueeze(1).expand(B, L - P, self.hidden_size)  # (B, L-P, C)

        return torch.cat([prefix_emb, target_emb], dim=1)  # (B, L, C)


class CyclicTPEs(nn.Module):
    """Cyclic Temporal Positional Embeddings.
    
    During training, TPEs are cyclically shifted with a random offset.
    During inference, TPEs are assigned chunk-by-chunk. When the cumulatively
    generated video exceeds L_train, a cyclic shift is applied.
    """

    def __init__(self, max_len: int, hidden_size: int):
        super().__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        # Learnable or sinusoidal? Paper uses sinusoidal TPEs
        self.tpe = nn.Parameter(get_sinusoidal_positional_encoding(max_len, hidden_size), requires_grad=False)

    def forward(self, L: int, offset: int = 0) -> torch.Tensor:
        """Get TPEs of length L with cyclic offset.
        
        Args:
            L: number of frames
            offset: cyclic offset for training shift
        
        Returns:
            tpe: (L, hidden_size)
        """
        # Cyclic shift
        indices = torch.arange(L, device=self.tpe.device) + offset
        indices = indices % self.max_len
        return self.tpe[indices]

    def get_inference_tpes(self, L: int, P_k: int) -> torch.Tensor:
        """Get TPEs for autoregressive inference.
        
        Args:
            L: chunk length (l)
            P_k: cumulative number of generated frames
        
        Returns:
            tpe: (L, hidden_size) with cyclic assignment
        """
        device = self.tpe.device
        # Assign cyclically: P_k, P_k+1, ..., P_k+L-1 mod max_len
        indices = (torch.arange(L, device=device) + P_k) % self.max_len
        return self.tpe[indices]


class SpatialPosEmbed(nn.Module):
    """Sinusoidal spatial positional embeddings following ViT."""

    def __init__(self, num_patches: int, hidden_size: int):
        super().__init__()
        self.num_patches = num_patches
        self.hidden_size = hidden_size
        self.pos_embed = nn.Parameter(get_sinusoidal_positional_encoding(num_patches, hidden_size), requires_grad=False)

    def forward(self) -> torch.Tensor:
        return self.pos_embed
