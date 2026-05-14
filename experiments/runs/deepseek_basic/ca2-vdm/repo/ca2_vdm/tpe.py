"""
Cyclic Temporal Positional Embeddings (Cyclic-TPEs) for Ca2-VDM.

Key insight: With KV-cache, TPEs are bound to keys and values at previous AR steps.
We cannot reassign TPEs from scratch when the cache is full (old frames dequeued).

Solution: Cyclic-TPEs - assign TPEs cyclically shifted with a random offset
during training, and use cyclic assignment during inference.

The design ensures that when P_k = P_max during inference and old frames are
dequeued, the new frames get TPEs starting from the beginning of the cyclic sequence.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


def get_sinusoidal_tpe(num_frames: int, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    Generate sinusoidal temporal positional embeddings.
    Following standard sinusoidal PE (as in ViT).
    
    Args:
        num_frames: number of temporal positions
        dim: embedding dimension
        max_period: maximum period for sinusoidal encoding
    
    Returns:
        tpe: (num_frames, dim)
    """
    position = torch.arange(num_frames, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(max_period) / dim)
    )
    tpe = torch.zeros(num_frames, dim)
    tpe[:, 0::2] = torch.sin(position * div_term)
    tpe[:, 1::2] = torch.cos(position * div_term)
    return tpe


class CyclicTPE(nn.Module):
    """
    Cyclic Temporal Positional Embeddings.
    
    During training:
    - Each sample is assigned a TPE sequence that is cyclically shifted
      with a random offset.
    - This trains the model to handle any cyclic position.
    
    During inference:
    - TPEs are assigned chunk-by-chunk as autoregression progresses.
    - When P_k reaches P_max, TPEs "wrap around" cyclically.
    - The denoising target gets TPEs indexed from the beginning after wrap.
    
    As shown in Figure 4(c): 
    - Without KV-cache (left): can reassign TPEs from scratch after dequeue.
    - With KV-cache (right): TPEs bound to cached K,V, so we use cyclic shift.
    """
    
    def __init__(
        self,
        dim: int,
        L_train: int,
        P_max: int,
        l: int,
        max_period: float = 10000.0,
    ):
        """
        Args:
            dim: embedding dimension
            L_train: maximum training length = P_max + l
            P_max: maximum conditional frames
            l: chunk length
            max_period: max period for sinusoidal
        """
        super().__init__()
        self.dim = dim
        self.L_train = L_train
        self.P_max = P_max
        self.l = l
        self.max_period = max_period
        
        # Base TPEs for L_train positions
        base_tpe = get_sinusoidal_tpe(L_train, dim, max_period)
        self.register_buffer('base_tpe', base_tpe)
    
    def get_tpe(self, offset: int, length: int) -> torch.Tensor:
        """
        Get TPEs with cyclic shift.
        
        Args:
            offset: cyclic offset (0 to L_train-1)
            length: number of TPEs to return
        
        Returns:
            tpe: (length, dim)
        """
        indices = (torch.arange(length) + offset) % self.L_train
        return self.base_tpe[indices]
    
    def forward(
        self,
        num_frames: int,
        cyclic_offset: int = 0,
    ) -> torch.Tensor:
        """
        Get TPEs for a sequence of frames.
        
        Args:
            num_frames: number of frames
            cyclic_offset: offset for cyclic shift
        
        Returns:
            tpe: (1, num_frames, 1, dim) ready for broadcasting
        """
        tpe = self.get_tpe(cyclic_offset, num_frames)
        return tpe.unsqueeze(0).unsqueeze(2)  # (1, num_frames, 1, dim)


class PositionalEmbeddings(nn.Module):
    """
    Combined spatial and temporal positional embeddings.
    
    Spatial PEs (SPE) are learned or sinusoidal, following ViT.
    Temporal PEs (TPE) use CyclicTPE.
    """
    
    def __init__(
        self,
        dim: int,
        H: int,
        W: int,
        L_train: int,
        P_max: int,
        l: int,
        use_learned_spe: bool = True,
    ):
        """
        Args:
            dim: embedding dimension
            H, W: spatial resolution after VAE encoding
            L_train: max training length
            P_max: max conditional frames
            l: chunk length
            use_learned_spe: if True, use learned SPE; else sinusoidal
        """
        super().__init__()
        self.dim = dim
        self.H = H
        self.W = W
        
        # Spatial positional embeddings
        if use_learned_spe:
            self.spe = nn.Parameter(torch.randn(1, 1, H * W, dim) * 0.02)
        else:
            spe = self._get_sinusoidal_spe(H, W, dim)
            self.register_buffer('spe', spe)
        
        # Temporal positional embeddings (cyclic)
        self.tpe = CyclicTPE(dim, L_train, P_max, l)
    
    def _get_sinusoidal_spe(self, H: int, W: int, dim: int) -> torch.Tensor:
        """Generate 2D sinusoidal spatial PE."""
        h_pos = torch.arange(H, dtype=torch.float32).unsqueeze(1)
        w_pos = torch.arange(W, dtype=torch.float32).unsqueeze(0)
        
        div_term = torch.exp(
            torch.arange(0, dim // 2, 1, dtype=torch.float32) * 
            (-math.log(10000.0) / (dim // 2))
        )
        
        # H positions
        pe_h = torch.zeros(H, W, dim // 2)
        pe_h[:, :, 0::2] = torch.sin(h_pos * div_term[0::2]).unsqueeze(1)
        pe_h[:, :, 1::2] = torch.cos(h_pos * div_term[1::2]).unsqueeze(1)
        
        # W positions
        pe_w = torch.zeros(H, W, dim // 2)
        pe_w[:, :, 0::2] = torch.sin(w_pos * div_term[0::2]).unsqueeze(0)
        pe_w[:, :, 1::2] = torch.cos(w_pos * div_term[1::2]).unsqueeze(0)
        
        pe = torch.cat([pe_h, pe_w], dim=-1)  # (H, W, dim)
        return pe.reshape(1, 1, H * W, dim)
    
    def forward(
        self,
        num_frames: int,
        cyclic_offset: int = 0,
    ) -> torch.Tensor:
        """
        Get combined SPE + TPE.
        
        Args:
            num_frames: number of frames
            cyclic_offset: offset for TPE cyclic shift
        
        Returns:
            pos_emb: (1, num_frames, H*W, dim)
        """
        tpe = self.tpe(num_frames, cyclic_offset)  # (1, num_frames, 1, dim)
        
        if hasattr(self, 'spe'):
            spe = self.spe  # (1, 1, H*W, dim)
        else:
            spe = self.spe  # parameter or buffer
        
        # Add SPE and TPE
        return spe + tpe  # broadcast: (1, num_frames, H*W, dim)
