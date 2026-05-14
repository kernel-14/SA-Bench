"""
Causal Temporal Attention and Prefix-Enhanced Spatial Attention
as described in Ca2-VDM paper.

Key components:
1. Causal Temporal Attention (Eq. 3): Lower triangular mask so each frame
   only attends to preceding frames.
2. Prefix-Enhanced Spatial Attention (Eq. 4): Spatial-wise concatenation
   of prefix frames to enhance guidance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalTemporalAttention(nn.Module):
    """
    Causal temporal attention with lower triangular mask.
    
    The input is permuted to treat spatial resolution H x W as batch dimension,
    then Q, K, V are projected. The attention mask ensures each frame only
    attends to its prefix frames.

    Eq. (3): CausalAttn(Q, K, V) = Softmax(Q K^T / sqrt(C') + M) V
    where M_{i,j} = -inf if i < j else 0.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def _get_causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        """Create lower triangular causal mask: M_{i,j} = -inf if i < j else 0."""
        mask = torch.triu(
            torch.ones(L, L, device=device) * float('-inf'), diagonal=1
        )
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: shape (B, L, H*W, C) - batch, frames, spatial, channels
            kv_cache: optional tuple of (K_cache, V_cache) from clean prefix frames
        
        Returns:
            out: shape (B, L, H*W, C)
            kv: tuple (K, V) for caching (clean, at t=0)
        """
        B, L, S, C = x.shape  # S = H*W
        
        # Project to Q, K, V
        qkv = self.qkv(x)  # (B, L, S, 3*C)
        qkv = qkv.reshape(B, L, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)  # (3, B, nH, L, S, d)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, nH, L, S, d)
        
        # Merge spatial and head dims for attention computation
        # (B, nH, L, S, d) -> (B, nH, L, S*d)? No, we compute per spatial grid
        # Following paper: treat spatial as batch dim
        # Rearrange: (B, nH, L, S, d) -> (B*S, nH, L, d)
        q = q.permute(0, 3, 1, 2, 4).reshape(B * S, self.num_heads, L, self.head_dim)
        k = k.permute(0, 3, 1, 2, 4).reshape(B * S, self.num_heads, L, self.head_dim)
        v = v.permute(0, 3, 1, 2, 4).reshape(B * S, self.num_heads, L, self.head_dim)
        
        # If KV cache is provided, prepend cached K, V
        if kv_cache is not None:
            k_cache, v_cache = kv_cache  # each: (B*S, nH, P_k, d)
            k = torch.cat([k_cache, k], dim=2)  # (B*S, nH, P_k+L, d)
            v = torch.cat([v_cache, v], dim=2)
        
        P_k = k.shape[2] - L if kv_cache is not None else 0
        total_L = k.shape[2]
        
        # Compute attention with causal mask
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B*S, nH, L, total_L)
        
        # Causal mask: each frame can only attend to prefix (including itself)
        # For the full sequence: positions [0, P_k) are prefix, [P_k, total_L) are current
        causal_mask = self._get_causal_mask(total_L, x.device)
        # Only apply to current frames attending: rows [P_k:] get the mask
        mask = torch.zeros(L, total_L, device=x.device)
        mask_portion = causal_mask[P_k:, :]
        mask = mask_portion
        attn = attn + mask.unsqueeze(0).unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)  # (B*S, nH, L, d)
        
        # Reshape back
        out = out.reshape(B, S, self.num_heads, L, self.head_dim)
        out = out.permute(0, 3, 1, 2, 4)  # (B, L, S, nH, d)
        out = out.reshape(B, L, S, C)
        out = self.proj(out)
        
        # Return KV for caching (only the current frames' K, V)
        # k,v are currently (B*S, nH, total_L, d), we need only current L frames
        if kv_cache is not None:
            k_current = k[:, :, P_k:, :]  # (B*S, nH, L, d)
            v_current = v[:, :, P_k:, :]
        else:
            k_current = k
            v_current = v
            
        return out, (k_current, v_current)


class PrefixEnhancedSpatialAttention(nn.Module):
    """
    Prefix-enhanced spatial attention.
    
    For each frame i, the spatial attention is enhanced by concatenating
    a sub-prefix of P' frames along the spatial dimension.
    
    Eq. (4):
    For i >= P (denoising target):
        K_bar(i) = W^K [h_0^{P-P'}; ...; h_0^{P-1}; h_t^i]
    For i < P (clean prefix):
        K_bar(i) = W^K [h_0^i; ...; h_0^i]  (self-repeat P' times)
        
    Same for V_bar(i).
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        prefix_len: int = 3,  # P' in paper
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.prefix_len = prefix_len
        assert dim % num_heads == 0
        
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def _get_prefix_kv(
        self,
        h: torch.Tensor,
        spatial_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        P: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build prefix-enhanced K and V.
        
        Args:
            h: hidden states (B, L, S, C) where S = H*W
            spatial_cache: (K_cache, V_cache) from previous chunk for clean prefix
            P: number of clean prefix frames in h
        
        Returns:
            K_bar, V_bar: each (B, L, (P'+1)*S, C)
        """
        B, L, S, C = h.shape
        P_prime = self.prefix_len
        
        # Compute K, V for each frame
        k_all = self.k(h)  # (B, L, S, C)
        v_all = self.v(h)  # (B, L, S, C)
        
        k_list = []
        v_list = []
        
        for i in range(L):
            if i < P:
                # Clean prefix part: self-repeat P' times
                # h_0^i broadcasted by self-repeat P' times
                k_i = k_all[:, i:i+1, :, :]  # (B, 1, S, C)
                v_i = v_all[:, i:i+1, :, :]
                # Repeat P' times along spatial dim
                k_enhanced = k_i.repeat(1, 1, P_prime, 1)  # (B, 1, P'*S, C)
                v_enhanced = v_i.repeat(1, 1, P_prime, 1)
                # In practice, we compute [h_0^i; ...; h_0^i] (P' copies) then project
                # But equivalent to repeating the projected K,V
                k_bar_i = torch.cat([k_enhanced, k_i], dim=2)  # (B, 1, (P'+1)*S, C)
                v_bar_i = torch.cat([v_enhanced, v_i], dim=2)
            else:
                # For denoising target frames: enhance with prefix K,V
                if spatial_cache is not None:
                    k_prefix, v_prefix = spatial_cache  # (B, P', S, C)
                    k_prefix = k_prefix.reshape(B, 1, P_prime * S, C)
                    v_prefix = v_prefix.reshape(B, 1, P_prime * S, C)
                elif P > 0:
                    # During training: use clean prefix frames h_0^{P-P':P}
                    prefix_start = max(0, P - P_prime)
                    k_prefix_frames = k_all[:, prefix_start:P, :, :]
                    v_prefix_frames = v_all[:, prefix_start:P, :, :]
                    actual_len = k_prefix_frames.shape[1]
                    if actual_len < P_prime:
                        pad_len = P_prime - actual_len
                        k_prefix_frames = torch.cat([
                            k_prefix_frames[:, :1, :, :].repeat(1, pad_len, 1, 1),
                            k_prefix_frames
                        ], dim=1)
                        v_prefix_frames = torch.cat([
                            v_prefix_frames[:, :1, :, :].repeat(1, pad_len, 1, 1),
                            v_prefix_frames
                        ], dim=1)
                    k_prefix = k_prefix_frames.reshape(B, 1, P_prime * S, C)
                    v_prefix = v_prefix_frames.reshape(B, 1, P_prime * S, C)
                else:
                    # No prefix frames (P=0 during cache writing):
                    # use self-repeat of current frame
                    k_prefix = k_all[:, i:i+1, :, :].repeat(1, 1, P_prime, 1).reshape(B, 1, P_prime * S, C)
                    v_prefix = v_all[:, i:i+1, :, :].repeat(1, 1, P_prime, 1).reshape(B, 1, P_prime * S, C)
                
                k_i = k_all[:, i:i+1, :, :]  # (B, 1, S, C)
                v_i = v_all[:, i:i+1, :, :]
                k_bar_i = torch.cat([k_prefix, k_i], dim=2)  # (B, 1, (P'+1)*S, C)
                v_bar_i = torch.cat([v_prefix, v_i], dim=2)
            
            k_list.append(k_bar_i)
            v_list.append(v_bar_i)
        
        k_bar = torch.cat(k_list, dim=1)  # (B, L, (P'+1)*S, C)
        v_bar = torch.cat(v_list, dim=1)
        
        return k_bar, v_bar
    
    def forward(
        self,
        h: torch.Tensor,
        spatial_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        P: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            h: (B, L, S, C) hidden states (S = H*W flattened)
            spatial_cache: (K_cache, V_cache) each (B, P', S, C)
            P: number of clean prefix frames
        
        Returns:
            out: (B, L, S, C)
            kv_cache_out: for next step (K, V) of current chunk frames
        """
        B, L, S, C = h.shape
        
        # Compute query for all frames
        q_bar = self.q(h)  # (B, L, S, C)
        
        # Compute prefix-enhanced K, V
        k_bar_all, v_bar_all = self._get_prefix_kv(h, spatial_cache, P)
        
        # Reshape for multi-head attention
        # Treat frames as batch, compute spatial attention
        q_bar = q_bar.reshape(B * L, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_bar_all = k_bar_all.reshape(B * L, (self.prefix_len + 1) * S, self.num_heads, self.head_dim)
        k_bar_all = k_bar_all.permute(0, 2, 1, 3)
        v_bar_all = v_bar_all.reshape(B * L, (self.prefix_len + 1) * S, self.num_heads, self.head_dim)
        v_bar_all = v_bar_all.permute(0, 2, 1, 3)
        
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q_bar, k_bar_all.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v_bar_all)  # (B*L, nH, S, d)
        out = out.permute(0, 2, 1, 3).reshape(B, L, S, C)
        out = self.proj(out)
        
        # Compute spatial KV cache for next chunk
        # We need K,V of the last P' frames (clean, not projected through prefix-enhancement)
        k_clean = self.k(h)  # (B, L, S, C)
        v_clean = self.v(h)
        # Take last P' frames (for caching)
        cache_len = min(self.prefix_len, L)
        k_cache_out = k_clean[:, -cache_len:, :, :]  # (B, P', S, C)
        v_cache_out = v_clean[:, -cache_len:, :, :]
        
        return out, (k_cache_out, v_cache_out)
