"""
NaViL Visual Encoder Module.

The visual encoder consists of a series of transformer layers with bidirectional
attention, processing image patches into visual tokens. It uses the same 
architectural backbone as the LLM (decoder-style transformer) but with 
bidirectional (full) attention and 2D-RoPE for spatial position encoding.

Architecture defined in Eq (1):
    V_{d,w}(I) = C ⊙ F_d^w ⊙ ... ⊙ F_2^w ⊙ F_1^w ⊙ P(I)
where:
    - P is the Patch Embedding Layer (stride=16)
    - F_i^w are transformer layers with hidden dim w
    - C is the Connector (pixel shuffle + MLP projection)
    - d is depth, w is width
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Patch Embedding Layer P(·).
    Converts raw image into patch embeddings using a 2D convolution.
    Default stride is 16, as specified in the paper.
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 2048, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        x = self.proj(x)  # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dim)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class BidirectionalAttention(nn.Module):
    """
    Bidirectional (full) self-attention for the visual encoder.
    Uses 2D-RoPE as described in paper (Section 5.1).
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, N, self.num_heads, self.head_dim)
        
        # Apply RoPE if provided
        if rope_cos is not None and rope_sin is not None:
            q, k = self._apply_rope(q, k, rope_cos, rope_sin)
        
        # Transpose for attention: (B, num_heads, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention (full bidirectional)
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, N, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)
        
        return self.o_proj(attn_output)
    
    @staticmethod
    def _apply_rope(q, k, cos, sin):
        """Apply rotary position embeddings."""
        q_dim = q.shape[-1]
        k_dim = k.shape[-1]
        
        # Handle even dimensions
        cos_q = cos[..., :q_dim // 2].unsqueeze(1)
        sin_q = sin[..., :q_dim // 2].unsqueeze(1)
        cos_k = cos[..., :k_dim // 2].unsqueeze(1)
        sin_k = sin[..., :k_dim // 2].unsqueeze(1)
        
        q_rot = q.reshape(*q.shape[:-1], -1, 2)
        q_rot = torch.stack([-q_rot[..., 1], q_rot[..., 0]], dim=-1).reshape(q.shape)
        q = q * cos_q + q_rot * sin_q
        
        k_rot = k.reshape(*k.shape[:-1], -1, 2)
        k_rot = torch.stack([-k_rot[..., 1], k_rot[..., 0]], dim=-1).reshape(k.shape)
        k = k * cos_k + k_rot * sin_k
        
        return q, k


class FeedForward(nn.Module):
    """Standard SwiGLU Feed-Forward Network."""
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class VisualEncoderLayer(nn.Module):
    """Single transformer layer for the visual encoder (bidirectional attention)."""
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = BidirectionalAttention(hidden_size, num_heads, dropout)
        self.norm2 = RMSNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, rope_cos, rope_sin)
        hidden_states = residual + hidden_states
        
        # Pre-norm FFN
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class VisualEncoder(nn.Module):
    """
    NaViL Visual Encoder V_{d,w}(·).
    
    Composed of:
    - Patch Embedding layer (stride 16)
    - d transformer layers with bidirectional attention and 2D-RoPE
    - A connector (pixel shuffle + MLP) — but connector is separate in the full model
    
    Args:
        depth: Number of transformer layers (d)
        width: Hidden dimension (w)
        mlp_width: Intermediate dimension for FFN (default 4x width)
        num_heads: Number of attention heads
        patch_size: Patch size for embedding (default 16)
        dropout: Dropout rate
    """
    def __init__(
        self,
        depth: int = 24,
        width: int = 1472,
        mlp_width: int = 5888,
        num_heads: int = 23,
        patch_size: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth = depth
        self.width = width
        
        self.patch_embed = PatchEmbedding(
            in_channels=3, 
            embed_dim=width, 
            patch_size=patch_size
        )
        
        self.layers = nn.ModuleList([
            VisualEncoderLayer(
                hidden_size=width,
                num_heads=num_heads,
                intermediate_size=mlp_width,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        
        self.norm = RMSNorm(width)
        
    def forward(
        self, 
        pixel_values: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) input images
            rope_cos, rope_sin: 2D-RoPE position encodings
            
        Returns:
            (B, N, width) visual token embeddings
        """
        x = self.patch_embed(pixel_values)
        
        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin)
            
        x = self.norm(x)
        return x
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    

def create_visual_encoder_from_params(
    target_params_millions: float,
    llm_hidden_size: int = 2048,
) -> VisualEncoder:
    """
    Create a visual encoder with approximately the target number of parameters.
    
    According to the paper, the parameter count N ≈ 12 × d × w².
    The paper explores depths {3, 6, 12, 24, 48} and widths 
    {4096, 2880, 2048, 1472, 1024} for 600M encoder.
    
    For NaViL-2B: depth=24, width=1472 → ~600M params
    For NaViL-9B: depth=32, width=1792 → ~1.2B params
    """
    # Build depth-width mapping from paper (Table 6)
    if target_params_millions <= 75:
        depth, width, mlp_width, num_heads = 6, 1024, 4096, 16
    elif target_params_millions <= 150:
        depth, width, mlp_width, num_heads = 12, 1024, 4096, 16
    elif target_params_millions <= 300:
        depth, width, mlp_width, num_heads = 12, 1472, 5888, 23
    elif target_params_millions <= 600:
        # NaViL-2B config
        depth, width, mlp_width, num_heads = 24, 1472, 5888, 23
    else:
        # NaViL-9B config (1.2B)
        depth, width, mlp_width, num_heads = 32, 1792, 7168, 28
        
    return VisualEncoder(
        depth=depth,
        width=width,
        mlp_width=mlp_width,
        num_heads=num_heads,
    )
