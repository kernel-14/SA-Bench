"""
Memory Attention for SAM 2.

The memory attention conditions the current frame features on past frames'
features and predictions, as well as on any new prompts.

Architecture (Section 4, Appendix D.1):
- Stacks L transformer blocks
- First block takes image encoding from current frame as input
- Each block performs:
  1. Self-attention (with 2D-RoPE)
  2. Cross-attention to memories (prompted/unprompted frames) and object pointers
  3. MLP
- Uses vanilla attention operations (enabling FlashAttention-2)
- Object pointer tokens excluded from RoPE (no spatial correspondence)
- Default L=4 layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding for memory attention self- and cross-attention."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply 2D RoPE.
        Args:
            x: [B, N, C] or [B*n_heads, N, head_dim]
            positions: [N, 2] or [B, N, 2] y,x positions (normalized or raw indices)
        """
        # Simplified: apply rotation based on position indices
        B, N, C = x.shape
        if positions.dim() == 2:
            positions = positions.unsqueeze(0).expand(B, -1, -1)

        pos_h = positions[..., 0].float()
        pos_w = positions[..., 1].float()

        freqs_h = pos_h.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)
        freqs_w = pos_w.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        x_reshaped = x.reshape(B, N, C // 2, 2)
        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
        return torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1).reshape(B, N, C)


class MemoryAttentionLayer(nn.Module):
    """Single layer of the memory attention module."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        # Self-attention
        self.self_attn_qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.self_attn_proj = nn.Linear(embed_dim, embed_dim)

        # Cross-attention to memory
        self.cross_attn_q = nn.Linear(embed_dim, embed_dim)
        self.cross_attn_kv = nn.Linear(embed_dim, embed_dim * 2)
        self.cross_attn_proj = nn.Linear(embed_dim, embed_dim)

        # MLP
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 2D RoPE for self-attention
        self.rope = RoPE2D(self.head_dim)

    def _self_attention(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Self-attention with 2D-RoPE."""
        B, N, C = x.shape
        qkv = self.self_attn_qkv(self.norm1(x))
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if pos is not None:
            # Apply RoPE to q and k
            q_rope = q.reshape(B * self.num_heads, N, self.head_dim)
            k_rope = k.reshape(B * self.num_heads, N, self.head_dim)
            pos_expanded = pos.unsqueeze(0).unsqueeze(2).expand(B, self.num_heads, N, 2)
            pos_flat = pos_expanded.reshape(B * self.num_heads, N, 2)
            q_rope = self.rope(q_rope, pos_flat)
            k_rope = self.rope(k_rope, pos_flat)
            q = q_rope.reshape(B, self.num_heads, N, self.head_dim)
            k = k_rope.reshape(B, self.num_heads, N, self.head_dim)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.self_attn_proj(out)

    def _cross_attention(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-attend to memory features."""
        B, N, C = x.shape
        _, M, _ = memory.shape

        q = self.cross_attn_q(self.norm2(x))
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        kv = self.cross_attn_kv(memory)
        kv = kv.reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        if memory_pos is not None:
            k_rope = k.reshape(B * self.num_heads, M, self.head_dim)
            k_rope = self.rope(k_rope, memory_pos.unsqueeze(0).expand(B, M, 2).reshape(B * self.num_heads, M, 2))
            k = k_rope.reshape(B, self.num_heads, M, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.cross_attn_proj(out)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] current frame features
            memory: [B, M, C] memory features (spatial + object pointers)
            pos: [N, 2] positions for self-attention RoPE
            memory_pos: [M, 2] positions for memory cross-attention RoPE

        Returns:
            [B, N, C] conditioned features
        """
        # Self-attention
        x = x + self._self_attention(x, pos)
        # Cross-attention to memory
        x = x + self._cross_attention(x, memory, memory_pos)
        # MLP
        x = x + self.mlp(self.norm3(x))
        return x


class MemoryAttention(nn.Module):
    """
    Memory attention module: stacks L transformer blocks.
    Conditions current frame features on past frames' memories and object pointers.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            MemoryAttentionLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Temporal position embedding for N recent frames (not for prompted frames)
        self.temporal_pos_embed = nn.Embedding(128, embed_dim)  # support up to 128 time steps

    def forward(
        self,
        image_features: torch.Tensor,
        memory_bank: "MemoryBankOutput",
        current_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            image_features: [B, C, H, W] current frame image features (from image encoder)
            memory_bank: MemoryBankOutput containing spatial memory and object pointers
            current_pos: [H*W, 2] position indices for RoPE on current frame

        Returns:
            [B, C, H, W] conditioned frame features
        """
        B, C, H, W = image_features.shape

        # Flatten image features to tokens
        x = image_features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        # Get memory features
        spatial_memory = memory_bank.spatial_memory  # [B, M_spatial, C]
        object_pointers = memory_bank.object_pointers  # [B, M_obj, C]

        # Concatenate memory: spatial features + object pointers
        memory = torch.cat([spatial_memory, object_pointers], dim=1)  # [B, M, C]

        # Get position info for RoPE
        if current_pos is None:
            ys = torch.arange(H, device=x.device).float()
            xs = torch.arange(W, device=x.device).float()
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            current_pos = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)

        # Apply temporal position embedding to memory features
        if memory_bank.temporal_positions is not None:
            temporal_emb = self.temporal_pos_embed(memory_bank.temporal_positions)
            # Add to spatial part of memory only (not to object pointers)
            n_spatial = spatial_memory.shape[1]
            memory[:, :n_spatial] = memory[:, :n_spatial] + temporal_emb

        # Memory positions for cross-attention RoPE
        memory_pos = memory_bank.memory_positions  # [M, 2]

        # Pass through each memory attention layer
        for layer in self.layers:
            x = layer(x, memory, current_pos, memory_pos)

        # Reshape back to 2D
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        return x
