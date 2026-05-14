"""
Memory attention module for SAM 2.

The memory attention module conditions the current frame's features on:
1. Memories of past frames (spatial feature maps from memory bank)
2. Object pointer vectors (high-level semantic information)

It uses L stacked transformer blocks with:
- Self-attention on current frame features
- Cross-attention to memories and object pointers
- MLP

2D Rotary Positional Embedding (RoPE) is used in self- and cross-attention layers.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_2d_sincos_position_embedding(
    h: int, w: int, embed_dim: int, temperature: float = 10000.0
) -> torch.Tensor:
    """Build 2D sinusoidal position embedding."""
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing='ij',
    )
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sincos PE"
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature ** omega)

    out_y = grid_y.flatten()[:, None] * omega[None, :]
    out_x = grid_x.flatten()[:, None] * omega[None, :]

    pos_emb = torch.cat([
        torch.sin(out_y), torch.cos(out_y),
        torch.sin(out_x), torch.cos(out_x),
    ], dim=1)  # (H*W, embed_dim)
    return pos_emb


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to query and key tensors."""
    # q, k: [B, num_heads, N, head_dim]
    # cos, sin: [N, head_dim]
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, N, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, N, head_dim]

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class RoPE2D(nn.Module):
    """2D Rotary Positional Embedding for memory attention."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

    def get_cos_sin(self, h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin for 2D RoPE."""
        # Use half the head_dim for each spatial dimension
        dim_per_axis = self.head_dim // 4  # each axis gets dim_per_axis * 2 (for sin/cos)

        # Frequencies for y and x axes
        freq_y = 1.0 / (self.theta ** (torch.arange(0, dim_per_axis, device=device).float() / dim_per_axis))
        freq_x = 1.0 / (self.theta ** (torch.arange(0, dim_per_axis, device=device).float() / dim_per_axis))

        # Grid positions
        y_pos = torch.arange(h, device=device).float()
        x_pos = torch.arange(w, device=device).float()

        # Outer products
        emb_y = torch.outer(y_pos, freq_y)  # [H, dim_per_axis]
        emb_x = torch.outer(x_pos, freq_x)  # [W, dim_per_axis]

        # Create 2D grid
        emb_y = emb_y.unsqueeze(1).expand(-1, w, -1)  # [H, W, dim_per_axis]
        emb_x = emb_x.unsqueeze(0).expand(h, -1, -1)  # [H, W, dim_per_axis]

        # Concatenate and flatten
        emb = torch.cat([emb_y, emb_x], dim=-1)  # [H, W, dim_per_axis*2]
        emb = emb.reshape(h * w, -1)  # [H*W, dim_per_axis*2]

        # Pad to head_dim if needed
        if emb.shape[-1] < self.head_dim:
            pad = torch.zeros(emb.shape[0], self.head_dim - emb.shape[-1], device=device)
            emb = torch.cat([emb, pad], dim=-1)

        cos = torch.cos(emb)
        sin = torch.sin(emb)
        return cos, sin


class MemoryAttentionLayer(nn.Module):
    """
    Single layer of memory attention.

    Performs:
    1. Self-attention on current frame features (with 2D RoPE)
    2. Cross-attention to memory bank (spatial features + object pointers)
    3. MLP
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
        pos_enc_at_attn: bool = False,
        pos_enc_at_cross_attn_queries: bool = True,
        pos_enc_at_cross_attn_keys: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention to memories
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        # MLP
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _with_pos_embed(self, tensor: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_pos: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: [B, N, C] current frame features
            memory: [B, M, C] memory features (spatial + object pointers)
            tgt_pos: [B, N, C] positional encoding for current frame
            memory_pos: [B, M, C] positional encoding for memory

        Returns:
            tgt: [B, N, C] updated current frame features
        """
        # Self-attention
        q = k = self._with_pos_embed(tgt, tgt_pos if self.pos_enc_at_attn else None)
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention to memory
        q = self._with_pos_embed(tgt, tgt_pos if self.pos_enc_at_cross_attn_queries else None)
        k = self._with_pos_embed(memory, memory_pos if self.pos_enc_at_cross_attn_keys else None)
        tgt2, _ = self.cross_attn_image(q, k, memory)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        # MLP
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class MemoryAttention(nn.Module):
    """
    Memory attention module for SAM 2.

    Conditions current frame features on memories from past frames.
    Uses L stacked transformer blocks.

    Memory bank contains:
    - Spatial memory features from N recent frames (with temporal position encoding)
    - Spatial memory features from M prompted frames (without temporal position encoding)
    - Object pointer vectors from each frame (lightweight semantic vectors)
    """

    def __init__(
        self,
        d_model: int = 256,
        pos_enc_at_input: bool = True,
        num_layers: int = 4,
        layer: Optional[nn.Module] = None,
        memory_dim: int = 64,
        object_ptr_dim: int = 256,
        num_object_ptr_tokens: int = 4,  # 256-dim pointer split into 4 x 64-dim tokens
    ):
        super().__init__()
        self.d_model = d_model
        self.pos_enc_at_input = pos_enc_at_input
        self.num_layers = num_layers
        self.memory_dim = memory_dim
        self.num_object_ptr_tokens = num_object_ptr_tokens

        if layer is None:
            layer = MemoryAttentionLayer(d_model=d_model)

        self.layers = nn.ModuleList([
            MemoryAttentionLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=2048,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Project memory features from memory_dim to d_model for cross-attention
        self.memory_proj = nn.Linear(memory_dim, d_model)

        # Project object pointer tokens for cross-attention
        # Object pointers are 256-dim, split into 4 x 64-dim tokens
        self.obj_ptr_proj = nn.Linear(object_ptr_dim // num_object_ptr_tokens, d_model)

        # Temporal position encoding for recent frame memories
        # Embedded as learned embeddings for each temporal position
        self.temporal_pos_embed = nn.Embedding(8, d_model)  # up to 8 recent frames

    def forward(
        self,
        curr_feats: torch.Tensor,
        curr_pos: torch.Tensor,
        memory_bank_feats: List[torch.Tensor],
        memory_bank_pos: List[torch.Tensor],
        object_ptrs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Condition current frame features on memory bank.

        Args:
            curr_feats: [B, C, H, W] current frame features from image encoder
            curr_pos: [B, C, H, W] positional encoding for current frame
            memory_bank_feats: list of [B, memory_dim, H, W] memory features
            memory_bank_pos: list of [B, C, H, W] positional encodings for memories
            object_ptrs: optional list of [B, object_ptr_dim] object pointer vectors

        Returns:
            output: [B, C, H, W] conditioned frame features
        """
        B, C, H, W = curr_feats.shape

        # Flatten spatial dimensions
        curr_feats_flat = curr_feats.flatten(2).permute(0, 2, 1)  # B HW C
        curr_pos_flat = curr_pos.flatten(2).permute(0, 2, 1)  # B HW C

        # Prepare memory tokens
        memory_tokens = []
        memory_pos_tokens = []

        for i, (mem_feat, mem_pos) in enumerate(zip(memory_bank_feats, memory_bank_pos)):
            # Project memory features to d_model
            mem_flat = mem_feat.flatten(2).permute(0, 2, 1)  # B HW memory_dim
            mem_proj = self.memory_proj(mem_flat)  # B HW d_model

            # Add temporal position encoding for recent frames
            if i < len(self.temporal_pos_embed.weight):
                temp_pos = self.temporal_pos_embed.weight[i].unsqueeze(0).unsqueeze(0)
                mem_proj = mem_proj + temp_pos

            memory_tokens.append(mem_proj)

            # Memory positional encoding
            if mem_pos is not None:
                mem_pos_flat = mem_pos.flatten(2).permute(0, 2, 1)  # B HW C
                memory_pos_tokens.append(mem_pos_flat)
            else:
                memory_pos_tokens.append(torch.zeros_like(mem_proj))

        # Add object pointer tokens
        if object_ptrs is not None:
            for ptr in object_ptrs:
                # Split 256-dim pointer into 4 x 64-dim tokens
                ptr_tokens = ptr.view(B, self.num_object_ptr_tokens, -1)  # B 4 64
                ptr_proj = self.obj_ptr_proj(ptr_tokens)  # B 4 d_model
                memory_tokens.append(ptr_proj)
                # Object pointers don't have spatial position encoding
                memory_pos_tokens.append(torch.zeros_like(ptr_proj))

        if memory_tokens:
            all_memory = torch.cat(memory_tokens, dim=1)  # B M d_model
            all_memory_pos = torch.cat(memory_pos_tokens, dim=1)  # B M d_model
        else:
            # No memory (first frame or image mode)
            all_memory = torch.zeros(B, 1, C, device=curr_feats.device)
            all_memory_pos = torch.zeros(B, 1, C, device=curr_feats.device)

        # Apply memory attention layers
        tgt = curr_feats_flat
        for layer in self.layers:
            tgt = layer(
                tgt=tgt,
                memory=all_memory,
                tgt_pos=curr_pos_flat,
                memory_pos=all_memory_pos,
            )

        tgt = self.norm(tgt)

        # Reshape back to spatial format
        output = tgt.permute(0, 2, 1).view(B, C, H, W)
        return output
