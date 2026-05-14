"""
Memory attention module for SAM 2.

Conditions the current frame's image features on the memory bank
(spatial memories from past frames + object pointer vectors).

Architecture (Section 4, Appendix D.1):
  - L=4 transformer blocks (default).
  - Each block: self-attention → cross-attention to memory bank → MLP.
  - Self-attention and cross-attention use 2d-RoPE positional embeddings.
  - Object pointer tokens are excluded from RoPE (no spatial correspondence).
  - Sinusoidal absolute positional embeddings added to spatial memory features.
  - Temporal position embeddings added to recent-frame memories (not prompted frames).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import MLP, RoPE2D


class MemoryAttentionLayer(nn.Module):
    """
    Single memory attention block:
      1. Self-attention on current frame features (with 2d-RoPE).
      2. Cross-attention to memory bank (spatial + pointer tokens, with 2d-RoPE for spatial).
      3. MLP.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        memory_dim: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Self-attention
        self.self_attn_q = nn.Linear(d_model, d_model)
        self.self_attn_k = nn.Linear(d_model, d_model)
        self.self_attn_v = nn.Linear(d_model, d_model)
        self.self_attn_out = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention to memory bank
        # Memory features are projected from memory_dim to d_model in MemoryAttention
        self.cross_attn_q = nn.Linear(d_model, d_model)
        self.cross_attn_k = nn.Linear(d_model, d_model)
        self.cross_attn_v = nn.Linear(d_model, d_model)
        self.cross_attn_out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # MLP
        self.mlp = MLP(d_model, dim_feedforward, d_model, num_layers=2, activation=nn.ReLU)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = dropout
        self.rope = RoPE2D(self.head_dim)

    def _split_heads(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        return x.reshape(B, N, self.nhead, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        B, H, N, D = x.shape
        return x.transpose(1, 2).reshape(B, N, H * D)

    def _attn(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        scale = math.sqrt(q.shape[-1])
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)
        return torch.matmul(attn, v)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        x_hw: Tuple[int, int],
        memory_hw: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """
        Args:
            x:         (B, H*W, d_model) — current frame features (flattened)
            memory:    (B, M_total, d_model) — projected memory tokens (already pos-encoded)
            x_hw:      (H, W) spatial dims of current frame
            memory_hw: unused, kept for API compatibility

        Returns:
            x: (B, H*W, d_model)
        """
        H, W = x_hw

        # --- Self-attention with 2d-RoPE ---
        q = self._split_heads(self.self_attn_q(x))
        k = self._split_heads(self.self_attn_k(x))
        v = self._split_heads(self.self_attn_v(x))
        q, k = self.rope(q, k, H, W)
        sa_out = self._attn(q, k, v)
        sa_out = self._merge_heads(sa_out)
        sa_out = self.self_attn_out(sa_out)
        x = self.norm1(x + sa_out)

        # --- Cross-attention to memory bank ---
        q = self._split_heads(self.cross_attn_q(x))
        k_mem = self._split_heads(self.cross_attn_k(memory))
        v_mem = self._split_heads(self.cross_attn_v(memory))
        # Apply RoPE to query (current frame spatial tokens)
        q_rope, _ = self.rope(q, q, H, W)
        ca_out = self._attn(q_rope, k_mem, v_mem)
        ca_out = self._merge_heads(ca_out)
        ca_out = self.cross_attn_out(ca_out)
        x = self.norm2(x + ca_out)

        # --- MLP ---
        x = self.norm3(x + self.mlp(x))
        return x


class MemoryAttention(nn.Module):
    """
    Stack of L memory attention layers.

    Conditions current frame features on:
      - Spatial memory features from N recent frames (with temporal pos embeddings)
      - Spatial memory features from M prompted frames (no temporal pos embeddings)
      - Object pointer vectors from all stored frames

    Memory bank layout (managed externally in SAM2 model):
      recent_memories:   list of (B, memory_dim, H_m, W_m) tensors
      prompted_memories: list of (B, memory_dim, H_m, W_m) tensors
      object_pointers:   list of (B, 4, 64) tensors (256-dim split into 4×64)
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        memory_dim: int = 64,
        max_recent_frames: int = 6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.memory_dim = memory_dim
        self.max_recent_frames = max_recent_frames

        self.layers = nn.ModuleList([
            MemoryAttentionLayer(d_model, nhead, dim_feedforward, dropout, memory_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Temporal position embeddings for recent frames (not prompted frames)
        self.temporal_pos_embed = nn.Embedding(max_recent_frames, memory_dim)

        # Project memory features to d_model for cross-attention
        self.memory_proj = nn.Linear(memory_dim, d_model)

        # Project object pointer tokens (64-dim) to d_model
        self.pointer_proj = nn.Linear(64, d_model)

    def _prepare_memory_tokens(
        self,
        recent_memories: List[Tensor],
        prompted_memories: List[Tensor],
        object_pointers: List[Tensor],
    ) -> Tensor:
        """
        Flatten and concatenate all memory tokens into a single sequence.
        Adds sinusoidal spatial positional encodings to spatial memory features.
        Adds temporal position embeddings to recent-frame memories only.

        Returns: (B, M_total, d_model)
        """
        all_tokens: List[Tensor] = []

        # Recent frame memories with temporal position embeddings
        for t, mem in enumerate(recent_memories):
            B, C, H, W = mem.shape
            # Add temporal position embedding
            t_idx = min(t, self.max_recent_frames - 1)
            t_emb = self.temporal_pos_embed.weight[t_idx]  # (memory_dim,)
            mem = mem + t_emb[None, :, None, None]

            # Project to d_model and add sinusoidal spatial pos encoding
            mem_flat = mem.flatten(2).permute(0, 2, 1)  # (B, H*W, memory_dim)
            mem_proj = self.memory_proj(mem_flat)        # (B, H*W, d_model)

            # Build sinusoidal pos encoding for this spatial size
            pos = self._build_spatial_pos(H, W, mem.device, mem.dtype)  # (H*W, d_model)
            mem_proj = mem_proj + pos.unsqueeze(0)

            all_tokens.append(mem_proj)

        # Prompted frame memories (no temporal pos embeddings)
        for mem in prompted_memories:
            B, C, H, W = mem.shape
            mem_flat = mem.flatten(2).permute(0, 2, 1)
            mem_proj = self.memory_proj(mem_flat)
            pos = self._build_spatial_pos(H, W, mem.device, mem.dtype)
            mem_proj = mem_proj + pos.unsqueeze(0)
            all_tokens.append(mem_proj)

        # Object pointer vectors (no RoPE, no spatial pos)
        for ptr in object_pointers:
            # ptr: (B, num_pointer_tokens, token_dim)
            ptr_proj = self.pointer_proj(ptr)  # (B, num_pointer_tokens, d_model)
            all_tokens.append(ptr_proj)

        if not all_tokens:
            return torch.zeros(1, 0, self.d_model)

        return torch.cat(all_tokens, dim=1)  # (B, M_total, d_model)

    def _build_spatial_pos(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Build sinusoidal 2D positional encoding for a H×W spatial grid. Returns (H*W, d_model)."""
        half = self.d_model // 2
        dim_t = torch.arange(half // 2, dtype=torch.float32, device=device)
        dim_t = 10000.0 ** (2 * dim_t / (half // 2))

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing="ij",
        )
        pos_x = grid_x.flatten()[:, None] / dim_t[None, :]
        pos_y = grid_y.flatten()[:, None] / dim_t[None, :]

        emb_x = torch.cat([pos_x.sin(), pos_x.cos()], dim=-1)  # (H*W, half)
        emb_y = torch.cat([pos_y.sin(), pos_y.cos()], dim=-1)
        emb = torch.cat([emb_x, emb_y], dim=-1)                # (H*W, d_model)
        return emb.to(dtype)

    def forward(
        self,
        current_features: Tensor,
        recent_memories: List[Tensor],
        prompted_memories: List[Tensor],
        object_pointers: List[Tensor],
    ) -> Tensor:
        """
        Args:
            current_features: (B, d_model, H, W) — unconditioned image embedding
            recent_memories:  list of (B, memory_dim, H_m, W_m)
            prompted_memories: list of (B, memory_dim, H_m, W_m)
            object_pointers:  list of (B, 4, 64)

        Returns:
            conditioned_features: (B, d_model, H, W)
        """
        B, C, H, W = current_features.shape

        # Add absolute sinusoidal positional embedding to current features
        pos = self._build_spatial_pos(H, W, current_features.device, current_features.dtype)
        x = current_features.flatten(2).permute(0, 2, 1)  # (B, H*W, d_model)
        x = x + pos.unsqueeze(0)

        # Prepare memory tokens
        has_memory = bool(recent_memories or prompted_memories or object_pointers)
        if has_memory:
            memory_tokens = self._prepare_memory_tokens(
                recent_memories, prompted_memories, object_pointers
            )
            if memory_tokens.shape[0] == 1 and B > 1:
                memory_tokens = memory_tokens.expand(B, -1, -1)
        else:
            memory_tokens = torch.zeros(B, 0, self.d_model, device=x.device, dtype=x.dtype)

        # Run memory attention layers
        for layer in self.layers:
            if memory_tokens.shape[1] > 0:
                x = layer(x, memory_tokens, x_hw=(H, W))
            else:
                # No memory: self-attention only (image mode, behaves like SAM)
                x = layer(x, x, x_hw=(H, W))

        x = self.norm(x)
        return x.permute(0, 2, 1).view(B, C, H, W)
