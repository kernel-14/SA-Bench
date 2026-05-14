"""
FacT (Factor Tuning) for Vision Transformers.

FacT stacks the weight matrices from all Transformer layers into a 3D tensor
W_FacT ∈ R^{12M × D × D} and learns an additive residual ΔW_FacT via
tensor decomposition.

Two variants:
  FacT_TT (Tensor-Train):
    ΔW_FacT = s * Σ ×_2 U^T ×_3 V^T
    where U ∈ R^{D×r}, V ∈ R^{D×r}, Σ ∈ R^{12L×r×r}

  FacT_TK (Tucker):
    ΔW_FacT = s * A ×_1 B^T ×_2 U^T ×_3 V^T
    where U ∈ R^{D×r}, V ∈ R^{D×r}, B ∈ R^{12L×r}, A ∈ R^{r×r×r}

The 12 weight matrices per layer are: W_Q, W_K, W_V, W_O (MSA) and W_1, W_2 (MLP).
However, W_1 ∈ R^{D×4D} and W_2 ∈ R^{4D×D}, so FacT handles them by splitting
into 4 sub-matrices each, giving 4+4+4 = 12 matrices of size D×D per layer.

Reference: Jie & Deng, "FacT: Factor-Tuning for Lightweight Adaptation on
Vision Transformer", AAAI 2023.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block


class FacTTTLayer(nn.Module):
    """
    FacT Tensor-Train decomposition for a single Transformer layer.

    Stores the per-layer core tensor Σ_m ∈ R^{12×r×r} and references
    the shared U, V factors.
    """

    def __init__(self, block: Block, embed_dim: int, rank: int, scale_factor: float,
                 shared_u: nn.Parameter, shared_v: nn.Parameter, layer_idx: int,
                 sigma: nn.Parameter):
        super().__init__()
        self.block = block
        self.embed_dim = embed_dim
        self.rank = rank
        self.scale = scale_factor
        self.shared_u = shared_u  # [D, r]
        self.shared_v = shared_v  # [D, r]
        self.sigma = sigma        # [12L, r, r] - slice for this layer
        self.layer_idx = layer_idx

    def _get_delta_w(self) -> List[torch.Tensor]:
        """
        Compute ΔW for all 12 weight matrices in this layer.
        ΔW_FacT = s * Σ ×_2 U^T ×_3 V^T
        For layer m: ΔW_i = s * U @ Σ[12m+i] @ V^T  for i in 0..11
        """
        U = self.shared_u  # [D, r]
        V = self.shared_v  # [D, r]
        deltas = []
        for i in range(12):
            sigma_i = self.sigma[self.layer_idx * 12 + i]  # [r, r]
            # ΔW_i = s * U @ sigma_i @ V^T ∈ R^{D×D}
            delta = self.scale * (U @ sigma_i @ V.t())
            deltas.append(delta)
        return deltas

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deltas = self._get_delta_w()
        D = self.embed_dim
        D4 = D * 4

        # Unpack deltas: indices 0-3 for Q,K,V,O; 4-11 for MLP W1 (4 blocks), W2 (4 blocks)
        dW_Q, dW_K, dW_V, dW_O = deltas[0], deltas[1], deltas[2], deltas[3]
        # W1 ∈ R^{D×4D}: split into 4 D×D blocks
        dW1 = torch.cat([deltas[4], deltas[5], deltas[6], deltas[7]], dim=1)  # [D, 4D]
        # W2 ∈ R^{4D×D}: split into 4 D×D blocks
        dW2 = torch.cat([deltas[8], deltas[9], deltas[10], deltas[11]], dim=0)  # [4D, D]

        # MSA sub-layer with LoRA-style updates
        h = self.block.norm1(x)
        B, N, C = h.shape

        # Apply delta to QKV
        attn = self.block.attn
        qkv = attn.qkv(h)  # [B, N, 3D]
        # Add deltas: Q gets dW_Q, K gets dW_K, V gets dW_V
        delta_q = h @ dW_Q.t()
        delta_k = h @ dW_K.t()
        delta_v = h @ dW_V.t()
        delta_qkv = torch.cat([delta_q, delta_k, delta_v], dim=-1)
        qkv = qkv + delta_qkv

        qkv = qkv.reshape(B, N, 3, attn.num_heads, C // attn.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scale = attn.scale
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = attn.attn_drop(attn_weights)
        h4 = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)

        # Apply delta to output projection W_O
        h4 = attn.proj(h4) + h4 @ dW_O.t()
        h4 = attn.proj_drop(h4)
        h4 = self.block.ls1(h4)
        h4 = self.block.drop_path1(h4)
        x = x + h4

        # MLP sub-layer with delta updates
        h = self.block.norm2(x)
        mlp = self.block.mlp
        # FC1 with delta
        h7 = mlp.act(mlp.fc1(h) + h @ dW1.t())
        h7 = mlp.drop1(h7)
        # FC2 with delta
        h8 = mlp.fc2(h7) + h7 @ dW2.t()
        h8 = mlp.drop2(h8)
        h8 = self.block.ls2(h8)
        h8 = self.block.drop_path2(h8)
        x = x + h8
        return x


class FacTTKLayer(nn.Module):
    """
    FacT Tucker decomposition for a single Transformer layer.
    ΔW_FacT = s * A ×_1 B^T ×_2 U^T ×_3 V^T
    For layer m, matrix i: ΔW_i = s * U @ (A ×_1 b_i) @ V^T
    where b_i = B[12m+i] ∈ R^r and A ∈ R^{r×r×r}
    """

    def __init__(self, block: Block, embed_dim: int, rank: int, scale_factor: float,
                 shared_u: nn.Parameter, shared_v: nn.Parameter,
                 shared_a: nn.Parameter, b_matrix: nn.Parameter, layer_idx: int):
        super().__init__()
        self.block = block
        self.embed_dim = embed_dim
        self.rank = rank
        self.scale = scale_factor
        self.shared_u = shared_u  # [D, r]
        self.shared_v = shared_v  # [D, r]
        self.shared_a = shared_a  # [r, r, r]
        self.b_matrix = b_matrix  # [12L, r]
        self.layer_idx = layer_idx

    def _get_delta_w(self) -> List[torch.Tensor]:
        """
        ΔW_i = s * U @ (b_i^T @ A.reshape(r, r*r)).reshape(r, r) @ V^T
        Equivalently: ΔW_i = s * U @ einsum('i,ijk->jk', b_i, A) @ V^T
        """
        U = self.shared_u  # [D, r]
        V = self.shared_v  # [D, r]
        A = self.shared_a  # [r, r, r]
        deltas = []
        for i in range(12):
            b_i = self.b_matrix[self.layer_idx * 12 + i]  # [r]
            # Core matrix for this weight: einsum('i,ijk->jk', b_i, A)
            core = torch.einsum('i,ijk->jk', b_i, A)  # [r, r]
            delta = self.scale * (U @ core @ V.t())  # [D, D]
            deltas.append(delta)
        return deltas

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deltas = self._get_delta_w()
        D = self.embed_dim

        dW_Q, dW_K, dW_V, dW_O = deltas[0], deltas[1], deltas[2], deltas[3]
        dW1 = torch.cat([deltas[4], deltas[5], deltas[6], deltas[7]], dim=1)
        dW2 = torch.cat([deltas[8], deltas[9], deltas[10], deltas[11]], dim=0)

        # MSA sub-layer
        h = self.block.norm1(x)
        B, N, C = h.shape
        attn = self.block.attn
        qkv = attn.qkv(h)
        delta_qkv = torch.cat([h @ dW_Q.t(), h @ dW_K.t(), h @ dW_V.t()], dim=-1)
        qkv = qkv + delta_qkv

        qkv = qkv.reshape(B, N, 3, attn.num_heads, C // attn.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_weights = (q @ k.transpose(-2, -1)) * attn.scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = attn.attn_drop(attn_weights)
        h4 = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        h4 = attn.proj(h4) + h4 @ dW_O.t()
        h4 = attn.proj_drop(h4)
        h4 = self.block.ls1(h4)
        h4 = self.block.drop_path1(h4)
        x = x + h4

        # MLP sub-layer
        h = self.block.norm2(x)
        mlp = self.block.mlp
        h7 = mlp.act(mlp.fc1(h) + h @ dW1.t())
        h7 = mlp.drop1(h7)
        h8 = mlp.fc2(h7) + h7 @ dW2.t()
        h8 = mlp.drop2(h8)
        h8 = self.block.ls2(h8)
        h8 = self.block.drop_path2(h8)
        x = x + h8
        return x


def apply_fact_tt(
    backbone: nn.Module,
    embed_dim: int,
    num_layers: int,
    rank: int = 16,
    scale_factor: float = 1.0,
) -> None:
    """
    Apply FacT_TT to a timm ViT backbone.
    Shared factors: U ∈ R^{D×r}, V ∈ R^{D×r}
    Per-layer core: Σ ∈ R^{12L×r×r}
    """
    # Shared factors
    shared_u = nn.Parameter(torch.empty(embed_dim, rank))
    shared_v = nn.Parameter(torch.empty(embed_dim, rank))
    nn.init.kaiming_uniform_(shared_u, a=math.sqrt(5))
    nn.init.kaiming_uniform_(shared_v, a=math.sqrt(5))

    # Core tensor: initialized to small random values
    sigma = nn.Parameter(torch.zeros(12 * num_layers, rank, rank))
    nn.init.normal_(sigma, std=0.01)

    # Register as backbone parameters
    backbone.fact_u = shared_u
    backbone.fact_v = shared_v
    backbone.fact_sigma = sigma

    new_blocks = nn.ModuleList()
    for i, block in enumerate(backbone.blocks):
        new_blocks.append(
            FacTTTLayer(block, embed_dim, rank, scale_factor, shared_u, shared_v, i, sigma)
        )
    backbone.blocks = new_blocks


def apply_fact_tk(
    backbone: nn.Module,
    embed_dim: int,
    num_layers: int,
    rank: int = 32,
    scale_factor: float = 1.0,
) -> None:
    """
    Apply FacT_TK to a timm ViT backbone.
    Shared factors: U ∈ R^{D×r}, V ∈ R^{D×r}, A ∈ R^{r×r×r}
    Per-layer: B ∈ R^{12L×r}
    """
    shared_u = nn.Parameter(torch.empty(embed_dim, rank))
    shared_v = nn.Parameter(torch.empty(embed_dim, rank))
    shared_a = nn.Parameter(torch.zeros(rank, rank, rank))
    b_matrix = nn.Parameter(torch.zeros(12 * num_layers, rank))

    nn.init.kaiming_uniform_(shared_u, a=math.sqrt(5))
    nn.init.kaiming_uniform_(shared_v, a=math.sqrt(5))
    nn.init.normal_(shared_a, std=0.01)
    nn.init.normal_(b_matrix, std=0.01)

    backbone.fact_u = shared_u
    backbone.fact_v = shared_v
    backbone.fact_a = shared_a
    backbone.fact_b = b_matrix

    new_blocks = nn.ModuleList()
    for i, block in enumerate(backbone.blocks):
        new_blocks.append(
            FacTTKLayer(
                block, embed_dim, rank, scale_factor,
                shared_u, shared_v, shared_a, b_matrix, i
            )
        )
    backbone.blocks = new_blocks
