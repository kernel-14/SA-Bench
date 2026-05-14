"""
Direct selective tuning PEFT methods:
  - BitFit: fine-tune only bias terms
  - LayerNorm: fine-tune only LayerNorm parameters
  - DiffFit: BitFit + LayerNorm + learnable scale factors after MSA and MLP
  - SSF: Scale & Shift Features at intermediate positions

References:
  Zaken et al., "BitFit", ACL 2022.
  Basu et al., "Strong Baselines for Parameter-Efficient Few-Shot Fine-Tuning", AAAI 2024.
  Xie et al., "DiffFit", arXiv 2023.
  Lian et al., "SSF", NeurIPS 2022.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


# ---------------------------------------------------------------------------
# BitFit
# ---------------------------------------------------------------------------

def apply_bitfit(backbone: nn.Module) -> None:
    """
    BitFit: unfreeze only bias terms in the backbone.
    Covers biases in:
      - patch embedding projection
      - Q/K/V projections in each MSA block
      - FC_attn (output projection) in each MSA block
      - FC1 and FC2 in each MLP block
      - LayerNorm blocks (both weight and bias, as LN params are bias-like)
    """
    for name, param in backbone.named_parameters():
        if "bias" in name:
            param.requires_grad = True
        # Also unfreeze patch embedding bias
        elif "patch_embed" in name and "bias" in name:
            param.requires_grad = True


# ---------------------------------------------------------------------------
# LayerNorm tuning
# ---------------------------------------------------------------------------

def apply_layernorm_tuning(backbone: nn.Module) -> None:
    """
    LayerNorm: unfreeze only the LayerNorm parameters (weight and bias).
    Each Transformer layer has two LN blocks: norm1 (before MSA) and norm2 (before MLP).
    """
    for name, param in backbone.named_parameters():
        if "norm" in name:
            param.requires_grad = True


# ---------------------------------------------------------------------------
# DiffFit
# ---------------------------------------------------------------------------

class DiffFitBlock(nn.Module):
    """
    DiffFit block: standard Transformer block with learnable scale factors
    γ1 and γ2 applied after MSA and MLP residuals respectively.
      h5 = γ1 * h5
      h9 = γ2 * h9
    """

    def __init__(self, block: Block, embed_dim: int):
        super().__init__()
        self.block = block
        self.gamma1 = nn.Parameter(torch.ones(embed_dim))
        self.gamma2 = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MSA sub-layer
        h = self.block.norm1(x)
        h = self.block.attn(h)
        h = self.block.ls1(h)
        h = self.block.drop_path1(h)
        x = self.gamma1 * (x + h)

        # MLP sub-layer
        h = self.block.norm2(x)
        h = self.block.mlp(h)
        h = self.block.ls2(h)
        h = self.block.drop_path2(h)
        x = self.gamma2 * (x + h)
        return x


def apply_difffit(backbone: nn.Module) -> None:
    """
    DiffFit: BitFit + LayerNorm + learnable scale factors γ after MSA and MLP.
    Replaces each block with DiffFitBlock and unfreezes bias + LN params.
    """
    embed_dim = backbone.embed_dim
    new_blocks = nn.ModuleList()
    for block in backbone.blocks:
        new_blocks.append(DiffFitBlock(block, embed_dim))
    backbone.blocks = new_blocks

    # Unfreeze bias terms and LayerNorm parameters
    for name, param in backbone.named_parameters():
        if "bias" in name or "norm" in name:
            param.requires_grad = True
        # gamma parameters in DiffFitBlock are already requires_grad=True


# ---------------------------------------------------------------------------
# SSF (Scale & Shift Features)
# ---------------------------------------------------------------------------

class SSFModule(nn.Module):
    """
    SSF module: element-wise scale and shift.
      SSF(h) = w ⊙ h + b
    where w, b ∈ R^D are learnable parameters.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.scale * h + self.shift


class SSFBlock(nn.Module):
    """
    SSF applied to intermediate features h2, h3, h5, h7, h8, h9.

    From Table 6:
      h2 = SSF2(h2)   [after LN1]
      h3 = SSF3(h3)   [after QKV projection, dim=3D]
      h5 = SSF5(h5)   [after MSA residual]
      h7 = SSF7(h7)   [after FC1 in MLP, dim=4D]
      h8 = SSF8(h8)   [after FC2 in MLP, dim=D]
      h9 = SSF9(h9)   [after MLP residual]
    """

    def __init__(self, block: Block, embed_dim: int):
        super().__init__()
        self.block = block
        self.embed_dim = embed_dim
        mlp_hidden_dim = int(embed_dim * 4)

        self.ssf_h2 = SSFModule(embed_dim)
        self.ssf_h3 = SSFModule(3 * embed_dim)  # Q, K, V concatenated
        self.ssf_h5 = SSFModule(embed_dim)
        self.ssf_h7 = SSFModule(mlp_hidden_dim)
        self.ssf_h8 = SSFModule(embed_dim)
        self.ssf_h9 = SSFModule(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h1 = x
        # h2 = LN1(x), apply SSF
        h2 = self.ssf_h2(self.block.norm1(x))

        # h3 = QKV projections, apply SSF to concatenated QKV
        attn = self.block.attn
        B, N, C = h2.shape
        qkv = attn.qkv(h2)  # [B, N, 3*D]
        qkv = self.ssf_h3(qkv)
        qkv = qkv.reshape(B, N, 3, attn.num_heads, C // attn.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Attention computation
        scale = attn.scale
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = attn.attn_drop(attn_weights)
        h4 = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        h4 = attn.proj(h4)
        h4 = attn.proj_drop(h4)

        # h5 = x + ls1(drop_path(h4)), apply SSF
        h4 = self.block.ls1(h4)
        h5 = x + self.block.drop_path1(h4)
        h5 = self.ssf_h5(h5)

        # h6 = LN2(h5)
        h6 = self.block.norm2(h5)

        # MLP forward with SSF at h7 and h8
        mlp = self.block.mlp
        h7 = mlp.act(mlp.fc1(h6))
        h7 = self.ssf_h7(h7)
        h7 = mlp.drop1(h7)
        h8 = mlp.fc2(h7)
        h8 = self.ssf_h8(h8)
        h8 = mlp.drop2(h8)

        # h9 = h5 + ls2(drop_path(h8)), apply SSF
        h8 = self.block.ls2(h8)
        h9 = h5 + self.block.drop_path2(h8)
        h9 = self.ssf_h9(h9)
        return h9


def apply_ssf(backbone: nn.Module, embed_dim: int) -> None:
    """Replace all transformer blocks with SSFBlock."""
    new_blocks = nn.ModuleList()
    for block in backbone.blocks:
        new_blocks.append(SSFBlock(block, embed_dim))
    backbone.blocks = new_blocks
