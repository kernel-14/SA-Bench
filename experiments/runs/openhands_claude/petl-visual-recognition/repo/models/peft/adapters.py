"""
Adapter-based PEFT methods:
  - Houlsby Adapter (Houl. Adapter): two adapters per layer, after MSA and MLP
  - Pfeiffer Adapter (Pfeif. Adapter): one adapter per layer, after MLP only
  - AdaptFormer: adapter parallel to MLP
  - ConvPass: convolutional adapter parallel to MSA and MLP
  - RepAdapter: linear adapter with group-wise transformation

All adapters follow the bottleneck structure:
  Adapter(h) = s * W_up * σ(W_down * h) + h
where r << D is the bottleneck dimension and s is a scaling factor.

References:
  Houlsby et al., "Parameter-Efficient Transfer Learning for NLP", ICML 2019.
  Pfeiffer et al., "AdapterFusion", EACL 2021.
  Chen et al., "AdaptFormer", NeurIPS 2022.
  Jie & Deng, "ConvPass", arXiv 2022.
  Luo et al., "RepAdapter", arXiv 2023.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, Attention, Mlp


class Adapter(nn.Module):
    """
    Standard bottleneck adapter:
      Adapter(h) = s * W_up * σ(W_down * h) + h

    Args:
        embed_dim: input/output feature dimension D
        bottleneck_dim: bottleneck dimension r (r << D)
        scale_factor: scalar multiplier s applied to adapter output
        act_layer: nonlinear activation σ
    """

    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int,
        scale_factor: float = 1.0,
        act_layer: nn.Module = nn.GELU(),
    ):
        super().__init__()
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.act = act_layer
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        self.scale = scale_factor

        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.scale * self.up_proj(self.act(self.down_proj(h))) + h


class HoulAdapterBlock(nn.Module):
    """
    Houlsby Adapter: inserts two adapters per Transformer layer.
      h5 = Adapter1(h5)   [after MSA + residual]
      h9 = Adapter2(h9)   [after MLP + residual]
    """

    def __init__(
        self,
        block: Block,
        embed_dim: int,
        bottleneck_dim: int,
        scale_factor: float = 1.0,
    ):
        super().__init__()
        self.block = block
        self.adapter_attn = Adapter(embed_dim, bottleneck_dim, scale_factor)
        self.adapter_mlp = Adapter(embed_dim, bottleneck_dim, scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MSA sub-layer with drop_path and optional LayerScale
        h = self.block.norm1(x)
        h = self.block.attn(h)
        h = self.block.ls1(h)
        h = self.block.drop_path1(h)
        x = x + h
        # Adapter after MSA (h5)
        x = self.adapter_attn(x)

        # MLP sub-layer with drop_path and optional LayerScale
        h = self.block.norm2(x)
        h = self.block.mlp(h)
        h = self.block.ls2(h)
        h = self.block.drop_path2(h)
        x = x + h
        # Adapter after MLP (h9)
        x = self.adapter_mlp(x)
        return x


class PfeifAdapterBlock(nn.Module):
    """
    Pfeiffer Adapter: inserts one adapter per Transformer layer, after MLP only.
      h9 = Adapter(h9)
    """

    def __init__(
        self,
        block: Block,
        embed_dim: int,
        bottleneck_dim: int,
        scale_factor: float = 1.0,
    ):
        super().__init__()
        self.block = block
        self.adapter_mlp = Adapter(embed_dim, bottleneck_dim, scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard MSA sub-layer
        h = self.block.norm1(x)
        h = self.block.attn(h)
        h = self.block.ls1(h)
        h = self.block.drop_path1(h)
        x = x + h

        # MLP sub-layer
        h = self.block.norm2(x)
        h = self.block.mlp(h)
        h = self.block.ls2(h)
        h = self.block.drop_path2(h)
        x = x + h
        # Adapter after MLP (h9)
        x = self.adapter_mlp(x)
        return x


class AdaptFormerBlock(nn.Module):
    """
    AdaptFormer: adapter in parallel with MLP block.
      h9 = h9 + Adapter(h7)
    where h7 is the input to the MLP (after LN2).
    """

    def __init__(
        self,
        block: Block,
        embed_dim: int,
        bottleneck_dim: int,
        scale_factor: float = 0.1,
    ):
        super().__init__()
        self.block = block
        self.adapter = Adapter(embed_dim, bottleneck_dim, scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard MSA sub-layer
        h = self.block.norm1(x)
        h = self.block.attn(h)
        h = self.block.ls1(h)
        h = self.block.drop_path1(h)
        x = x + h

        # MLP sub-layer (h6 = norm2(x), h7 = mlp input)
        h6 = self.block.norm2(x)
        # Parallel: MLP output + Adapter(h7)
        mlp_out = self.block.mlp(h6)
        adapter_out = self.adapter(h6)
        mlp_out = self.block.ls2(mlp_out)
        h = self.block.drop_path2(mlp_out)
        x = x + h + adapter_out
        return x


class ConvPassAdapter(nn.Module):
    """
    ConvPass adapter: convolutional bottleneck module.
      Convpass(h) = s * W_up * σ(Conv2d(σ(W_down * h)))

    The 1D token sequence is reshaped to 2D spatial grid for convolution.
    """

    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int,
        scale_factor: float = 1.0,
        kernel_size: int = 3,
        xavier_init: bool = True,
    ):
        super().__init__()
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.conv = nn.Conv2d(
            bottleneck_dim, bottleneck_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=1,
        )
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        self.act = nn.GELU()
        self.scale = scale_factor

        if xavier_init:
            nn.init.xavier_uniform_(self.down_proj.weight)
            nn.init.xavier_uniform_(self.up_proj.weight)
        else:
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, 1+N, D] where N = H*W/P^2
        B, L, D = h.shape
        # Separate CLS token from patch tokens
        cls = h[:, :1, :]
        patches = h[:, 1:, :]  # [B, N, D]

        N = patches.shape[1]
        grid_size = int(math.sqrt(N))

        # Down projection
        patches = self.act(self.down_proj(patches))  # [B, N, r]

        # Reshape to 2D for convolution
        patches_2d = patches.reshape(B, grid_size, grid_size, -1)
        patches_2d = patches_2d.permute(0, 3, 1, 2)  # [B, r, H, W]
        patches_2d = self.act(self.conv(patches_2d))
        patches_2d = patches_2d.permute(0, 2, 3, 1).reshape(B, N, -1)  # [B, N, r]

        # Up projection
        patches = self.up_proj(patches_2d)  # [B, N, D]

        # Reconstruct with CLS (no conv on CLS token)
        out = torch.cat([cls, patches], dim=1)
        return self.scale * out


class ConvPassBlock(nn.Module):
    """
    ConvPass: convolutional adapter parallel to MSA and MLP.
      h5 = Convpass1(h2) + h5
      h9 = Convpass2(h7) + h9
    """

    def __init__(
        self,
        block: Block,
        embed_dim: int,
        bottleneck_dim: int,
        scale_factor: float = 1.0,
        kernel_size: int = 3,
        xavier_init: bool = True,
    ):
        super().__init__()
        self.block = block
        self.convpass_attn = ConvPassAdapter(
            embed_dim, bottleneck_dim, scale_factor, kernel_size, xavier_init
        )
        self.convpass_mlp = ConvPassAdapter(
            embed_dim, bottleneck_dim, scale_factor, kernel_size, xavier_init
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h1 = x, h2 = norm1(x)
        h2 = self.block.norm1(x)
        # MSA output
        attn_out = self.block.attn(h2)
        attn_out = self.block.ls1(attn_out)
        attn_out = self.block.drop_path1(attn_out)
        # h5 = attn_out + x + Convpass1(h2)
        x = x + attn_out + self.convpass_attn(h2)

        # h6 = norm2(x), h7 = mlp input
        h6 = self.block.norm2(x)
        mlp_out = self.block.mlp(h6)
        mlp_out = self.block.ls2(mlp_out)
        mlp_out = self.block.drop_path2(mlp_out)
        # h9 = mlp_out + x + Convpass2(h6)
        x = x + mlp_out + self.convpass_mlp(h6)
        return x


class RepAdapterModule(nn.Module):
    """
    RepAdapter: linear adapter with group-wise transformation.
      RepAdapter(h) = s * φ_up(φ_down(h)) + h
      φ_down(h) = W_down @ h
      φ_up(h̃) = [W_g1 @ h̃_g1, ..., W_gG @ h̃_gG]

    After training, can be re-parameterized into the original weights.
    """

    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int,
        scale_factor: float = 1.0,
        num_groups: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.scale = scale_factor
        self.num_groups = num_groups

        assert bottleneck_dim % num_groups == 0
        assert embed_dim % num_groups == 0

        self.down_proj = nn.Linear(embed_dim, bottleneck_dim, bias=False)
        # Group-wise up projection: G separate linear layers
        self.up_projs = nn.ModuleList([
            nn.Linear(bottleneck_dim // num_groups, embed_dim // num_groups, bias=False)
            for _ in range(num_groups)
        ])
        self.bias = nn.Parameter(torch.zeros(embed_dim))

        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        for up in self.up_projs:
            nn.init.zeros_(up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, L, D]
        h_down = self.down_proj(h)  # [B, L, r]
        # Split into groups
        chunks = h_down.chunk(self.num_groups, dim=-1)  # G x [B, L, r/G]
        out_chunks = [self.up_projs[i](chunks[i]) for i in range(self.num_groups)]
        h_up = torch.cat(out_chunks, dim=-1)  # [B, L, D]
        return self.scale * (h_up + self.bias) + h


class RepAdapterBlock(nn.Module):
    """
    RepAdapter: sequential linear adapters after MSA and MLP.
      h2 = RepAdapter1(h2)  [applied to LN output before MSA]
      h7 = RepAdapter2(h7)  [applied to LN output before MLP]

    Note: From Table 6, RepAdapter is placed at h2 and h7 (before MSA/MLP).
    """

    def __init__(
        self,
        block: Block,
        embed_dim: int,
        bottleneck_dim: int,
        scale_factor: float = 1.0,
        num_groups: int = 2,
    ):
        super().__init__()
        self.block = block
        self.rep_adapter_attn = RepAdapterModule(
            embed_dim, bottleneck_dim, scale_factor, num_groups
        )
        self.rep_adapter_mlp = RepAdapterModule(
            embed_dim, bottleneck_dim, scale_factor, num_groups
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h2 = norm1(x), apply RepAdapter before MSA
        h2 = self.block.norm1(x)
        h2 = self.rep_adapter_attn(h2)
        attn_out = self.block.attn(h2)
        attn_out = self.block.ls1(attn_out)
        attn_out = self.block.drop_path1(attn_out)
        x = x + attn_out

        # h7 = norm2(x), apply RepAdapter before MLP
        h6 = self.block.norm2(x)
        h6 = self.rep_adapter_mlp(h6)
        mlp_out = self.block.mlp(h6)
        mlp_out = self.block.ls2(mlp_out)
        mlp_out = self.block.drop_path2(mlp_out)
        x = x + mlp_out
        return x


def apply_houl_adapter(
    backbone: nn.Module,
    embed_dim: int,
    bottleneck_dim: int = 16,
    scale_factor: float = 0.1,
) -> None:
    """Replace all transformer blocks with HoulAdapterBlock."""
    new_blocks = nn.ModuleList()
    for block in backbone.blocks:
        new_blocks.append(
            HoulAdapterBlock(block, embed_dim, bottleneck_dim, scale_factor)
        )
    backbone.blocks = new_blocks


def apply_pfeif_adapter(
    backbone: nn.Module,
    embed_dim: int,
    bottleneck_dim: int = 16,
    scale_factor: float = 0.1,
) -> None:
    """Replace all transformer blocks with PfeifAdapterBlock."""
    new_blocks = nn.ModuleList()
    for block in backbone.blocks:
        new_blocks.append(
            PfeifAdapterBlock(block, embed_dim, bottleneck_dim, scale_factor)
        )
    backbone.blocks = new_blocks


def apply_adaptformer(
    backbone: nn.Module,
    embed_dim: int,
    bottleneck_dim: int = 16,
    scale_factor: float = 0.1,
) -> None:
    """Replace all transformer blocks with AdaptFormerBlock."""
    new_blocks = nn.ModuleList()
    for block in backbone.blocks:
        new_blocks.append(
            AdaptFormerBlock(block, embed_dim, bottleneck_dim, scale_factor)
        )
    backbone.blocks = new_blocks


def apply_convpass(
    backbone: nn.Module,
    embed_dim: int,
    bottleneck_dim: int = 8,
    scale_factor: float = 1.0,
    kernel_size: int = 3,
    xavier_init: bool = True,
) -> None:
    """Replace all transformer blocks with ConvPassBlock."""
    new_blocks = nn.ModuleList()
    for block in backbone.blocks:
        new_blocks.append(
            ConvPassBlock(
                block, embed_dim, bottleneck_dim, scale_factor, kernel_size, xavier_init
            )
        )
    backbone.blocks = new_blocks


def apply_repadapter(
    backbone: nn.Module,
    embed_dim: int,
    bottleneck_dim: int = 16,
    scale_factor: float = 1.0,
    num_groups: int = 2,
) -> None:
    """Replace all transformer blocks with RepAdapterBlock."""
    new_blocks = nn.ModuleList()
    for block in backbone.blocks:
        new_blocks.append(
            RepAdapterBlock(block, embed_dim, bottleneck_dim, scale_factor, num_groups)
        )
    backbone.blocks = new_blocks
