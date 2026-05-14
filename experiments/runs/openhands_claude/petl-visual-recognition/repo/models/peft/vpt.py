"""
Visual Prompt Tuning (VPT) - Shallow and Deep variants.

Reference: Jia et al., "Visual Prompt Tuning", ECCV 2022.

VPT-Shallow: prepends l learnable prompt tokens to the input of the first
  Transformer layer only. The prompt outputs are passed to subsequent layers.

VPT-Deep: prepends l learnable prompt tokens to the input of every
  Transformer layer. The prompt outputs are discarded at the end of each layer.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class VPTShallowBlock(nn.Module):
    """
    Wraps a timm Block to pass through prompt tokens from the first layer.
    The prompts are prepended to the sequence and their outputs are kept.
    """

    def __init__(self, block: Block):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VPTDeepBlock(nn.Module):
    """
    Wraps a timm Block for VPT-Deep: prepends fresh prompts at each layer
    and discards the prompt outputs, keeping only the patch + CLS tokens.

    The prompts for this layer are stored as a parameter on this module.
    """

    def __init__(self, block: Block, num_prompts: int, embed_dim: int):
        super().__init__()
        self.block = block
        self.num_prompts = num_prompts
        self.prompts = nn.Parameter(torch.zeros(1, num_prompts, embed_dim))
        nn.init.trunc_normal_(self.prompts, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # x: [B, 1+N, D] (CLS + patch tokens)
        prompts = self.prompts.expand(B, -1, -1)
        # Prepend prompts: [B, num_prompts + 1 + N, D]
        x_with_prompts = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)
        out = self.block(x_with_prompts)
        # Discard prompt outputs, keep CLS + patch tokens
        return torch.cat([out[:, :1], out[:, 1 + self.num_prompts:]], dim=1)


class VPTShallowViT(nn.Module):
    """
    VPT-Shallow: adds l learnable prompts to the input of the first layer.
    The prompt outputs are propagated through all subsequent layers.
    """

    def __init__(self, backbone: nn.Module, num_prompts: int):
        super().__init__()
        self.backbone = backbone
        self.num_prompts = num_prompts
        embed_dim = backbone.embed_dim
        self.prompts = nn.Parameter(torch.zeros(1, num_prompts, embed_dim))
        nn.init.trunc_normal_(self.prompts, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Patch embedding
        x = self.backbone.patch_embed(x)
        # CLS token
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        # Prepend prompts to the first layer input
        prompts = self.prompts.expand(B, -1, -1)
        x = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)

        # Forward through all blocks (prompts propagate naturally)
        for block in self.backbone.blocks:
            x = block(x)

        x = self.backbone.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        # CLS token is at position 0
        return x[:, 0]


def apply_vpt_shallow(backbone: nn.Module, num_prompts: int = 50) -> None:
    """
    Apply VPT-Shallow to a timm ViT backbone in-place.
    Adds a learnable prompt parameter and patches forward_features.

    timm's _pos_embed already handles CLS token prepending and position
    embedding addition, so we call it directly and then insert prompts
    between the CLS token and patch tokens.
    """
    embed_dim = backbone.embed_dim
    backbone.vpt_prompts = nn.Parameter(torch.zeros(1, num_prompts, embed_dim))
    nn.init.trunc_normal_(backbone.vpt_prompts, std=0.02)
    backbone.vpt_num_prompts = num_prompts

    def vpt_shallow_forward_features(x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # patch_embed: [B, N, D]
        x = backbone.patch_embed(x)
        # _pos_embed: prepends CLS token and adds position embeddings → [B, 1+N, D]
        x = backbone._pos_embed(x)
        x = backbone.patch_drop(x)
        x = backbone.norm_pre(x)

        # Insert prompts between CLS token and patch tokens
        prompts = backbone.vpt_prompts.expand(B, -1, -1)
        x = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)  # [B, 1+P+N, D]

        x = backbone.blocks(x)
        x = backbone.norm(x)
        return x

    backbone.forward_features = vpt_shallow_forward_features


def apply_vpt_deep(backbone: nn.Module, num_prompts: int = 10) -> None:
    """
    Apply VPT-Deep to a timm ViT backbone in-place.
    Replaces each transformer block with a VPTDeepBlock.
    """
    embed_dim = backbone.embed_dim
    new_blocks = nn.ModuleList()
    for block in backbone.blocks:
        new_blocks.append(VPTDeepBlock(block, num_prompts, embed_dim))
    backbone.blocks = new_blocks

    def vpt_deep_forward_features(x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = backbone.patch_embed(x)
        # _pos_embed prepends CLS token → [B, 1+N, D]
        x = backbone._pos_embed(x)
        x = backbone.patch_drop(x)
        x = backbone.norm_pre(x)
        # Each VPTDeepBlock handles prompt injection and removal
        x = backbone.blocks(x)
        x = backbone.norm(x)
        return x

    backbone.forward_features = vpt_deep_forward_features
