"""
LoRA (Low-Rank Adaptation) for Vision Transformers.

LoRA parameterizes weight updates via low-rank decomposition:
  W + ΔW = W + W_down @ W_up
where W_down ∈ R^{r×D}, W_up ∈ R^{D×r}, r << D.

Applied to Q and V projection weights in each MSA block:
  h3 = LoRA(h2) + h3
  LoRA(h2) = [W_down^Q @ W_up^Q @ h2, 0, W_down^V @ W_up^V @ h2]

W_up is initialized to zero so ΔW = 0 at the start of training.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, Attention


class LoRAAttention(nn.Module):
    """
    Attention block with LoRA applied to Q and V projections.

    The QKV projection is split: Q and V get low-rank updates,
    K is unchanged.
    """

    def __init__(self, attn: Attention, embed_dim: int, rank: int):
        super().__init__()
        self.attn = attn
        self.embed_dim = embed_dim
        self.rank = rank

        # LoRA for Q: W_down^Q ∈ R^{r×D}, W_up^Q ∈ R^{D×r}
        self.lora_q_down = nn.Linear(embed_dim, rank, bias=False)
        self.lora_q_up = nn.Linear(rank, embed_dim, bias=False)

        # LoRA for V: W_down^V ∈ R^{r×D}, W_up^V ∈ R^{D×r}
        self.lora_v_down = nn.Linear(embed_dim, rank, bias=False)
        self.lora_v_up = nn.Linear(rank, embed_dim, bias=False)

        # Initialize: W_down with kaiming, W_up with zeros (so ΔW=0 initially)
        nn.init.kaiming_uniform_(self.lora_q_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_up.weight)
        nn.init.kaiming_uniform_(self.lora_v_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_v_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # Original QKV projection
        qkv = self.attn.qkv(x)  # [B, N, 3*D]
        qkv = qkv.reshape(B, N, 3, self.attn.num_heads, C // self.attn.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B, nh, N, dh]

        # LoRA updates for Q and V
        delta_q = self.lora_q_up(self.lora_q_down(x))  # [B, N, D]
        delta_v = self.lora_v_up(self.lora_v_down(x))  # [B, N, D]

        # Reshape deltas to match multi-head format
        dh = C // self.attn.num_heads
        delta_q = delta_q.reshape(B, N, self.attn.num_heads, dh).permute(0, 2, 1, 3)
        delta_v = delta_v.reshape(B, N, self.attn.num_heads, dh).permute(0, 2, 1, 3)

        q = q + delta_q
        v = v + delta_v

        # Standard attention computation
        scale = self.attn.scale
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.attn.attn_drop(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        out = self.attn.proj(out)
        out = self.attn.proj_drop(out)
        return out


class LoRABlock(nn.Module):
    """Transformer block with LoRA-augmented attention."""

    def __init__(self, block: Block, embed_dim: int, rank: int):
        super().__init__()
        self.block = block
        self.lora_attn = LoRAAttention(block.attn, embed_dim, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MSA sub-layer with LoRA
        h = self.block.norm1(x)
        h = self.lora_attn(h)
        h = self.block.ls1(h)
        h = self.block.drop_path1(h)
        x = x + h

        # Standard MLP sub-layer
        h = self.block.norm2(x)
        h = self.block.mlp(h)
        h = self.block.ls2(h)
        h = self.block.drop_path2(h)
        x = x + h
        return x


def apply_lora(backbone: nn.Module, embed_dim: int, rank: int = 16) -> None:
    """Replace all transformer blocks with LoRABlock."""
    new_blocks = nn.ModuleList()
    for block in backbone.blocks:
        new_blocks.append(LoRABlock(block, embed_dim, rank))
    backbone.blocks = new_blocks
