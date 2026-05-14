"""Memory attention module for SAM 2.

Stacks L transformer blocks that condition current frame features on memories.
Each block: self-attention -> cross-attention to memories -> cross-attention to object pointers -> MLP.

Uses:
- Sinusoidal absolute positional embeddings
- 2D Rotary Positional Embedding (RoPE) in self-attention and cross-attention
- Object pointers excluded from RoPE (no spatial correspondence)
- Vanilla attention operations (compatible with FlashAttention)
"""

from typing import Optional

import torch
import torch.nn as nn

from config import MemoryAttentionConfig
from transformer import TransformerBlock, get_sinusoidal_pos_embed


class MemoryAttention(nn.Module):
    """L-block transformer conditioning current frame on memory bank.

    Architecture:
    1. Self-attention on current frame tokens
    2. Cross-attention to spatial memory features (from memory encoder)
    3. Cross-attention to object pointers
    4. MLP
    """
    def __init__(self, config: MemoryAttentionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        # L transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                use_rope=config.use_rope,
                use_cross_attn=config.memory_cross_attn,
            )
            for _ in range(config.num_layers)
        ])

        # Additional cross-attention layer for object pointers (after memory cross-attention)
        if config.pointer_cross_attn:
            from transformer import MultiheadAttention
            self.pointer_attn = nn.ModuleList([
                MultiheadAttention(config.d_model, config.num_heads, config.dropout,
                                  use_rope=False, is_cross_attn=True)
                for _ in range(config.num_layers)
            ])
            self.pointer_norm = nn.ModuleList([
                nn.LayerNorm(config.d_model) for _ in range(config.num_layers)
            ])
        else:
            self.pointer_attn = None
            self.pointer_norm = None

        self.use_abs_pos = config.use_abs_pos
        self.use_rope = config.use_rope

    def forward(self, x: torch.Tensor, memory_features: Optional[torch.Tensor] = None,
                object_pointers: Optional[torch.Tensor] = None,
                pos_embed: Optional[torch.Tensor] = None,
                spatial_size: int = 64) -> torch.Tensor:
        """Condition current frame tokens on memory.

        Args:
            x: [B, N, C] current frame tokens from image encoder
            memory_features: [B, M, C] spatial memory features from memory bank (flattened)
            object_pointers: [B, P, C] object pointer tokens from memory bank
            pos_embed: [N, C] absolute positional embedding for spatial tokens
            spatial_size: spatial dimension (H or W) for RoPE

        Returns:
            [B, N, C] conditioned frame tokens
        """
        if pos_embed is not None:
            x = x + pos_embed.unsqueeze(0).to(x.dtype)

        for i, block in enumerate(self.blocks):
            # Self-attention + cross-attention to spatial memory
            x = block(x, memory_features, rope_h=spatial_size, rope_w=spatial_size)

            # Cross-attention to object pointers
            if self.pointer_attn is not None and object_pointers is not None:
                x = x + self.pointer_attn[i](
                    self.pointer_norm[i](x), object_pointers, object_pointers
                )

        return x


def build_memory_attention(config: MemoryAttentionConfig) -> MemoryAttention:
    """Factory function for MemoryAttention."""
    return MemoryAttention(config)
