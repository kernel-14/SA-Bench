
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .layers import MLP, RotaryPositionalEmbedding

class Attention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        use_rope: bool = False,
        rope_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embedding_dim, embedding_dim)

        self.use_rope = use_rope
        if use_rope:
            assert rope_dim is not None
            self.rope = RotaryPositionalEmbedding(rope_dim)

    def forward(self, x: torch.Tensor, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs attention. If ref is provided, performs cross-attention, else self-attention.
        x: query (B, N, C)
        ref: key/value (B, M, C), if None, x is used for key/value (self-attention)
        """
        B, N, C = x.shape
        kv_input = x if ref is None else ref

        # Self-attention
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = qkv[0] # (B, num_heads, N, head_dim)

        # Cross-attention
        if ref is not None:
            kv_qkv = self.qkv(kv_input).reshape(B, kv_input.shape[1], 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv_qkv[1], kv_qkv[2] # (B, num_heads, M, head_dim)
        else: # Self-attention for k, v
            k, v = qkv[1], qkv[2] # (B, num_heads, N, head_dim)

        if self.use_rope:
            # Apply RoPE to query and key if enabled
            q_rope_input = q.permute(0,2,1,3).reshape(B, N, -1)
            k_rope_input = k.permute(0,2,1,3).reshape(B, k.shape[2], -1)
            
            q = self.rope(q_rope_input).reshape(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)
            k = self.rope(k_rope_input).reshape(B, k.shape[2], self.num_heads, self.head_dim).permute(0,2,1,3)
            

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MemoryAttentionBlock(nn.Module):
    """
    A single Transformer block for memory attention.
    Performs self-attention followed by cross-attention to memory features and object pointers.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        use_rope: bool = True, # Use RoPE in attention layers
        rope_dim: int = 64, # Dimension for RoPE
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(embedding_dim)
        self.attn = Attention(
            embedding_dim,
            num_heads,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_dim=rope_dim,
        )

        self.norm2 = norm_layer(embedding_dim)
        self.cross_attn_mem = Attention(
            embedding_dim,
            num_heads,
            qkv_bias=qkv_bias,
            use_rope=use_rope, # RoPE for cross-attention with spatial memory
            rope_dim=rope_dim,
        )
        self.cross_attn_obj_ptr = Attention(
            embedding_dim,
            num_heads,
            qkv_bias=qkv_bias,
            use_rope=False, # No RoPE for object pointers
        )

        self.norm3 = norm_layer(embedding_dim)
        self.mlp = MLP(
            in_features=embedding_dim,
            hidden_features=int(embedding_dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self,
        x: torch.Tensor, # Current frame features (from image encoder)
        memory_features: torch.Tensor, # Spatial memory features from memory bank
        object_pointers: torch.Tensor, # Object pointers from memory bank
    ) -> torch.Tensor:
        # Self-attention on current frame features
        x = x + self.attn(self.norm1(x))

        # Cross-attention to spatial memory features
        x = x + self.cross_attn_mem(self.norm2(x), memory_features)

        # Cross-attention to object pointers
        x = x + self.cross_attn_obj_ptr(self.norm2(x), object_pointers) # Uses the same norm2, check paper for details

        # MLP
        x = x + self.mlp(self.norm3(x))

        return x

class MemoryAttention(nn.Module):
    """
    Stacks multiple MemoryAttentionBlocks to form the Memory Attention module.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_layers: int, # L in the paper
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        use_rope: bool = True,
        rope_dim: int = 64,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MemoryAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                act_layer=act_layer,
                norm_layer=norm_layer,
                use_rope=use_rope,
                rope_dim=rope_dim,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor, # Current frame feature (from image encoder)
        memory_features: torch.Tensor, # Spatial memory features from memory bank
        object_pointers: torch.Tensor, # Object pointers from memory bank
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory_features, object_pointers)
        return x

