import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Type, Optional
import math
class Attention(nn.Module):
    """
    A basic Attention block. Simplified for mock purposes.
    """
    def __init__(
        self, embedding_dim: int, num_heads: int, q_bias: bool = False, kv_bias: bool = False
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=q_bias)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=kv_bias)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=kv_bias)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, N_q, C = q.shape
        _, N_k, _ = k.shape
        q = self.q_proj(q).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = (attn_weights @ v).transpose(1, 2).reshape(B, N_q, C)
        out = self.out_proj(out)
        return out
class MLP(nn.Module):
    """
    Simple MLP for heads.
    """
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, activation: Type[nn.Module] = nn.GELU
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.activation = activation()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding for sparse prompts, similar to SAM.
    Generates random positional embeddings that are then scaled.
    """
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None:
            scale = 2 * math.pi
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((num_pos_feats, 2)),
        )
    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords @ self.positional_encoding_gaussian_matrix.to(coords.dtype)
        coords = 2 * math.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        grid = torch.ones((h, w), device=self.positional_encoding_gaussian_matrix.device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe # (H, W, C)
    def forward_with_coords(self, coords_input: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(self.positional_encoding_gaussian_matrix.device))
class RotaryPositionEmbedding(nn.Module):
    """
    RoPE as described in Su et al., 2021 and Heo et al., 2024.
    """
    def __init__(self, dim, seq_len_interpolation_factor=None):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.cached_p_sincos = None
    def forward(self, qk, seq_len=None):
        # qk: (B, N, C)
        if seq_len is None: seq_len = qk.shape[1]
        if self.seq_len_interpolation_factor is not None:
            seq_len = seq_len * self.seq_len_interpolation_factor
        t = torch.arange(seq_len, device=qk.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum(i,j-        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        # Apply rotary to the last dimension
        # For 2D RoPE, this would typically involve creating a 2D grid and applying rotation.
        # The paper specifies 2d-RoPE, so a more complex grid-based rotation would be needed.
        def rotate_half(x):
            x = x.reshape(x.shape[:-1] + (-1, 2))
            x1, x2 = x.unbind(dim=-1)
            return torch.cat((-x2, x1), dim=-1).reshape(x.shape[:-2] + (-1,))
        return (qk * cos) + (rotate_half(qk) * sin)
class MemoryAttentionBlock(nn.Module):
    """
    Single block for Memory Attention.
    Performs self-attention, then cross-attention to memory bank and object pointers.
    """
    def __init__(
        self, embedding_dim: int, num_heads: int, mlp_dim: int, activation: Type[nn.Module] = nn.GELU
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.self_attn = Attention(embedding_dim, num_heads, q_bias=True, kv_bias=True)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.cross_attn_memory = Attention(embedding_dim, num_heads, q_bias=True, kv_bias=True)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.cross_attn_object_pointers = Attention(embedding_dim, num_heads, q_bias=True, kv_bias=True)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, 2, activation)
        self.rope = RotaryPositionEmbedding(embedding_dim // num_heads)
    def forward(
        self, 
        x: torch.Tensor, # Current frame image embedding (B, H*W, C)
        memory_features: torch.Tensor, # From MemoryBank (B, N_mem, C)
        object_pointers: torch.Tensor, # From MemoryBank (B, N_obj, C)
        image_pe: torch.Tensor, # Positional encoding for image (B, H*W, C)
        memory_pe: torch.Tensor, # Positional encoding for memory (B, N_mem, C)
    ) -> torch.Tensor:
        # Self-attention with RoPE
        q_self = x + image_pe
        q_self = self.rope(q_self)
        x = x + self.self_attn(q_self, q_self, x)
        x = self.norm1(x)
        # Cross-attention to memory features with RoPE
        q_cross_mem = x + image_pe
        q_cross_mem = self.rope(q_cross_mem)
        k_mem = memory_features + memory_pe # Assume memory_pe is already 2D RoPE compatible for memory features
        k_mem = self.rope(k_mem)
        x = x + self.cross_attn_memory(q_cross_mem, k_mem, memory_features)
        x = self.norm2(x)
        # Cross-attention to object pointers (no RoPE as they lack spatial correspondence)
        q_cross_obj = x + image_pe # Image features are query
        q_cross_obj = self.rope(q_cross_obj)
        x = x + self.cross_attn_object_pointers(q_cross_obj, object_pointers, object_pointers) # Object pointers are key/value
        x = self.norm3(x)
        # MLP
        x = x + self.mlp(x)
        x = self.norm4(x)
        return x
class MemoryAttention(nn.Module):
    """
    Memory attention module. Stacks multiple MemoryAttentionBlock layers.
    """
    def __init__(
        self, 
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        depth: int = 4, # Default depth for memory attention
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MemoryAttentionBlock(embedding_dim, num_heads, mlp_dim, activation)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(embedding_dim)
    def forward(
        self, 
        image_embedding: torch.Tensor, # Current frame image embedding (B, C, H, W)
        image_pe: torch.Tensor, # Positional encoding for current frame (1, H, W, C)
        memory_features: torch.Tensor, # From MemoryBank (B, N_mem_tokens, C)
        object_pointers: torch.Tensor, # From MemoryBank (B, N_obj_tokens, C)
        memory_pe: torch.Tensor, # Positional encoding for memory features (B, N_mem_tokens, C)
    ) -> torch.Tensor:
        # Flatten image_embedding and image_pe for transformer
        B, C, H, W = image_embedding.shape
        x = image_embedding.view(B, C, H * W).permute(0, 2, 1) # (B, H*W, C)
        image_pe_flat = image_pe.view(1, H * W, C).expand(B, -1, -1) # (B, H*W, C)
        for layer in self.layers:
            x = layer(x, memory_features, object_pointers, image_pe_flat, memory_pe)
        x = self.final_norm(x)
        # Reshape back to (B, C, H, W)
        conditioned_image_embedding = x.permute(0, 2, 1).view(B, C, H, W)
        return conditioned_image_embedding
# Example usage
if __name__ == "__main__":
    embed_dim = 256
    image_h_w = 1024 // 16
    batch_size = 1
    num_memory_tokens = 5 # Example number of memory tokens
    num_object_pointers = 2 # Example number of object pointer tokens
    image_embedding = torch.randn(batch_size, embed_dim, image_h_w, image_h_w)
    image_pe_generator = PositionEmbeddingRandom(embed_dim // 2)
    image_pe = image_pe_generator( (image_h_w, image_h_w) ).unsqueeze(0) # (1, H, W, C)
    memory_features = torch.randn(batch_size, num_memory_tokens, embed_dim)
    object_pointers = torch.randn(batch_size, num_object_pointers, embed_dim)
    memory_pe = torch.randn(batch_size, num_memory_tokens, embed_dim) # Placeholder
    mem_attn = MemoryAttention(
        embedding_dim=embed_dim,
        num_heads=8,
        mlp_dim=embed_dim * 4,
        depth=4,
    )
    conditioned_image_embedding = mem_attn(image_embedding, memory_features, object_pointers, image_pe, memory_pe)
    print(f"Conditioned image embedding shape: {conditioned_image_embedding.shape}")
    assert conditioned_image_embedding.shape == image_embedding.shape
    print("MemoryAttention output shape matches input image embedding shape.")
