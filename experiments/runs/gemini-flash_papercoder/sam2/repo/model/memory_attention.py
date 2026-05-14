import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

# Placeholder for Config type hint to avoid circular import with config.py
# In a real project, this would be 'from config import Config'
Config = Any


class MLP(nn.Module):
    """
    A simple two-layer MLP with GELU activation and optional dropout.
    """
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RotaryPositionalEmbedding(nn.Module):
    """
    Implements 2D Rotary Positional Embedding (RoPE) as described in
    Su et al. (2021) and Heo et al. (2024).

    This module applies rotation to query and key embeddings based on their
    2D spatial coordinates.
    """
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        # Cache cis (complex-valued) frequencies
        self.head_dim = head_dim
        self.base = base
        
        # Ensure dim is even for complex number pairing
        if head_dim % 2 != 0:
            raise ValueError(f"RotaryPositionalEmbedding dim ({head_dim}) must be an even number.")

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute the rotary embedding table (cis) for max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float32)
        # freqs will be (max_seq_len, head_dim/2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) 
        
        # Apply complex exponential for rotation
        cis = torch.polar(torch.ones_like(freqs), freqs) # (max_seq_len, head_dim/2)
        self.register_buffer("cis", cis)

    def _rotate_queries_and_keys(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Applies rotation to a tensor `x` using cosine and sine components.
        `x` is assumed to be (..., D) where D is the feature dimension.
        `cos` and `sin` are assumed to be (..., D/2) (real part)
        """
        # Split into two halves for complex multiplication
        # x_split is (..., D/2, 2)
        x_split = x.reshape(*x.shape[:-1], -1, 2) 
        x_real, x_imag = x_split[..., 0], x_split[..., 1]
        
        # Apply rotation (x_real + i*x_imag) * (cos + i*sin)
        rotated_real = x_real * cos - x_imag * sin
        rotated_imag = x_real * sin + x_imag * cos

        # Recombine
        # rotated_x becomes (..., D)
        rotated_x = torch.cat([rotated_real, rotated_imag], dim=-1)
        rotated_x = rotated_x.reshape(*x.shape[:-1], self.head_dim)
        return rotated_x
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, spatial_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies 2D RoPE to query and key tensors.

        Args:
            q (torch.Tensor): Query tensor. Shape (B, L_q, NumHeads, HeadDim).
            k (torch.Tensor): Key tensor. Shape (B, L_k, NumHeads, HeadDim).
            spatial_shape (Tuple[int, int]): (H_spatial, W_spatial) of the original 2D layout
                                           before flattening to sequence length.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
        """
        H_spatial, W_spatial = spatial_shape
        batch_size, num_heads, seq_len_q, head_dim = q.shape

        if seq_len_q != H_spatial * W_spatial:
            # This RoPE implementation is strictly for 2D grids flattened.
            # If sequence length doesn't match, it's problematic.
            # As per design, spatial_shape refers to the query's spatial shape,
            # and it should apply when query is from a 2D grid.
            warnings.warn(f"RoPE: Query sequence length ({seq_len_q}) does not match spatial_shape ({H_spatial}x{W_spatial}). RoPE might not apply correctly.")
            # For simplicity, if it doesn't match, we return original
            return q, k

        # Get x and y coordinates for each position in the flattened sequence
        # Ensure coordinates are on the correct device
        pos_y, pos_x = torch.meshgrid(torch.arange(H_spatial, device=q.device),
                                      torch.arange(W_spatial, device=q.device), indexing='ij')
        
        # Flatten coordinates
        pos_x = pos_x.reshape(-1) # (seq_len_q,)
        pos_y = pos_y.reshape(-1) # (seq_len_q,)

        # Slice precomputed cis for x and y coordinates
        # cis is (max_seq_len, head_dim/2)
        cis_x = self.cis[pos_x].unsqueeze(1) # (seq_len_q, 1, head_dim/2)
        cis_y = self.cis[pos_y].unsqueeze(1) # (seq_len_q, 1, head_dim/2)

        # Expand for multi-head dimension
        cis_x = cis_x.expand(-1, num_heads, -1) # (seq_len_q, num_heads, head_dim/2)
        cis_y = cis_y.expand(-1, num_heads, -1) # (seq_len_q, num_heads, head_dim/2)

        # Cosine and Sine components for x and y
        cos_x, sin_x = cis_x.real, cis_x.imag
        cos_y, sin_y = cis_y.real, cis_y.imag

        # Apply rotation for each dimension (x then y)
        # q, k are (B, NumHeads, SeqLen, HeadDim)
        # cos/sin are (SeqLen, NumHeads, HeadDim/2)
        # Need to reorder cos/sin to (1, NumHeads, SeqLen, HeadDim/2) for broadcasting
        cos_x = cos_x.permute(1, 0, 2).unsqueeze(0) # (1, NumHeads, SeqLen, HeadDim/2)
        sin_x = sin_x.permute(1, 0, 2).unsqueeze(0)
        cos_y = cos_y.permute(1, 0, 2).unsqueeze(0)
        sin_y = sin_y.permute(1, 0, 2).unsqueeze(0)

        # First apply for x
        q_rotated_x = self._rotate_queries_and_keys(q, cos_x, sin_x)
        k_rotated_x = self._rotate_queries_and_keys(k, cos_x, sin_x)
        
        # Then for y
        q_rotated_xy = self._rotate_queries_and_keys(q_rotated_x, cos_y, sin_y)
        k_rotated_xy = self._rotate_queries_and_keys(k_rotated_x, cos_y, sin_y)

        return q_rotated_xy, k_rotated_xy


class Attention(nn.Module):
    """
    Multi-head attention mechanism supporting optional 2D RoPE.
    """
    def __init__(self, q_dim: int, kv_dim: int, hidden_dim: int, num_heads: int, drop_rate: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.q_proj = nn.Linear(q_dim, hidden_dim, bias=True)
        self.k_proj = nn.Linear(kv_dim, hidden_dim, bias=True)
        self.v_proj = nn.Linear(kv_dim, hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn_drop = nn.Dropout(drop_rate)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(
        self,
        query: torch.Tensor, # (B, L_q, C_q)
        key: torch.Tensor,    # (B, L_kv, C_kv)
        value: torch.Tensor,  # (B, L_kv, C_kv)
        rope_fn: Optional[RotaryPositionalEmbedding] = None, # The RoPE module instance
        spatial_shape: Optional[Tuple[int, int]] = None, # (H, W) for 2D RoPE
        apply_rope: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            query (torch.Tensor): Query features. Shape (B, L_q, C_q).
            key (torch.Tensor): Key features. Shape (B, L_kv, C_kv).
            value (torch.Tensor): Value features. Shape (B, L_kv, C_kv).
            rope_fn (Optional[RotaryPositionalEmbedding]): RoPE instance to apply.
            spatial_shape (Optional[Tuple[int, int]]): (H, W) if applying 2D RoPE.
            apply_rope (bool): Whether to apply RoPE.

        Returns:
            torch.Tensor: Output features after attention. Shape (B, L_q, hidden_dim).
        """
        B, L_q, C_q = query.shape
        _, L_kv, C_kv = key.shape

        q = self.q_proj(query).reshape(B, L_q, self.num_heads, self.head_dim)
        k = self.k_proj(key).reshape(B, L_kv, self.num_heads, self.head_dim)
        v = self.v_proj(value).reshape(B, L_kv, self.num_heads, self.head_dim)
        
        # Transpose for batching (B, NumHeads, SeqLen, HeadDim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if apply_rope and rope_fn is not None and spatial_shape is not None:
            # Apply 2D RoPE to Q and K
            q, k = rope_fn(q, k, spatial_shape) # (B, NumHeads, L_q, HeadDim), (B, NumHeads, L_kv, HeadDim)
        elif apply_rope and (rope_fn is None or spatial_shape is None):
            warnings.warn("apply_rope=True but rope_fn or spatial_shape is None. RoPE not applied.")


        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, NumHeads, L_q, L_kv)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Multiply by values
        x = (attn @ v).transpose(1, 2).reshape(B, L_q, -1) # (B, L_q, hidden_dim)

        # Output projection
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x


class MemoryAttentionBlock(nn.Module):
    """
    A single transformer block in MemoryAttention, combining self-attention,
    two cross-attention layers (spatial memory and object pointers), and an MLP.
    """
    def __init__(self, hidden_dim: int, num_heads: int, use_2d_rope: bool, drop_rate: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_2d_rope = use_2d_rope

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = Attention(
            q_dim=hidden_dim, kv_dim=hidden_dim, hidden_dim=hidden_dim, num_heads=num_heads, drop_rate=drop_rate
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        # Cross-attention to spatial memory features (keys and values have hidden_dim after projection)
        self.cross_attn_mem = Attention(
            q_dim=hidden_dim, kv_dim=hidden_dim, hidden_dim=hidden_dim, num_heads=num_heads, drop_rate=drop_rate
        )

        self.norm3 = nn.LayerNorm(hidden_dim)
        # Cross-attention to object pointer features (keys and values have hidden_dim after projection)
        self.cross_attn_obj = Attention(
            q_dim=hidden_dim, kv_dim=hidden_dim, hidden_dim=hidden_dim, num_heads=num_heads, drop_rate=drop_rate
        )

        self.norm4 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * 4, drop=drop_rate) # Typical MLP expansion factor is 4

        # RoPE instance for this block (shared for self and spatial cross attention queries)
        # Max sequence length for RoPE: Hiera-B+ output stride 16 for 1024x1024 image is 64x64.
        # So max seq len for RoPE applied to image embeddings will be 64*64=4096.
        if use_2d_rope:
            self.rope_fn = RotaryPositionalEmbedding(head_dim=hidden_dim // num_heads, max_seq_len=4096)
        else:
            self.rope_fn = None # No RoPE if not used

    def forward(
        self,
        query_features: torch.Tensor, # (B, L_q, hidden_dim)
        spatial_memory_features: torch.Tensor, # (B, L_mem, hidden_dim)
        object_pointer_features: torch.Tensor, # (B, L_obj, hidden_dim)
        spatial_query_shape: Tuple[int, int], # (H_q, W_q) to apply 2D RoPE correctly to query_features
    ) -> torch.Tensor:
        """
        Forward pass for a single MemoryAttentionBlock.

        Args:
            query_features (torch.Tensor): Features of the current frame, acting as query. (B, L_q, hidden_dim).
            spatial_memory_features (torch.Tensor): Features from the memory bank (spatial). (B, L_mem, hidden_dim).
            object_pointer_features (torch.Tensor): Features from the memory bank (object pointers). (B, L_obj, hidden_dim).
            spatial_query_shape (Tuple[int, int]): Original (H, W) dimensions of the query features for 2D RoPE.

        Returns:
            torch.Tensor: Updated query features after attention and MLP. (B, L_q, hidden_dim).
        """
        # Self-Attention
        # The paper specifies "The first one taking the image encoding from the current frame as input. Each block performs self-attention"
        norm_query_sa = self.norm1(query_features)
        query_features = query_features + self.self_attn(
            norm_query_sa, norm_query_sa, norm_query_sa,
            rope_fn=self.rope_fn, spatial_shape=spatial_query_shape, apply_rope=self.use_2d_rope
        )

        # Cross-Attention to Spatial Memory Features
        # "followed by cross-attention to memories of (prompted/unprompted) frames"
        norm_query_mem_ca = self.norm2(query_features)
        query_features = query_features + self.cross_attn_mem(
            norm_query_mem_ca, spatial_memory_features, spatial_memory_features,
            rope_fn=self.rope_fn, spatial_shape=spatial_query_shape, # Apply RoPE to query, not memory keys if memory is non-spatial
                                                                     # Given the paper: "cross-attends to both spatial memory features"
                                                                     # This implies memory features also have spatial positions.
                                                                     # If memory_features are from different spatial resolutions, then it is more complex.
                                                                     # Assuming for now, spatial_shape is solely for the query_features' 2D positions.
            apply_rope=self.use_2d_rope
        )

        # Cross-Attention to Object Pointers
        # "and object pointers (see below), stored in a memory bank"
        # Object pointers do not have spatial correspondence, so RoPE is not applied here.
        norm_query_obj_ca = self.norm3(query_features)
        query_features = query_features + self.cross_attn_obj(
            norm_query_obj_ca, object_pointer_features, object_pointer_features,
            rope_fn=None, spatial_shape=None, apply_rope=False # Explicitly disable RoPE for object pointers
        )

        # MLP
        # "followed by an MLP."
        norm_query_mlp = self.norm4(query_features)
        query_features = query_features + self.mlp(norm_query_mlp)

        return query_features


class MemoryAttention(nn.Module):
    """
    The main MemoryAttention module for SAM2.
    It takes current frame features, memory features, and object pointers,
    and applies a stack of transformer blocks to condition the current frame features.
    """

    def __init__(self, config: Config):
        """
        Initializes the MemoryAttention module.

        Args:
            config (Config): The global configuration object.
        """
        super().__init__()
        self._config = config

        self.num_layers: int = self._config.get("model.memory_attention.num_layers", 4)
        self.hidden_dim: int = self._config.get("model.memory_attention.hidden_dim", 256)
        self.num_heads: int = self._config.get("model.memory_attention.num_heads", 8)
        self.use_2d_rope: bool = self._config.get("model.memory_attention.use_2d_rope", True)
        self.memory_feature_dim: int = self._config.get("model.memory_bank.memory_feature_dim", 64)
        # object_pointer_token_dim is the dimension of each of the 4 split tokens
        self.object_pointer_token_dim: int = self._config.get("model.memory_bank.object_pointer_dim", 256) // 4

        # Validate dimensions
        if self.hidden_dim <= 0 or self.num_heads <= 0:
            raise ValueError("hidden_dim and num_heads must be positive integers.")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if self.memory_feature_dim <= 0 or self.object_pointer_token_dim <= 0:
            raise ValueError("memory_feature_dim and object_pointer_token_dim must be positive integers.")


        # Projection for memory features if their dimension differs from hidden_dim
        if self.memory_feature_dim != self.hidden_dim:
            self.memory_proj = nn.Linear(self.memory_feature_dim, self.hidden_dim)
        else:
            self.memory_proj = nn.Identity()

        # Projection for object pointer tokens if their dimension differs from hidden_dim
        if self.object_pointer_token_dim != self.hidden_dim:
            self.object_pointer_proj = nn.Linear(self.object_pointer_token_dim, self.hidden_dim)
        else:
            self.object_pointer_proj = nn.Identity()

        # Stack of MemoryAttentionBlocks
        self.blocks = nn.ModuleList([
            MemoryAttentionBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                use_2d_rope=self.use_2d_rope
            )
            for _ in range(self.num_layers)
        ])

        # Temporal positional embedding layer
        # The `MemoryBank` design specifies `temporal_pos_embeddings` as `current_frame_idx - stored_frame_idx`
        # for recent frames, and `0` for prompted frames.
        # This implies positive integers for recent frames.
        # Let's consider `max_recent_frames` from MemoryBank as max relative positive offset.
        # The `temporal_pos_embeddings` provided by `MemoryBank` would be `[1, 2, ..., N]` for recent.
        # And `[0]` for prompted frames. So, we need embeddings for `0` up to `max_recent_frames`.
        self.max_relative_time_steps = self._config.get("model.memory_bank.max_recent_frames", 6)
        self.temporal_embedding_layer = nn.Embedding(
            num_embeddings=self.max_relative_time_steps + 1, # e.g., 0 for prompted, 1..6 for recent
            embedding_dim=self.hidden_dim
        )

    def forward(
        self,
        current_frame_features: torch.Tensor, # (B, C, H, W)
        memory_features: List[torch.Tensor],  # List of (C_mem, H_mem, W_mem)
        object_pointers: List[torch.Tensor],  # List of (4, C_obj_token)
        temporal_pos_embeddings: List[torch.Tensor], # List of (1) integer tensors
    ) -> torch.Tensor:
        """
        Performs the forward pass through the MemoryAttention module.

        Args:
            current_frame_features (torch.Tensor): Features of the current frame from the ImageEncoder.
                                                   Shape (B, C, H_orig, W_orig).
            memory_features (List[torch.Tensor]): List of spatial memory feature tensors from MemoryBank.
                                                  Each tensor has shape (C_mem, H_mem, W_mem).
            object_pointers (List[torch.Tensor]): List of object pointer tensors (split into 4 tokens) from MemoryBank.
                                                  Each tensor has shape (4, C_obj_token).
            temporal_pos_embeddings (List[torch.Tensor]): List of 1-element tensors, each containing
                                                          the relative temporal position (int) for its corresponding
                                                          memory feature. 0 for prompted frames, positive for recent frames.

        Returns:
            torch.Tensor: Conditioned current frame features. Shape (B, H_orig * W_orig, hidden_dim).
        """
        B, C_orig, H_orig, W_orig = current_frame_features.shape

        # 1. Prepare current_frame_features
        # Reshape from (B, C, H, W) to (B, H*W, C)
        query_features = current_frame_features.permute(0, 2, 3, 1).reshape(B, H_orig * W_orig, C_orig)
        spatial_query_shape = (H_orig, W_orig)

        # 2. Prepare spatial_memory_features
        if memory_features:
            # memory_features is a List[Tensor] where each Tensor is (C_mem, H_mem, W_mem)
            # Stack them to (N_mem, C_mem, H_mem, W_mem)
            batched_memory_features = torch.stack(memory_features, dim=0)
            N_mem, C_mem, H_mem, W_mem = batched_memory_features.shape
            
            # Project to hidden_dim
            projected_memory_features = self.memory_proj(
                batched_memory_features.permute(0, 2, 3, 1) # (N_mem, H_mem, W_mem, C_mem)
            ) # (N_mem, H_mem, W_mem, hidden_dim)

            # Flatten spatial dimensions for attention (N_mem, H_mem*W_mem, hidden_dim)
            spatial_memory_kv = projected_memory_features.reshape(N_mem, H_mem * W_mem, self.hidden_dim)
            
            # Add temporal positional embeddings
            # temporal_pos_embeddings is List[Tensor(1)] with integer values
            temporal_indices = torch.cat(temporal_pos_embeddings, dim=0).long() # (N_mem,)
            
            # Clamp indices to valid range if any are out of bounds (shouldn't happen with correct MemoryBank)
            temporal_indices = torch.clamp(temporal_indices, 0, self.max_relative_time_steps)
            
            temporal_embedding = self.temporal_embedding_layer(temporal_indices) # (N_mem, hidden_dim)
            
            # Expand temporal_embedding to match spatial_memory_kv's sequence length and add.
            # (N_mem, hidden_dim) -> (N_mem, 1, hidden_dim) broadcast to (N_mem, H_mem*W_mem, hidden_dim)
            spatial_memory_kv = spatial_memory_kv + temporal_embedding.unsqueeze(1)
            
            # For cross-attention, we concatenate all memories along the sequence dimension
            # (B, N_mem * H_mem * W_mem, hidden_dim)
            # Since SAM2 processes one object per video at a time, B will typically be 1.
            spatial_memory_features_flat = spatial_memory_kv.reshape(1, -1, self.hidden_dim).expand(B, -1, -1)
        else:
            spatial_memory_features_flat = torch.empty(B, 0, self.hidden_dim, device=query_features.device)


        # 3. Prepare object_pointer_features
        if object_pointers:
            # object_pointers is List[Tensor] where each Tensor is (4, C_obj_token)
            # Concatenate them: (N_total_obj_tokens, C_obj_token)
            batched_object_pointers = torch.cat(object_pointers, dim=0)
            
            # Project to hidden_dim
            object_pointer_kv = self.object_pointer_proj(batched_object_pointers) # (N_total_obj_tokens, hidden_dim)
            
            # Expand to batch size
            object_pointer_features_flat = object_pointer_kv.unsqueeze(0).expand(B, -1, -1)
        else:
            object_pointer_features_flat = torch.empty(B, 0, self.hidden_dim, device=query_features.device)

        # 4. Pass through MemoryAttentionBlock(s)
        for i, block in enumerate(self.blocks):
            query_features = block(
                query_features,
                spatial_memory_features=spatial_memory_features_flat,
                object_pointer_features=object_pointer_features_flat,
                spatial_query_shape=spatial_query_shape,
            )

        return query_features

