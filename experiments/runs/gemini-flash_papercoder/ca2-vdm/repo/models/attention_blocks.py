import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class CausalTemporalAttention(nn.Module):
    """
    Implements causal temporal attention, where each frame can only attend
    to itself and its preceding frames.

    This module is designed to replace standard bidirectional temporal attention
    in a Transformer block to enable KV-caching.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64):
        """
        Initializes the CausalTemporalAttention module.

        Args:
            dim (int): The input feature dimension of the hidden states (e.g., model_channels).
            heads (int): The number of attention heads. Defaults to 8.
            dim_head (int): The dimension of each attention head. Defaults to 64.
                            If not specified, it's calculated as dim // heads.
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim_head if dim_head is not None else dim // heads
        
        # Ensure that dim is divisible by heads for proper multi-head attention
        if self.dim % self.heads != 0:
            raise ValueError(
                f"dim {self.dim} must be divisible by heads {self.heads} for CausalTemporalAttention"
            )

        self.scale = self.head_dim ** -0.5

        # Linear projections for query, key, value
        # bias=False is common in Transformers especially with LayerNorm.
        self.to_q = nn.Linear(self.dim, self.dim, bias=False)
        self.to_k = nn.Linear(self.dim, self.dim, bias=False)
        self.to_v = nn.Linear(self.dim, self.dim, bias=False)

        # Final output projection
        self.to_out = nn.Linear(self.dim, self.dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for causal temporal attention.

        Args:
            query (torch.Tensor): Input query tensor. Shape (B_effective, L_seq, D).
                                  B_effective is typically B * H * W, L_seq is number of frames.
            key (torch.Tensor): Input key tensor. Shape (B_effective, L_seq, D).
            value (torch.Tensor): Input value tensor. Shape (B_effective, L_seq, D).
            attention_mask (Optional[torch.Tensor]): A causal mask to prevent attending to future frames.
                                                    Expected shape: (L_seq, L_seq) or (1, L_seq, L_seq).
                                                    Should contain -torch.inf for masked positions and 0 for unmasked.
                                                    If None, no mask is applied (behaves like standard self-attention).

        Returns:
            torch.Tensor: Output tensor after causal temporal attention. Shape (B_effective, L_seq, D).
        """
        h = self.heads
        
        # Apply linear projections
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)

        # Reshape for multi-head attention:
        # (B_effective, L_seq, D) -> (B_effective, L_seq, H, head_dim) -> (B_effective, H, L_seq, head_dim)
        # using -1 for the batch size allows handling both B_effective * L_chunk for spatial attention
        # and B_effective for temporal attention when a single frame is processed.
        q = q.view(q.shape[0], -1, h, self.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], -1, h, self.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], -1, h, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        # (B_effective, H, L_seq_Q, head_dim) @ (B_effective, H, head_dim, L_seq_K) -> (B_effective, H, L_seq_Q, L_seq_K)
        # Note: L_seq_Q and L_seq_K might be different if K,V include cached items.
        sim = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Apply causal mask
        if attention_mask is not None:
            # Ensure mask is on the same device as sim
            attention_mask = attention_mask.to(sim.device)
            # Unsqueeze for batch (if mask is 1,L,L) and heads dimensions for broadcasting
            # The mask effectively becomes (1, 1, L_seq, L_seq) or (B_effective, 1, L_seq, L_seq)
            # to match sim's shape (B_effective, H, L_seq, L_seq)
            while attention_mask.ndim < sim.ndim:
                attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 2 else attention_mask.unsqueeze(1)
            sim = sim + attention_mask

        # Softmax to get attention probabilities
        attn = F.softmax(sim, dim=-1)

        # Weighted sum of values
        # (B_effective, H, L_seq_Q, L_seq_K) @ (B_effective, H, L_seq_K, head_dim) -> (B_effective, H, L_seq_Q, head_dim)
        out = torch.matmul(attn, v)

        # Reshape back to original dimension:
        # (B_effective, H, L_seq_Q, head_dim) -> (B_effective, L_seq_Q, H, head_dim) -> (B_effective, L_seq_Q, D)
        out = out.transpose(1, 2).contiguous().view(out.shape[0], -1, self.dim)

        # Final output projection
        return self.to_out(out)


class PrefixEnhancedSpatialAttention(nn.Module):
    """
    Implements spatial attention enhanced by a short sub-prefix of clean frames.
    This module concatenates pre-computed KVs from prefix frames to the current
    frames' KVs before attention computation, providing local context guidance.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, prefix_sub_len: int = 3):
        """
        Initializes the PrefixEnhancedSpatialAttention module.

        Args:
            dim (int): The input feature dimension of the hidden states.
            heads (int): The number of attention heads. Defaults to 8.
            dim_head (int): The dimension of each attention head. Defaults to 64.
                            If not specified, it's calculated as dim // heads.
            prefix_sub_len (int): The length P' of the sub-prefix used for enhancement. Defaults to 3.
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim_head if dim_head is not None else dim // heads
        self.prefix_sub_len = prefix_sub_len

        if self.dim % self.heads != 0:
            raise ValueError(
                f"dim {self.dim} must be divisible by heads {self.heads} for PrefixEnhancedSpatialAttention"
            )

        self.scale = self.head_dim ** -0.5

        # Linear projections for query, key, value
        # bias=False is common in Transformers especially with LayerNorm.
        self.to_q = nn.Linear(self.dim, self.dim, bias=False)
        self.to_k = nn.Linear(self.dim, self.dim, bias=False)
        self.to_v = nn.Linear(self.dim, self.dim, bias=False)

        # Final output projection
        self.to_out = nn.Linear(self.dim, self.dim)

    def forward(self, hidden_states: torch.Tensor, prefix_kvs: Optional[Dict[str, torch.Tensor]] = None, use_prefix_enhancement: bool = True) -> torch.Tensor:
        """
        Forward pass for prefix-enhanced spatial attention.

        Args:
            hidden_states (torch.Tensor): The current hidden states for spatial attention.
                                          Shape: (B_effective, S_seq, D).
                                          B_effective is typically B * L_chunk, S_seq is H * W.
            prefix_kvs (Optional[Dict[str, torch.Tensor]]): A dictionary containing pre-computed
                                                            and cached key/value tensors from the
                                                            clean spatial prefix.
                                                            Expected structure: {'K': K_tensor, 'V': V_tensor}.
                                                            K_tensor and V_tensor should be in multi-head format:
                                                            (B_effective, heads, prefix_sub_len * S_seq, head_dim).
                                                            If None, no prefix enhancement is applied.
            use_prefix_enhancement (bool): Flag to enable/disable prefix enhancement.
                                           If False, prefix_kvs is ignored. Defaults to True.

        Returns:
            torch.Tensor: Output tensor after prefix-enhanced spatial attention.
                          Shape: (B_effective, S_seq, D).
        """
        h = self.heads

        # Apply linear projections for current hidden states
        q_current = self.to_q(hidden_states)
        k_current = self.to_k(hidden_states)
        v_current = self.to_v(hidden_states)

        # Reshape current K, V for multi-head attention:
        # (B_effective, S_seq, D) -> (B_effective, S_seq, H, head_dim) -> (B_effective, H, S_seq, head_dim)
        q_current = q_current.view(q_current.shape[0], -1, h, self.head_dim).transpose(1, 2)
        k_current = k_current.view(k_current.shape[0], -1, h, self.head_dim).transpose(1, 2)
        v_current = v_current.view(v_current.shape[0], -1, h, self.head_dim).transpose(1, 2)

        # Initialize combined keys and values with current ones
        k_combined = k_current
        v_combined = v_current

        # Combine keys and values with prefix KVs if enabled and available
        if use_prefix_enhancement and prefix_kvs is not None and prefix_kvs.get('K') is not None and prefix_kvs.get('V') is not None:
            # Ensure prefix KVs are on the same device as current hidden_states
            prefix_k_tensor = prefix_kvs['K'].to(hidden_states.device)
            prefix_v_tensor = prefix_kvs['V'].to(hidden_states.device)

            # Concatenate prefix KVs with current KVs along the sequence dimension (-2)
            # The prefix KVs are expected to already be in the multi-head format
            # (B_effective, heads, prefix_sub_len * S_seq, head_dim).
            k_combined = torch.cat([prefix_k_tensor, k_current], dim=-2)
            v_combined = torch.cat([prefix_v_tensor, v_current], dim=-2)

        # Calculate attention scores
        # (B_effective, H, S_seq_Q, head_dim) @ (B_effective, H, head_dim, Combined_S_seq_K) -> (B_effective, H, S_seq_Q, Combined_S_seq_K)
        sim = torch.matmul(q_current, k_combined.transpose(-1, -2)) * self.scale

        # Softmax to get attention probabilities
        attn = F.softmax(sim, dim=-1)

        # Weighted sum of values
        # (B_effective, H, S_seq_Q, Combined_S_seq_K) @ (B_effective, H, Combined_S_seq_K, head_dim) -> (B_effective, H, S_seq_Q, head_dim)
        out = torch.matmul(attn, v_combined)

        # Reshape back to original dimension:
        # (B_effective, H, S_seq_Q, head_dim) -> (B_effective, S_seq_Q, H, head_dim) -> (B_effective, S_seq_Q, D)
        out = out.transpose(1, 2).contiguous().view(out.shape[0], -1, self.dim)

        # Final output projection
        return self.to_out(out)

