## model/attention.py
"""Multi-head self-attention with QK-Norm and RoPE for OLMoE.

Implements the attention layer used in every OLMoEBlock. Key features
distinguishing this from vanilla multi-head attention:

  1. QK-Norm: RMSNorm applied to Q and K per-head, BEFORE RoPE (Section 4.2.5)
  2. RoPE: Rotary position embeddings applied after QK-Norm (Table 10)
  3. No biases: All linear projections use bias=False (Table 10)
  4. Full attention: No GQA/MQA, 16 independent heads (Table 10)
  5. Flash attention: Uses F.scaled_dot_product_attention with is_causal=True

Configuration values used (from config.yaml):
  model.hidden_dim: 2048
  model.num_heads: 16
  model.max_seq_len: 4096
  model.rope_theta: 10000.0
  model.use_qk_norm: true
  model.rms_norm_eps: 1.0e-05
  model.use_bias: false

Derived constants:
  head_dim = hidden_dim // num_heads = 2048 // 16 = 128
  scale = 1 / sqrt(head_dim) = 1 / sqrt(128) ≈ 0.0884 (applied by SDPA)

Data flow:
  x: (B, S, 2048)
      ↓ q_proj, k_proj, v_proj  [bias=False]
  q, k, v: (B, S, 2048)
      ↓ reshape + transpose
  q, k, v: (B, 16, S, 128)
      ↓ q_norm, k_norm  [RMSNorm(128), if use_qk_norm=True]
  q, k: (B, 16, S, 128)
      ↓ apply_rotary (RoPE)
  q, k: (B, 16, S, 128)
      ↓ F.scaled_dot_product_attention(is_causal=True)
  attn_out: (B, 16, S, 128)
      ↓ transpose + contiguous + view
  attn_out: (B, S, 2048)
      ↓ o_proj  [bias=False]
  output: (B, S, 2048)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import OLMoEConfig
from model.rms_norm import RMSNorm
from model.rope import RotaryEmbedding


class OLMoEAttention(nn.Module):
    """Multi-head self-attention layer for OLMoE with QK-Norm and RoPE.

    This module implements the attention sub-layer of each OLMoEBlock.
    It is called with pre-normalized input (the block applies attn_norm
    before calling this module) and returns the attention output of the
    same shape, which is then added to the residual stream.

    Attributes:
        hidden_dim: Model hidden dimension (2048 for OLMoE-1B-7B).
        num_heads: Number of attention heads (16 for OLMoE-1B-7B).
        head_dim: Per-head dimension = hidden_dim // num_heads = 128.
        use_qk_norm: Whether to apply RMSNorm to Q and K before RoPE.
        q_proj: Query projection, Linear(2048, 2048, bias=False).
        k_proj: Key projection, Linear(2048, 2048, bias=False).
        v_proj: Value projection, Linear(2048, 2048, bias=False).
        o_proj: Output projection, Linear(2048, 2048, bias=False).
        q_norm: RMSNorm(128) for query normalization (if use_qk_norm).
        k_norm: RMSNorm(128) for key normalization (if use_qk_norm).
        rope: RotaryEmbedding(dim=128, max_seq_len=4096, theta=10000.0).

    Example:
        >>> config = OLMoEConfig()
        >>> attn = OLMoEAttention(config)
        >>> x = torch.randn(2, 4096, 2048)
        >>> out = attn(x)
        >>> out.shape
        torch.Size([2, 4096, 2048])
    """

    def __init__(self, config: OLMoEConfig) -> None:
        """Initialize OLMoEAttention.

        Creates all projection layers, optional QK-Norm modules, and the
        shared RoPE instance. All linear layers use bias=False per Table 10.

        Args:
            config: OLMoEConfig instance containing all architecture
                    hyperparameters. Key fields used:
                    - hidden_dim (2048): input/output dimension
                    - num_heads (16): number of attention heads
                    - max_seq_len (4096): for RoPE cache precomputation
                    - rope_theta (10000.0): RoPE base frequency
                    - use_qk_norm (True): whether to apply QK-Norm
                    - rms_norm_eps (1e-5): epsilon for RMSNorm
                    - use_bias (False): whether to use biases in projections
        """
        super().__init__()

        self.hidden_dim: int = config.hidden_dim
        self.num_heads: int = config.num_heads
        self.head_dim: int = config.hidden_dim // config.num_heads
        self.use_qk_norm: bool = config.use_qk_norm

        # Validate that hidden_dim is divisible by num_heads.
        # This is also checked in OLMoEConfig.__post_init__ but we guard here
        # for safety when config is constructed manually.
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads}). "
                f"Got head_dim = {self.hidden_dim / self.num_heads}."
            )

        # ------------------------------------------------------------------
        # Linear projections — all bias=False per Table 10 ("Biases: -")
        # ------------------------------------------------------------------
        # Query projection: maps hidden_dim -> hidden_dim (all heads combined)
        self.q_proj: nn.Linear = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=config.use_bias
        )
        # Key projection: maps hidden_dim -> hidden_dim
        self.k_proj: nn.Linear = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=config.use_bias
        )
        # Value projection: maps hidden_dim -> hidden_dim
        self.v_proj: nn.Linear = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=config.use_bias
        )
        # Output projection: maps hidden_dim -> hidden_dim
        self.o_proj: nn.Linear = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=config.use_bias
        )

        # ------------------------------------------------------------------
        # QK-Norm (Section 4.2.5, config.yaml: use_qk_norm: true)
        #
        # Applied to Q and K per-head BEFORE RoPE. The norm operates on
        # the head_dim=128 dimension. When applied to tensors of shape
        # (B, num_heads, S, head_dim), RMSNorm normalizes each 128-dim
        # vector independently (last dimension), which is exactly per-head
        # normalization.
        #
        # These parameters ARE subject to weight_decay=0.1 per Section 4.2.3.
        # The optimizer in training/optimizer.py must NOT exclude them.
        # ------------------------------------------------------------------
        if self.use_qk_norm:
            self.q_norm: Optional[RMSNorm] = RMSNorm(
                dim=self.head_dim, eps=config.rms_norm_eps
            )
            self.k_norm: Optional[RMSNorm] = RMSNorm(
                dim=self.head_dim, eps=config.rms_norm_eps
            )
        else:
            self.q_norm = None
            self.k_norm = None

        # ------------------------------------------------------------------
        # Rotary Position Embedding (Table 10: pos_emb = RoPE, theta = 10000)
        #
        # One shared RoPE instance per attention layer. The cos/sin caches
        # are precomputed in RotaryEmbedding.__init__ for all positions up
        # to max_seq_len=4096 and sliced at runtime.
        # ------------------------------------------------------------------
        self.rope: RotaryEmbedding = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute multi-head self-attention with QK-Norm and RoPE.

        The input x is expected to be pre-normalized (the OLMoEBlock applies
        attn_norm before calling this method). The output is added to the
        residual stream by the block.

        Args:
            x: Pre-normalized input tensor of shape (batch, seq_len, hidden_dim).
               For OLMoE-1B-7B: (B, S, 2048) where S <= 4096.
            attention_mask: Optional attention mask tensor. If None, a causal
               mask is applied automatically via is_causal=True in SDPA.
               If provided, should be a float tensor of shape broadcastable to
               (B, num_heads, S, S) with 0.0 for attended positions and
               -inf for masked positions. When provided, is_causal is set to
               False and the mask is passed directly to SDPA.
            position_ids: Optional position indices of shape (batch, seq_len)
               or (seq_len,). If None, positions are assumed to be
               0, 1, ..., seq_len-1 (standard sequential positions).
               Used to index into the RoPE cos/sin cache for non-contiguous
               positions (e.g., during generation or with packed sequences).

        Returns:
            Attention output tensor of shape (batch, seq_len, hidden_dim),
            same shape and dtype as the input x.

        Shape example:
            Input:  (2, 4096, 2048)
            Output: (2, 4096, 2048)
        """
        batch_size: int = x.shape[0]
        seq_len: int = x.shape[1]

        # ------------------------------------------------------------------
        # Step 1: Project input to Q, K, V
        # All projections: (B, S, hidden_dim) -> (B, S, hidden_dim)
        # ------------------------------------------------------------------
        q: Tensor = self.q_proj(x)  # (B, S, 2048)
        k: Tensor = self.k_proj(x)  # (B, S, 2048)
        v: Tensor = self.v_proj(x)  # (B, S, 2048)

        # ------------------------------------------------------------------
        # Step 2: Reshape to multi-head format
        # (B, S, hidden_dim) -> (B, S, num_heads, head_dim) -> (B, num_heads, S, head_dim)
        # For OLMoE-1B-7B: (B, S, 2048) -> (B, 16, S, 128)
        # ------------------------------------------------------------------
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (B, num_heads, S, head_dim) = (B, 16, S, 128)

        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)  # (B, 16, S, 128)

        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.transpose(1, 2)  # (B, 16, S, 128)

        # ------------------------------------------------------------------
        # Step 3: Apply QK-Norm BEFORE RoPE (Section 4.2.5)
        #
        # RMSNorm normalizes over the last dimension (head_dim=128).
        # Applied to (B, 16, S, 128) tensors: normalizes each 128-dim
        # vector independently, which is per-head normalization.
        #
        # Placement: BEFORE RoPE, as applying norm after RoPE would
        # interfere with the rotary structure. The norm stabilizes Q/K
        # magnitudes before rotation to prevent large attention logits.
        # ------------------------------------------------------------------
        if self.use_qk_norm and self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)  # (B, 16, S, 128) -> (B, 16, S, 128)
            k = self.k_norm(k)  # (B, 16, S, 128) -> (B, 16, S, 128)

        # ------------------------------------------------------------------
        # Step 4: Apply Rotary Position Embeddings to Q and K
        #
        # Get cos/sin from the precomputed cache, then apply rotation.
        # Two cases:
        #   (a) position_ids is None: standard sequential positions 0..S-1
        #       -> use rope(seq_len) which slices cache[:seq_len]
        #   (b) position_ids is provided: non-contiguous positions
        #       -> index into the full cache using position_ids
        # ------------------------------------------------------------------
        if position_ids is None:
            # Standard case: sequential positions 0, 1, ..., seq_len-1
            cos, sin = self.rope(seq_len=seq_len)
            # cos, sin: (seq_len, head_dim) = (S, 128)
            q, k = RotaryEmbedding.apply_rotary(q, k, cos, sin)
        else:
            # Non-contiguous positions (e.g., generation, packed sequences)
            # position_ids: (batch, seq_len) or (seq_len,)
            # Get full cache up to max_seq_len
            cos_full, sin_full = self.rope(seq_len=self.rope.max_seq_len)
            # cos_full, sin_full: (max_seq_len, head_dim)

            # Flatten position_ids for indexing, then reshape for broadcasting
            if position_ids.dim() == 1:
                # (seq_len,) -> index directly
                cos = cos_full[position_ids]  # (seq_len, head_dim)
                sin = sin_full[position_ids]  # (seq_len, head_dim)
            else:
                # (batch, seq_len) -> index per batch item
                # For simplicity, use the first batch item's positions
                # (in practice, all items in a packed batch have the same positions)
                cos = cos_full[position_ids[0]]  # (seq_len, head_dim)
                sin = sin_full[position_ids[0]]  # (seq_len, head_dim)

            q, k = RotaryEmbedding.apply_rotary(q, k, cos, sin)

        # ------------------------------------------------------------------
        # Step 5: Scaled dot-product attention
        #
        # Uses F.scaled_dot_product_attention (PyTorch >= 2.0) which:
        #   - Dispatches to Flash Attention when available (CUDA, BF16/FP16)
        #   - Applies 1/sqrt(head_dim) scaling internally
        #   - Handles causal masking via is_causal=True
        #
        # Causal mask strategy:
        #   - If attention_mask is None: use is_causal=True (efficient path)
        #   - If attention_mask is provided: use is_causal=False and pass
        #     the mask directly (handles variable-length sequences, eval, etc.)
        #
        # No dropout: config.dropout=0.0, paper does not mention attn dropout.
        # ------------------------------------------------------------------
        if attention_mask is None:
            # Standard causal attention — most efficient path
            attn_output: Tensor = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
        else:
            # Explicit mask provided — disable is_causal to avoid double masking
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        # attn_output shape: (B, num_heads, S, head_dim) = (B, 16, S, 128)

        # ------------------------------------------------------------------
        # Step 6: Reshape attention output and project
        #
        # (B, num_heads, S, head_dim) -> (B, S, num_heads, head_dim) -> (B, S, hidden_dim)
        # .contiguous() is required before .view() because transpose creates
        # a non-contiguous tensor.
        # ------------------------------------------------------------------
        attn_output = attn_output.transpose(1, 2)  # (B, S, num_heads, head_dim)
        attn_output = attn_output.contiguous()      # ensure contiguous memory layout
        attn_output = attn_output.view(
            batch_size, seq_len, self.hidden_dim
        )  # (B, S, 2048)

        # Final output projection: (B, S, 2048) -> (B, S, 2048)
        output: Tensor = self.o_proj(attn_output)

        return output

    def extra_repr(self) -> str:
        """Return extra representation string for printing the module.

        Returns:
            String showing key configuration values.
        """
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"use_qk_norm={self.use_qk_norm}"
        )
