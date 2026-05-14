"""
Gated Attention for Large Language Models
==========================================
Implements the gating variants described in:
  "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"

Key contributions:
  - Five gating positions: G1 (SDPA output), G2 (value), G3 (key), G4 (query), G5 (dense output)
  - Two granularities: elementwise and headwise
  - Head-specific vs head-shared gating
  - Multiplicative vs additive gating
  - Sigmoid vs SiLU activation
  - NS-sigmoid (non-sparse sigmoid) variant for ablation

The best-performing variant is head-specific, elementwise, multiplicative sigmoid gating
applied after SDPA output (G1).
"""

import math
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingPosition(Enum):
    """Position where gating is applied in the attention layer."""
    G1_SDPA_OUTPUT = "sdpa_output"   # After SDPA (best performing)
    G2_VALUE = "value"               # After value projection
    G3_KEY = "key"                   # After key projection
    G4_QUERY = "query"               # After query projection
    G5_DENSE_OUTPUT = "dense_output" # After final output projection


class GatingGranularity(Enum):
    """Granularity of gating scores."""
    ELEMENTWISE = "elementwise"  # Per-dimension gating (n x q x d_k)
    HEADWISE = "headwise"        # Per-head scalar gating (n x q)


class GatingType(Enum):
    """Whether gating is multiplicative or additive."""
    MULTIPLICATIVE = "multiplicative"
    ADDITIVE = "additive"


class GatingActivation(Enum):
    """Activation function for gating scores."""
    SIGMOID = "sigmoid"
    SILU = "silu"
    IDENTITY = "identity"
    RMSNORM = "rmsnorm"
    NS_SIGMOID = "ns_sigmoid"  # Non-sparse sigmoid: 0.5 + 0.5 * sigmoid(x)


def ns_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Non-sparse sigmoid: constrains gating scores to [0.5, 1.0].
    
    Used in ablation study (Sec 4.2) to show that sparsity is important.
    NS-sigmoid(x) = 0.5 + 0.5 * sigmoid(x)
    """
    return 0.5 + 0.5 * torch.sigmoid(x)


class HeadRMSNorm(nn.Module):
    """RMSNorm applied per attention head.
    
    Used as a non-linearity between W_V and W_O (Sec 4.1).
    Inspired by Sun et al. (2023) and Ye et al. (2024).
    """
    
    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(head_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., head_dim)
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class GatingModule(nn.Module):
    """
    A gating module that can be applied at various positions in the attention layer.
    
    Implements: Y' = Y * sigma(X * W_theta)  [multiplicative]
             or Y' = Y + sigma(X * W_theta)  [additive]
    
    where sigma is the activation function, X is the input used to compute gating scores,
    and W_theta are learnable parameters.
    
    Args:
        d_model: Model hidden dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads (for GQA)
        head_dim: Dimension per head
        position: Where gating is applied
        granularity: Elementwise or headwise gating
        head_specific: Whether each head has its own gating scores
        gating_type: Multiplicative or additive
        activation: Activation function for gating scores
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        position: GatingPosition = GatingPosition.G1_SDPA_OUTPUT,
        granularity: GatingGranularity = GatingGranularity.ELEMENTWISE,
        head_specific: bool = True,
        gating_type: GatingType = GatingType.MULTIPLICATIVE,
        activation: GatingActivation = GatingActivation.SIGMOID,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.position = position
        self.granularity = granularity
        self.head_specific = head_specific
        self.gating_type = gating_type
        self.activation = activation
        
        # Determine output dimension of gating scores
        # For G1 (SDPA output) and G4 (query): num_heads heads
        # For G2 (value) and G3 (key): num_kv_heads heads
        # For G5 (dense output): d_model
        if position in (GatingPosition.G1_SDPA_OUTPUT, GatingPosition.G4_QUERY):
            target_heads = num_heads
        elif position in (GatingPosition.G2_VALUE, GatingPosition.G3_KEY):
            target_heads = num_kv_heads
        else:  # G5
            target_heads = None
        
        # Compute gate output size
        if position == GatingPosition.G5_DENSE_OUTPUT:
            gate_out_dim = d_model
        elif granularity == GatingGranularity.ELEMENTWISE:
            if head_specific:
                gate_out_dim = target_heads * head_dim
            else:
                # Head-shared: single set of scores shared across heads
                gate_out_dim = head_dim
        else:  # HEADWISE
            if head_specific:
                gate_out_dim = target_heads
            else:
                gate_out_dim = 1
        
        self.gate_out_dim = gate_out_dim
        self.target_heads = target_heads
        
        # Gate projection: maps from d_model to gate_out_dim
        if activation == GatingActivation.RMSNORM:
            # RMSNorm doesn't need a projection; applied directly to SDPA output
            self.norm = HeadRMSNorm(head_dim)
            self.gate_proj = None
        else:
            self.gate_proj = nn.Linear(d_model, gate_out_dim, bias=False)
            self.norm = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize gate projection weights to zero.
        
        Zero initialization ensures the model starts as the baseline
        (sigmoid(0) = 0.5, so initial gating is 0.5 * Y, but this
        gets learned quickly during training).
        """
        if self.gate_proj is not None:
            nn.init.zeros_(self.gate_proj.weight)
    
    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gating activation function."""
        if self.activation == GatingActivation.SIGMOID:
            return torch.sigmoid(x)
        elif self.activation == GatingActivation.SILU:
            return F.silu(x)
        elif self.activation == GatingActivation.IDENTITY:
            return x
        elif self.activation == GatingActivation.NS_SIGMOID:
            return ns_sigmoid(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def forward(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        num_heads: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply gating to y using x to compute gating scores.
        
        Args:
            y: Tensor to be gated. Shape depends on position:
               - G1: (batch, seq, num_heads, head_dim)
               - G2/G3: (batch, seq, num_kv_heads, head_dim)
               - G5: (batch, seq, d_model)
            x: Input used to compute gating scores. Shape: (batch, seq, d_model)
            num_heads: Override for number of heads (used for G1 with GQA)
        
        Returns:
            Gated output with same shape as y.
        """
        if self.activation == GatingActivation.RMSNORM:
            return self._apply_rmsnorm_gating(y)
        
        batch_size, seq_len = x.shape[:2]
        
        # Compute raw gating scores from input x
        gate_scores = self.gate_proj(x)  # (batch, seq, gate_out_dim)
        gate_scores = self._apply_activation(gate_scores)
        
        # Reshape gate scores to match y's shape
        if self.position == GatingPosition.G5_DENSE_OUTPUT:
            # y: (batch, seq, d_model), gate_scores: (batch, seq, d_model)
            pass
        elif self.granularity == GatingGranularity.ELEMENTWISE:
            if self.head_specific:
                # gate_scores: (batch, seq, target_heads * head_dim)
                # -> (batch, seq, target_heads, head_dim)
                gate_scores = gate_scores.view(
                    batch_size, seq_len, self.target_heads, self.head_dim
                )
            else:
                # Head-shared: (batch, seq, head_dim) -> broadcast over heads
                gate_scores = gate_scores.view(
                    batch_size, seq_len, 1, self.head_dim
                )
        else:  # HEADWISE
            if self.head_specific:
                # gate_scores: (batch, seq, target_heads)
                # -> (batch, seq, target_heads, 1) for broadcasting
                gate_scores = gate_scores.view(
                    batch_size, seq_len, self.target_heads, 1
                )
            else:
                gate_scores = gate_scores.view(batch_size, seq_len, 1, 1)
        
        # Apply gating
        if self.gating_type == GatingType.MULTIPLICATIVE:
            return y * gate_scores
        else:  # ADDITIVE
            return y + gate_scores
    
    def _apply_rmsnorm_gating(self, y: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to each head's output (non-linearity without parameters).
        
        This corresponds to the SDPA GroupNorm variant in Table 3 (row 5).
        Applying RMSNorm independently to each head's output introduces
        non-linearity between W_V and W_O without adding gate parameters.
        """
        # y: (batch, seq, num_heads, head_dim)
        return self.norm(y)


class GatedMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with optional gating at various positions.
    
    Supports Group Query Attention (GQA) as used in the paper's experiments.
    
    The default best-performing configuration is:
      - position=G1_SDPA_OUTPUT
      - granularity=ELEMENTWISE
      - head_specific=True
      - gating_type=MULTIPLICATIVE
      - activation=SIGMOID
    
    Args:
        d_model: Model hidden dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads (for GQA; if None, uses num_heads)
        head_dim: Dimension per head (if None, computed as d_model // num_heads)
        dropout: Attention dropout probability
        gating_position: Where to apply gating (None = no gating = baseline)
        gating_granularity: Elementwise or headwise
        head_specific: Whether each head has its own gating scores
        gating_type: Multiplicative or additive
        gating_activation: Activation function for gating scores
        max_seq_len: Maximum sequence length (for RoPE)
        rope_base: RoPE base frequency
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        gating_position: Optional[GatingPosition] = GatingPosition.G1_SDPA_OUTPUT,
        gating_granularity: GatingGranularity = GatingGranularity.ELEMENTWISE,
        head_specific: bool = True,
        gating_type: GatingType = GatingType.MULTIPLICATIVE,
        gating_activation: GatingActivation = GatingActivation.SIGMOID,
        max_seq_len: int = 4096,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim if head_dim is not None else d_model // num_heads
        self.dropout = dropout
        self.gating_position = gating_position
        self.scale = self.head_dim ** -0.5
        
        # GQA: number of query heads per KV head
        self.num_groups = self.num_heads // self.num_kv_heads
        
        # QKV projections
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        
        # Gating module (if enabled)
        self.gate = None
        if gating_position is not None:
            self.gate = GatingModule(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                position=gating_position,
                granularity=gating_granularity,
                head_specific=head_specific,
                gating_type=gating_type,
                activation=gating_activation,
            )
        
        # RoPE embeddings
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base
        self._build_rope_cache(max_seq_len)
    
    def _build_rope_cache(self, seq_len: int):
        """Build rotary position embedding cache."""
        theta = 1.0 / (
            self.rope_base ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
            )
        )
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, theta)
        # cos and sin: (seq_len, head_dim/2)
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)
    
    def _apply_rope(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply rotary position embeddings to x.
        
        Args:
            x: (batch, seq, num_heads, head_dim)
            offset: Position offset for KV cache
        """
        seq_len = x.shape[1]
        cos = self.rope_cos[offset:offset + seq_len]  # (seq, head_dim/2)
        sin = self.rope_sin[offset:offset + seq_len]
        
        # Split into even/odd pairs
        x1 = x[..., ::2]   # (batch, seq, heads, head_dim/2)
        x2 = x[..., 1::2]
        
        # Reshape for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim/2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        
        # Standard RoPE rotation
        out = torch.empty_like(x)
        out[..., ::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x2 * cos + x1 * sin
        return out
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of gated multi-head attention.
        
        Args:
            x: Input tensor (batch, seq, d_model)
            attention_mask: Optional mask (batch, 1, seq, seq) or (batch, seq)
            position_ids: Optional position IDs for RoPE
            past_key_value: Optional KV cache tuple
            use_cache: Whether to return updated KV cache
        
        Returns:
            output: (batch, seq, d_model)
            past_key_value: Updated KV cache if use_cache=True
        """
        batch_size, seq_len, _ = x.shape
        
        # QKV projections
        q = self.q_proj(x)  # (batch, seq, num_heads * head_dim)
        k = self.k_proj(x)  # (batch, seq, num_kv_heads * head_dim)
        v = self.v_proj(x)  # (batch, seq, num_kv_heads * head_dim)
        
        # Reshape to (batch, seq, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply gating at query/key/value positions if configured
        if self.gate is not None:
            if self.gating_position == GatingPosition.G4_QUERY:
                q = self.gate(q, x)
            elif self.gating_position == GatingPosition.G3_KEY:
                k = self.gate(k, x)
            elif self.gating_position == GatingPosition.G2_VALUE:
                v = self.gate(v, x)
        
        # Apply RoPE
        q = self._apply_rope(q)
        k = self._apply_rope(k)
        
        # Handle KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        
        new_past_key_value = (k, v) if use_cache else None
        kv_seq_len = k.shape[1]
        
        # Expand KV for GQA: repeat KV heads to match query heads
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=2)
            v = v.repeat_interleave(self.num_groups, dim=2)
        
        # Transpose for attention: (batch, num_heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        else:
            # Create causal mask
            causal_mask = torch.full(
                (seq_len, kv_seq_len),
                float('-inf'),
                device=x.device,
                dtype=x.dtype,
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        if self.dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # Weighted sum of values: (batch, num_heads, seq, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # Transpose back: (batch, seq, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Apply gating at SDPA output (G1) - the best-performing position
        # This is the key contribution: head-specific sigmoid gate after SDPA
        if self.gate is not None and self.gating_position == GatingPosition.G1_SDPA_OUTPUT:
            attn_output = self.gate(attn_output, x)
        
        # Concatenate heads: (batch, seq, num_heads * head_dim)
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # Final output projection
        output = self.o_proj(attn_output)
        
        # Apply gating at dense output (G5)
        if self.gate is not None and self.gating_position == GatingPosition.G5_DENSE_OUTPUT:
            output = self.gate(output, x)
        
        return output, new_past_key_value
    
    def get_num_gate_params(self) -> int:
        """Return number of parameters added by gating."""
        if self.gate is None or self.gate.gate_proj is None:
            return 0
        return sum(p.numel() for p in self.gate.gate_proj.parameters())
