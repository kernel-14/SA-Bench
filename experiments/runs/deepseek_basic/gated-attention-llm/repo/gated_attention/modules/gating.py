"""
Gated Attention Mechanisms for Large Language Models.

This module implements the gating variants explored in the paper:
"Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"

Key gating positions (Fig. 1):
  G1: After SDPA output (most effective)
  G2: After value projection
  G3: After key projection
  G4: After query projection
  G5: After dense output projection

Gating granularities:
  - Headwise: Single scalar per attention head
  - Elementwise: Per-dimension modulation

Gating modes:
  - Multiplicative: Y' = Y * sigma(X @ W_theta)
  - Additive: Y' = Y + sigma(X @ W_theta)

Activation functions:
  - Sigmoid: sigma(x) = 1/(1+exp(-x)), range [0,1]
  - SiLU: x * sigmoid(x), unbounded range
  - NS-sigmoid: 0.5 + 0.5*sigmoid(x), range [0.5, 1.0]
"""

import math
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingPosition(Enum):
    """Position where gating is applied in the attention computation."""
    NONE = "none"                # No gating (baseline)
    G1_SDPA_OUTPUT = "g1"       # After Scaled Dot-Product Attention (Eq 3)
    G2_VALUE = "g2"              # After value projection (Eq 1, V)
    G3_KEY = "g3"                # After key projection (Eq 1, K)
    G4_QUERY = "g4"              # After query projection (Eq 1, Q)
    G5_DENSE_OUTPUT = "g5"       # After final dense output layer (Eq 4)


class GatingGranularity(Enum):
    """Granularity of the gating scores."""
    HEADWISE = "headwise"        # One scalar per head: shape (n, num_heads)
    ELEMENTWISE = "elementwise"  # Per-dimension: shape (n, num_heads * head_dim)


class GatingMode(Enum):
    """How the gate modulates the target."""
    MULTIPLICATIVE = "multiplicative"  # Y' = Y * gate
    ADDITIVE = "additive"              # Y' = Y + gate


class GatingScope(Enum):
    """Whether each head has its own gate or shares."""
    HEAD_SPECIFIC = "head_specific"    # Each head has distinct gating scores
    HEAD_SHARED = "head_shared"        # Gating scores shared across heads


class ActivationType(Enum):
    """Activation function for computing gating scores."""
    SIGMOID = "sigmoid"
    SILU = "silu"
    IDENTITY = "identity"
    NS_SIGMOID = "ns_sigmoid"  # Non-sparse sigmoid: 0.5 + 0.5 * sigmoid(x)


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def _get_activation(act_type: ActivationType):
    """Return the activation function corresponding to the given type."""
    if act_type == ActivationType.SIGMOID:
        return torch.sigmoid
    elif act_type == ActivationType.SILU:
        return F.silu
    elif act_type == ActivationType.IDENTITY:
        return lambda x: x
    elif act_type == ActivationType.NS_SIGMOID:
        return lambda x: 0.5 + 0.5 * torch.sigmoid(x)
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


# ---------------------------------------------------------------------------
# Gating parameter projection
# ---------------------------------------------------------------------------

class GateProjection(nn.Module):
    """Learnable linear projection for computing gating scores.

    Given input X (n, d_model), computes gate = activation(X @ W_theta).

    The shape of W_theta depends on:
      - score_shape: dimensionality of the gating scores
      - scope: whether head-specific or head-shared
    """

    def __init__(
        self,
        d_model: int,
        score_dim: int,
        activation: ActivationType = ActivationType.SIGMOID,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.score_dim = score_dim
        self.activation_type = activation
        self.act_fn = _get_activation(activation)
        self.weight = nn.Parameter(torch.empty(d_model, score_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(score_dim))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)
        # Following the paper, gate parameters can be zero-initialized
        # for input-independent gating experiments (Sec 4.2, row 6)

    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, score_dim={self.score_dim}, "
                f"activation={self.activation_type.value}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model) or (seq_len, d_model)
        gate_logits = torch.matmul(x, self.weight)
        if self.bias is not None:
            gate_logits = gate_logits + self.bias
        return self.act_fn(gate_logits)


# ---------------------------------------------------------------------------
# Main Gated Attention Module
# ---------------------------------------------------------------------------

@dataclass
class GatedAttentionConfig:
    """Configuration for a gated attention layer.

    This captures all variants explored in the paper (Tables 1-4).
    """
    # Position of gating
    position: GatingPosition = GatingPosition.G1_SDPA_OUTPUT
    # Granularity of gating scores
    granularity: GatingGranularity = GatingGranularity.ELEMENTWISE
    # Multiplicative or additive
    mode: GatingMode = GatingMode.MULTIPLICATIVE
    # Head-specific or head-shared
    scope: GatingScope = GatingScope.HEAD_SPECIFIC
    # Activation function
    activation: ActivationType = ActivationType.SIGMOID

    # Attention architecture
    d_model: int = 2048
    num_heads: int = 32
    num_kv_heads: int = 4        # For GQA
    head_dim: int = 128
    max_seq_len: int = 4096

    # Gating specifics
    gate_bias: bool = False
    # Input source for gate computation:
    #   'query': use the current query hidden state (query-dependent)
    #   'key_value': use the key/value hidden state
    gate_input_source: str = "query"

    @property
    def num_query_heads(self) -> int:
        return self.num_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.num_kv_heads

    @property
    def num_kv_groups(self) -> int:
        """Number of query heads per key-value head (for GQA)."""
        return self.num_heads // self.num_kv_heads


class GatedAttention(nn.Module):
    """Multi-head gated attention supporting all variants from the paper.

    Implements the full gated attention mechanism described in Sections 2-4.
    Supports:
      - Grouped Query Attention (GQA)
      - Rotary Position Embeddings (RoPE)
      - Five gating positions (G1-G5)
      - Headwise/elementwise gating
      - Head-specific/head-shared gating
      - Multiplicative/additive gating
      - Sigmoid/SiLU/NS-sigmoid/Identity activations
    """

    def __init__(self, config: GatedAttentionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups

        # QKV projections
        self.q_proj = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)

        # Build gate projections based on position and configuration
        self.gate_proj = self._build_gate_projection()

        # RoPE (optional, can be set externally)
        self.rope_cache = None

    def _build_gate_projection(self) -> Optional[nn.Module]:
        """Build the gate projection layer(s) based on configuration."""
        cfg = self.config

        # No gating for baseline
        if cfg.position == GatingPosition.NONE:
            return None

        # G5 is applied after the dense output, which is at d_model dimension
        if cfg.position == GatingPosition.G5_DENSE_OUTPUT:
            return GateProjection(
                d_model=self.d_model,
                score_dim=self.d_model,
                activation=cfg.activation,
                bias=cfg.gate_bias,
            )

        # Determine score shape based on position, granularity, and scope
        score_dim = self._compute_score_dim()

        if score_dim == 0:
            return None  # No gating needed

        return GateProjection(
            d_model=self.d_model,
            score_dim=score_dim,
            activation=cfg.activation,
            bias=cfg.gate_bias,
        )

    def _compute_score_dim(self) -> int:
        """Compute the dimensionality of gating scores.

        Following Table 1 'Score Shape' column:
          - SDPA Elementwise G1: n × q × dk  (q * head_dim)
          - SDPA Headwise G1: n × q          (q)
          - V Elementwise G2: n × k × dk     (k * head_dim, where k = num_kv_heads)
          - K Elementwise G3: n × k × dk
          - Q Elementwise G4: n × q × dk
          - Dense Output G5: n × d_model
        """
        cfg = self.config
        if cfg.position == GatingPosition.NONE:
            return 0
        is_headwise = (cfg.granularity == GatingGranularity.HEADWISE)
        is_shared = (cfg.scope == GatingScope.HEAD_SHARED)

        if cfg.position in (GatingPosition.G1_SDPA_OUTPUT, GatingPosition.G4_QUERY):
            n_heads = self.num_heads  # query heads
        elif cfg.position in (GatingPosition.G2_VALUE, GatingPosition.G3_KEY):
            n_heads = self.num_kv_heads  # key-value heads
        else:
            n_heads = self.num_heads

        if is_headwise:
            if is_shared:
                return 1     # Single scalar shared across all heads
            else:
                return n_heads  # One scalar per head
        else:  # elementwise
            if is_shared:
                return self.head_dim  # Per-head-dim, shared across heads
            else:
                return n_heads * self.head_dim  # Full elementwise

    def _reshape_for_gating(
        self, x: torch.Tensor, num_heads: int
    ) -> torch.Tensor:
        """Reshape outputs for gating application.

        Returns tensor of shape (batch, seq_len, num_heads, head_dim).
        """
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, num_heads, self.head_dim)

    def _compute_gate_scores(
        self, hidden_states: torch.Tensor, num_heads: int
    ) -> torch.Tensor:
        """Compute gating scores from hidden states.

        Args:
            hidden_states: (batch, seq_len, d_model)
            num_heads: number of heads to gate over

        Returns:
            gate_scores: shape depends on configuration:
              - headwise head-specific: (batch, seq_len, num_heads)
              - headwise head-shared: (batch, seq_len, 1) broadcast to all heads
              - elementwise head-specific: (batch, seq_len, num_heads, head_dim)
              - elementwise head-shared: (batch, seq_len, head_dim) broadcast
        """
        cfg = self.config
        if self.gate_proj is None:
            return None

        gate_logits = self.gate_proj(hidden_states)
        # gate_logits: (batch, seq_len, score_dim)

        is_headwise = (cfg.granularity == GatingGranularity.HEADWISE)
        is_shared = (cfg.scope == GatingScope.HEAD_SHARED)

        if is_headwise:
            if is_shared:
                # Single scalar -> (batch, seq_len, 1, 1) for proper broadcasting
                gate_scores = gate_logits.view(*gate_logits.shape[:2], 1, 1)
            else:
                # Per-head scalar -> (batch, seq_len, num_heads, 1) for broadcasting over head_dim
                gate_scores = gate_logits.view(*gate_logits.shape[:2], num_heads, 1)
        else:  # elementwise
            if is_shared:
                # Per-head-dim -> (batch, seq_len, 1, head_dim) for broadcasting over heads
                gate_scores = gate_logits.view(*gate_logits.shape[:2], 1, self.head_dim)
            else:
                # Per-head per-dim -> (batch, seq_len, num_heads, head_dim)
                gate_scores = gate_logits.view(
                    *gate_logits.shape[:2], num_heads, self.head_dim
                )

        return gate_scores

    def _apply_gate(
        self, target: torch.Tensor, gate_scores: torch.Tensor
    ) -> torch.Tensor:
        """Apply gating modulation to target tensor.

        Args:
            target: tensor to be gated
            gate_scores: gating scores (will broadcast)

        Returns:
            gated output: target * gate (multiplicative) or target + gate (additive)
        """
        if gate_scores is None:
            return target

        if self.config.mode == GatingMode.MULTIPLICATIVE:
            return target * gate_scores
        elif self.config.mode == GatingMode.ADDITIVE:
            return target + gate_scores
        else:
            raise ValueError(f"Unknown gating mode: {self.config.mode}")

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value heads to match query heads for GQA."""
        batch, seq_len, num_kv_heads, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, None, :].expand(
            batch, seq_len, num_kv_heads, n_rep, head_dim
        )
        return hidden_states.reshape(batch, seq_len, num_kv_heads * n_rep, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        """Forward pass implementing the gated attention.

        Returns:
            attn_output: gated attention output
            attn_weights: attention weights (if output_attentions=True)
            past_key_value: updated KV cache (if use_cache=True)
        """
        cfg = self.config
        batch, seq_len, _ = hidden_states.shape

        # Stage 1: QKV projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape: (batch, seq_len, num_heads * head_dim) -> (batch, seq_len, num_heads, head_dim)
        query_states = query_states.view(batch, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # --- Gating at G4 (after Q projection) ---
        if cfg.position == GatingPosition.G4_QUERY:
            gate_scores = self._compute_gate_scores(hidden_states, self.num_heads)
            query_states = self._apply_gate(query_states, gate_scores)
        # --- Gating at G3 (after K projection) ---
        if cfg.position == GatingPosition.G3_KEY:
            gate_scores = self._compute_gate_scores(hidden_states, self.num_kv_heads)
            key_states = self._apply_gate(key_states, gate_scores)
        # --- Gating at G2 (after V projection) ---
        if cfg.position == GatingPosition.G2_VALUE:
            gate_scores = self._compute_gate_scores(hidden_states, self.num_kv_heads)
            value_states = self._apply_gate(value_states, gate_scores)

        # Apply RoPE if provided
        if self.rope_cache is not None:
            cos, sin = self.rope_cache
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)

        past_kv = (key_states, value_states) if use_cache else None

        # Repeat KV heads for GQA
        key_states = self._repeat_kv(key_states, self.num_kv_groups)
        value_states = self._repeat_kv(value_states, self.num_kv_groups)

        # Stage 2: Scaled Dot-Product Attention
        # Compute attention scores: (batch, num_heads, seq_len, kv_seq_len)
        query_states_t = query_states.transpose(1, 2)
        key_states_t = key_states.transpose(1, 2)
        value_states_t = value_states.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query_states_t, key_states_t.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax normalization
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(query_states.dtype)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states_t)
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # --- Gating at G1 (after SDPA output) ---
        if cfg.position == GatingPosition.G1_SDPA_OUTPUT:
            gate_scores = self._compute_gate_scores(hidden_states, self.num_heads)
            attn_output = self._apply_gate(attn_output, gate_scores)

        # Stage 3: Multi-head concatenation
        attn_output = attn_output.reshape(batch, seq_len, self.num_heads * self.head_dim)

        # Stage 4: Output projection
        attn_output = self.o_proj(attn_output)

        # --- Gating at G5 (after dense output) ---
        if cfg.position == GatingPosition.G5_DENSE_OUTPUT:
            if self.gate_proj is not None:
                gate_scores = self.gate_proj(hidden_states)  # (batch, seq_len, d_model)
                attn_output = self._apply_gate(attn_output, gate_scores)

        if output_attentions:
            return attn_output, attn_weights, past_kv

        return attn_output, None, past_kv


# ---------------------------------------------------------------------------
# RoPE utilities
# ---------------------------------------------------------------------------

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embeddings to query and key."""
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2024)."""

    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_gated_attention(
    d_model: int = 2048,
    num_heads: int = 32,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    position: str = "g1",
    granularity: str = "elementwise",
    mode: str = "multiplicative",
    scope: str = "head_specific",
    activation: str = "sigmoid",
    max_seq_len: int = 4096,
) -> GatedAttention:
    """Factory function to create a GatedAttention module from string args.

    This mirrors the configuration used throughout the paper's experiments.

    Args:
        d_model: Model hidden dimension
        num_heads: Number of query attention heads
        num_kv_heads: Number of key-value heads (for GQA)
        head_dim: Dimension per attention head
        position: One of 'g1', 'g2', 'g3', 'g4', 'g5'
        granularity: 'headwise' or 'elementwise'
        mode: 'multiplicative' or 'additive'
        scope: 'head_specific' or 'head_shared'
        activation: 'sigmoid', 'silu', 'identity', 'ns_sigmoid'
        max_seq_len: Maximum sequence length

    Returns:
        Configured GatedAttention module
    """
    position_map = {
        "none": GatingPosition.NONE,
        "g1": GatingPosition.G1_SDPA_OUTPUT,
        "g2": GatingPosition.G2_VALUE,
        "g3": GatingPosition.G3_KEY,
        "g4": GatingPosition.G4_QUERY,
        "g5": GatingPosition.G5_DENSE_OUTPUT,
    }
    granularity_map = {
        "headwise": GatingGranularity.HEADWISE,
        "elementwise": GatingGranularity.ELEMENTWISE,
    }
    mode_map = {
        "multiplicative": GatingMode.MULTIPLICATIVE,
        "additive": GatingMode.ADDITIVE,
    }
    scope_map = {
        "head_specific": GatingScope.HEAD_SPECIFIC,
        "head_shared": GatingScope.HEAD_SHARED,
    }
    activation_map = {
        "sigmoid": ActivationType.SIGMOID,
        "silu": ActivationType.SILU,
        "identity": ActivationType.IDENTITY,
        "ns_sigmoid": ActivationType.NS_SIGMOID,
    }

    config = GatedAttentionConfig(
        position=position_map[position],
        granularity=granularity_map[granularity],
        mode=mode_map[mode],
        scope=scope_map[scope],
        activation=activation_map[activation],
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
    )

    return GatedAttention(config)
