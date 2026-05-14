"""
Gated Attention variants from the paper:
"Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"

Implements all gating positions (G1-G5), granularities (elementwise/headwise),
head-specific/shared modes, multiplicative/additive modes, and activation functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

from config import ModelConfig


def compute_rope_freqs(dim: int, max_seq_len: int, base: float = 10000.0):
    """Compute RoPE frequencies."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    """Apply RoPE to query and key tensors."""
    seq_len = q.shape[1]
    cos_slice = cos[offset:offset + seq_len].unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dim)
    sin_slice = sin[offset:offset + seq_len].unsqueeze(0).unsqueeze(2)
    q_embed = (q * cos_slice) + (rotate_half(q) * sin_slice)
    k_embed = (k * cos_slice) + (rotate_half(k) * sin_slice)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class GatedMLP(nn.Module):
    """FFN with SwiGLU activation (standard in paper)."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    """Apply activation function to scores."""
    if activation == "sigmoid":
        return torch.sigmoid(x)
    elif activation == "ns_sigmoid":
        return 0.5 + 0.5 * torch.sigmoid(x)
    elif activation == "silu":
        return F.silu(x)
    elif activation == "identity":
        return x
    else:
        raise ValueError(f"Unknown activation: {activation}")


class GatedSDPA(nn.Module):
    """
    Multi-head attention with configurable gating at various positions.

    Supports:
    - G1: Gating after SDPA output (pre-W_o)
    - G2: Gating after value projection
    - G3: Gating after key projection
    - G4: Gating after query projection
    - G5: Gating after output projection (post-W_o)
    - Granularity: elementwise or headwise
    - Head-specific or head-shared gating
    - Multiplicative or additive gating
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_query_heads = config.n_query_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_model = config.d_model
        self.d_head = config.d_head
        self.max_seq_len = config.max_seq_len
        self.rope_base = config.rope_base

        # GQA: query heads grouped into groups sharing KV heads
        self.n_kv_groups = config.n_query_heads // config.n_kv_heads

        # Linear projections
        self.w_q = nn.Linear(config.d_model, config.n_query_heads * config.d_head, bias=False)
        self.w_k = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.w_v = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.w_o = nn.Linear(config.n_query_heads * config.d_head, config.d_model, bias=False)

        # RoPE cache
        cos, sin = compute_rope_freqs(config.d_head, config.max_seq_len, config.rope_base)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        # Gating layers (created based on config)
        self.gating_position = config.gating_position
        self.gating_granularity = config.gating_granularity
        self.gating_head_specific = config.gating_head_specific
        self.gating_mode = config.gating_mode
        self.gating_activation = config.gating_activation

        self.gate_G1 = None
        self.gate_G2 = None
        self.gate_G3 = None
        self.gate_G4 = None
        self.gate_G5 = None

        self._init_gates()

    def _init_gates(self):
        """Initialize gating parameter matrices based on config."""
        if self.gating_position is None:
            return

        pos = self.gating_position
        granularity = self.gating_granularity
        head_specific = self.gating_head_specific

        # Determine dimensions for the gate
        if pos in ("G1", "G2", "G3", "G4"):
            # Compute which heads we're dealing with
            n_heads = self.n_query_heads if pos not in ("G2", "G3") else self.n_kv_heads

            if granularity == "elementwise":
                gate_dim = n_heads * self.d_head if head_specific else self.d_head
            elif granularity == "headwise":
                gate_dim = n_heads if head_specific else 1
            else:
                raise ValueError(f"Unknown granularity: {granularity}")
        elif pos == "G5":
            # G5 is after W_o, dimension is d_model
            gate_dim = self.d_model
        else:
            raise ValueError(f"Unknown gating position: {pos}")

        # Create the gate parameter
        gate = nn.Parameter(torch.zeros(self.d_model, gate_dim))

        if pos == "G1":
            self.gate_G1 = gate
        elif pos == "G2":
            self.gate_G2 = gate
        elif pos == "G3":
            self.gate_G3 = gate
        elif pos == "G4":
            self.gate_G4 = gate
        elif pos == "G5":
            self.gate_G5 = gate

    def _resolve_head_scores(
        self,
        scores: torch.Tensor,
        n_heads: int,
        head_specific: bool,
    ) -> torch.Tensor:
        """
        Resolve gating scores for multi-head application.

        For head-shared: broadcast across heads.
        For SDPA head-shared elementwise (paper row 12): average over query head dim.
        """
        if head_specific:
            return scores
        # head-shared: expand to all heads
        if scores.shape[-1] == 1:
            # single scalar per token -> broadcast over all heads and dims
            return scores.unsqueeze(-1)
        # Need to broadcast head-dim feature across n_heads
        return scores.unsqueeze(-2).expand(*scores.shape[:-1], n_heads, scores.shape[-1])

    def _apply_gate(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        gate_param: nn.Parameter,
        output_shape: tuple,
        mode: str,
        activation: str,
    ) -> torch.Tensor:
        """Apply gating: y' = y * sigma(x @ W_gate) or y' = y + sigma(x @ W_gate)."""
        batch, seq_len = x.shape[0], x.shape[1]

        # Compute gate scores
        gate_scores = torch.matmul(
            x.reshape(batch, seq_len, -1),
            gate_param
        )

        # Reshape to output_shape
        target_shape = (batch, seq_len) + output_shape
        gate_scores = gate_scores.reshape(target_shape)

        gate_scores = apply_activation(gate_scores, activation)

        if mode == "multiplicative":
            return y * gate_scores
        elif mode == "additive":
            return y + gate_scores
        else:
            raise ValueError(f"Unknown gating mode: {mode}")

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, seq_len, d_model)
            attention_mask: optional causal mask
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # QKV projections
        q = self.w_q(x).view(batch, seq_len, self.n_query_heads, self.d_head)
        k = self.w_k(x).view(batch, seq_len, self.n_kv_heads, self.d_head)
        v = self.w_v(x).view(batch, seq_len, self.n_kv_heads, self.d_head)

        # Gating at Q, K, V positions (G4, G3, G2)
        if self.gate_G4 is not None:
            output_shape = (self.n_query_heads, self.d_head) if self.gating_granularity == "elementwise" else (self.n_query_heads,)
            if not self.gating_head_specific:
                output_shape = (self.d_head,) if self.gating_granularity == "elementwise" else (1,)
            q = self._apply_gate(
                q, x, self.gate_G4, output_shape,
                self.gating_mode, self.gating_activation,
            )

        if self.gate_G3 is not None:
            output_shape = (self.n_kv_heads, self.d_head) if self.gating_granularity == "elementwise" else (self.n_kv_heads,)
            if not self.gating_head_specific:
                output_shape = (self.d_head,) if self.gating_granularity == "elementwise" else (1,)
            k = self._apply_gate(
                k, x, self.gate_G3, output_shape,
                self.gating_mode, self.gating_activation,
            )

        if self.gate_G2 is not None:
            output_shape = (self.n_kv_heads, self.d_head) if self.gating_granularity == "elementwise" else (self.n_kv_heads,)
            if not self.gating_head_specific:
                output_shape = (self.d_head,) if self.gating_granularity == "elementwise" else (1,)
            v = self._apply_gate(
                v, x, self.gate_G2, output_shape,
                self.gating_mode, self.gating_activation,
            )

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, self.cos, self.sin)

        # Expand KV heads for GQA
        if self.n_kv_groups > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.n_kv_groups, -1)
            k = k.reshape(batch, seq_len, self.n_query_heads, self.d_head)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.n_kv_groups, -1)
            v = v.reshape(batch, seq_len, self.n_query_heads, self.d_head)

        # Scaled dot-product attention
        scale = self.d_head ** 0.5
        q = q.transpose(1, 2)  # (B, H, S, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (B, H, S, D)

        # Reshape SDPA output: (B, H, S, D) -> (B, S, H*D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        # Gating at SDPA output (G1)
        if self.gate_G1 is not None:
            # Compute gate scores from current queries x
            gate_scores = torch.matmul(x, self.gate_G1)  # (B, S, gate_dim)
            gate_scores = apply_activation(gate_scores, self.gating_activation)

            if self.gating_granularity == "elementwise" and self.gating_head_specific:
                # (B, S, H*D) -> reshape for per-head per-dim gating
                gate_scores = gate_scores.view(batch, seq_len, self.n_query_heads, self.d_head)
            elif self.gating_granularity == "elementwise" and not self.gating_head_specific:
                # Head-shared: one d_head vector per token, broadcast across heads
                gate_scores = gate_scores.unsqueeze(2)  # (B, S, 1, D)
                gate_scores = gate_scores.expand(-1, -1, self.n_query_heads, -1)
            elif self.gating_granularity == "headwise" and self.gating_head_specific:
                # One scalar per head, expand to d_head dims
                gate_scores = gate_scores.unsqueeze(-1)  # (B, S, H, 1)
                gate_scores = gate_scores.expand(-1, -1, -1, self.d_head)
            elif self.gating_granularity == "headwise" and not self.gating_head_specific:
                # One scalar, expand to all heads and dims
                gate_scores = gate_scores.unsqueeze(-1).unsqueeze(-1)  # (B, S, 1, 1)
                gate_scores = gate_scores.expand(-1, -1, self.n_query_heads, self.d_head)

            # Apply gate
            attn_output_view = attn_output.view(batch, seq_len, self.n_query_heads, self.d_head)
            if self.gating_mode == "multiplicative":
                attn_output_view = attn_output_view * gate_scores
            else:
                attn_output_view = attn_output_view + gate_scores

            attn_output = attn_output_view.reshape(batch, seq_len, -1)

        # Final output projection
        output = self.w_o(attn_output)

        # Gating after output projection (G5)
        if self.gate_G5 is not None:
            output_shape = (self.d_model,)
            output = self._apply_gate(
                output, x, self.gate_G5, output_shape,
                self.gating_mode, self.gating_activation,
            )

        return output


class GatedAttentionRef(nn.Module):
    """
    Cleaner reference implementation of Gated SDPA output gating (G1).

    This is the best-performing variant from the paper.
    Implements G1 elementwise, head-specific, multiplicative sigmoid gating.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cfg = config
        self.n_query_heads = config.n_query_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_model = config.d_model
        self.d_head = config.d_head
        self.max_seq_len = config.max_seq_len
        self.rope_base = config.rope_base
        self.n_kv_groups = config.n_query_heads // config.n_kv_heads

        self.w_q = nn.Linear(config.d_model, config.n_query_heads * config.d_head, bias=False)
        self.w_k = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.w_v = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.w_o = nn.Linear(config.n_query_heads * config.d_head, config.d_model, bias=False)

        # G1 gate: elementwise, head-specific
        self.gate = nn.Linear(config.d_model, config.n_query_heads * config.d_head, bias=False)

        cos, sin = compute_rope_freqs(config.d_head, config.max_seq_len, config.rope_base)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        B, S, _ = x.shape

        q = self.w_q(x).view(B, S, self.n_query_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)

        # Apply RoPE
        cos_s = self.cos[:S].unsqueeze(0).unsqueeze(2)
        sin_s = self.sin[:S].unsqueeze(0).unsqueeze(2)

        q_embed = (q * cos_s) + (rotate_half(q) * sin_s)

        cos_s_k = cos_s[:, :, :self.n_kv_heads]
        sin_s_k = sin_s[:, :, :self.n_kv_heads]
        k_embed = (k * cos_s_k) + (rotate_half(k) * sin_s_k)

        # Expand KV for GQA
        if self.n_kv_groups > 1:
            k_embed = k_embed.repeat_interleave(self.n_kv_groups, dim=2)
            v = v.repeat_interleave(self.n_kv_groups, dim=2)

        # SDPA
        scale = self.d_head ** 0.5
        attn_weights = torch.matmul(q_embed, k_embed.transpose(-2, -1)) / scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # G1 Gate: elementwise, head-specific
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, S, H, D)
        gate_scores = self.gate(x).view(B, S, self.n_query_heads, self.d_head)
        gate_scores = apply_activation(gate_scores, self.cfg.gating_activation)
        gated_output = attn_output * gate_scores

        # Output projection
        gated_output = gated_output.reshape(B, S, -1)
        output = self.w_o(gated_output)

        return output


def build_gated_attention(config: ModelConfig) -> nn.Module:
    """Factory to build the appropriate gated attention module."""
    if config.gating_position == "G1" and config.gating_granularity == "elementwise" and config.gating_head_specific and config.gating_activation == "sigmoid":
        return GatedAttentionRef(config)
    return GatedSDPA(config)
