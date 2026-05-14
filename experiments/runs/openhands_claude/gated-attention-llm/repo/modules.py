"""
Core gating modules implementing the mechanisms described in Sec. 2.2.

The gating formulation (Eq. 5):
    Y' = g(Y, X, W_theta, sigma) = Y ⊙ sigma(X W_theta)

Five positions (Fig. 1):
    G1 – after SDPA output (best performing)
    G2 – after value projection
    G3 – after key projection
    G4 – after query projection
    G5 – after final dense output layer

Variants explored:
    - Granularity: elementwise vs headwise
    - Head specificity: head-specific vs head-shared
    - Mode: multiplicative vs additive
    - Activation: sigmoid, SiLU, identity, ns_sigmoid, rmsnorm
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GatingConfig


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def ns_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Non-Sparse sigmoid: constrains output to [0.5, 1.0] (Eq. 9 in paper).

    NS-sigmoid(x) = 0.5 + 0.5 * sigmoid(x)
    Used in ablation to study the effect of removing sparsity while keeping
    non-linearity.
    """
    return 0.5 + 0.5 * torch.sigmoid(x)


# ---------------------------------------------------------------------------
# RMSNorm (used as a non-linearity substitute at G1, Table 3 row 5)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# Core gating module
# ---------------------------------------------------------------------------

class GatingModule(nn.Module):
    """Implements the gating mechanism from Eq. 5.

    Args:
        cfg:        GatingConfig specifying all gating choices.
        d_input:    Dimensionality of X (the input used to compute gate scores).
        d_gate:     Dimensionality of Y (the tensor being gated).
        num_heads:  Number of attention heads (needed for head-specific gating).
        head_dim:   Per-head dimension (d_k).
    """

    def __init__(
        self,
        cfg: GatingConfig,
        d_input: int,
        d_gate: int,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_gate = d_gate

        if cfg.activation == "rmsnorm":
            # RMSNorm applied per-head to the SDPA output (Table 3 row 5).
            # No learnable gate projection; just normalise each head's output.
            self.head_norms = nn.ModuleList(
                [RMSNorm(head_dim) for _ in range(num_heads)]
            )
            self.gate_proj = None
            return

        # Determine output dimension of the gate projection
        if cfg.granularity == "headwise":
            # One scalar per head → shape (n, num_heads)
            gate_out_dim = num_heads
        else:
            # elementwise: one value per element of Y
            if cfg.head_specific:
                # Each head has its own d_k-dimensional gate vector
                # Gate projection maps d_input → num_heads * head_dim
                gate_out_dim = num_heads * head_dim
            else:
                # Head-shared: single d_k-dimensional vector shared across heads
                # (paper: average over query head dim to get n × d_k)
                gate_out_dim = head_dim

        if cfg.activation == "silu" and cfg.mode == "additive":
            # Additive gating: Y' = Y + SiLU(X W_theta)
            # The gate output must match d_gate exactly
            gate_out_dim = d_gate

        self.gate_proj = nn.Linear(d_input, gate_out_dim, bias=False)
        nn.init.zeros_(self.gate_proj.weight)  # zero-init for stable training start

        self.head_norms = None

    def _apply_activation(self, logits: torch.Tensor) -> torch.Tensor:
        act = self.cfg.activation
        if act == "sigmoid":
            return torch.sigmoid(logits)
        elif act == "silu":
            return F.silu(logits)
        elif act == "ns_sigmoid":
            return ns_sigmoid(logits)
        elif act == "identity":
            return logits
        else:
            raise ValueError(f"Unknown activation: {act}")

    def forward(self, Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Apply gating.

        Args:
            Y: Tensor to be gated.
               For G1/G2: shape (batch, n, num_heads, head_dim) or
                          (batch, n, num_heads * head_dim).
               For G5:    shape (batch, n, d_model).
            X: Input used to compute gate scores, shape (batch, n, d_input).

        Returns:
            Gated tensor Y' with the same shape as Y.
        """
        cfg = self.cfg

        # RMSNorm variant: normalise each head independently
        if cfg.activation == "rmsnorm":
            return self._apply_rmsnorm(Y)

        logits = self.gate_proj(X)  # (batch, n, gate_out_dim)
        scores = self._apply_activation(logits)

        if cfg.mode == "additive":
            # Y' = Y + scores
            # For multi-head Y (batch, n, num_heads, head_dim), reshape scores to match
            if Y.ndim == 4 and scores.ndim == 3:
                batch, n = Y.shape[0], Y.shape[1]
                scores = scores.view(batch, n, self.num_heads, self.head_dim)
            return Y + scores

        # Multiplicative gating: Y' = Y * scores
        return self._apply_multiplicative(Y, scores)

    def _apply_rmsnorm(self, Y: torch.Tensor) -> torch.Tensor:
        """Apply per-head RMSNorm to SDPA output (Table 3 row 5)."""
        # Y: (batch, n, num_heads, head_dim)
        batch, n, h, d = Y.shape
        out = torch.stack(
            [self.head_norms[i](Y[:, :, i, :]) for i in range(h)], dim=2
        )
        return out

    def _apply_multiplicative(
        self, Y: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.cfg
        batch = Y.shape[0]
        n = Y.shape[1]

        # G5 flat case: Y is (batch, n, d_model) — no head dimension
        if Y.ndim == 3:
            # scores: (batch, n, d_model) or (batch, n, num_heads)
            # Either way, broadcast directly
            return Y * scores

        # Multi-head cases: Y is (batch, n, num_heads, head_dim)
        if cfg.granularity == "headwise":
            # scores: (batch, n, num_heads) → broadcast over head_dim
            scores = scores.unsqueeze(-1)  # (batch, n, num_heads, 1)
            return Y * scores

        # elementwise
        if cfg.head_specific:
            # scores: (batch, n, num_heads * head_dim)
            scores = scores.view(batch, n, self.num_heads, self.head_dim)
            return Y * scores
        else:
            # head-shared: scores shape (batch, n, head_dim)
            # broadcast across all heads
            scores = scores.unsqueeze(2)  # (batch, n, 1, head_dim)
            return Y * scores


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2024 / RoFormer)."""

    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        positions = torch.arange(seq_len).float()
        freqs = torch.outer(positions, theta)  # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = q.shape[2]
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len

        cos = self.cos_cached[:, :, :seq_len, :].to(q.dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(q.dtype)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos[:, :, :k.shape[2], :] + self._rotate_half(k) * sin[:, :, :k.shape[2], :]
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Scaled Dot-Product Attention with optional gating
# ---------------------------------------------------------------------------

class GatedMultiHeadAttention(nn.Module):
    """Multi-head attention with GQA and configurable gating (Sec. 2.2).

    Supports all five gating positions (G1–G5) and all variant combinations
    described in Table 1 of the paper.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 4096,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
        gating_cfg: Optional[GatingConfig] = None,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads  # GQA groups
        self.dropout = dropout

        if gating_cfg is None:
            gating_cfg = GatingConfig(position="none")
        self.gating_cfg = gating_cfg

        # QKV projections
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)

        self.rotary_emb = RotaryEmbedding(head_dim, base=rope_base, max_seq_len=max_seq_len)

        # Build gating modules for the requested position(s)
        self.gate_G1 = self.gate_G2 = self.gate_G3 = self.gate_G4 = self.gate_G5 = None

        pos = gating_cfg.position
        if pos == "G1":
            # Gate applied to SDPA output: Y shape (batch, n, num_heads, head_dim)
            # X is the hidden state of the current query token: (batch, n, d_model)
            self.gate_G1 = GatingModule(
                cfg=gating_cfg,
                d_input=d_model,
                d_gate=num_heads * head_dim,
                num_heads=num_heads,
                head_dim=head_dim,
            )
        elif pos == "G2":
            # Gate applied to value projection output: Y shape (batch, n, num_kv_heads, head_dim)
            # X is the hidden state of each key/value token
            self.gate_G2 = GatingModule(
                cfg=gating_cfg,
                d_input=d_model,
                d_gate=num_kv_heads * head_dim,
                num_heads=num_kv_heads,
                head_dim=head_dim,
            )
        elif pos == "G3":
            # Gate applied to key projection output
            self.gate_G3 = GatingModule(
                cfg=gating_cfg,
                d_input=d_model,
                d_gate=num_kv_heads * head_dim,
                num_heads=num_kv_heads,
                head_dim=head_dim,
            )
        elif pos == "G4":
            # Gate applied to query projection output
            self.gate_G4 = GatingModule(
                cfg=gating_cfg,
                d_input=d_model,
                d_gate=num_heads * head_dim,
                num_heads=num_heads,
                head_dim=head_dim,
            )
        elif pos == "G5":
            # Gate applied to final dense output: Y shape (batch, n, d_model)
            # Elementwise: gate proj d_model → d_model
            self.gate_G5 = GatingModule(
                cfg=gating_cfg,
                d_input=d_model,
                d_gate=d_model,
                num_heads=1,
                head_dim=d_model,  # treated as flat d_model-dim vector
            )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Expand KV heads to match query heads for GQA."""
        if self.num_kv_groups == 1:
            return x
        batch, n, h, d = x.shape
        x = x[:, :, :, None, :].expand(batch, n, h, self.num_kv_groups, d)
        return x.reshape(batch, n, h * self.num_kv_groups, d)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Args:
            x:              (batch, seq_len, d_model)
            attention_mask: (batch, 1, seq_len, seq_len) additive mask
            past_key_value: cached (K, V) for generation
            use_cache:      whether to return updated KV cache

        Returns:
            output:         (batch, seq_len, d_model)
            past_key_value: updated cache (or None)
            attn_weights:   (batch, num_heads, seq_len, seq_len) for analysis
        """
        batch, seq_len, _ = x.shape

        # QKV projections (Eq. 1)
        q = self.q_proj(x)  # (batch, n, num_heads * head_dim)
        k = self.k_proj(x)  # (batch, n, num_kv_heads * head_dim)
        v = self.v_proj(x)  # (batch, n, num_kv_heads * head_dim)

        # Reshape to (batch, n, heads, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # G4: gate on query projection output
        if self.gate_G4 is not None:
            q = self.gate_G4(q, x)

        # G3: gate on key projection output
        if self.gate_G3 is not None:
            k = self.gate_G3(k, x)

        # G2: gate on value projection output (Eq. 7 non-linearity)
        if self.gate_G2 is not None:
            v = self.gate_G2(v, x)

        # Transpose to (batch, heads, n, head_dim) for attention computation
        q = q.transpose(1, 2)  # (batch, num_heads, n, head_dim)
        k = k.transpose(1, 2)  # (batch, num_kv_heads, n, head_dim)
        v = v.transpose(1, 2)  # (batch, num_kv_heads, n, head_dim)

        # Apply RoPE
        q, k = self.rotary_emb(q, k, seq_len=seq_len)

        # KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        new_cache = (k, v) if use_cache else None

        kv_seq_len = k.shape[2]

        # Expand KV heads for GQA
        k = self._repeat_kv(k.transpose(1, 2)).transpose(1, 2)
        v = self._repeat_kv(v.transpose(1, 2)).transpose(1, 2)

        # SDPA (Eq. 2)
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        # (batch, num_heads, seq_len, kv_seq_len)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)

        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Weighted sum of values: (batch, num_heads, seq_len, head_dim)
        sdpa_out = torch.matmul(attn_weights, v)

        # Transpose back: (batch, seq_len, num_heads, head_dim)
        sdpa_out = sdpa_out.transpose(1, 2).contiguous()

        # G1: gate on SDPA output (Eq. 8 non-linearity — best variant)
        if self.gate_G1 is not None:
            sdpa_out = self.gate_G1(sdpa_out, x)

        # Flatten heads: (batch, seq_len, num_heads * head_dim)
        sdpa_out = sdpa_out.view(batch, seq_len, self.num_heads * self.head_dim)

        # Final output projection (Eq. 4)
        out = self.o_proj(sdpa_out)

        # G5: gate on dense output
        if self.gate_G5 is not None:
            out = self.gate_G5(out, x)

        return out, new_cache, attn_weights
