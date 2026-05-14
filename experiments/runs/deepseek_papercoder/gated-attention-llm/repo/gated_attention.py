## gated_attention.py
"""
Implements the GatedAttention module – a drop-in replacement for standard multi-head
attention that supports various gating mechanisms as described in Section 2.2 of
"Gated Attention for Large Language Models".

The module exposes a forward interface that optionally applies a sigmoid (or SiLU)
based gate at one of five positions (G₁–G₅), with selectable granularity, head
specificity, and multiplicative/additive operation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class GatedAttention(nn.Module):
    """
    Multi-head attention block with an optional gating mechanism.

    Args:
        config: A dictionary sub‑section of the full configuration
                (should correspond to `config["model"]`). It must contain the following keys:
                - hidden_size: int
                - num_attention_heads: int
                - num_key_value_heads: int
                - head_dim: int
                - use_gated_attention: bool
                - gate_position: str (one of "SDPA_output", "value", "key", "query", "dense_output")
                - gate_granularity: str (one of "elementwise", "headwise")
                - gate_head_specific: bool
                - gate_activation: str (one of "sigmoid", "silu")
                - gate_operation: str (one of "multiply", "add")
    """

    def __init__(self, config: Dict):
        super().__init__()

        # Unpack configuration
        hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.use_gated = config.get("use_gated_attention", False)
        self.gate_position = config.get("gate_position", "SDPA_output")
        self.gate_granularity = config.get("gate_granularity", "elementwise")
        self.gate_head_specific = config.get("gate_head_specific", True)
        gate_activation = config.get("gate_activation", "sigmoid")
        self.gate_operation = config.get("gate_operation", "multiply")

        # Linear projections for Q, K, V, output
        self.Q = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.K = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.V = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.O = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

        # Gating projection (only instantiated when gating is enabled)
        if self.use_gated:
            gate_out_dim = self._gate_output_dim(config)
            self.W_gate = nn.Linear(hidden_size, gate_out_dim, bias=False)
            # Activation function
            if gate_activation == "sigmoid":
                self.activation_fn = torch.sigmoid
            elif gate_activation == "silu":
                self.activation_fn = F.silu
            else:
                raise ValueError(f"Unsupported gate activation: {gate_activation}")
        else:
            self.W_gate = None
            self.activation_fn = None

        # These will be populated during forward passes and consumed by analysis hooks.
        self.gate_scores: Optional[torch.Tensor] = None
        self.attention_weights: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute gated (or standard) multi-head attention.

        Args:
            hidden_states: input tensor of shape [batch, seq_len, hidden_size]
            attention_mask: optional float mask of shape [batch, 1, seq_len, seq_len]
                            or [seq_len, seq_len]. If None, a causal mask is applied.

        Returns:
            Output tensor of shape [batch, seq_len, hidden_size]
        """
        B, T, _ = hidden_states.shape

        # 1. Linear projections to Q, K, V
        Q = self.Q(hidden_states)
        K = self.K(hidden_states)
        V = self.V(hidden_states)

        # 2. Reshape for multi-head: [B, T, num_heads * head_dim] -> [B, num_heads, T, head_dim]
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, q, T, d_k]
        K = K.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, k, T, d_k]
        V = V.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Handle Grouped‑Query Attention (GQA): replicate K and V to match query heads
        if self.num_heads > self.num_kv_heads:
            expand_factor = self.num_heads // self.num_kv_heads
            K = K.repeat_interleave(expand_factor, dim=1)  # [B, q, T, d_k]
            V = V.repeat_interleave(expand_factor, dim=1)

        # 4. Apply gating *before* attention (positions G₂–G₄)
        if self.use_gated and self.gate_position != "SDPA_output" and self.gate_position != "dense_output":
            gate_scores = self._compute_gate_scores(hidden_states)
            if self.gate_position == "value":
                # Gate V before expansion: compute gate on original KV‑heads shape
                gate_scores_for_v = self._reshape_gate_for_kv(gate_scores)  # [B, k, T, d_k]
                V_orig = V.view(B, self.num_kv_heads, -1, self.head_dim) if self.num_heads != self.num_kv_heads else V
                V_orig = self._apply_gate(V_orig, gate_scores_for_v)
                # reshape back to [B, q, T, d_k] (the gate is already replicated along kv heads inside _apply_gate)
                V = V_orig.view(B, self.num_kv_heads, -1, self.head_dim).repeat_interleave(expand_factor, dim=1)
                # recompute V to match original shape after gating
                V = V.view(B, self.num_heads, T, self.head_dim)
            elif self.gate_position == "key":
                gate_scores_for_k = self._reshape_gate_for_kv(gate_scores)
                K_orig = K.view(B, self.num_kv_heads, -1, self.head_dim) if self.num_heads != self.num_kv_heads else K
                K_orig = self._apply_gate(K_orig, gate_scores_for_k)
                K = K_orig.view(B, self.num_kv_heads, -1, self.head_dim).repeat_interleave(expand_factor, dim=1)
                K = K.view(B, self.num_heads, T, self.head_dim)
            elif self.gate_position == "query":
                gate_scores_for_q = self._reshape_gate_for_q(gate_scores)
                Q = self._apply_gate(Q, gate_scores_for_q)
            # store the applied gate scores for analysis
            self.gate_scores = gate_scores.detach()
        else:
            self.gate_scores = None  # will be set later for G₁/G₅

        # 5. Scaled dot‑product attention
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, q, T, T]

        # Build causal mask if none provided
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=hidden_states.device),
                diagonal=1,
            )
            attn_scores = attn_scores + causal_mask
        else:
            # Assume attention_mask is broadcastable (e.g., [B, 1, T, T])
            attn_scores = attn_scores + attention_mask

        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, q, T, T]
        self.attention_weights = attn_weights.detach()

        attn_output = torch.matmul(attn_weights, V)  # [B, q, T, d_k]

        # 6. Reshape back to [B, T, q * d_k]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)

        # 7. Apply gating *after* attention (position G₁ – SDPA output) if not already done
        if self.use_gated and self.gate_position == "SDPA_output":
            gate_scores = self._compute_gate_scores(hidden_states)
            # Reshape gating scores to match [B, T, q, d_k] for elementwise / [B, T, q, 1] for headwise
            gate_scores_reshaped = self._reshape_gate_for_q(gate_scores, target_shape_is_heads=False)  # returns shape (B, T, q, d_k) or broadcastable
            # Apply gating before the output projection
            attn_output_view = attn_output.view(B, T, self.num_heads, self.head_dim)
            gated_attn = self._apply_gate(attn_output_view, gate_scores_reshaped)
            attn_output = gated_attn.reshape(B, T, self.num_heads * self.head_dim)
            self.gate_scores = gate_scores.detach()

        # 8. Output projection
        output = self.O(attn_output)  # [B, T, hidden_size]

        # 9. Gating after final dense output (position G₅) – rarely used, ineffective
        if self.use_gated and self.gate_position == "dense_output":
            gate_scores = self._compute_gate_scores(hidden_states)
            # Gate scores must match hidden_size
            gate_scores_reshaped = gate_scores  # already [B, T, hidden_size] from projection
            output = self._apply_gate(output, gate_scores_reshaped)
            self.gate_scores = gate_scores.detach()
        elif self.use_gated and self.gate_position not in ("value", "key", "query", "SDPA_output", "dense_output"):
            raise ValueError(f"Unsupported gate position: {self.gate_position}")

        return output

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _gate_output_dim(self, config: Dict) -> int:
        """
        Determine the number of gating scores produced by the gate linear layer.

        This mapping follows Table 1 and the gate variants in Section 2.2.
        """
        pos = self.gate_position
        granularity = self.gate_granularity
        head_specific = self.gate_head_specific

        # G₁ / G₄ (query side) use query heads, G₂ / G₃ use key-value heads
        if pos in ("query", "SDPA_output"):
            n_heads = self.num_heads
        elif pos in ("key", "value"):
            n_heads = self.num_kv_heads
        elif pos == "dense_output":
            if granularity == "elementwise":
                return config["hidden_size"]
            else:  # headwise (a single scalar per token) – not used in the paper
                return 1
        else:
            raise ValueError(f"Unknown gate_position: '{pos}'")

        if granularity == "elementwise":
            if head_specific:
                return n_heads * self.head_dim
            else:
                # Head‑shared elementwise: project to full head*dim, later average over heads
                return n_heads * self.head_dim
        else:  # headwise
            if head_specific:
                return n_heads
            else:
                # Head‑shared headwise: project to n_heads, then average
                return n_heads

    def _compute_gate_scores(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states, apply the activation function and return the raw gate scores.

        Returns:
            Tensor of shape [B, T, gate_out_dim] (not yet reshaped).
        """
        raw = self.W_gate(hidden_states)
        return self.activation_fn(raw)

    def _reshape_gate_for_q(
        self,
        gate_scores: torch.Tensor,
        target_shape_is_heads: bool = False,
    ) -> torch.Tensor:
        """
        Reshape gate scores for positions that modulate the query‑aligned tensors
        (SDPA output G₁ or query projection G₄). The returned shape can broadcast
        with a tensor of shape (B, T, q, head_dim) or (B, T, q, 1) depending on
        granularity.

        Args:
            gate_scores: Raw gate scores from _compute_gate_scores,
                         shape [B, T, gate_out_dim].
            target_shape_is_heads: If True, the gate is applied to the per‑head tensor
                          *after* expansion (i.e., shape [B, T, q, head_dim]).
                          If False, it will be applied to the raw Q or attn output
                          before reshaping (for G₄, used with Q directly; for G₁ we
                          first reshape attn output anyway).

        Returns:
            A tensor that can broadcast with the target tensor.
        """
        B, T, _ = gate_scores.shape

        if self.gate_granularity == "elementwise":
            if self.gate_head_specific:
                # Already projected to [q * d_k] -> reshape to [B, T, q, d_k]
                return gate_scores.view(B, T, self.num_heads, self.head_dim)
            else:
                # Head‑shared: average over the head dimension, then expand
                # Gate was projected to [q * d_k]; we view as [B, T, q, d_k], mean over q dim.
                gates_2d = gate_scores.view(B, T, self.num_heads, self.head_dim)
                gates_shared = gates_2d.mean(dim=2, keepdim=True)  # [B, T, 1, d_k]
                return gates_shared.expand(-1, -1, self.num_heads, -1)
        else:  # headwise
            if self.gate_head_specific:
                # Projected to [q] -> shape [B, T, q, 1] (so it broadcasts over head_dim)
                return gate_scores.unsqueeze(-1)  # [B, T, q, 1]
            else:
                # Head‑shared: gate was projected to [q]; average -> [B, T, 1, 1]
                gates_shared = gate_scores.mean(dim=-1, keepdim=True).unsqueeze(-1)  # [B, T, 1, 1]
                return gates_shared

    def _reshape_gate_for_kv(
        self,
        gate_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reshape gate scores for positions that modulate key/value projections (G₂, G₃).
        The target tensor has shape [B, T, k, head_dim] (before GQA expansion).

        Returns:
            A tensor that can broadcast with [B, T, k, head_dim] or [B, T, k, 1].
        """
        B, T, _ = gate_scores.shape

        if self.gate_granularity == "elementwise":
            if self.gate_head_specific:
                return gate_scores.view(B, T, self.num_kv_heads, self.head_dim)
            else:
                gates_2d = gate_scores.view(B, T, self.num_kv_heads, self.head_dim)
                gates_shared = gates_2d.mean(dim=2, keepdim=True)  # [B, T, 1, d_k]
                return gates_shared.expand(-1, -1, self.num_kv_heads, -1)
        else:  # headwise
            if self.gate_head_specific:
                return gate_scores.unsqueeze(-1)  # [B, T, k, 1]
            else:
                gates_shared = gate_scores.mean(dim=-1, keepdim=True).unsqueeze(-1)  # [B, T, 1, 1]
                return gates_shared

    def _apply_gate(
        self, modulated_tensor: torch.Tensor, gate_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply multiplicative or additive gating to a tensor.

        Args:
            modulated_tensor: tensor to which the gate is applied (shape must be
                             broadcastable with gate_scores).
            gate_scores: gating scores (sigmoid/SiLU outputs).

        Returns:
            Gated tensor of the same shape as modulated_tensor.
        """
        if self.gate_operation == "multiply":
            return modulated_tensor * gate_scores
        elif self.gate_operation == "add":
            return modulated_tensor + gate_scores
        else:
            raise ValueError(f"Unsupported gate operation: {self.gate_operation}")

