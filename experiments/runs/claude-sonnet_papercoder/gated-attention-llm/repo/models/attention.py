## models/attention.py
"""Gated Multi-Head Attention with GQA, RoPE, and all paper gate variants.

This module implements GatedMultiHeadAttention, a drop-in replacement for
standard multi-head attention that supports all gating variants described in
"Gated Attention for Large Language Models: Non-linearity, Sparsity, and
Attention-Sink-Free".

Supported gate positions (Fig. 1 of the paper):
    G1: After SDPA output — best performing (Table 1 row 5)
    G2: After value projection (Table 1 row 6)
    G3: After key projection (Table 1 row 7)
    G4: After query projection (Table 1 row 8)
    G5: After final dense output layer (Table 1 row 9)
    none: Baseline (no gating)

Architecture details:
    - Group Query Attention (GQA): q=32 query heads, k=4 KV heads for MoE;
      q=32, k=8 for dense. Paper Sec 3.1: "We adopt group query attention (GQA)".
    - RoPE positional encoding with YaRN extension support.
    - Causal (autoregressive) masking.
    - BF16 training compatible.
    - Flash attention via F.scaled_dot_product_attention when available.

Config values used (from config.yaml):
    model.d_model: Hidden dimension (2048 for dense, 4096 for MoE)
    model.num_heads: Query heads q (32 for both)
    model.num_kv_heads: KV heads k (8 for dense, 4 for MoE)
    model.d_k: Per-head dimension (64 for dense, 128 for MoE)
    model.max_seq_len: Maximum sequence length (4096)
    rope.base: RoPE base frequency (10000.0)
    gate.position: Gate position ('G1'–'G5' or 'none')
    gate.granularity: 'elementwise' or 'headwise'
    gate.head_specific: bool
    gate.gate_type: 'multiplicative' or 'additive'
    gate.activation: 'sigmoid', 'silu', 'identity', 'ns_sigmoid', 'rmsnorm', 'silu_only'
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gate_module import GateModule
from models.rope import RoPEEmbedding


class GatedMultiHeadAttention(nn.Module):
    """Multi-head attention with configurable gating at five positions.

    Implements standard GQA-based multi-head attention augmented with an
    optional GateModule that can be applied at positions G1–G5. The gate
    input is always the pre-norm hidden state X (query-dependent for G1),
    making the gate scores input-dependent as described in Sec 4.2.

    The forward pass follows this sequence:
        1. QKV projections
        2. Reshape to per-head tensors
        3. Apply RoPE to Q and K
        4. Apply G4 gate (query) if configured
        5. Apply G3 gate (key) if configured
        6. Apply G2 gate (value) if configured
        7. GQA expansion of K and V
        8. Scaled dot-product attention (SDPA)
        9. Apply G1 gate (SDPA output) if configured  ← best performing
        10. Concatenate heads and apply W_O
        11. Apply G5 gate (dense output) if configured

    Attributes:
        d_model: Hidden dimension of the model.
        num_heads: Number of query heads (q).
        num_kv_heads: Number of key-value heads (k) for GQA.
        d_k: Per-head dimension.
        gate_position: Position identifier for the gate ('G1'–'G5' or 'none').
        W_Q: Query projection, shape [d_model, num_heads * d_k].
        W_K: Key projection, shape [d_model, num_kv_heads * d_k].
        W_V: Value projection, shape [d_model, num_kv_heads * d_k].
        W_O: Output projection, shape [num_heads * d_k, d_model].
        gate: GateModule instance, or None if gate_position == 'none'.
        rope: RoPEEmbedding for positional encoding.
    """

    def __init__(
        self,
        d_model: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        d_k: int = 64,
        max_seq_len: int = 4096,
        rope_base: float = 10000.0,
        gate_position: str = "none",
        gate_granularity: str = "elementwise",
        gate_head_specific: bool = True,
        gate_type: str = "multiplicative",
        gate_activation: str = "sigmoid",
        gate_input_independent: bool = False,
    ) -> None:
        """Initialize GatedMultiHeadAttention.

        Args:
            d_model: Hidden dimension of the model. From model.d_model.
                Default 2048 (dense 1.7B). MoE uses 4096.
            num_heads: Number of query heads q. From model.num_heads.
                Default 32 (paper Table 1 baseline).
            num_kv_heads: Number of key-value heads k for GQA.
                From model.num_kv_heads. Default 8 (dense). MoE uses 4.
            d_k: Per-head dimension. From model.d_k.
                Default 64 (dense). MoE uses 128.
            max_seq_len: Maximum sequence length. From model.max_seq_len.
                Default 4096 (paper Sec 3.1).
            rope_base: RoPE base frequency. From rope.base.
                Default 10000.0 (paper Sec 4.4: "increase from 10k to 1M").
            gate_position: Where to apply the gate. From gate.position.
                One of 'G1', 'G2', 'G3', 'G4', 'G5', or 'none'.
                Paper finding: 'G1' yields best results (Table 1).
            gate_granularity: 'elementwise' or 'headwise'. From gate.granularity.
                Paper default: 'elementwise'.
            gate_head_specific: Whether each head has independent gate weights.
                From gate.head_specific. Paper: head-specific is better (Sec 3.2.1).
            gate_type: 'multiplicative' or 'additive'. From gate.gate_type.
                Paper default: 'multiplicative' (Sec 3.2.1).
            gate_activation: Activation function name. From gate.activation.
                Paper default: 'sigmoid' (Sec 2.2).
            gate_input_independent: If True, uses zero-init parameter instead
                of input-dependent projection. From gate.input_independent.
                Table 4 row 6 ablation.

        Raises:
            ValueError: If num_heads is not divisible by num_kv_heads.
        """
        super().__init__()

        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads "
                f"({num_kv_heads}) for GQA."
            )

        # Store dimensions as instance attributes
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads
        self.d_k: int = d_k
        self.gate_position: str = gate_position

        # GQA repeat factor: how many query heads share each KV head
        self._gqa_repeat: int = num_heads // num_kv_heads

        # ---------------------------------------------------------------------------
        # Linear projections — all without bias (standard for modern LLMs)
        # Paper Sec 2.1: Q = X W_Q, K = X W_K, V = X W_V
        # W_K and W_V project to num_kv_heads * d_k (GQA design)
        # ---------------------------------------------------------------------------
        self.W_Q: nn.Linear = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.W_K: nn.Linear = nn.Linear(d_model, num_kv_heads * d_k, bias=False)
        self.W_V: nn.Linear = nn.Linear(d_model, num_kv_heads * d_k, bias=False)
        self.W_O: nn.Linear = nn.Linear(num_heads * d_k, d_model, bias=False)

        # ---------------------------------------------------------------------------
        # Gate module — instantiated only when a gate position is specified
        # ---------------------------------------------------------------------------
        if gate_position != "none":
            self.gate: Optional[GateModule] = GateModule(
                position=gate_position,
                granularity=gate_granularity,
                head_specific=gate_head_specific,
                gate_type=gate_type,
                activation=gate_activation,
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                d_k=d_k,
                input_independent=gate_input_independent,
            )
        else:
            self.gate = None

        # ---------------------------------------------------------------------------
        # RoPE positional embedding
        # Applied to Q and K (not V), per standard transformer practice.
        # Paper Sec 4.4: base starts at 10k, extended to 1M for long-context.
        # ---------------------------------------------------------------------------
        self.rope: RoPEEmbedding = RoPEEmbedding(
            d_k=d_k,
            max_seq_len=max_seq_len,
            base=rope_base,
        )

    def _apply_gqa_expand(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expand K and V from num_kv_heads to num_heads for GQA.

        Each KV head is repeated `num_heads // num_kv_heads` times to match
        the number of query heads. For the MoE model (q=32, k=4), each KV
        head is repeated 8 times.

        Paper Sec 3.1: "We adopt group query attention (GQA) for the attention
        part." with q=32, k=4 for MoE and q=32, k=8 for dense.

        Args:
            K: Key tensor, shape [batch, seq, num_kv_heads, d_k].
            V: Value tensor, shape [batch, seq, num_kv_heads, d_k].

        Returns:
            Tuple of (K_expanded, V_expanded), each shape [batch, seq, num_heads, d_k].
        """
        if self._gqa_repeat == 1:
            # No expansion needed (standard MHA where num_heads == num_kv_heads)
            return K, V

        # torch.repeat_interleave repeats each element along dim=2 (head dim).
        # KV head i maps to query heads i*repeat through (i+1)*repeat - 1.
        K_expanded = torch.repeat_interleave(K, repeats=self._gqa_repeat, dim=2)
        V_expanded = torch.repeat_interleave(V, repeats=self._gqa_repeat, dim=2)

        return K_expanded, V_expanded

    def _sdpa(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute scaled dot-product attention.

        Implements: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        as described in paper Sec 2.1 (Eq. 2).

        Uses F.scaled_dot_product_attention (flash attention) when available
        and attention weights are not needed. Falls back to manual computation
        for analysis (when return_attn_weights=True).

        Args:
            Q: Query tensor, shape [batch, seq, num_heads, d_k].
            K: Key tensor, shape [batch, seq, num_heads, d_k] (GQA-expanded).
            V: Value tensor, shape [batch, seq, num_heads, d_k] (GQA-expanded).
            mask: Optional additive causal mask, shape [1, 1, seq, seq] or
                [batch, 1, seq, seq]. 0 for allowed positions, -inf for masked.
                If None, no masking is applied.
            return_attn_weights: If True, forces manual computation to return
                attention weight tensor. If False, uses flash attention when
                available (faster, no attention weights returned).

        Returns:
            Tuple of:
                - output: Attention output, shape [batch, seq, num_heads, d_k].
                - attn_weights: Attention weight tensor of shape
                  [batch, num_heads, seq, seq] if return_attn_weights=True,
                  else None (flash attention path).
        """
        batch_size: int = Q.shape[0]
        seq_len: int = Q.shape[1]

        # Transpose to [batch, num_heads, seq, d_k] for matrix multiplication
        Q_t = Q.transpose(1, 2)  # [batch, num_heads, seq, d_k]
        K_t = K.transpose(1, 2)  # [batch, num_heads, seq, d_k]
        V_t = V.transpose(1, 2)  # [batch, num_heads, seq, d_k]

        # --- Flash attention path (training, no weights needed) ---
        use_flash = (
            hasattr(F, "scaled_dot_product_attention")
            and not return_attn_weights
        )

        if use_flash:
            # F.scaled_dot_product_attention handles scaling, masking, and softmax.
            # is_causal=True generates the causal mask internally when mask is None.
            if mask is not None:
                # Pass explicit mask; disable is_causal to avoid double masking.
                output_t = F.scaled_dot_product_attention(
                    Q_t, K_t, V_t,
                    attn_mask=mask,
                    is_causal=False,
                )
            else:
                output_t = F.scaled_dot_product_attention(
                    Q_t, K_t, V_t,
                    is_causal=True,
                )
            # Transpose back: [batch, num_heads, seq, d_k] → [batch, seq, num_heads, d_k]
            output = output_t.transpose(1, 2)
            return output, None

        # --- Manual computation path (analysis or older PyTorch) ---
        # Compute scaled attention scores: [batch, num_heads, seq, seq]
        scale: float = 1.0 / math.sqrt(self.d_k)
        scores = torch.matmul(Q_t, K_t.transpose(-2, -1)) * scale

        # Apply additive causal mask
        if mask is not None:
            scores = scores + mask
        else:
            # Build causal mask on-the-fly: upper triangle is -inf
            # Shape: [1, 1, seq, seq] — broadcasts over batch and heads
            causal_mask = torch.full(
                (seq_len, seq_len),
                fill_value=float("-inf"),
                device=Q.device,
                dtype=scores.dtype,
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Softmax over key dimension: [batch, num_heads, seq, seq]
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values: [batch, num_heads, seq, d_k]
        output_t = torch.matmul(attn_weights, V_t)

        # Transpose back: [batch, seq, num_heads, d_k]
        output = output_t.transpose(1, 2)

        # Detach attention weights — only needed for analysis, not gradients
        return output, attn_weights.detach()

    def forward(
        self,
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Full forward pass with configurable gating at positions G1–G5.

        The gate input is always X (pre-norm hidden states), making G1 gate
        scores query-dependent. This is the key design choice described in
        Sec 4.2: "SDPA output gating scores are derived from the hidden states
        corresponding to the current query."

        Args:
            X: Pre-norm hidden states (post-RMSNorm, pre-attention).
                Shape [batch, seq, d_model].
            mask: Optional additive causal mask. Shape [1, 1, seq, seq] or
                [batch, 1, seq, seq]. If None, causal mask is built internally.
            position_ids: Token position indices for RoPE. Shape [batch, seq]
                or [seq]. If None, defaults to sequential [0, 1, ..., seq-1].
            return_attn_weights: If True, forces manual SDPA computation to
                return attention weights for analysis. Default False (uses
                flash attention when available).

        Returns:
            Tuple of:
                - output: Attention output, shape [batch, seq, d_model].
                - attn_weights: Attention weights [batch, num_heads, seq, seq]
                  if return_attn_weights=True, else None.
        """
        batch_size: int = X.shape[0]
        seq_len: int = X.shape[1]

        # Save original input for gate computation.
        # Critical: gate scores are always computed from X (query-dependent).
        # Paper Sec 4.2: "gating score sparsity is more effective when
        # query-dependent rather than determined by the key and value."
        X_gate_input: torch.Tensor = X

        # -----------------------------------------------------------------------
        # Step 1: QKV Projections
        # Paper Sec 2.1 (Eq. 1): Q = X W_Q, K = X W_K, V = X W_V
        # -----------------------------------------------------------------------
        Q: torch.Tensor = self.W_Q(X)  # [batch, seq, num_heads * d_k]
        K: torch.Tensor = self.W_K(X)  # [batch, seq, num_kv_heads * d_k]
        V: torch.Tensor = self.W_V(X)  # [batch, seq, num_kv_heads * d_k]

        # -----------------------------------------------------------------------
        # Step 2: Reshape to per-head tensors
        # -----------------------------------------------------------------------
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.d_k)

        # -----------------------------------------------------------------------
        # Step 3: Apply RoPE to Q and K (not V)
        # Paper Sec 4.4: RoPE base starts at 10k, extended to 1M for long-context.
        # -----------------------------------------------------------------------
        Q = self.rope.apply_rotary(Q, position_ids)
        K = self.rope.apply_rotary(K, position_ids)

        # -----------------------------------------------------------------------
        # Step 4: Apply G4 gate (query gate) if configured
        # Gate score shape (elementwise head-specific): [batch, seq, num_heads, d_k]
        # Table 1 row 8: "q Elementwise G4" — minimal improvement over baseline.
        # -----------------------------------------------------------------------
        if self.gate is not None and self.gate.position == "G4":
            Q = self.gate.forward(Q, X_gate_input)

        # -----------------------------------------------------------------------
        # Step 5: Apply G3 gate (key gate) if configured
        # Gate score shape (elementwise head-specific): [batch, seq, num_kv_heads, d_k]
        # Table 1 row 7: "k Elementwise G3" — no improvement over baseline.
        # -----------------------------------------------------------------------
        if self.gate is not None and self.gate.position == "G3":
            K = self.gate.forward(K, X_gate_input)

        # -----------------------------------------------------------------------
        # Step 6: Apply G2 gate (value gate) if configured
        # Applied BEFORE GQA expansion — operates on num_kv_heads dimension.
        # Gate score shape (elementwise head-specific): [batch, seq, num_kv_heads, d_k]
        # Paper Sec 4.1 (Eq. 7): corresponds to Non-Linearity-Map(X_j W_V^k).
        # Table 1 row 6: "v Elementwise G2" — notable PPL improvement.
        # -----------------------------------------------------------------------
        if self.gate is not None and self.gate.position == "G2":
            V = self.gate.forward(V, X_gate_input)

        # -----------------------------------------------------------------------
        # Step 7: GQA expansion — expand K and V to num_heads
        # Each KV head is repeated (num_heads // num_kv_heads) times.
        # For MoE: 32 // 4 = 8 repeats. For dense: 32 // 8 = 4 repeats.
        # -----------------------------------------------------------------------
        K, V = self._apply_gqa_expand(K, V)
        # K, V now: [batch, seq, num_heads, d_k]

        # -----------------------------------------------------------------------
        # Step 8: Scaled dot-product attention
        # Paper Sec 2.1 (Eq. 2): Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
        # -----------------------------------------------------------------------
        sdpa_out, attn_weights = self._sdpa(
            Q, K, V, mask=mask, return_attn_weights=return_attn_weights
        )
        # sdpa_out: [batch, seq, num_heads, d_k]
        # attn_weights: [batch, num_heads, seq, seq] or None

        # -----------------------------------------------------------------------
        # Step 9: Apply G1 gate (SDPA output gate) if configured
        # PRIMARY CONTRIBUTION of the paper (Sec 1, Table 1 row 5).
        # Gate input X_gate_input makes scores query-dependent (Sec 4.2).
        # Paper Sec 4.1 (Eq. 8): Non-Linearity-Map(sum_j S_ij^k * X_j W_V^k).
        # Gate score shape (elementwise head-specific): [batch, seq, num_heads, d_k]
        # Paper finding: "up to 0.2 PPL reduction and 2 points on MMLU" (Sec 1).
        # -----------------------------------------------------------------------
        if self.gate is not None and self.gate.position == "G1":
            sdpa_out = self.gate.forward(sdpa_out, X_gate_input)

        # -----------------------------------------------------------------------
        # Step 10: Concatenate heads and apply output projection
        # Paper Sec 2.1 (Eq. 4): O = MultiHead(Q,K,V) W_O
        # -----------------------------------------------------------------------
        # Ensure contiguous memory layout before reshape
        output: torch.Tensor = sdpa_out.contiguous().reshape(
            batch_size, seq_len, self.num_heads * self.d_k
        )
        output = self.W_O(output)  # [batch, seq, d_model]

        # -----------------------------------------------------------------------
        # Step 11: Apply G5 gate (dense output gate) if configured
        # Paper Sec 4.1: G5 does NOT address the low-rank problem between W_V
        # and W_O, which is why it shows no improvement (Table 1 row 9).
        # Gate score shape: [batch, seq, d_model]
        # -----------------------------------------------------------------------
        if self.gate is not None and self.gate.position == "G5":
            output = self.gate.forward(output, X_gate_input)

        return output, attn_weights

    def get_attention_weights(
        self,
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Analysis-only method returning attention weight tensors.

        Forces manual SDPA computation (bypasses flash attention) to obtain
        the full attention weight matrix. Used by AttentionSinkAnalyzer to
        compute the proportion of attention allocated to the first token
        (Fig. 2, Table 4 'F-Attn' column).

        Paper finding (Sec 4.3): Baseline model allocates 46.7% of attention
        to the first token on average; gated model reduces this to 4.8%.

        Args:
            X: Pre-norm hidden states, shape [batch, seq, d_model].
            mask: Optional causal mask. If None, built internally.

        Returns:
            Attention weights tensor, shape [batch, num_heads, seq, seq].
            Entry [b, h, i, j] is the attention weight from query position i
            to key position j in head h of batch element b.

        Note:
            This method runs in no_grad context for efficiency. The returned
            tensor is already detached (from _sdpa's detach() call).
        """
        with torch.no_grad():
            _, attn_weights = self.forward(
                X,
                mask=mask,
                position_ids=None,
                return_attn_weights=True,
            )

        # attn_weights is guaranteed non-None when return_attn_weights=True
        assert attn_weights is not None, (
            "attn_weights should not be None when return_attn_weights=True"
        )
        return attn_weights

    def extra_repr(self) -> str:
        """Return human-readable module description for print(model).

        Returns:
            String summarizing the attention configuration.
        """
        gate_info: str = (
            f"gate={self.gate.position}/{self.gate.granularity}/"
            f"{'head_specific' if self.gate.head_specific else 'head_shared'}/"
            f"{self.gate.activation}"
            if self.gate is not None
            else "gate=none"
        )
        return (
            f"d_model={self.d_model}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"d_k={self.d_k}, "
            f"gqa_repeat={self._gqa_repeat}, "
            f"{gate_info}"
        )
