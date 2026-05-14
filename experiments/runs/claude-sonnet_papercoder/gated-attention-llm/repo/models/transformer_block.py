## models/transformer_block.py
"""Transformer block combining gated attention and SwiGLU FFN with optional sandwich norm.

This module implements two classes:
    - FeedForward: SwiGLU feed-forward network sublayer (Shazeer, 2020).
    - TransformerBlock: Full pre-norm transformer block combining GatedMultiHeadAttention
      and FeedForward, with optional sandwich normalization for training stability.

TransformerBlock is the fundamental building block consumed by both DenseModel and
MoEModel. It stores attention weights in self.last_attn_weights after each forward
pass, enabling post-hoc analysis by AttentionSinkAnalyzer and GatingScoreAnalyzer
without requiring explicit hooks.

Architecture follows the pre-norm (pre-LN) design standard for modern LLMs:
    X = X + [norm_attn_out()] attention(norm1(X))
    X = X + [norm_ffn_out()] ffn(norm2(X))

Sandwich normalization (Ding et al., 2021) is an optional training stability
baseline tested in Table 2 row 7 of the paper. It applies an additional RMSNorm
to sublayer outputs before the residual addition, preventing large FFN activations
from entering the residual stream (Appendix A.3, A.5).

Config values used (from config.yaml):
    model.d_model: Hidden dimension (2048 for dense, 4096 for MoE)
    model.ffn_dim: FFN intermediate dimension (8192 for dense 28L, 4096 for MoE)
    model.sandwich_norm: Whether to apply sandwich normalization (default: false)
    gate.*: All gate configuration fields (consumed by GatedMultiHeadAttention)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import GatedMultiHeadAttention


class FeedForward(nn.Module):
    """SwiGLU feed-forward network sublayer.

    Implements the gated linear unit variant from Shazeer (2020) used in
    modern LLMs (Llama, Qwen2.5, etc.). The SwiGLU formulation is:

        FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))

    This is referenced in the paper's related work (Sec 5.1) as a standard
    component in open-source LLMs following Yang et al. (2024a).

    Compared to standard FFN (two linear layers), SwiGLU uses three linear
    layers but with the same effective parameter budget when d_ffn is scaled
    by 2/3 relative to the standard 4*d_model width.

    Attributes:
        d_model: Input and output hidden dimension.
        d_ffn: Intermediate (hidden) dimension of the FFN.
        gate_proj: First gating projection, shape [d_model, d_ffn].
        up_proj: Second up projection, shape [d_model, d_ffn].
        down_proj: Down projection back to d_model, shape [d_ffn, d_model].
    """

    def __init__(self, d_model: int, d_ffn: int) -> None:
        """Initialize SwiGLU feed-forward network.

        Args:
            d_model: Input and output hidden dimension. From model.d_model.
                Default 2048 (dense 1.7B). MoE uses 4096.
            d_ffn: Intermediate dimension. From model.ffn_dim (possibly reduced
                by DenseModel when ffn_reduce_for_gate=True to maintain parameter
                parity with the gated baseline). Default 8192 for dense 28L.

        Note:
            All linear layers use bias=False, consistent with modern LLM practice
            and the paper's architecture (Yang et al., 2024a conventions).
        """
        super().__init__()

        self.d_model: int = d_model
        self.d_ffn: int = d_ffn

        # SwiGLU projections — all without bias (standard for modern LLMs)
        # gate_proj and up_proj both project d_model → d_ffn
        # down_proj projects d_ffn → d_model
        self.gate_proj: nn.Linear = nn.Linear(d_model, d_ffn, bias=False)
        self.up_proj: nn.Linear = nn.Linear(d_model, d_ffn, bias=False)
        self.down_proj: nn.Linear = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SwiGLU feed-forward transformation.

        Applies the gated linear unit: FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))

        The SiLU (Sigmoid Linear Unit) activation on gate_proj provides the
        gating signal that modulates the up_proj output elementwise.

        Args:
            x: Input tensor of shape [batch, seq, d_model].

        Returns:
            Output tensor of shape [batch, seq, d_model].
        """
        # SwiGLU: element-wise product of SiLU-gated and linear paths
        # F.silu(x) = x * sigmoid(x) — the SiLU activation
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def extra_repr(self) -> str:
        """Return human-readable module description for print(model).

        Returns:
            String summarizing the FFN configuration.
        """
        return f"d_model={self.d_model}, d_ffn={self.d_ffn}, activation=SwiGLU"


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with gated attention and SwiGLU FFN.

    Implements the standard pre-norm transformer block used in both DenseModel
    and MoEModel. The block applies:
        1. RMSNorm → GatedMultiHeadAttention → [optional sandwich norm] → residual add
        2. RMSNorm → FeedForward → [optional sandwich norm] → residual add

    Sandwich normalization (Ding et al., 2021) is an optional training stability
    baseline. When enabled, an additional RMSNorm is applied to each sublayer's
    output before the residual addition. This prevents large FFN activations from
    entering the residual stream (paper Appendix A.3, A.5).

    The block stores attention weights in self.last_attn_weights after each forward
    pass. This enables post-hoc analysis without explicit hooks:
        - AttentionSinkAnalyzer reads last_attn_weights to compute first-token
          attention proportions (Fig. 2, Table 4 'F-Attn' column).
        - GatingScoreAnalyzer reads gate scores via model.get_gate_scores_all_layers().

    For MoEModel compatibility, the ffn attribute can be replaced after construction:
        block = TransformerBlock(config, layer_idx)
        block.ffn = MoELayer(config)  # Replace FeedForward with MoELayer

    Attributes:
        layer_idx: Layer index (0-based) for analysis and debugging.
        attention: GatedMultiHeadAttention module with configurable gate.
        ffn: FeedForward (SwiGLU) module, or MoELayer when replaced by MoEModel.
        norm1: Pre-attention RMSNorm.
        norm2: Pre-FFN RMSNorm.
        sandwich_norm: Whether sandwich normalization is enabled.
        norm_attn_out: Post-attention RMSNorm (only when sandwich_norm=True).
        norm_ffn_out: Post-FFN RMSNorm (only when sandwich_norm=True).
        last_attn_weights: Stored attention weights from the most recent forward
            pass. Shape [batch, num_heads, seq, seq], or None if flash attention
            was used (which does not return attention weights).
    """

    def __init__(self, config: object, layer_idx: int = 0) -> None:
        """Initialize transformer block from config.

        Constructs all sublayers based on the config object. The config is
        expected to have a nested structure matching config.yaml:
            config.model.d_model
            config.model.ffn_dim
            config.model.sandwich_norm
            config.model.num_heads
            config.model.num_kv_heads
            config.model.d_k
            config.model.max_seq_len
            config.rope.base
            config.gate.position
            config.gate.granularity
            config.gate.head_specific
            config.gate.gate_type
            config.gate.activation

        Args:
            config: Configuration object (OmegaConf DictConfig or similar).
                Must expose model.*, gate.*, and rope.* fields as described above.
            layer_idx: Zero-based layer index. Stored for analysis purposes
                (e.g., AttentionSinkAnalyzer targets layer_idx=21 for Fig. 2,
                and Appendix A.3 identifies layer 5 as the source of massive
                activations in the baseline model).
        """
        super().__init__()

        # Store layer index for analysis (paper references specific layers)
        self.layer_idx: int = layer_idx

        # Extract model dimensions from config
        d_model: int = int(config.model.d_model)
        ffn_dim: int = int(config.model.ffn_dim)
        sandwich_norm: bool = bool(config.model.sandwich_norm)

        # Store sandwich_norm flag for use in forward()
        self.sandwich_norm: bool = sandwich_norm

        # -----------------------------------------------------------------------
        # Attention sublayer
        # GatedMultiHeadAttention handles all gate variants (G1–G5 or none).
        # Gate configuration is read from config.gate.* fields.
        # -----------------------------------------------------------------------
        # Extract gate configuration with safe defaults matching config.yaml
        gate_position: str = str(getattr(config.gate, "position", "none"))
        gate_granularity: str = str(getattr(config.gate, "granularity", "elementwise"))
        gate_head_specific: bool = bool(getattr(config.gate, "head_specific", True))
        gate_type: str = str(getattr(config.gate, "gate_type", "multiplicative"))
        gate_activation: str = str(getattr(config.gate, "activation", "sigmoid"))
        gate_input_independent: bool = bool(
            getattr(config.gate, "input_independent", False)
        )

        # Extract attention dimensions from config
        num_heads: int = int(config.model.num_heads)
        num_kv_heads: int = int(config.model.num_kv_heads)
        d_k: int = int(config.model.d_k)
        max_seq_len: int = int(config.model.max_seq_len)
        rope_base: float = float(config.rope.base)

        self.attention: GatedMultiHeadAttention = GatedMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            d_k=d_k,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            gate_position=gate_position,
            gate_granularity=gate_granularity,
            gate_head_specific=gate_head_specific,
            gate_type=gate_type,
            gate_activation=gate_activation,
            gate_input_independent=gate_input_independent,
        )

        # -----------------------------------------------------------------------
        # FFN sublayer (SwiGLU)
        # Uses ffn_dim from config, which may already be reduced by DenseModel
        # when ffn_reduce_for_gate=True to maintain parameter parity.
        # Paper Sec 3.2.2: "we reduce the width of FFN to maintain the parameter size"
        # -----------------------------------------------------------------------
        self.ffn: nn.Module = FeedForward(d_model=d_model, d_ffn=ffn_dim)

        # -----------------------------------------------------------------------
        # Normalization layers
        # Pre-norm architecture: norm applied to input before each sublayer.
        # Paper follows modern LLM conventions (Yang et al., 2024a).
        # -----------------------------------------------------------------------
        self.norm1: nn.RMSNorm = nn.RMSNorm(d_model)
        self.norm2: nn.RMSNorm = nn.RMSNorm(d_model)

        # -----------------------------------------------------------------------
        # Sandwich normalization (optional)
        # Paper Table 2 row 7: "sandwich norm" baseline for training stability.
        # Ding et al. (2021): apply LayerNorm to sublayer outputs before residual.
        # Paper Appendix A.5: prevents large FFN activations from entering residual.
        # Only instantiated when sandwich_norm=True to avoid wasting parameters.
        # -----------------------------------------------------------------------
        self.norm_attn_out: Optional[nn.RMSNorm] = None
        self.norm_ffn_out: Optional[nn.RMSNorm] = None

        if self.sandwich_norm:
            self.norm_attn_out = nn.RMSNorm(d_model)
            self.norm_ffn_out = nn.RMSNorm(d_model)

        # -----------------------------------------------------------------------
        # Analysis storage
        # Stores attention weights from the most recent forward pass.
        # Used by AttentionSinkAnalyzer without requiring explicit hooks.
        # May be None if flash attention path was used (no weights returned).
        # Paper Fig. 2: baseline allocates 46.7% attention to first token;
        # gated model reduces this to 4.8%.
        # -----------------------------------------------------------------------
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the transformer block.

        Implements the pre-norm transformer block with optional sandwich norm:

        Without sandwich norm (standard pre-norm):
            attn_out = attention(norm1(X))
            X = X + attn_out
            ffn_out = ffn(norm2(X))
            X = X + ffn_out

        With sandwich norm (Ding et al., 2021):
            attn_out = norm_attn_out(attention(norm1(X)))
            X = X + attn_out
            ffn_out = norm_ffn_out(ffn(norm2(X)))
            X = X + ffn_out

        The attention module receives the post-norm input (norm1(X)) as both
        the query/key/value input and the gate input X. This makes G1 gate
        scores query-dependent (Sec 4.2): the gate is computed from the current
        token's normalized representation.

        Attention weights are stored in self.last_attn_weights (detached) for
        post-hoc analysis. They may be None when flash attention is used.

        Args:
            X: Input hidden states, shape [batch, seq, d_model].
                This is the residual stream input (post-previous-block output).
            mask: Optional additive causal attention mask.
                Shape [1, 1, seq, seq] or [batch, 1, seq, seq].
                Values: 0.0 for allowed positions, -inf for masked positions.
                If None, GatedMultiHeadAttention builds the causal mask internally.
            position_ids: Token position indices for RoPE.
                Shape [batch, seq] or [seq].
                If None, GatedMultiHeadAttention defaults to sequential positions.

        Returns:
            Output hidden states of shape [batch, seq, d_model].
            Same shape as input X (residual connection preserves dimensions).
        """
        # -----------------------------------------------------------------------
        # Attention sublayer with pre-norm
        # -----------------------------------------------------------------------
        # Step 1: Pre-attention RMSNorm
        # attn_input is the normalized hidden state passed to attention.
        # This is also the gate input X inside GatedMultiHeadAttention,
        # making G1 gate scores query-dependent (paper Sec 4.2).
        attn_input: torch.Tensor = self.norm1(X)

        # Step 2: Gated multi-head attention
        # Returns (output, attn_weights) where attn_weights may be None
        # if F.scaled_dot_product_attention (flash attention) was used.
        attn_out: torch.Tensor
        attn_weights: Optional[torch.Tensor]
        attn_out, attn_weights = self.attention(
            attn_input,
            mask=mask,
            position_ids=position_ids,
            return_attn_weights=False,  # Use flash attention during training
        )

        # Step 3: Store attention weights for analysis (detached to avoid memory leak)
        # AttentionSinkAnalyzer reads this to compute first-token attention proportions.
        # Paper Fig. 2 and Table 4 'F-Attn' column use these stored weights.
        self.last_attn_weights = (
            attn_weights.detach() if attn_weights is not None else None
        )

        # Step 4: Optional sandwich norm on attention output
        # Paper Table 2 row 7: sandwich norm prevents large attention outputs
        # from entering the residual stream, improving training stability.
        if self.sandwich_norm:
            # norm_attn_out is guaranteed non-None when sandwich_norm=True
            attn_out = self.norm_attn_out(attn_out)  # type: ignore[misc]

        # Step 5: Residual addition
        X = X + attn_out

        # -----------------------------------------------------------------------
        # FFN sublayer with pre-norm
        # -----------------------------------------------------------------------
        # Step 6: Pre-FFN RMSNorm
        ffn_input: torch.Tensor = self.norm2(X)

        # Step 7: SwiGLU feed-forward (or MoELayer when replaced by MoEModel)
        ffn_out: torch.Tensor = self.ffn(ffn_input)

        # Step 8: Optional sandwich norm on FFN output
        # Paper Appendix A.3: FFN at layer 5 produces massive activations in
        # the baseline. Sandwich norm prevents these from entering the residual.
        if self.sandwich_norm:
            # norm_ffn_out is guaranteed non-None when sandwich_norm=True
            ffn_out = self.norm_ffn_out(ffn_out)  # type: ignore[misc]

        # Step 9: Residual addition
        X = X + ffn_out

        return X

    def extra_repr(self) -> str:
        """Return human-readable module description for print(model).

        Returns:
            String summarizing the block configuration.
        """
        return (
            f"layer_idx={self.layer_idx}, "
            f"sandwich_norm={self.sandwich_norm}"
        )
