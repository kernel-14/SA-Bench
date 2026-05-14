## models/transformer/transformer_blocks.py
"""AdaLN Transformer Block for the NFIG Transformer.

Implements the AdaLNTransformerBlock class — the core building block of the
NFIG Transformer. Each block applies Adaptive Layer Normalization (AdaLN-Zero)
for class-conditional generation, following the DiT/VAR design pattern.

The block structure per forward pass:
    x → AdaLN-modulated norm → BlockwiseCausalAttention → gated residual
      → AdaLN-modulated norm → FFN → gated residual → x

AdaLN-Zero initialization ensures each block starts as an identity function,
which is critical for stable training of the 16-block NFIG Transformer.

Paper references:
    - Section 3.2: "decoder-only transformer framework and block-wise causal attention"
    - Section 4.1: "VAR Transformer backbone with a depth of 16"
    - Table 5 row 4: "+ AdaLN" component in ablation study

Config values used (config.yaml nfig section):
    hidden_dim:  1024   (D — transformer hidden dimension)
    num_heads:   16     (H — number of attention heads)
    ffn_ratio:   4      (expansion ratio for FFN intermediate dimension)
    dropout:     0.0    (no dropout per config)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.transformer.attention import BlockwiseCausalAttention


class AdaLNTransformerBlock(nn.Module):
    """Transformer block with Adaptive Layer Normalization (AdaLN-Zero).

    Implements a single transformer layer with class-conditional modulation
    via AdaLN-Zero. The conditioning vector (class embedding) is projected
    to 6 modulation parameters that control scale, shift, and gating for
    both the attention and FFN sublayers.

    AdaLN-Zero formulation (DiT-style):
        modulation = adaLN_modulation(cond)  # [B, 6*D]
        shift_1, scale_1, gate_1, shift_2, scale_2, gate_2 = split(modulation)

        # Attention sublayer:
        x_norm1 = norm1(x) * (1 + scale_1) + shift_1
        x = x + gate_1 * attn(x_norm1, attn_mask)

        # FFN sublayer:
        x_norm2 = norm2(x) * (1 + scale_2) + shift_2
        x = x + gate_2 * ffn(x_norm2)

    The `(1 + scale)` formulation ensures identity behavior when scale=0
    (at initialization). The `gate` starts at 0 (AdaLN-Zero init), making
    each block a no-op at initialization — critical for 16-block stability.

    Attributes:
        attn: BlockwiseCausalAttention module for frequency-band-aware attention.
        ffn: Two-layer MLP (hidden_dim → hidden_dim*ffn_ratio → hidden_dim).
        norm1: LayerNorm without affine parameters (AdaLN provides scale/shift).
        norm2: LayerNorm without affine parameters (AdaLN provides scale/shift).
        adaLN_modulation: Linear(hidden_dim, 6*hidden_dim) for conditioning.
            Zero-initialized (AdaLN-Zero) for training stability.
        hidden_dim: Transformer hidden dimension D = 1024.
        ffn_ratio: FFN expansion ratio = 4 (intermediate dim = 4096).
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        ffn_ratio: float = 4.0,
    ) -> None:
        """Initialize the AdaLNTransformerBlock.

        Args:
            hidden_dim: Transformer hidden dimension D.
                From config.nfig.hidden_dim = 1024.
                Must be divisible by num_heads.
                Must be a positive integer.
            num_heads: Number of attention heads H.
                From config.nfig.num_heads = 16.
                Must be a positive integer that divides hidden_dim evenly.
            ffn_ratio: Expansion ratio for the FFN intermediate dimension.
                From config.nfig.ffn_ratio = 4.
                FFN intermediate dimension = hidden_dim * ffn_ratio = 4096.
                Must be a positive number.

        Raises:
            ValueError: If hidden_dim is not divisible by num_heads.
            ValueError: If hidden_dim, num_heads, or ffn_ratio are not positive.
        """
        super().__init__()

        # --- Input validation ---
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be a positive integer, got {hidden_dim}."
            )
        if num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, got {num_heads}."
            )
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}. "
                f"Got hidden_dim % num_heads = {hidden_dim % num_heads}."
            )
        if ffn_ratio <= 0:
            raise ValueError(
                f"ffn_ratio must be a positive number, got {ffn_ratio}."
            )

        # Store configuration as attributes for downstream access.
        self.hidden_dim: int = hidden_dim
        self.ffn_ratio: float = ffn_ratio

        # Compute FFN intermediate dimension.
        ffn_hidden_dim: int = int(hidden_dim * ffn_ratio)  # 1024 * 4 = 4096

        # ------------------------------------------------------------------ #
        # 1. Attention sublayer: BlockwiseCausalAttention
        # ------------------------------------------------------------------ #
        # Handles multi-head scaled dot-product attention with the block-wise
        # causal mask enforcing the NFIG autoregressive constraint.
        # The attn_mask is NOT built here — it is passed in from NFIGTransformer.
        self.attn: BlockwiseCausalAttention = BlockwiseCausalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )

        # ------------------------------------------------------------------ #
        # 2. FFN sublayer: two-layer MLP with GELU activation
        # ------------------------------------------------------------------ #
        # Standard transformer FFN: Linear → GELU → Linear.
        # No dropout (config.nfig.dropout = 0.0).
        # Applied position-wise (same weights for all token positions).
        self.ffn: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(ffn_hidden_dim, hidden_dim, bias=True),
        )
        self._init_ffn_weights()

        # ------------------------------------------------------------------ #
        # 3. Layer normalization (without affine parameters)
        # ------------------------------------------------------------------ #
        # elementwise_affine=False: AdaLN provides scale and shift externally.
        # Having both LayerNorm affine params AND AdaLN modulation would be
        # redundant and could interfere with the AdaLN-Zero initialization.
        # eps=1e-6: standard for transformer layer norms (avoids division by zero).
        self.norm1: nn.LayerNorm = nn.LayerNorm(
            hidden_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.norm2: nn.LayerNorm = nn.LayerNorm(
            hidden_dim,
            elementwise_affine=False,
            eps=1e-6,
        )

        # ------------------------------------------------------------------ #
        # 4. AdaLN modulation: conditioning vector → 6 modulation parameters
        # ------------------------------------------------------------------ #
        # Maps the class conditioning vector [B, hidden_dim] to 6 modulation
        # parameters [B, 6 * hidden_dim]:
        #   [shift_1, scale_1, gate_1, shift_2, scale_2, gate_2]
        #
        # Each parameter has shape [B, hidden_dim] after splitting.
        # - shift_1, scale_1: modulate norm1 output before attention
        # - gate_1: scale attention output before residual add
        # - shift_2, scale_2: modulate norm2 output before FFN
        # - gate_2: scale FFN output before residual add
        #
        # AdaLN-Zero initialization: both weight and bias are set to zero.
        # This ensures all 6 modulation parameters start at zero:
        #   - scale_1 = scale_2 = 0 → (1 + scale) = 1 → identity scaling
        #   - shift_1 = shift_2 = 0 → no shift
        #   - gate_1 = gate_2 = 0 → residual branches contribute nothing
        # The block therefore starts as an identity function, which is critical
        # for stable training of the 16-block NFIG Transformer.
        self.adaLN_modulation: nn.Linear = nn.Linear(
            hidden_dim,
            6 * hidden_dim,
            bias=True,
        )
        self._init_adaLN_weights()

    def _init_ffn_weights(self) -> None:
        """Initialize FFN weights using standard transformer initialization.

        The first linear layer uses Kaiming uniform (PyTorch default for Linear).
        The second linear layer (output of FFN residual branch) uses a small
        normal initialization to reduce the initial magnitude of FFN outputs,
        complementing the AdaLN-Zero gate initialization.

        This follows the pattern from GPT-2 and subsequent work where the
        output projection of each residual branch is scaled down.
        """
        # First FFN layer: default PyTorch initialization (Kaiming uniform).
        # nn.Linear already applies this by default; explicit for clarity.
        first_linear: nn.Linear = self.ffn[0]  # type: ignore[index]
        nn.init.kaiming_uniform_(first_linear.weight, nonlinearity="linear")
        if first_linear.bias is not None:
            nn.init.zeros_(first_linear.bias)

        # Second FFN layer (output projection): small normal initialization.
        # Reduces initial FFN output magnitude, complementing gate_2=0 init.
        second_linear: nn.Linear = self.ffn[2]  # type: ignore[index]
        nn.init.normal_(second_linear.weight, mean=0.0, std=0.02)
        if second_linear.bias is not None:
            nn.init.zeros_(second_linear.bias)

    def _init_adaLN_weights(self) -> None:
        """Initialize AdaLN modulation weights to zero (AdaLN-Zero).

        Zero-initializes both weight and bias of adaLN_modulation so that
        all 6 modulation parameters (shift_1, scale_1, gate_1, shift_2,
        scale_2, gate_2) start at zero.

        Effect at initialization:
            - scale_1 = scale_2 = 0 → (1 + scale) = 1 → identity scaling
            - shift_1 = shift_2 = 0 → no additive shift
            - gate_1 = gate_2 = 0 → residual branches contribute nothing

        This makes each block an identity function at initialization, which
        is the AdaLN-Zero design from DiT (Peebles & Xie 2023) and is
        critical for stable training of deep transformers (16 blocks here).
        """
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        attn_mask: Tensor,
    ) -> Tensor:
        """Apply one AdaLN transformer block to the token sequence.

        Implements the full AdaLN-Zero transformer block:

            Step 1: Compute 6 modulation parameters from conditioning vector.
            Step 2: Attention sublayer with AdaLN pre-norm and gated residual.
            Step 3: FFN sublayer with AdaLN pre-norm and gated residual.

        The block-wise causal attention mask is passed through to the attention
        module without modification. It is precomputed once in NFIGTransformer
        and shared across all 16 blocks.

        Args:
            x: Token sequence of shape (B, T, D) where:
                - B: batch size
                - T: total_tokens = 680 (config.nfig.total_tokens)
                - D: hidden_dim = 1024 (config.nfig.hidden_dim)
                Values are transformer hidden states in an unconstrained range.
            cond: Class conditioning vector of shape (B, D).
                Output of the class embedding lookup in NFIGTransformer.
                For CFG training: some entries use the null class embedding
                (class_id = 1000, config.nfig.null_class_id).
                For CFG inference: two forward passes are made — one with
                the real class embedding and one with the null class embedding.
            attn_mask: Block-wise causal attention mask of shape (T, T), dtype bool.
                Entry [i, j] = True means token i is allowed to attend to token j.
                Precomputed by NFIGTransformer._build_attn_mask() using
                BlockwiseCausalAttention.build_causal_mask().
                Must be on the same device as x.

        Returns:
            Updated token sequence of shape (B, T, D).
            Same shape as input x. Gradients flow through all operations.

        Raises:
            RuntimeError: If x.shape[-1] != hidden_dim (dimension mismatch).
            RuntimeError: If cond.shape != (B, hidden_dim).
            RuntimeError: If attn_mask.shape[0] != x.shape[1] (token count mismatch).
        """
        B: int
        T: int
        D: int
        B, T, D = x.shape

        # --- Validate input dimensions ---
        if D != self.hidden_dim:
            raise RuntimeError(
                f"Input hidden dimension D={D} does not match "
                f"expected hidden_dim={self.hidden_dim}. "
                f"Input x shape: {tuple(x.shape)}."
            )

        if cond.shape != (B, self.hidden_dim):
            raise RuntimeError(
                f"Conditioning vector shape {tuple(cond.shape)} does not match "
                f"expected (B={B}, hidden_dim={self.hidden_dim}). "
                "Ensure cond is the class embedding of shape (B, hidden_dim)."
            )

        if attn_mask.shape[0] != T or attn_mask.shape[1] != T:
            raise RuntimeError(
                f"attn_mask shape {tuple(attn_mask.shape)} does not match "
                f"expected (T, T) = ({T}, {T}) where T = x.shape[1]. "
                "Ensure the mask was built with the same scale_factors as the "
                "token sequence length."
            )

        # ------------------------------------------------------------------ #
        # Step 1: Compute AdaLN modulation parameters from conditioning vector
        # ------------------------------------------------------------------ #
        # cond: [B, D] → modulation: [B, 6*D]
        modulation: Tensor = self.adaLN_modulation(cond)

        # Split into 6 equal chunks along the last dimension.
        # Each chunk: [B, D]
        # Order: shift_1, scale_1, gate_1, shift_2, scale_2, gate_2
        (
            shift_1,   # [B, D] — additive shift for attention pre-norm
            scale_1,   # [B, D] — multiplicative scale for attention pre-norm
            gate_1,    # [B, D] — gate for attention residual branch
            shift_2,   # [B, D] — additive shift for FFN pre-norm
            scale_2,   # [B, D] — multiplicative scale for FFN pre-norm
            gate_2,    # [B, D] — gate for FFN residual branch
        ) = modulation.chunk(6, dim=-1)

        # Unsqueeze to [B, 1, D] for broadcasting over the token dimension T.
        # This allows element-wise operations with x of shape [B, T, D].
        shift_1 = shift_1.unsqueeze(1)  # [B, 1, D]
        scale_1 = scale_1.unsqueeze(1)  # [B, 1, D]
        gate_1 = gate_1.unsqueeze(1)    # [B, 1, D]
        shift_2 = shift_2.unsqueeze(1)  # [B, 1, D]
        scale_2 = scale_2.unsqueeze(1)  # [B, 1, D]
        gate_2 = gate_2.unsqueeze(1)    # [B, 1, D]

        # ------------------------------------------------------------------ #
        # Step 2: Attention sublayer with AdaLN pre-norm and gated residual
        # ------------------------------------------------------------------ #

        # Pre-norm: apply LayerNorm (no affine) then AdaLN modulation.
        # norm1 normalizes x to zero mean and unit variance per token.
        # AdaLN then applies learned scale and shift from the conditioning vector.
        # (1 + scale_1): identity when scale_1=0 (at initialization).
        x_norm1: Tensor = self.norm1(x)                              # [B, T, D]
        x_norm1 = x_norm1 * (1.0 + scale_1) + shift_1               # [B, T, D]

        # Apply block-wise causal attention.
        # attn_mask enforces the NFIG frequency-band causal constraint.
        attn_out: Tensor = self.attn(x_norm1, attn_mask)             # [B, T, D]

        # Gated residual connection.
        # gate_1 starts at 0 (AdaLN-Zero init), so this branch contributes
        # nothing at initialization. Gradually opens during training.
        x = x + gate_1 * attn_out                                    # [B, T, D]

        # ------------------------------------------------------------------ #
        # Step 3: FFN sublayer with AdaLN pre-norm and gated residual
        # ------------------------------------------------------------------ #

        # Pre-norm: apply LayerNorm (no affine) then AdaLN modulation.
        # Same pattern as the attention sublayer but with separate parameters.
        x_norm2: Tensor = self.norm2(x)                              # [B, T, D]
        x_norm2 = x_norm2 * (1.0 + scale_2) + shift_2               # [B, T, D]

        # Apply FFN: Linear(D→4D) → GELU → Linear(4D→D).
        # Applied position-wise (same weights for all T token positions).
        ffn_out: Tensor = self.ffn(x_norm2)                          # [B, T, D]

        # Gated residual connection.
        # gate_2 starts at 0 (AdaLN-Zero init), same as gate_1.
        x = x + gate_2 * ffn_out                                     # [B, T, D]

        return x

    def extra_repr(self) -> str:
        """Return a human-readable string with key block configuration.

        Returns:
            String describing the block's hidden dimension, FFN ratio,
            and AdaLN initialization status.
        """
        ffn_hidden_dim: int = int(self.hidden_dim * self.ffn_ratio)
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"ffn_ratio={self.ffn_ratio}, "
            f"ffn_hidden_dim={ffn_hidden_dim}, "
            f"adaLN_zero_init=True, "
            f"norm_affine=False"
        )
