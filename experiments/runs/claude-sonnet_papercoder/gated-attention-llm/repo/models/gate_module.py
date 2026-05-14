## models/gate_module.py
"""Configurable gating module for all gated attention variants.

This module implements every gating variant described in the paper
"Gated Attention for Large Language Models: Non-linearity, Sparsity,
and Attention-Sink-Free", covering Tables 1, 3, 4, and 6.

The GateModule class is a drop-in component used by GatedMultiHeadAttention.
It handles all combinations of:
    - Position: G1 (SDPA output), G2 (value), G3 (key), G4 (query), G5 (dense output)
    - Granularity: elementwise or headwise
    - Head specificity: head-specific or head-shared
    - Gate type: multiplicative or additive
    - Activation: sigmoid, silu, identity, ns_sigmoid, rmsnorm, silu_only
    - Input independence: standard (query-dependent) or input-independent ablation

Config values used (from config.yaml):
    gate.position: 'G1' through 'G5' or 'none'
    gate.granularity: 'elementwise' or 'headwise'
    gate.head_specific: bool
    gate.gate_type: 'multiplicative' or 'additive'
    gate.activation: 'sigmoid', 'silu', 'identity', 'ns_sigmoid', 'rmsnorm', 'silu_only'
    model.d_model: hidden dimension (2048 for dense, 4096 for MoE)
    model.num_heads: query heads q (32 for both)
    model.num_kv_heads: key-value heads k (8 for dense, 4 for MoE)
    model.d_k: per-head dimension (64 for dense, 128 for MoE)
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Valid configuration constants
# ---------------------------------------------------------------------------
_VALID_POSITIONS = frozenset({"G1", "G2", "G3", "G4", "G5"})
_VALID_GRANULARITIES = frozenset({"elementwise", "headwise"})
_VALID_GATE_TYPES = frozenset({"multiplicative", "additive"})
_VALID_ACTIVATIONS = frozenset(
    {"sigmoid", "silu", "identity", "ns_sigmoid", "rmsnorm", "silu_only"}
)


class GateModule(nn.Module):
    """Configurable gating module supporting all paper variants.

    Implements the gating mechanism:
        Y' = g(Y, X, W_θ, σ) = Y ⊙ σ(X W_θ)   [multiplicative]
        Y' = Y + σ(X W_θ)                         [additive]

    where Y is the tensor being modulated, X is the pre-norm hidden state
    (input to the attention layer), W_θ are learnable gate parameters, and
    σ is an activation function.

    Attributes:
        position: Gate position identifier ('G1'–'G5').
        granularity: 'elementwise' or 'headwise'.
        head_specific: Whether each head has independent gate weights.
        gate_type: 'multiplicative' or 'additive'.
        activation: Name of the activation function.
        d_model: Input hidden dimension.
        num_heads: Number of query heads (q).
        num_kv_heads: Number of key-value heads (k).
        d_k: Per-head dimension.
        input_independent: If True, uses zero-init parameter instead of projection.
        gate_proj: Linear projection W_θ, or None for special variants.
        gate_param: Learnable parameter for input-independent variant, or None.
        norm: RMSNorm for the 'rmsnorm' activation variant, or None.
        activation_fn: Callable activation function, or None for rmsnorm/silu_only.
    """

    def __init__(
        self,
        position: str = "G1",
        granularity: str = "elementwise",
        head_specific: bool = True,
        gate_type: str = "multiplicative",
        activation: str = "sigmoid",
        d_model: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        d_k: int = 128,
        input_independent: bool = False,
    ) -> None:
        """Initialize the gate module and build all learnable components.

        Args:
            position: Where to apply the gate. One of 'G1' (SDPA output),
                'G2' (value projection), 'G3' (key projection),
                'G4' (query projection), 'G5' (dense output).
                Paper finding: G1 yields best results (Table 1).
            granularity: 'elementwise' for per-dimension gating (score shape
                matches Y's last two dims) or 'headwise' for a single scalar
                per head per token. Paper default: 'elementwise'.
            head_specific: If True, each attention head has its own W_θ
                (head-specific). If False, W_θ is shared across heads
                (head-shared). Paper Sec 3.2.1: head-specific is better.
            gate_type: 'multiplicative' (Y' = Y * σ(XW_θ)) or 'additive'
                (Y' = Y + σ(XW_θ)). Paper default: 'multiplicative'.
            activation: Activation function name. Options:
                - 'sigmoid': σ(x) = 1/(1+e^{-x}), scores in [0,1]. Default.
                - 'silu': σ(x) = x * sigmoid(x), unbounded.
                - 'identity': σ(x) = x, no non-linearity in activation.
                - 'ns_sigmoid': 0.5 + 0.5*sigmoid(x), scores in [0.5,1].
                  Ablation for sparsity analysis (Table 4 row 7).
                - 'rmsnorm': Apply per-head RMSNorm to Y, no projection.
                  Table 3 row 5 (GroupNorm baseline).
                - 'silu_only': Apply SiLU directly to Y, no projection.
                  Table 3 row 6 (no added params).
            d_model: Hidden dimension of the model. From model.d_model.
                Default 2048 (dense 1.7B). MoE uses 4096.
            num_heads: Number of query heads q. From model.num_heads.
                Default 32 (paper Table 1 baseline).
            num_kv_heads: Number of key-value heads k for GQA.
                From model.num_kv_heads. Default 8 (dense). MoE uses 4.
            d_k: Per-head dimension. From model.d_k. Default 128
                (paper Table 1: dk=128 for all methods).
            input_independent: If True, uses zero-initialized nn.Parameter
                instead of a linear projection. Ignores X in forward().
                Table 4 row 6 ablation to test query-dependency importance.

        Raises:
            ValueError: If any argument is not in the valid set.
        """
        super().__init__()

        # Validate inputs
        if position not in _VALID_POSITIONS:
            raise ValueError(
                f"position must be one of {_VALID_POSITIONS}, got '{position}'"
            )
        if granularity not in _VALID_GRANULARITIES:
            raise ValueError(
                f"granularity must be one of {_VALID_GRANULARITIES}, got '{granularity}'"
            )
        if gate_type not in _VALID_GATE_TYPES:
            raise ValueError(
                f"gate_type must be one of {_VALID_GATE_TYPES}, got '{gate_type}'"
            )
        if activation not in _VALID_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {_VALID_ACTIVATIONS}, got '{activation}'"
            )

        # Store configuration
        self.position: str = position
        self.granularity: str = granularity
        self.head_specific: bool = head_specific
        self.gate_type: str = gate_type
        self.activation: str = activation
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads
        self.d_k: int = d_k
        self.input_independent: bool = input_independent

        # Initialize learnable components to None; populated by _build_projection
        self.gate_proj: Optional[nn.Linear] = None
        self.gate_param: Optional[nn.Parameter] = None
        self.norm: Optional[nn.RMSNorm] = None
        self.activation_fn: Optional[Callable] = None

        # Build projection and activation
        self._build_projection()
        self.activation_fn = self._get_activation_fn()

    def _build_projection(self) -> None:
        """Build the gate projection matrix W_θ or special parameter.

        Sets self.gate_proj (nn.Linear), self.gate_param (nn.Parameter),
        or self.norm (nn.RMSNorm) depending on the variant. At most one
        of these will be non-None after this call.

        The output feature count of W_θ is determined by the combination
        of position, granularity, and head_specific, as described in the
        Logic Analysis section.

        Parameter count verification against Table 1 (MoE: d_model=4096,
        num_heads=32, num_kv_heads=4, d_k=128, 24 layers):
            G1 elementwise head-specific: 4096*(32*128)*24 ≈ 201M ✓
            G1 headwise head-specific:    4096*32*24       ≈ 1.6M ✓
            G2 elementwise head-specific: 4096*(4*128)*24  ≈ 25M  ✓
            G5 dense output:              4096*4096*24     ≈ 100M (approx) ✓
        """
        # --- Special variant: rmsnorm ---
        # Apply per-head RMSNorm to Y directly. No projection needed.
        # Paper Table 3 row 5: "SDPA GroupNorm (RMSNorm per head)"
        if self.activation == "rmsnorm":
            self.norm = nn.RMSNorm(self.d_k)
            # gate_proj and gate_param remain None
            return

        # --- Special variant: silu_only ---
        # Apply SiLU directly to Y. No projection, no added parameters.
        # Paper Table 3 row 6: "SDPA SiLU only (no gate projection)"
        if self.activation == "silu_only":
            # gate_proj, gate_param, norm all remain None
            return

        # --- Special variant: input_independent ---
        # Zero-initialized learnable parameter, ignores input X.
        # Paper Table 4 row 6: tests importance of query-dependency.
        if self.input_independent:
            # Shape [num_heads, d_k] — after sigmoid gives 0.5 everywhere initially
            self.gate_param = nn.Parameter(torch.zeros(self.num_heads, self.d_k))
            # gate_proj remains None
            return

        # --- G5: Dense output gate ---
        # Gates the full d_model output of W_O. No head structure.
        # Paper Table 1 row 9: "Dense Output G5", score shape n × d_model.
        if self.position == "G5":
            self.gate_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            return

        # --- Standard cases: G1, G2, G3, G4 ---
        # Determine the number of head slots based on position.
        # G1 (SDPA output) and G4 (query) use query heads (num_heads = q).
        # G2 (value) and G3 (key) use KV heads (num_kv_heads = k).
        if self.position in ("G1", "G4"):
            n_heads_for_gate: int = self.num_heads
        else:  # G2, G3
            n_heads_for_gate = self.num_kv_heads

        # Determine out_features based on granularity and head_specific.
        if self.granularity == "elementwise":
            if self.head_specific:
                # Each head has its own d_k-dimensional gate vector.
                # Score shape after reshape: [B, n, n_heads, d_k]
                out_features: int = n_heads_for_gate * self.d_k
            else:
                # Head-shared: single d_k vector broadcast across all heads.
                # Paper: "average over the query head dimension q to obtain
                # an n × d_k score from the original n × q × d_k"
                # We directly project to d_k (equivalent, more efficient).
                # Score shape after unsqueeze: [B, n, 1, d_k]
                out_features = self.d_k
        else:  # headwise
            if self.head_specific:
                # One scalar per head per token.
                # Score shape after unsqueeze: [B, n, n_heads, 1]
                out_features = n_heads_for_gate
            else:
                # Single scalar for all heads.
                # Score shape after unsqueeze: [B, n, 1, 1]
                out_features = 1

        self.gate_proj = nn.Linear(self.d_model, out_features, bias=False)

    def _get_activation_fn(self) -> Optional[Callable]:
        """Build and return the activation function callable.

        For 'rmsnorm' and 'silu_only', returns None because these variants
        are handled directly in forward() without a separate activation step.

        Returns:
            Callable activation function, or None for rmsnorm/silu_only.

        Raises:
            ValueError: If activation is not recognized (defensive check).
        """
        if self.activation == "sigmoid":
            return torch.sigmoid

        elif self.activation == "silu":
            return F.silu

        elif self.activation == "identity":
            # No activation — used for additive gating ablation (Table 3 row 7).
            # Y' = Y + X W_θ (linear, no non-linearity in gate output)
            return lambda x: x

        elif self.activation == "ns_sigmoid":
            # Non-Sparse sigmoid: constrains scores to [0.5, 1.0].
            # Paper Sec 4.2: "NS-sigmoid(x) = 0.5 + 0.5 * sigmoid(x)"
            # Ensures non-linearity while removing sparsity.
            # Used to validate that sparsity (not just non-linearity) matters.
            return lambda x: 0.5 + 0.5 * torch.sigmoid(x)

        elif self.activation == "rmsnorm":
            # Handled in forward() via self.norm. No activation_fn needed.
            return None

        elif self.activation == "silu_only":
            # Handled in forward() by applying F.silu directly to Y.
            return None

        else:
            # Should never reach here due to __init__ validation
            raise ValueError(f"Unknown activation: '{self.activation}'")

    def get_gate_scores(self, X: torch.Tensor) -> torch.Tensor:
        """Compute gate scores from input X and reshape for broadcasting with Y.

        The gate scores are computed as σ(X W_θ) and reshaped to be
        broadcast-compatible with the tensor Y being gated.

        Args:
            X: Pre-norm hidden states, shape [batch, seq, d_model].
                This is always the input to the attention layer (query-dependent
                for G1, key/value-dependent for G2/G3).

        Returns:
            Gate scores tensor shaped for elementwise multiplication with Y:
                - G1/G4 elementwise head-specific:  [batch, seq, num_heads, d_k]
                - G1/G4 elementwise head-shared:    [batch, seq, 1, d_k]
                - G1/G4 headwise head-specific:     [batch, seq, num_heads, 1]
                - G1/G4 headwise head-shared:        [batch, seq, 1, 1]
                - G2/G3 elementwise head-specific:  [batch, seq, num_kv_heads, d_k]
                - G2/G3 headwise head-specific:     [batch, seq, num_kv_heads, 1]
                - G5:                               [batch, seq, d_model]
                - input_independent:                [1, 1, num_heads, d_k]

        Raises:
            RuntimeError: If called when activation is 'rmsnorm' or 'silu_only'
                (these variants do not use gate scores).
        """
        if self.activation in ("rmsnorm", "silu_only"):
            raise RuntimeError(
                f"get_gate_scores() should not be called for activation='{self.activation}'. "
                "These variants are handled directly in forward()."
            )

        # --- Input-independent gate ---
        # Uses zero-initialized parameter, ignores X entirely.
        # Paper Table 4 row 6: tests importance of query-dependency.
        if self.input_independent:
            # gate_param shape: [num_heads, d_k]
            # Apply sigmoid, broadcast to [1, 1, num_heads, d_k]
            scores = torch.sigmoid(self.gate_param)
            return scores.unsqueeze(0).unsqueeze(0)

        # --- Standard projection-based gate scores ---
        # Compute raw scores: [batch, seq, out_features]
        raw: torch.Tensor = self.gate_proj(X)

        # Apply activation function
        scores: torch.Tensor = self.activation_fn(raw)

        batch_size: int = X.shape[0]
        seq_len: int = X.shape[1]

        # --- G5: No head structure, return as-is ---
        if self.position == "G5":
            # scores shape: [batch, seq, d_model] — matches Y for G5
            return scores

        # --- Determine head count for reshape ---
        if self.position in ("G1", "G4"):
            n_heads: int = self.num_heads
        else:  # G2, G3
            n_heads = self.num_kv_heads

        # --- Reshape based on granularity and head_specific ---
        if self.granularity == "elementwise":
            if self.head_specific:
                # [batch, seq, n_heads * d_k] → [batch, seq, n_heads, d_k]
                return scores.view(batch_size, seq_len, n_heads, self.d_k)
            else:
                # head-shared: [batch, seq, d_k] → [batch, seq, 1, d_k]
                # Broadcasts over all heads during multiplication with Y
                return scores.unsqueeze(2)

        else:  # headwise
            if self.head_specific:
                # [batch, seq, n_heads] → [batch, seq, n_heads, 1]
                # Broadcasts over d_k dimension during multiplication with Y
                return scores.unsqueeze(-1)
            else:
                # [batch, seq, 1] → [batch, seq, 1, 1]
                # Broadcasts over both num_heads and d_k
                return scores.unsqueeze(-1)

    def forward(self, Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Apply gating to tensor Y using input X to compute gate scores.

        Implements the gating formula from the paper (Eq. 5):
            Multiplicative: Y' = Y ⊙ σ(X W_θ)
            Additive:        Y' = Y + σ(X W_θ)

        Special variants (rmsnorm, silu_only) bypass the projection entirely.

        Args:
            Y: Tensor being modulated. Shape depends on gate position:
                - G1: [batch, seq, num_heads, d_k]  (SDPA output per head)
                - G2: [batch, seq, num_kv_heads, d_k]  (value projection output)
                - G3: [batch, seq, num_kv_heads, d_k]  (key projection output)
                - G4: [batch, seq, num_heads, d_k]  (query projection output)
                - G5: [batch, seq, d_model]  (final attention output)
            X: Pre-norm hidden states, shape [batch, seq, d_model].
                Always the input to the attention layer. For G1, this makes
                the gate query-dependent (current token's representation).
                For G2/G3, X is the same input but gates key/value outputs.

        Returns:
            Gated tensor Y' with the same shape as Y.
        """
        # --- Special variant: rmsnorm ---
        # Apply per-head RMSNorm to Y directly. No gate projection.
        # Paper Table 3 row 5: introduces non-linearity without input-dependency.
        # Y shape: [batch, seq, num_heads, d_k]
        # nn.RMSNorm(d_k) normalizes the last dimension, broadcasting correctly.
        if self.activation == "rmsnorm":
            return self.norm(Y)

        # --- Special variant: silu_only ---
        # Apply SiLU directly to Y. No projection, no added parameters.
        # Paper Table 3 row 6: tests whether non-linearity alone (without
        # learned gating) provides benefit. Shows modest PPL reduction.
        if self.activation == "silu_only":
            return F.silu(Y)

        # --- Multiplicative gating ---
        # Y' = Y ⊙ σ(X W_θ)
        # Paper default (Sec 2.2): "Unless otherwise specified, we employ
        # head-specific, multiplicative gating utilizing the sigmoid activation"
        if self.gate_type == "multiplicative":
            gate_scores = self.get_gate_scores(X)
            return Y * gate_scores

        # --- Additive gating ---
        # Y' = Y + σ(X W_θ)
        # Paper Sec 2.2: "Additive Gating: Y' = Y + σ(Xθ)"
        # Uses SiLU due to unbounded output range (Table 1 row 14).
        # Identity activation used for Table 3 row 7 ablation.
        else:  # gate_type == 'additive'
            raw: torch.Tensor = self.gate_proj(X)
            gate_output: torch.Tensor = self.activation_fn(raw)

            # Reshape gate_output to match Y's shape for addition.
            # Y shape: [batch, seq, num_heads, d_k] for G1/G4
            #          [batch, seq, num_kv_heads, d_k] for G2/G3
            #          [batch, seq, d_model] for G5
            gate_output = gate_output.view_as(Y)
            return Y + gate_output

    def count_params(self) -> int:
        """Count the number of learnable parameters added by this gate module.

        Used to verify parameter counts against Table 1's "Added Param" column.
        Note: multiply by num_layers to get total added params for the full model.

        Returns:
            Number of parameters in this gate module (int).

        Example (MoE-15A2B, d_model=4096, num_heads=32, num_kv_heads=4, d_k=128):
            G1 elementwise head-specific per layer: 4096 * (32*128) = 16,777,216
            × 24 layers ≈ 201M total ✓ (Table 1 row 5)

            G1 headwise head-specific per layer: 4096 * 32 = 131,072
            × 24 layers ≈ 1.6M total ✓ (Table 1 row 10)

            G2 elementwise head-specific per layer: 4096 * (4*128) = 2,097,152
            × 24 layers ≈ 25M total ✓ (Table 1 row 6)
        """
        total: int = 0

        if self.gate_proj is not None:
            total += self.gate_proj.weight.numel()

        if self.gate_param is not None:
            total += self.gate_param.numel()

        if self.norm is not None:
            # nn.RMSNorm has a learnable scale parameter of shape [d_k]
            if self.norm.weight is not None:
                total += self.norm.weight.numel()

        return total

    def extra_repr(self) -> str:
        """Return a string representation of the module's configuration.

        Used by PyTorch's print(model) for human-readable module descriptions.

        Returns:
            String summarizing the gate configuration.
        """
        return (
            f"position={self.position}, "
            f"granularity={self.granularity}, "
            f"head_specific={self.head_specific}, "
            f"gate_type={self.gate_type}, "
            f"activation={self.activation}, "
            f"d_model={self.d_model}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"d_k={self.d_k}, "
            f"input_independent={self.input_independent}, "
            f"added_params={self.count_params():,}"
        )
