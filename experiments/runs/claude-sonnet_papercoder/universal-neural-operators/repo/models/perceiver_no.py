```python
## models/perceiver_no.py
"""
Perceiver IO-based Neural Operator backbone for the multi-physics neural
operator pretraining framework described in:
  "Towards Universal Neural Operators through Multiphysics Pretraining"

Implements the Perceiver IO-based NO described in Section 3 of the paper:

    "The introduction of Perceiver enabled the encoding of information with
    a smaller number of latent feature arrays, internal to operator blocks,
    thereby operating with more abstract feature arrays and maintaining a
    limited number of parameters. As the operators we employ, we use blocks
    based on the Perceiver IO, where the mapping is performed with a
    symmetrical cross-attention mechanism for outputs, which mirrors the
    cross-attention block for constructing representations of latent arrays
    from the input process."

    "The latent variables and input state are combined first with the
    cross-attention block, where keys and values are obtained from
    FNO-based mapping from the inputs K1 = FNO_K1(X), V1 = FNO_V1(X),
    and latent variables are taken as queries Q1 = L."

Architecture per block:
  1. Encode:  cross-attention(Q=latent, K=FNO_K(x), V=FNO_V(x))
  2. Process: self-attention(latent, latent, latent)
  3. Decode:  cross-attention(Q=x_tokens, K=latent, V=latent)

Classes:
  PerceiverBlock  - one encode-process-decode cycle with FNO-mapped K/V
  PerceiverNO     - stacks n_blocks PerceiverBlocks as the shared backbone

Tensor layout convention (Shared Knowledge #1):
  Channel-first: [B, C, L] for 1D, [B, C, H, W] for 2D.
  Internally flattened to [B, seq_len, d_model] for attention operations.

Config alignment (config.yaml):
  models.perceiver_no.hidden_dim: 128     -> hidden_dim parameter
  models.perceiver_no.n_modes: 16         -> n_modes for FNO blocks
  models.perceiver_no.n_layers: 2         -> n_blocks parameter
  models.perceiver_no.n_dims: 2           -> n_dims parameter
  models.perceiver_no.target_params: 1e8  -> approximate parameter count
  models.perceiver_no.perceiver.latent_dim: 256  -> latent_dim parameter
  models.perceiver_no.perceiver.n_latents: 64    -> n_latents parameter
  models.perceiver_no.perceiver.n_heads: 8       -> n_heads parameter
  models.perceiver_no.perceiver.n_blocks: 2      -> n_blocks parameter

Integration with AdapterFramework:
  PerceiverNO serves as the backbone argument to AdapterFramework.__init__.
  During pretraining: all parameters (latent arrays, FNO blocks, attention
  weights) are trainable.
  During fine-tuning: AdapterFramework.freeze_backbone() freezes all
  PerceiverNO parameters; only adapter parameters are updated.

Dependencies: torch, torch.nn, einops, models/fno_backbone.py.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from einops import rearrange as _einops_rearrange
    _EINOPS_AVAILABLE: bool = True
except ImportError:
    _EINOPS_AVAILABLE = False
    _einops_rearrange = None  # type: ignore[assignment]

from models.fno_backbone import FNOBlock

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

_logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: spatial flatten / unflatten
# ---------------------------------------------------------------------------


def _flatten_spatial(x: Tensor) -> Tuple[Tensor, Tuple[int, ...]]:
    """Flatten spatial dimensions of a channel-first tensor to a token sequence.

    Converts channel-first spatial tensors to the [B, seq_len, C] format
    expected by nn.MultiheadAttention with batch_first=True.

    Supports both 1D ([B, C, L]) and 2D ([B, C, H, W]) inputs.

    Args:
        x: Channel-first tensor.
            Shape [B, C, L] for 1D problems.
            Shape [B, C, H, W] for 2D problems.

    Returns:
        Tuple of:
          - tokens: Tensor of shape [B, seq_len, C] where
              seq_len = L for 1D, seq_len = H*W for 2D.
          - spatial_shape: Tuple of spatial dimensions (L,) for 1D or
              (H, W) for 2D. Used by _unflatten_spatial to restore shape.

    Raises:
        ValueError: If x has fewer than 3 or more than 4 dimensions.
    """
    if x.ndim == 3:
        # 1D: [B, C, L] -> [B, L, C]
        B, C, L = x.shape
        tokens: Tensor = x.permute(0, 2, 1).contiguous()  # [B, L, C]
        return tokens, (L,)
    elif x.ndim == 4:
        # 2D: [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = x.shape
        # permute to [B, H, W, C] then reshape to [B, H*W, C]
        tokens = x.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)
        return tokens, (H, W)
    else:
        raise ValueError(
            f"_flatten_spatial expects 3D [B, C, L] or 4D [B, C, H, W] input, "
            f"got {x.ndim}D tensor with shape {tuple(x.shape)}."
        )


def _unflatten_spatial(
    tokens: Tensor,
    spatial_shape: Tuple[int, ...],
) -> Tensor:
    """Restore spatial structure from a flattened token sequence.

    Inverse of _flatten_spatial. Converts [B, seq_len, C] back to
    channel-first spatial format.

    Args:
        tokens: Flattened token tensor of shape [B, seq_len, C].
        spatial_shape: Tuple of spatial dimensions returned by
            _flatten_spatial: (L,) for 1D or (H, W) for 2D.

    Returns:
        Channel-first spatial tensor:
          Shape [B, C, L] for 1D (spatial_shape = (L,)).
          Shape [B, C, H, W] for 2D (spatial_shape = (H, W)).

    Raises:
        ValueError: If spatial_shape has an unsupported number of elements.
    """
    B, seq_len, C = tokens.shape

    if len(spatial_shape) == 1:
        # 1D: [B, L, C] -> [B, C, L]
        L: int = spatial_shape[0]
        return tokens.permute(0, 2, 1).contiguous()  # [B, C, L]
    elif len(spatial_shape) == 2:
        # 2D: [B, H*W, C] -> [B, C, H, W]
        H: int = spatial_shape[0]
        W: int = spatial_shape[1]
        # reshape to [B, H, W, C] then permute to [B, C, H, W]
        return (
            tokens.reshape(B, H, W, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # [B, C, H, W]
    else:
        raise ValueError(
            f"_unflatten_spatial expects spatial_shape of length 1 or 2, "
            f"got length {len(spatial_shape)}: {spatial_shape}."
        )


# ---------------------------------------------------------------------------
# Feed-forward network helper
# ---------------------------------------------------------------------------


class _FeedForward(nn.Module):
    """Two-layer MLP with GELU activation for transformer-style FFN blocks.

    Implements: Linear(d_in, d_hidden) -> GELU -> Linear(d_hidden, d_in)

    Used after self-attention on the latent array to increase expressivity,
    following standard transformer practice (Vaswani et al., 2017).

    Attributes:
        _fc1: First linear layer, d_in -> d_hidden.
        _act: GELU activation.
        _fc2: Second linear layer, d_hidden -> d_in.
    """

    def __init__(self, d_in: int, d_hidden: int) -> None:
        """Initialise _FeedForward.

        Args:
            d_in: Input and output dimension.
            d_hidden: Hidden dimension (typically d_in * 4 for transformers).
        """
        super().__init__()
        self._fc1: nn.Linear = nn.Linear(d_in, d_hidden)
        self._act: nn.GELU = nn.GELU()
        self._fc2: nn.Linear = nn.Linear(d_hidden, d_in)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the two-layer MLP.

        Args:
            x: Input tensor of shape [..., d_in].

        Returns:
            Output tensor of shape [..., d_in].
        """
        return self._fc2(self._act(self._fc1(x)))


# ---------------------------------------------------------------------------
# PerceiverBlock
# ---------------------------------------------------------------------------


class PerceiverBlock(nn.Module):
    """One Perceiver IO encode-process-decode cycle with FNO-mapped K/V.

    Implements the Perceiver IO block described in Section 3 of the paper.
    The block performs three operations:

    1. **Encode** (cross-attention: latent attends to FNO-processed input):
       - K1 = FNO_K(x)  — FNO-mapped keys from input (non-local structure)
       - V1 = FNO_V(x)  — FNO-mapped values from input
       - latent = cross_attn(Q=latent, K=proj_k(K1), V=proj_v(V1))

    2. **Process** (self-attention on latent):
       - latent = self_attn(Q=latent, K=latent, V=latent)
       - latent = ffn(latent)

    3. **Decode** (cross-attention: input attends to latent):
       - x_out = cross_attn(Q=x_tokens, K=proj_lk(latent), V=proj_lv(latent))

    The FNO blocks for K and V capture non-local spatial structure before
    the attention mechanism, giving the latent array access to globally-
    informed features rather than purely local ones. This is the key
    architectural distinction from standard Perceiver IO.

    The latent array is a learnable parameter of shape [n_latents, latent_dim]
    that acts as a fixed-size bottleneck, compressing the input into a
    smaller representation. The same latent array is used for all samples
    in a batch (expanded via unsqueeze(0).expand(B, -1, -1)).

    Attributes:
        hidden_dim: Input/output channel dimension (matches FNO backbone).
        latent_dim: Dimension of the latent array.
        n_latents: Number of latent vectors.
        n_heads: Number of attention heads.
        n_modes: Number of Fourier modes for internal FNO blocks.
        n_dims: Spatial dimensionality (1 or 2).
        latent_array: Learnable latent parameter, shape [n_latents, latent_dim].
        fno_k: FNOBlock for computing K1 = FNO_K(x).
        fno_v: FNOBlock for computing V1 = FNO_V(x).
        proj_k: Linear projection from hidden_dim to latent_dim for keys.
        proj_v: Linear projection from hidden_dim to latent_dim for values.
        cross_attn_encode: Cross-attention for encoding (latent attends to input).
        norm_encode: LayerNorm after encoding cross-attention.
        self_attn: Self-attention on latent.
        norm_self: LayerNorm after self-attention.
        ffn_latent: Feed-forward network on latent.
        norm_ffn_latent: LayerNorm after FFN.
        proj_latent_k: Linear projection from latent_dim to hidden_dim for decode keys.
        proj_latent_v: Linear projection from latent_dim to hidden_dim for decode values.
        cross_attn_decode: Cross-attention for decoding (input attends to latent).
        norm_decode: LayerNorm after decoding cross-attention.

    Example::

        block = PerceiverBlock(
            hidden_dim=128, latent_dim=256, n_latents=64,
            n_heads=8, n_modes=16, n_dims=2,
        )
        x = torch.randn(4, 128, 64, 64)   # [B, C, H, W]
        out = block(x)                     # [B, 128, 64, 64]
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 256,
        n_latents: int = 64,
        n_heads: int = 8,
        n_modes: int = 16,
        n_dims: int = 2,
    ) -> None:
        """Initialise PerceiverBlock.

        Args:
            hidden_dim: Input/output channel dimension. Must match the
                hidden_dim of the surrounding FNO backbone and adapters.
                From config.yaml models.perceiver_no.hidden_dim (default 128).
                Must be divisible by n_heads.
            latent_dim: Dimension of the latent array. Controls the
                expressiveness of the bottleneck representation.
                From config.yaml models.perceiver_no.perceiver.latent_dim
                (default 256). Must be divisible by n_heads.
            n_latents: Number of latent vectors. Controls the size of the
                bottleneck. Smaller n_latents -> more compression, faster
                self-attention. From config.yaml
                models.perceiver_no.perceiver.n_latents (default 64).
            n_heads: Number of attention heads for all attention layers.
                From config.yaml models.perceiver_no.perceiver.n_heads
                (default 8). Must divide both hidden_dim and latent_dim.
            n_modes: Number of Fourier modes for the internal FNO blocks
                (fno_k and fno_v). From config.yaml
                models.perceiver_no.n_modes (default 16).
            n_dims: Spatial dimensionality. 1 for 1D problems (Burgers,
                Advection), 2 for 2D problems (NS, RD, Gray-Scott, Heat).
                From config.yaml models.perceiver_no.n_dims (default 2).

        Raises:
            ValueError: If hidden_dim is not divisible by n_heads.
            ValueError: If latent_dim is not divisible by n_heads.
            ValueError: If any argument is non-positive.
            ValueError: If n_dims is not 1 or 2.
        """
        super().__init__()

        # ── Validate arguments ────────────────────────────────────────────
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if n_latents <= 0:
            raise ValueError(f"n_latents must be positive, got {n_latents}.")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}.")
        if n_modes <= 0:
            raise ValueError(f"n_modes must be positive, got {n_modes}.")
        if n_dims not in (1, 2):
            raise ValueError(
                f"n_dims must be 1 or 2, got {n_dims}. "
                f"3D problems are not currently supported."
            )
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by "
                f"n_heads={n_heads}. "
                f"Got hidden_dim % n_heads = {hidden_dim % n_heads}."
            )
        if latent_dim % n_heads != 0:
            raise ValueError(
                f"latent_dim={latent_dim} must be divisible by "
                f"n_heads={n_heads}. "
                f"Got latent_dim % n_heads = {latent_dim % n_heads}."
            )

        # ── Store hyperparameters ─────────────────────────────────────────
        self.hidden_dim: int = hidden_dim
        self.latent_dim: int = latent_dim
        self.n_latents: int = n_latents
        self.n_heads: int = n_heads
        self.n_modes: int = n_modes
        self.n_dims: int = n_dims

        # ── Learnable latent array ────────────────────────────────────────
        # Shape: [n_latents, latent_dim] — not batch-indexed.
        # Expanded to [B, n_latents, latent_dim] in forward().
        # Initialized with truncated normal (std=0.02) following standard
        # transformer practice (e.g., ViT, BERT initialization).
        self.latent_array: nn.Parameter = nn.Parameter(
            torch.empty(n_latents, latent_dim)
        )
        nn.init.trunc_normal_(self.latent_array, std=0.02)

        # ── FNO blocks for K and V (encode step) ──────────────────────────
        # K1 = FNO_K(x): FNO-processed keys from input.
        # V1 = FNO_V(x): FNO-processed values from input.
        # Separate weights — they learn different projections of the input.
        # Both operate in channel-first format [B, hidden_dim, *spatial].
        self.fno_k: FNOBlock = FNOBlock(
            hidden_dim=hidden_dim,
            n_modes=n_modes,
            n_dims=n_dims,
            activation="gelu",
        )
        self.fno_v: FNOBlock = FNOBlock(
            hidden_dim=hidden_dim,
            n_modes=n_modes,
            n_dims=n_dims,
            activation="gelu",
        )

        # ── Projections: hidden_dim -> latent_dim (for encode K, V) ───────
        # FNO blocks output hidden_dim channels, but cross_attn_encode
        # expects keys/values of dimension latent_dim (since Q=latent has
        # latent_dim). Linear projections bridge this gap.
        self.proj_k: nn.Linear = nn.Linear(hidden_dim, latent_dim)
        self.proj_v: nn.Linear = nn.Linear(hidden_dim, latent_dim)

        # ── Cross-attention: encode (latent attends to FNO-processed input) ─
        # Q = latent [B, n_latents, latent_dim]
        # K = proj_k(FNO_K(x)) [B, seq_len, latent_dim]
        # V = proj_v(FNO_V(x)) [B, seq_len, latent_dim]
        # Output: updated latent [B, n_latents, latent_dim]
        self.cross_attn_encode: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.norm_encode: nn.LayerNorm = nn.LayerNorm(latent_dim)

        # ── Self-attention: process (latent attends to itself) ─────────────
        # Q = K = V = latent [B, n_latents, latent_dim]
        # Output: refined latent [B, n_latents, latent_dim]
        self.self_attn: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.norm_self: nn.LayerNorm = nn.LayerNorm(latent_dim)

        # ── Feed-forward network on latent ────────────────────────────────
        # Applied after self-attention to increase expressivity.
        # Hidden dim = latent_dim * 4 follows standard transformer practice.
        self.ffn_latent: _FeedForward = _FeedForward(
            d_in=latent_dim,
            d_hidden=latent_dim * 4,
        )
        self.norm_ffn_latent: nn.LayerNorm = nn.LayerNorm(latent_dim)

        # ── Projections: latent_dim -> hidden_dim (for decode K, V) ───────
        # The decode cross-attention uses hidden_dim (matching the input
        # token dimension), so the latent must be projected down.
        self.proj_latent_k: nn.Linear = nn.Linear(latent_dim, hidden_dim)
        self.proj_latent_v: nn.Linear = nn.Linear(latent_dim, hidden_dim)

        # ── Cross-attention: decode (input attends to latent) ─────────────
        # Q = x_tokens [B, seq_len, hidden_dim]
        # K = proj_latent_k(latent) [B, n_latents, hidden_dim]
        # V = proj_latent_v(latent) [B, n_latents, hidden_dim]
        # Output: updated x_tokens [B, seq_len, hidden_dim]
        self.cross_attn_decode: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.norm_decode: nn.LayerNorm = nn.LayerNorm(hidden_dim)

        # ── Log parameter count ───────────────────────────────────────────
        n_params: int = sum(p.numel() for p in self.parameters())
        _logger.debug(
            "PerceiverBlock: hidden_dim=%d, latent_dim=%d, n_latents=%d, "
            "n_heads=%d, n_modes=%d, n_dims=%d. Parameters: %d.",
            hidden_dim,
            latent_dim,
            n_latents,
            n_heads,
            n_modes,
            n_dims,
            n_params,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply one Perceiver IO encode-process-decode cycle.

        Implements the full block described in the paper:
          1. Compute FNO-mapped keys and values from input.
          2. Encode: latent cross-attends to FNO-processed input.
          3. Process: latent self-attends to itself + FFN.
          4. Decode: input cross-attends to refined latent.

        All attention operations use post-norm residual connections
        (add then normalize), following standard transformer practice.

        Args:
            x: Input feature tensor from LiftingAdapter or previous block.
                Shape [B, hidden_dim, L] for 1D problems (n_dims=1).
                Shape [B, hidden_dim, H, W] for 2D problems (n_dims=2).

        Returns:
            Output feature tensor with the same shape as x.
            The Perceiver IO block is shape-preserving: input and output
            both have hidden_dim channels and the same spatial dimensions.

        Raises:
            ValueError: If x has an unsupported number of dimensions
                (must be 3 for 1D or 4 for 2D).
            ValueError: If the channel dimension of x does not match
                hidden_dim.
        """
        # ── Validate input ────────────────────────────────────────────────
        if x.ndim not in (3, 4):
            raise ValueError(
                f"PerceiverBlock expects 3D [B, C, L] or 4D [B, C, H, W] "
                f"input, got {x.ndim}D tensor with shape {tuple(x.shape)}."
            )

        c_in: int = x.shape[1]
        if c_in != self.hidden_dim:
            raise ValueError(
                f"Input channel dimension C={c_in} does not match "
                f"hidden_dim={self.hidden_dim}. "
                f"The LiftingAdapter must project to hidden_dim channels "
                f"before passing features to PerceiverBlock."
            )

        batch_size: int = x.shape[0]

        # ── Step 1: Compute FNO-mapped keys and values ────────────────────
        # Both operate in channel-first format [B, hidden_dim, *spatial].
        # K1 = FNO_K(x): captures non-local spatial structure for keys.
        # V1 = FNO_V(x): captures non-local spatial structure for values.
        k1: Tensor = self.fno_k(x)   # [B, hidden_dim, *spatial]
        v1: Tensor = self.fno_v(x)   # [B, hidden_dim, *spatial]

        # ── Step 2: Flatten spatial dims to token sequences ───────────────
        # Convert channel-first to [B, seq_len, C] for attention.
        # Store spatial_shape for unflattening after decode.
        x_tokens: Tensor
        spatial_shape: Tuple[int, ...]
        x_tokens, spatial_shape = _flatten_spatial(x)    # [B, seq_len, hidden_dim]

        k1_tokens: Tensor
        k1_tokens, _ = _flatten_spatial(k1)              # [B, seq_len, hidden_dim]

        v1_tokens: Tensor
        v1_tokens, _ = _flatten_spatial(v1)              # [B, seq_len, hidden_dim]

        # ── Step 3: Project K1, V1 to latent_dim for encoding ─────────────
        # FNO outputs hidden_dim; cross_attn_encode expects latent_dim.
        k1_proj: Tensor = self.proj_k(k1_tokens)   # [B, seq_len, latent_dim]
        v1_proj: Tensor = self.proj_v(v1_tokens)   # [B, seq_len, latent_dim]

        # ── Step 4: Expand latent array to batch dimension ────────────────
        # latent_array: [n_latents, latent_dim] (not batch-indexed)
        # Expand to [B, n_latents, latent_dim] for batched attention.
        # Using expand (not repeat) to avoid memory allocation.
        latent: Tensor = self.latent_array.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, n_latents, latent_dim]

        # Make contiguous for attention (expand creates non-contiguous view)
        latent = latent.contiguous()

        # ── Step 5: Cross-attention encode (latent attends to input) ──────
        # Q = latent [B, n_latents, latent_dim]
        # K = k1_proj [B, seq_len, latent_dim]
        # V = v1_proj [B, seq_len, latent_dim]
        # Output: latent_enc [B, n_latents, latent_dim]
        latent_enc: Tensor
        latent_enc, _ = self.cross_attn_encode(
            query=latent,
            key=k1_proj,
            value=v1_proj,
        )  # [B, n_latents, latent_dim]

        # Post-norm residual connection
        latent = self.norm_encode(latent + latent_enc)  # [B, n_latents, latent_dim]

        # ── Step 6: Self-attention on latent ──────────────────────────────
        # Q = K = V = latent [B, n_latents, latent_dim]
        # Output: latent_self [B, n_latents, latent_dim]
        latent_self: Tensor
        latent_self, _ = self.self_attn(
            query=latent,
            key=latent,
            value=latent,
        )  # [B, n_latents, latent_dim]

        # Post-norm residual connection
        latent = self.norm_self(latent + latent_self)  # [B, n_latents, latent_dim]

        # ── Step 7: Feed-forward network on latent ────────────────────────
        # Increases expressivity of the latent representation.
        latent_ffn: Tensor = self.ffn_latent(latent)  # [B, n_latents, latent_dim]

        # Post-norm residual connection
        latent = self.norm_ffn_latent(latent + latent_ffn)  # [B, n_latents, latent_dim]

        # ── Step 8: Project latent to hidden_dim for decoding ─────────────
        # Decode cross-attention uses hidden_dim (matching x_tokens dim).
        latent_k: Tensor = self.proj_latent_k(latent)  # [B, n_latents, hidden_dim]
        latent_v: Tensor = self.proj_latent_v(latent)  # [B, n_latents, hidden_dim]

        # ── Step 9: Cross-attention decode (input attends to latent) ──────
        # Q = x_tokens [B, seq_len, hidden_dim]
        # K = latent_k [B, n_latents, hidden_dim]
        # V = latent_v [B, n_latents, hidden_dim]
        # Output: x_dec [B, seq_len, hidden_dim]
        x_dec: Tensor
        x_dec, _ = self.cross_attn_decode(
            query=x_tokens,
            key=latent_k,
            value=latent_v,
        )  # [B, seq_len, hidden_dim]

        # Post-norm residual connection
        x_out_tokens: Tensor = self.norm_decode(
            x_tokens + x_dec
        )  # [B, seq_len, hidden_dim]

        # ── Step 10: Unflatten back to spatial format ─────────────────────
        # Restore channel-first layout: [B, seq_len, C] -> [B, C, *spatial]
        x_out: Tensor = _unflatten_spatial(
            x_out_tokens, spatial_shape
        )  # [B, hidden_dim, *spatial]

        _logger.debug(
            "PerceiverBlock.forward: input %s -> output %s.",
            tuple(x.shape),
            tuple(x_out.shape),
        )

        return x_out


# ---------------------------------------------------------------------------
# PerceiverNO
# ---------------------------------------------------------------------------


class PerceiverNO(nn.Module):
    """Perceiver IO-based Neural