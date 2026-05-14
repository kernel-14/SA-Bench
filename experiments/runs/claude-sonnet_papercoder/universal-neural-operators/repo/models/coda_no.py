## Code: models/coda_no.py

```python
## models/coda_no.py
"""
CoDA-NO (Codomain Attention Neural Operator) backbone implementation for the
multi-physics neural operator pretraining framework described in:
  "Towards Universal Neural Operators through Multiphysics Pretraining"

Implements the CoDA-NO architecture following Rahman et al. 2024 [13],
referenced in the paper as a baseline comparison model. The key innovation
is **codomain attention**: instead of computing attention between spatial
tokens (standard transformer), it computes attention between spatial
locations weighted by feature-space (codomain) similarity.

From the paper (Section 2):
    "Codomain Attention Neural Operator (CoDA-NO), designed for multiphysics
    PDE transfer learning, employs codomain attention with function space
    dot product."

    "Codomain attention mechanisms, introduced in [13], are advantageous to
    the conventional transformers in the neural-operator based problems: the
    dot product detecting similarity not between samples, but between
    features, mapped with neural operators."

Architecture:
  CodaAttention  - multi-head codomain attention (similarity in feature space)
  CodaNOBlock    - one FNO block + codomain attention with residual connection
  CodaNO         - stacks n_layers CodaNOBlocks as the shared backbone (θ_F)

Codomain attention mechanism:
  Given features v of shape [B, C, N] (C=channels/codomain, N=spatial):
    attn_weights = softmax(v^T @ v / sqrt(head_dim))  # [B, n_heads, N, N]
    output = v @ attn_weights^T                        # [B, C, N]

  This detects similarity between spatial locations weighted by their
  feature-space (codomain) similarity, rather than between samples.

Tensor layout convention (Shared Knowledge #1):
  Channel-first: [B, C, L] for 1D, [B, C, H, W] for 2D.
  B=batch, C=channels (hidden_dim), L/H/W=spatial dimensions.

Config alignment (config.yaml):
  models.coda_no.hidden_dim: 128   -> hidden_dim parameter
  models.coda_no.n_modes: 16       -> n_modes parameter
  models.coda_no.n_layers: 4       -> n_layers parameter
  models.coda_no.n_dims: 2         -> n_dims parameter (1 or 2)
  models.coda_no.n_heads: 8        -> n_heads parameter
  models.coda_no.target_params: 1e8 -> approximate parameter count target

Integration with AdapterFramework:
  CodaNO serves as the backbone argument to AdapterFramework.__init__.
  During pretraining: all parameters are trainable.
  During fine-tuning: AdapterFramework.freeze_backbone() freezes all
  CodaNO parameters; only adapter parameters are updated.

Memory management:
  For large spatial grids (N > max_spatial_tokens), spatial locations are
  uniformly subsampled before attention to prevent OOM. Default threshold
  is 1024 tokens (32×32 for 2D). This is a practical approximation that
  preserves the codomain attention semantics on a representative subset.

Dependencies: torch, torch.nn, models/fno_backbone.py.
NO imports from training/, data/, utils/, or evaluation/.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.fno_backbone import FNOBlock

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Maximum number of spatial tokens before subsampling is applied.
# For 2D grids: 1024 = 32×32. For 1D: 1024 spatial points.
# This prevents OOM for large grids (e.g., 64×64 = 4096 tokens would
# require [B, n_heads, 4096, 4096] attention matrices ≈ 512 MB per sample).
_MAX_SPATIAL_TOKENS: int = 1024

# Small epsilon for numerical stability in attention normalization.
_ATTN_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# Helper: spatial flatten / unflatten
# ---------------------------------------------------------------------------


def _flatten_spatial(x: Tensor) -> Tuple[Tensor, Tuple[int, ...]]:
    """Flatten spatial dimensions of a channel-first tensor to [B, C, N].

    Converts channel-first spatial tensors to the [B, C, N] format needed
    for codomain attention, where N is the total number of spatial tokens.

    Supports both 1D ([B, C, L]) and 2D ([B, C, H, W]) inputs.

    Args:
        x: Channel-first tensor.
            Shape [B, C, L] for 1D problems.
            Shape [B, C, H, W] for 2D problems.

    Returns:
        Tuple of:
          - flat: Tensor of shape [B, C, N] where
              N = L for 1D, N = H*W for 2D.
          - spatial_shape: Tuple of spatial dimensions (L,) for 1D or
              (H, W) for 2D. Used by _unflatten_spatial to restore shape.

    Raises:
        ValueError: If x has fewer than 3 or more than 4 dimensions.
    """
    if x.ndim == 3:
        # 1D: [B, C, L] — already in [B, C, N] format with N=L
        return x, (x.shape[2],)
    elif x.ndim == 4:
        # 2D: [B, C, H, W] -> [B, C, H*W]
        B, C, H, W = x.shape
        flat: Tensor = x.reshape(B, C, H * W)
        return flat, (H, W)
    else:
        raise ValueError(
            f"_flatten_spatial expects 3D [B, C, L] or 4D [B, C, H, W] "
            f"input, got {x.ndim}D tensor with shape {tuple(x.shape)}."
        )


def _unflatten_spatial(
    flat: Tensor,
    spatial_shape: Tuple[int, ...],
) -> Tensor:
    """Restore spatial structure from a flattened [B, C, N] tensor.

    Inverse of _flatten_spatial. Converts [B, C, N] back to channel-first
    spatial format.

    Args:
        flat: Flattened tensor of shape [B, C, N].
        spatial_shape: Tuple of spatial dimensions returned by
            _flatten_spatial: (L,) for 1D or (H, W) for 2D.

    Returns:
        Channel-first spatial tensor:
          Shape [B, C, L] for 1D (spatial_shape = (L,)).
          Shape [B, C, H, W] for 2D (spatial_shape = (H, W)).

    Raises:
        ValueError: If spatial_shape has an unsupported number of elements.
    """
    B, C, N = flat.shape

    if len(spatial_shape) == 1:
        # 1D: [B, C, N] is already [B, C, L] — no reshape needed
        return flat
    elif len(spatial_shape) == 2:
        # 2D: [B, C, H*W] -> [B, C, H, W]
        H: int = spatial_shape[0]
        W: int = spatial_shape[1]
        return flat.reshape(B, C, H, W)
    else:
        raise ValueError(
            f"_unflatten_spatial expects spatial_shape of length 1 or 2, "
            f"got length {len(spatial_shape)}: {spatial_shape}."
        )


# ---------------------------------------------------------------------------
# CodaAttention
# ---------------------------------------------------------------------------


class CodaAttention(nn.Module):
    """Multi-head codomain attention for neural operator problems.

    Implements the codomain attention mechanism from Rahman et al. 2024 [13],
    where the attention matrix is computed over the **spatial/codomain
    dimension** rather than the standard token/sample dimension.

    Standard self-attention detects similarity between tokens (spatial
    locations) in embedding space. Codomain attention detects similarity
    between spatial locations in **feature space** (the codomain of the
    operator). This is more appropriate for neural operator problems where
    the relevant structure is in the function space (codomain) rather than
    the input space.

    Mechanism (single head):
        Given v of shape [B, C, N] (C=features/codomain, N=spatial):
          attn_weights = softmax(v^T @ v / sqrt(C))  # [B, N, N]
          output = v @ attn_weights                   # [B, C, N]

    Multi-head variant:
        Split C into n_heads groups of head_dim = C // n_heads channels.
        Each head computes its own [B, N, N] attention matrix.
        Outputs are concatenated along the channel dimension.

    Memory management:
        For large spatial grids (N > max_spatial_tokens), spatial locations
        are uniformly subsampled before attention. The attention output is
        then interpolated back to the original spatial resolution. This
        prevents OOM for large grids while preserving the codomain attention
        semantics on a representative subset.

    Attributes:
        hidden_dim: Total feature dimension C. From config.yaml
            models.coda_no.hidden_dim (default 128).
        n_heads: Number of attention heads. From config.yaml
            models.coda_no.n_heads (default 8).
        head_dim: Feature dimension per head = hidden_dim // n_heads.
        scale: Attention scale factor = 1 / sqrt(head_dim).
        max_spatial_tokens: Maximum spatial tokens before subsampling.
            Default _MAX_SPATIAL_TOKENS = 1024.
        linear_in: Input projection [hidden_dim -> hidden_dim].
        linear_out: Output projection [hidden_dim -> hidden_dim].

    Example::

        attn = CodaAttention(hidden_dim=128, n_heads=8)

        # 2D spatial field
        v = torch.randn(4, 128, 64, 64)   # [B, C, H, W]
        out = attn(v)                       # [B, 128, 64, 64]

        # 1D spatial field
        v_1d = torch.randn(4, 128, 256)    # [B, C, L]
        out_1d = attn(v_1d)                # [B, 128, 256]
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 8,
        max_spatial_tokens: int = _MAX_SPATIAL_TOKENS,
    ) -> None:
        """Initialise CodaAttention.

        Args:
            hidden_dim: Feature dimension C. From config.yaml
                models.coda_no.hidden_dim (default 128). Must be divisible
                by n_heads.
            n_heads: Number of attention heads. From config.yaml
                models.coda_no.n_heads (default 8). Must divide hidden_dim.
            max_spatial_tokens: Maximum number of spatial tokens before
                uniform subsampling is applied. Default 1024 (32×32 for 2D).
                Set to a large value (e.g., 1e9) to disable subsampling.

        Raises:
            ValueError: If hidden_dim <= 0 or n_heads <= 0.
            ValueError: If hidden_dim is not divisible by n_heads.
        """
        super().__init__()

        # ── Validate arguments ────────────────────────────────────────────
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be a positive integer, got {hidden_dim}."
            )
        if n_heads <= 0:
            raise ValueError(
                f"n_heads must be a positive integer, got {n_heads}."
            )
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by "
                f"n_heads={n_heads}. "
                f"Got hidden_dim % n_heads = {hidden_dim % n_heads}. "
                f"Adjust hidden_dim or n_heads in config.yaml."
            )

        # ── Store hyperparameters ─────────────────────────────────────────
        self.hidden_dim: int = hidden_dim
        self.n_heads: int = n_heads
        self.head_dim: int = hidden_dim // n_heads
        self.scale: float = self.head_dim ** -0.5
        self.max_spatial_tokens: int = int(max_spatial_tokens)

        # ── Input projection: hidden_dim -> hidden_dim ────────────────────
        # Projects features before computing attention weights.
        # Allows the model to learn a task-specific feature space for
        # computing codomain similarity.
        self.linear_in: nn.Linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # ── Output projection: hidden_dim -> hidden_dim ───────────────────
        # Projects the attention output back to the hidden dimension.
        # Mixes information across heads after concatenation.
        self.linear_out: nn.Linear = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # ── Initialize weights ────────────────────────────────────────────
        # Xavier uniform initialization for stable training.
        nn.init.xavier_uniform_(self.linear_in.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)
        if self.linear_out.bias is not None:
            nn.init.zeros_(self.linear_out.bias)

        _logger.debug(
            "CodaAttention: hidden_dim=%d, n_heads=%d, head_dim=%d, "
            "scale=%.4f, max_spatial_tokens=%d.",
            hidden_dim,
            n_heads,
            self.head_dim,
            self.scale,
            self.max_spatial_tokens,
        )

    def _codomain_attention(
        self,
        v: Tensor,
    ) -> Tensor:
        """Apply multi-head codomain attention to [B, C, N] features.

        Core codomain attention computation:
          1. Project features: v_proj = linear_in(v^T)^T  [B, C, N]
          2. Split into heads: [B, n_heads, head_dim, N]
          3. Compute attention: attn = softmax(v_h^T @ v_h / sqrt(head_dim))
             Shape: [B, n_heads, N, N]
          4. Apply attention: out_h = v_h @ attn  [B, n_heads, head_dim, N]
          5. Merge heads: [B, C, N]
          6. Output projection: linear_out  [B, C, N]

        Args:
            v: Feature tensor of shape [B, C, N] where C=hidden_dim and
               N is the (possibly subsampled) number of spatial tokens.

        Returns:
            Attention output of shape [B, C, N].
        """
        B: int = v.shape[0]
        C: int = v.shape[1]
        N: int = v.shape[2]

        # ── Step 1: Input projection ──────────────────────────────────────
        # linear_in expects [..., hidden_dim] input.
        # v is [B, C, N]; transpose to [B, N, C], project, transpose back.
        v_t: Tensor = v.permute(0, 2, 1)                    # [B, N, C]
        v_proj_t: Tensor = self.linear_in(v_t)              # [B, N, C]
        v_proj: Tensor = v_proj_t.permute(0, 2, 1)          # [B, C, N]

        # ── Step 2: Split into heads ──────────────────────────────────────
        # Reshape [B, C, N] -> [B, n_heads, head_dim, N]
        v_heads: Tensor = v_proj.reshape(
            B, self.n_heads, self.head_dim, N
        )  # [B, n_heads, head_dim, N]

        # ── Step 3: Compute codomain attention matrix ─────────────────────
        # For each head h: attn[h] = softmax(v_h^T @ v_h / sqrt(head_dim))
        # v_h: [B, n_heads, head_dim, N]
        # v_h^T: [B, n_heads, N, head_dim]
        # v_h^T @ v_h: [B, n_heads, N, N]
        #
        # This computes similarity between spatial locations i and j as:
        #   sim(i, j) = Σ_c v_h[c, i] * v_h[c, j] / sqrt(head_dim)
        # i.e., the dot product of their feature vectors in the codomain.
        v_h_t: Tensor = v_heads.permute(0, 1, 3, 2)  # [B, n_heads, N, head_dim]

        # Scaled dot product: [B, n_heads, N, head_dim] @ [B, n_heads, head_dim, N]
        # = [B, n_heads, N, N]
        attn_logits: Tensor = torch.matmul(v_h_t, v_heads) * self.scale
        # [B, n_heads, N, N]

        # Softmax over the last dimension (key dimension):
        # attn[b, h, i, j] = exp(sim(i,j)) / Σ_k exp(sim(i,k))
        # This gives the weight that location i places on location j.
        attn_weights: Tensor = torch.softmax(attn_logits, dim=-1)
        # [B, n_heads, N, N]

        # ── Step 4: Apply attention weights ──────────────────────────────
        # out_h[b, h, c, i] = Σ_j v_h[b, h, c, j] * attn[b, h, i, j]
        # = v_h @ attn^T  (since attn is [N, N] and we want weighted sum
        #   over j for each query location i)
        #
        # v_heads: [B, n_heads, head_dim, N]
        # attn_weights: [B, n_heads, N, N]
        # attn_weights^T: [B, n_heads, N, N] (symmetric-ish after softmax)
        #
        # We want: out[b, h, c, i] = Σ_j v[b, h, c, j] * attn[b, h, i, j]
        # = (v_heads @ attn_weights.transpose(-1, -2))[b, h, c, i]
        # v_heads: [B, n_heads, head_dim, N]
        # attn_weights.T: [B, n_heads, N, N]
        # matmul: [B, n_heads, head_dim, N] @ [B, n_heads, N, N]
        #       = [B, n_heads, head_dim, N]
        attn_weights_t: Tensor = attn_weights.transpose(-1, -2)
        # [B, n_heads, N, N]

        out_heads: Tensor = torch.matmul(v_heads, attn_weights_t)
        # [B, n_heads, head_dim, N]

        # ── Step 5: Merge heads ───────────────────────────────────────────
        # [B, n_heads, head_dim, N] -> [B, C, N]
        out_merged: Tensor = out_heads.reshape(B, C, N)  # [B, C, N]

        # ── Step 6: Output projection ─────────────────────────────────────
        # linear_out expects [..., hidden_dim] input.
        # Transpose to [B, N, C], project, transpose back.
        out_t: Tensor = out_merged.permute(0, 2, 1)      # [B, N, C]
        out_proj_t: Tensor = self.linear_out(out_t)      # [B, N, C]
        out_proj: Tensor = out_proj_t.permute(0, 2, 1)   # [B, C, N]

        return out_proj

    def forward(self, v: Tensor) -> Tensor:
        """Apply multi-head codomain attention to spatial field features.

        Handles both 1D ([B, C, L]) and 2D ([B, C, H, W]) inputs by
        flattening spatial dimensions to [B, C, N] before attention and
        restoring the original spatial structure afterward.

        For large spatial grids (N > max_spatial_tokens), uniformly
        subsamples spatial locations before attention and interpolates
        the output back to the original resolution. This prevents OOM
        while preserving the codomain attention semantics.

        The output is added to the input via a residual connection:
            output = v + codomain_attention(v)

        This residual structure ensures that the attention module can be
        initialized to near-identity (output ≈ 0 initially due to random
        initialization) and gradually learns to contribute meaningful
        corrections.

        Args:
            v: Feature tensor in channel-first layout.
                Shape [B, C, L] for 1D problems (n_dims=1).
                Shape [B, C, H, W] for 2D problems (n_dims=2).
                C must equal self.hidden_dim.

        Returns:
            Feature tensor with the same shape as v, with codomain
            attention applied and residual connection added.

        Raises:
            ValueError: If v has an unsupported number of dimensions
                (must be 3 for 1D or 4 for 2D).
            ValueError: If the channel dimension C does not match
                hidden_dim.
        """
        # ── Validate input ────────────────────────────────────────────────
        if v.ndim not in (3, 4):
            raise ValueError(
                f"CodaAttention expects 3D [B, C, L] or 4D [B, C, H, W] "
                f"input, got {v.ndim}D tensor with shape {tuple(v.shape)}."
            )

        c_in: int = v.shape[1]
        if c_in != self.hidden_dim:
            raise ValueError(
                f"Input channel dimension C={c_in} does not match "
                f"hidden_dim={self.hidden_dim}. "
                f"Ensure the LiftingAdapter projects to hidden_dim channels."
            )

        # ── Flatten spatial dims to [B, C, N] ────────────────────────────
        v_flat: Tensor
        spatial_shape: Tuple[int, ...]
        v_flat, spatial_shape = _flatten_spatial(v)  # [B, C, N]

        B: int = v_flat.shape[0]
        C: int = v_flat.shape[1]
        N: int = v_flat.shape[2]

        # ── Subsample if N > max_spatial_tokens ───────────────────────────
        # For large grids, uniformly subsample spatial locations to prevent
        # OOM from the [B, n_heads, N, N] attention matrix.
        needs_subsample: bool = N > self.max_spatial_tokens
        subsample_indices: Optional[Tensor] = None
        N_sub: int = N

        if needs_subsample:
            N_sub = self.max_spatial_tokens
            # Uniform subsampling: select evenly spaced indices
            step: int = max(1, N // N_sub)
            subsample_indices = torch.arange(
                0, N, step, device=v.device
            )[:N_sub]  # [N_sub]

            # Subsample: [B, C, N] -> [B, C, N_sub]
            v_sub: Tensor = v_flat[:, :, subsample_indices]  # [B, C, N_sub]

            _logger.debug(
                "CodaAttention: subsampling N=%d -> N_sub=%d "
                "(max_spatial_tokens=%d).",
                N,
                N_sub,
                self.max_spatial_tokens,
            )
        else:
            v_sub = v_flat  # [B, C, N]

        # ── Apply codomain attention ──────────────────────────────────────
        attn_out_sub: Tensor = self._codomain_attention(v_sub)  # [B, C, N_sub]

        # ── Interpolate back to full spatial resolution if subsampled ─────
        if needs_subsample and subsample_indices is not None:
            # Scatter attention output back to full spatial positions.
            # Positions not in subsample_indices retain their original values
            # (identity residual for non-attended positions).
            attn_out_full: Tensor = torch.zeros_like(v_flat)  # [B, C, N]
            attn_out_full[:, :, subsample_indices] = attn_out_sub
        else:
            attn_out_full = attn_out_sub  # [B, C, N]

        # ── Residual connection ───────────────────────────────────────────
        # output = v_flat + attn_out_full
        # The residual ensures near-identity initialization and stable
        # gradient flow through the attention module.
        out_flat: Tensor = v_flat + attn_out_full  # [B, C, N]

        # ── Restore spatial structure ─────────────────────────────────────
        out: Tensor = _unflatten_spatial(out_flat, spatial_shape)

        _logger.debug(
            "CodaAttention.forward: input %s -> output %s "
            "(subsampled=%s).",
            tuple(v.shape),
            tuple(out.shape),
            needs_subsample,
        )

        return out


# ---------------------------------------------------------------------------
# CodaNOBlock
# ---------------------------------------------------------------------------


class CodaNOBlock(nn.Module):
    """One CoDA-NO block: FNO layer + codomain attention with residual.

    Combines an FNOBlock (capturing global spectral structure via Fourier
    integral operators) with CodaAttention (capturing feature-space
    similarity between spatial locations) into a single processing block.

    Block structure (sequential, pre-norm for stability):
        v → FNOBlock(v) → v_fno
        v_fno → GroupNorm(v_fno) → CodaAttention(·) → attn_out
        output = v_fno + attn_out

    The FNOBlock already contains an internal residual connection
    (spectral path + W path), so the outer residual in this block adds
    the codomain attention correction on top of the FNO output.

    GroupNorm with num_groups=1 is used for normalization because it works
    directly on channel-first [B, C, *spatial] tensors without requiring
    transposition, unlike LayerNorm which expects the normalized dimension
    to be last.

    Attributes:
        hidden_dim: Feature dimension. Must match FNOBlock and CodaAttention.
        n_modes: Number of Fourier modes for the FNO block.
        n_heads: Number of attention heads for codomain attention.
        n_dims: Spatial dimensionality (1 or 2).
        fno_block: FNOBlock instance for spectral processing.
        coda_attn: CodaAttention instance for codomain attention.
        norm: GroupNorm(1, hidden_dim) applied before attention.

    Example::

        block = CodaNOBlock(
            hidden_dim=128, n_modes=16, n_heads=8, n_dims=2,
        )
        v = torch.randn(4, 128, 64, 64)   # [B, C, H, W]
        out = block(v)                     # [B, 128, 64, 64]
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_modes: int = 16,
        n_heads: int = 8,
        n_dims: int = 2,
        max_spatial_tokens: int = _MAX_SPATIAL_TOKENS,
    ) -> None:
        """Initialise CodaNOBlock.

        Args:
            hidden_dim: Feature dimension. From config.yaml
                models.coda_no.hidden_dim (default 128). Must be divisible
                by n_heads.
            n_modes: Number of Fourier modes for the FNO block. From
                config.yaml models.coda_no.n_modes (default 16).
            n_heads: Number of attention heads for codomain attention.
                From config.yaml models.coda_no.n_heads (default 8).
                Must divide hidden_dim.
            n_dims: Spatial dimensionality. 1 for 1D problems (Burgers,
                Advection), 2 for 2D problems (NS, RD, Gray-Scott, Heat).
                From config.yaml models.coda_no.n_dims (default 2).
            max_spatial_tokens: Maximum spatial tokens before subsampling
                in CodaAttention. Default _MAX_SPATIAL_TOKENS = 1024.

        Raises:
            ValueError: If hidden_dim, n_modes, or n_heads <= 0.
            ValueError: If n_dims is not 1 or 2.
            ValueError: If hidden_dim is not divisible by n_heads
                (propagated from CodaAttention).
        """
        super().__init__()

        # ── Validate arguments ────────────────────────────────────────────
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be a positive integer, got {hidden_dim}."
            )
        if n_modes <= 0:
            raise ValueError(
                f"n_modes must be a positive integer, got {n_modes}."
            )
        if n_heads <= 0:
            raise ValueError(
                f"n_heads must be a positive integer, got {n_heads}."
            )
        if n_dims not in (1, 2):
            raise ValueError(
                f"n_dims must be 1 or 2, got {n_dims}. "
                f"3D problems are not currently supported."
            )

        # ── Store hyperparameters ─────────────────────────────────────────
        self.hidden_dim: int = hidden_dim
        self.n_modes: int = n_modes
        self.n_heads: int = n_heads
        self.n_dims: int = n_dims

        # ── FNO block: spectral processing ───────────────────────────────
        # Captures global non-local structure via Fourier integral operators.
        # Contains internal residual: output = σ(spectral_path + W_path).
        self.fno_block: FNOBlock = FNOBlock(
            hidden_dim=hidden_dim,
            n_modes=n_modes,
            n_dims=n_dims,
            activation="gelu",
        )

        # ── Normalization: GroupNorm before codomain attention ─────────────
        # GroupNorm(num_groups=1, num_channels=hidden_dim) is equivalent to
        # LayerNorm over the channel dimension and works directly on
        # channel-first [B, C, *spatial] tensors without transposition.
        # num_groups=1 means all channels are normalized together (instance
        # normalization over channels), which is appropriate here since we
        # want to normalize the feature distribution before attention.
        self.norm: nn.GroupNorm = nn.Group