## Code: models/mamba_fno.py

```python
## models/mamba_fno.py
"""
MambaFNO backbone implementation for the multi-physics neural operator
pretraining framework described in:
  "Towards Universal Neural Operators through Multiphysics Pretraining"

Implements the post-lifting Mamba preconditioner combined with the FNO
backbone, forming the MambaFNO architecture described in Section 3 of the
paper:

    "inserting a Mamba-SSM module M_φ after the lifting map L allows the
    model to encode long-range temporal and spatial dependencies directly
    in the hidden representation."

The composition is:
    v (lifted features) → MambaModule → ṽ (preconditioned) → FNOBackbone → v_out

The Mamba module acts as a "latent preconditioner" that aligns embeddings
with dominant dynamical motifs (transport, diffusion, oscillation) before
the Fourier integral layers process them. This improves transfer learning
efficiency by reducing the spectral rank of the input to the FNO blocks.

Classes:
  MambaModule  - wraps mamba_ssm.Mamba with spatial reshape logic
  MambaFNO     - composes MambaModule + FNOBackbone as a unified backbone

Tensor layout convention (Shared Knowledge #1):
  Channel-first: [B, C, L] for 1D, [B, C, H, W] for 2D.
  B=batch, C=channels (hidden_dim), L/H/W=spatial dimensions.

Config alignment (config.yaml):
  models.mamba_fno.hidden_dim: 64       -> hidden_dim parameter
  models.mamba_fno.n_modes: 16          -> n_modes parameter
  models.mamba_fno.n_layers: 4          -> n_layers parameter
  models.mamba_fno.n_dims: 2            -> n_dims parameter (1 or 2)
  models.mamba_fno.target_params: 1e7   -> approximate parameter count target
  models.mamba_fno.mamba.d_state: 16    -> d_state parameter (Mamba default)
  models.mamba_fno.mamba.d_conv: 4      -> d_conv parameter (Mamba default)
  models.mamba_fno.mamba.expand: 2      -> expand parameter (Mamba default)

Integration with AdapterFramework:
  MambaFNO serves as the backbone argument to AdapterFramework.__init__.
  During pretraining: all parameters (Mamba + FNO) are trainable.
  During fine-tuning: AdapterFramework.freeze_backbone() freezes all
  MambaFNO parameters; only adapter parameters are updated.

Dependencies: torch, torch.nn, mamba_ssm (optional, CUDA required),
              models/fno_backbone.py.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from models.fno_backbone import FNOBackbone

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional mamba_ssm import
# ---------------------------------------------------------------------------

# mamba_ssm requires CUDA and a specific installation procedure.
# We guard the import and provide a clear error message if unavailable.
try:
    from mamba_ssm import Mamba as _MambaSSM  # type: ignore[import]

    _MAMBA_AVAILABLE: bool = True
    _logger.debug("mamba_ssm imported successfully.")
except ImportError:
    _MAMBA_AVAILABLE = False
    _MambaSSM = None  # type: ignore[assignment, misc]
    _logger.warning(
        "mamba_ssm is not installed. MambaModule and MambaFNO will raise "
        "ImportError when instantiated. Install with: "
        "pip install mamba-ssm causal-conv1d"
    )


# ---------------------------------------------------------------------------
# MambaModule
# ---------------------------------------------------------------------------


class MambaModule(nn.Module):
    """Post-lifting Mamba SSM preconditioner for spatial PDE fields.

    Wraps ``mamba_ssm.Mamba`` to handle the tensor layout mismatch between
    spatial field tensors (channel-first: ``[B, C, H, W]`` or ``[B, C, L]``)
    and the sequence format Mamba expects (``[B, seq_len, d_model]``).

    The module implements the causal convolution described in the paper:

        ṽ_0(x,t) = (M_φ v_0)(x,t) = Σ_{τ≤t} K_τ v_0(x, t-τ)

    where K_τ are learnable convolution kernels defining the causal
    recurrence. By flattening spatial dimensions into a sequence, Mamba
    captures long-range spatial correlations across the entire grid.

    This acts as a "latent preconditioner": embeddings are aligned with
    dominant dynamical motifs (transport, diffusion, oscillation) common
    across PDEs, so that when passed into the Fourier integral layers, the
    effective operator acts on inputs of reduced variability and lower
    spectral rank.

    Reshape strategy:
      2D input [B, C, H, W]:
        permute(0,2,3,1) → [B, H, W, C] → reshape → [B, H*W, C]
        apply Mamba → [B, H*W, C]
        reshape → [B, H, W, C] → permute(0,3,1,2) → [B, C, H, W]

      1D input [B, C, L]:
        permute(0,2,1) → [B, L, C]
        apply Mamba → [B, L, C]
        permute(0,2,1) → [B, C, L]

    The channel dimension C maps to Mamba's d_model, so each spatial
    location is treated as a "token" with C-dimensional features.

    Attributes:
        hidden_dim: Feature dimension (d_model for Mamba). Equals the
            hidden_dim of the FNO backbone.
        d_state: SSM state dimension. Default 16 (Mamba paper default,
            config.yaml models.mamba_fno.mamba.d_state).
        d_conv: Local convolution width in Mamba. Default 4 (config.yaml
            models.mamba_fno.mamba.d_conv).
        expand: Expansion factor for inner dimension. Default 2 (config.yaml
            models.mamba_fno.mamba.expand). Inner dim = hidden_dim * expand.
        _mamba: The underlying mamba_ssm.Mamba instance.

    Example::

        module = MambaModule(hidden_dim=64, d_state=16, d_conv=4, expand=2)

        # 2D spatial field
        v_2d = torch.randn(8, 64, 32, 32).cuda()   # [B, C, H, W]
        out_2d = module(v_2d)                        # [B, 64, 32, 32]

        # 1D spatial field
        v_1d = torch.randn(8, 64, 256).cuda()        # [B, C, L]
        out_1d = module(v_1d)                         # [B, 64, 256]
    """

    def __init__(
        self,
        hidden_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        """Initialise MambaModule.

        Args:
            hidden_dim: Feature dimension. This is both the channel count C
                of the spatial field tensors and the d_model argument to
                mamba_ssm.Mamba. From config.yaml models.mamba_fno.hidden_dim
                (default 64).
            d_state: SSM state dimension. Controls the expressiveness of the
                state space model. Default 16 (Mamba paper default, config.yaml
                models.mamba_fno.mamba.d_state).
            d_conv: Local convolution width in the Mamba block. Default 4
                (Mamba paper default, config.yaml
                models.mamba_fno.mamba.d_conv).
            expand: Expansion factor for the inner dimension of the Mamba
                block. Inner dimension = hidden_dim * expand. Default 2
                (Mamba paper default, config.yaml
                models.mamba_fno.mamba.expand).

        Raises:
            ImportError: If mamba_ssm is not installed. Install with:
                pip install mamba-ssm causal-conv1d
            ValueError: If hidden_dim, d_state, d_conv, or expand <= 0.
        """
        super().__init__()

        # ── Check mamba_ssm availability ──────────────────────────────────
        if not _MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm is required for MambaModule but is not installed. "
                "Install it with: pip install mamba-ssm causal-conv1d\n"
                "Note: mamba_ssm requires CUDA and a compatible GPU. "
                "CPU-only environments are not supported."
            )

        # ── Validate arguments ────────────────────────────────────────────
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be a positive integer, got {hidden_dim}."
            )
        if d_state <= 0:
            raise ValueError(
                f"d_state must be a positive integer, got {d_state}."
            )
        if d_conv <= 0:
            raise ValueError(
                f"d_conv must be a positive integer, got {d_conv}."
            )
        if expand <= 0:
            raise ValueError(
                f"expand must be a positive integer, got {expand}."
            )

        # ── Store hyperparameters ─────────────────────────────────────────
        self.hidden_dim: int = hidden_dim
        self.d_state: int = d_state
        self.d_conv: int = d_conv
        self.expand: int = expand

        # ── Instantiate Mamba SSM ─────────────────────────────────────────
        # d_model = hidden_dim: each spatial token has hidden_dim features.
        # The Mamba block internally uses d_model * expand as the inner
        # dimension for its selective state space computation.
        self._mamba: nn.Module = _MambaSSM(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        _logger.info(
            "MambaModule: hidden_dim=%d (d_model), d_state=%d, "
            "d_conv=%d, expand=%d. Inner dim=%d.",
            hidden_dim,
            d_state,
            d_conv,
            expand,
            hidden_dim * expand,
        )

    def forward(self, v: Tensor) -> Tensor:
        """Apply the Mamba SSM preconditioner to spatial field features.

        Handles both 1D (``[B, C, L]``) and 2D (``[B, C, H, W]``) inputs
        by dynamically detecting the number of spatial dimensions from the
        input tensor shape. The spatial dimensions are flattened into a
        single sequence dimension for Mamba processing, then reshaped back.

        The output has the same shape as the input — this is a
        shape-preserving transformation. No channel count changes occur.

        Args:
            v: Lifted feature tensor from LiftingAdapter.
                Shape ``[B, C, L]`` for 1D problems (Burgers, Advection).
                Shape ``[B, C, H, W]`` for 2D problems (NS, RD, Heat).
                C must equal ``self.hidden_dim``.
                Must be on CUDA (mamba_ssm requires GPU).

        Returns:
            Preconditioned feature tensor with the same shape as ``v``.
            The Mamba module aligns features with dominant dynamical motifs
            before they are processed by the Fourier integral layers.

        Raises:
            RuntimeError: If the input tensor is on CPU (mamba_ssm requires
                CUDA). Move the model and data to GPU before calling forward.
            ValueError: If ``v`` has an unsupported number of dimensions
                (must be 3 for 1D or 4 for 2D).
            ValueError: If the channel dimension C does not match
                ``self.hidden_dim``.
        """
        # ── Validate device ───────────────────────────────────────────────
        # mamba_ssm.Mamba requires CUDA. Provide a clear error message
        # rather than letting the CUDA extension raise a cryptic error.
        if not v.is_cuda:
            raise RuntimeError(
                "MambaModule requires CUDA tensors. "
                "The input tensor is on CPU. "
                "Move the model and data to GPU: "
                "model.cuda() and tensor.cuda() before calling forward. "
                "If you need CPU support, consider using FNOBackbone directly "
                "without the Mamba preconditioner."
            )

        # ── Validate input dimensions ─────────────────────────────────────
        if v.ndim not in (3, 4):
            raise ValueError(
                f"MambaModule expects 3D [B, C, L] or 4D [B, C, H, W] input, "
                f"got {v.ndim}D tensor with shape {tuple(v.shape)}."
            )

        # ── Validate channel dimension ────────────────────────────────────
        c_in: int = v.shape[1]
        if c_in != self.hidden_dim:
            raise ValueError(
                f"Input channel dimension C={c_in} does not match "
                f"hidden_dim={self.hidden_dim}. "
                f"The LiftingAdapter must project to hidden_dim channels "
                f"before passing features to MambaModule."
            )

        # ── Store original dtype for output consistency ───────────────────
        # mamba_ssm may require float32; cast if needed and restore after.
        original_dtype: torch.dtype = v.dtype
        if v.dtype != torch.float32:
            v = v.to(torch.float32)

        # ── Dispatch to 1D or 2D reshape logic ───────────────────────────
        # Dimensionality is detected dynamically from the input tensor shape,
        # not from a stored n_dims attribute. This makes the module robust
        # to mixed-dimensionality usage.
        if v.ndim == 3:
            out: Tensor = self._forward_1d(v)
        else:  # v.ndim == 4
            out = self._forward_2d(v)

        # ── Restore original dtype ────────────────────────────────────────
        if out.dtype != original_dtype:
            out = out.to(original_dtype)

        return out

    def _forward_1d(self, v: Tensor) -> Tensor:
        """Apply Mamba to a 1D spatial field.

        Reshape strategy:
          [B, C, L] → permute(0,2,1) → [B, L, C] → Mamba → [B, L, C]
          → permute(0,2,1) → [B, C, L]

        The spatial length L becomes the sequence dimension for Mamba.
        Each spatial location is a token with C=hidden_dim features.

        Args:
            v: Input tensor of shape [B, C, L], dtype float32, on CUDA.

        Returns:
            Output tensor of shape [B, C, L], same dtype and device as input.
        """
        batch_size: int = v.shape[0]
        channels: int = v.shape[1]
        seq_len: int = v.shape[2]  # L

        # [B, C, L] → [B, L, C]: spatial dim becomes sequence, channel becomes d_model
        v_seq: Tensor = v.permute(0, 2, 1).contiguous()  # [B, L, C]

        # Apply Mamba SSM: [B, L, C] → [B, L, C]
        v_out_seq: Tensor = self._mamba(v_seq)  # [B, L, C]

        # [B, L, C] → [B, C, L]: restore channel-first layout
        v_out: Tensor = v_out_seq.permute(0, 2, 1).contiguous()  # [B, C, L]

        _logger.debug(
            "_forward_1d: input [%d, %d, %d] → output [%d, %d, %d].",
            batch_size, channels, seq_len,
            v_out.shape[0], v_out.shape[1], v_out.shape[2],
        )

        return v_out

    def _forward_2d(self, v: Tensor) -> Tensor:
        """Apply Mamba to a 2D spatial field.

        Reshape strategy:
          [B, C, H, W] → permute(0,2,3,1) → [B, H, W, C]
          → reshape → [B, H*W, C] → Mamba → [B, H*W, C]
          → reshape → [B, H, W, C] → permute(0,3,1,2) → [B, C, H, W]

        The flattened spatial grid H*W becomes the sequence dimension for
        Mamba. Each grid point is a token with C=hidden_dim features.

        Args:
            v: Input tensor of shape [B, C, H, W], dtype float32, on CUDA.

        Returns:
            Output tensor of shape [B, C, H, W], same dtype and device as
            input.
        """
        batch_size: int = v.shape[0]
        channels: int = v.shape[1]
        height: int = v.shape[2]   # H
        width: int = v.shape[3]    # W
        seq_len: int = height * width  # H*W

        # [B, C, H, W] → [B, H, W, C]: move channel to last dim
        v_hwc: Tensor = v.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]

        # [B, H, W, C] → [B, H*W, C]: flatten spatial dims into sequence
        v_seq: Tensor = v_hwc.reshape(batch_size, seq_len, channels)  # [B, H*W, C]

        # Apply Mamba SSM: [B, H*W, C] → [B, H*W, C]
        v_out_seq: Tensor = self._mamba(v_seq)  # [B, H*W, C]

        # [B, H*W, C] → [B, H, W, C]: restore spatial structure
        v_out_hwc: Tensor = v_out_seq.reshape(batch_size, height, width, channels)

        # [B, H, W, C] → [B, C, H, W]: restore channel-first layout
        v_out: Tensor = v_out_hwc.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        _logger.debug(
            "_forward_2d: input [%d, %d, %d, %d] → output [%d, %d, %d, %d].",
            batch_size, channels, height, width,
            v_out.shape[0], v_out.shape[1], v_out.shape[2], v_out.shape[3],
        )

        return v_out


# ---------------------------------------------------------------------------
# MambaFNO
# ---------------------------------------------------------------------------


class MambaFNO(nn.Module):
    """MambaFNO backbone: post-lifting Mamba preconditioner + FNO layers.

    Composes ``MambaModule`` (post-lifting SSM preconditioner) with
    ``FNOBackbone`` (Fourier integral layers) into a single backbone module
    for use with ``AdapterFramework``.

    The architecture implements the paper's description (Section 3):

        "inserting a Mamba-SSM module M_φ after the lifting map L allows
        the model to encode long-range temporal and spatial dependencies
        directly in the hidden representation. [...] This step acts as a
        latent preconditioner: embeddings are aligned with dominant
        dynamical motifs (transport, diffusion, oscillation) common across
        PDEs, so that when passed into the Fourier integral layers, the
        effective operator acts on inputs of reduced variability and lower
        spectral rank. Consequently, the composition F_t ∘ M_φ yields more
        stable training and improves efficiency in transferring pre-trained
        representations to new PDEs during fine-tuning."

    Data flow:
        v (lifted features from LiftingAdapter)
          → MambaModule._mamba(v)     # SSM preconditioner
          → FNOBackbone._fno_blocks   # Fourier integral layers
          → v_out (to ProjectionAdapter)

    Both ``_mamba`` and ``_backbone`` operate on the same ``hidden_dim``
    channel count. The Mamba module is shape-preserving (no channel count
    changes), so no intermediate projection is needed.

    During pretraining: all parameters (Mamba + FNO blocks) are trainable.
    During fine-tuning: ``AdapterFramework.freeze_backbone()`` sets
    ``requires_grad=False`` on all MambaFNO parameters; only the new
    adapter parameters are updated.

    Attributes:
        hidden_dim: Shared channel dimension throughout the backbone.
            From config.yaml models.mamba_fno.hidden_dim (default 64).
        n_modes: Number of Fourier modes per spatial dimension.
            From config.yaml models.mamba_fno.n_modes (default 16).
        n_layers: Number of FNO blocks in the backbone.
            From config.yaml models.mamba_fno.n_layers (default 4).
        d_state: Mamba SSM state dimension.
            From config.yaml models.mamba_fno.mamba.d_state (default 16).
        d_conv: Mamba local convolution width.
            From config.yaml models.mamba_fno.mamba.d_conv (default 4).
        expand: Mamba expansion factor.
            From config.yaml models.mamba_fno.mamba.expand (default 2).
        n_dims: Spatial dimensionality (1 or 2).
            From config.yaml models.mamba_fno.n_dims (default 2).
        _mamba: MambaModule instance (post-lifting preconditioner).
        _backbone: FNOBackbone instance (Fourier integral layers).

    Example::

        # 2D MambaFNO (default, for NS/RD/Heat problems)
        model = MambaFNO(
            hidden_dim=64, n_modes=16, n_layers=4,
            d_state=16, d_conv=4, expand=2, n_dims=2,
        )
        v = torch.randn(8, 64, 64, 64).cuda()   # [B, hidden_dim, H, W]
        out = model(v)                            # [B, 64, 64, 64]

        # 1D MambaFNO (for Burgers/Advection problems)
        model_1d = MambaFNO(
            hidden_dim=64, n_modes=16, n_layers=4,
            d_state=16, d_conv=4, expand=2, n_dims=1,
        )
        v_1d = torch.randn(8, 64, 256).cuda()    # [B, hidden_dim, L]
        out_1d = model_1d(v_1d)                   # [B, 64, 256]
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_modes: int = 16,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_dims: int = 2,
        activation: str = "gelu",
    ) -> None:
        """Initialise MambaFNO.

        Args:
            hidden_dim: Shared channel dimension for both the Mamba module
                and the FNO backbone. From config.yaml
                models.mamba_fno.hidden_dim (default 64). This is d_model
                for Mamba and the hidden dimension for all FNO blocks.
            n_modes: Number of Fourier modes to retain per spatial dimension
                in the FNO blocks. From config.yaml models.mamba_fno.n_modes
                (default 16). For 2D, the same n_modes is used for both H
                and W dimensions.
            n_layers: Number of FNO blocks in the backbone. From config.yaml
                models.mamba_fno.n_layers (default 4).
            d_state: Mamba SSM state dimension. Controls the expressiveness
                of the state space model. From config.yaml
                models.mamba_fno.mamba.d_state (default 16).
            d_conv: Local convolution width in the Mamba block. From
                config.yaml models.mamba_fno.mamba.d_conv (default 4).
            expand: Expansion factor for the Mamba inner dimension. Inner
                dimension = hidden_dim * expand. From config.yaml
                models.mamba_fno.mamba.expand (default 2).
            n_dims: Spatial dimensionality. 1 for 1D problems (Burgers,
                Advection), 2 for 2D problems (NS, RD, Gray-Scott, Heat).
                From config.yaml models.mamba_fno.n_dims (default 2).
            activation: Activation function name for FNO blocks. From
                config.yaml models.mamba_fno.activation (default 'gelu').
                One of 'gelu', 'relu', 'tanh', 'silu', 'leaky_relu'.

        Raises:
            ImportError: If mamba_ssm is not installed (propagated from
                MambaModule.__init__).
            ValueError: If any argument is invalid (propagated from
                MambaModule or FNOBackbone).
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
        if n_layers <= 0:
            raise ValueError(
                f"n_layers must be a positive integer, got {n_layers}."
            )
        if n_dims not in (1, 2):
            raise ValueError(
                f"n_dims must be 1 or 2, got {n_dims}. "
                f"3D problems are not currently supported."
            )

        # ── Store hyperparameters ─────────────────────────────────────────
        self.hidden_dim: int = hidden_dim
        self.n_modes: int = n_modes
        self.n_layers: int = n_layers
        self.d_state: int = d_state
        self.d_conv: int = d_conv
        self.expand: int = expand
        self.n_dims: int = n_dims
        self.activation: str = activation

        # ── Post-lifting Mamba preconditioner ─────────────────────────────
        # Inserted between LiftingAdapter and FNO blocks.
        # Aligns lifted embeddings with dominant dynamical motifs before
        # the Fourier integral layers process them.
        # d_model = hidden_dim: each spatial token has hidden_dim features.
        self._mamba: MambaModule = MambaModule(
            hidden_dim=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # ── FNO backbone (Fourier integral layers) ────────────────────────
        # Shared backbone θ_F frozen during fine-tuning.
        # Processes preconditioned features from MambaModule.
        self._backbone: FNOBackbone = FNOBackbone(
            hidden_dim=hidden_dim,
            n_modes=n_modes,
            n_layers=n_layers,
            n_dims=n_dims,
            activation=activation,
        )

        # ── Log parameter counts ──────────────────────────────────────────
        n_mamba_params: int = sum(
            p.numel() for p in self._mamba.parameters()
        )
        n_backbone_params: int = sum(
            p.numel() for p in self._backbone.parameters()
        )
        n_total_params: int = n_mamba_params + n_backbone_params

        _logger.info(
            "MambaFNO: hidden_dim=%d, n_modes=%d, n_layers=%d, "
            "d_state=%d, d_conv=%d, expand=%d, n_dims=%d, "
            "activation='%s'. "
            "Parameters: mamba=%d, backbone=%d, total=%d.",
            hidden_dim,
            n_modes,
            n_layers,
            d_state,
            d_conv,
            expand,
            n_dims,
            activation,
            n_mamba_params,
            n_backbone_params,
            n_total_params,
        )

    def forward(self, v: Tensor) -> Tensor:
        """Apply the MambaFNO backbone to lifted feature tensors.

        Implements the two-step composition described in the paper:
          1. Mamba preconditioner: aligns features with dynamical motifs
          2. FNO Fourier integral layers: capture non-local spatial structure

        The composition F_t ∘ M_φ yields more stable training and improves
        efficiency in transferring pre-trained representations to new PDEs
        during fine-tuning (Section 3 of the paper).

        No activation or normalization is applied between the Mamba module
        and the FNO blocks — the FNO blocks contain their own activations
        (GELU by default), and the paper does not describe any intermediate
        processing between M_φ and F_t.

        Args:
            v: Lifted feature tensor from LiftingAdapter.
                Shape ``[B, hidden_dim, L]`` for 1D problems (n_dims=1).
                Shape ``[B, hidden_dim, H, W]`` for 2D problems (n_dims=2).
                Must be on CUDA (required by mamba_ssm).

        Returns:
            Processed feature tensor with the same shape as ``v``.
            Passed to ProjectionAdapter in AdapterFramework.forward().

        Raises:
            RuntimeError: If the input tensor is on CPU (propagated from
                MambaModule.forward — mamba_ssm requires CUDA).
            ValueError: If ``v`` has an unsupported number of dimensions
                or if the channel dimension does not match hidden_dim
                (propagated from MambaModule.forward).

        Note:
            During fine-tuning with AdapterFramework.freeze_backbone(),
            all parameters of this module have requires_grad=False.
            The forward pass still executes normally (inference mode),
            but no gradients are computed for _mamba or _backbone parameters.
        """
        # ── Step 1: Post-lifting Mamba preconditioner ─────────────────────
        # Aligns lifted embeddings with dominant dynamical motifs.
        # Input:  v  — [B, hidden_dim, *spatial]
        # Output: v_tilde — [B, hidden_dim, *spatial] (same shape)
        v_