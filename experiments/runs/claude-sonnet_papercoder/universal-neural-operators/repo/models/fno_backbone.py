```python
## models/fno_backbone.py
"""
FNO backbone implementation for the multi-physics neural operator pretraining
framework described in:
  "Towards Universal Neural Operators through Multiphysics Pretraining"

Implements the shared FNO backbone (θ_F) that is frozen during fine-tuning.
The backbone sits between problem-specific lifting and projection adapters:

    Input a → LiftingAdapter → FNOBackbone → ProjectionAdapter → Output u

The backbone transforms [B, hidden_dim, *spatial] → [B, hidden_dim, *spatial]
without any physics-specific logic, enabling clean adapter-based transfer.

Classes:
  SpectralConv1d  - Fourier integral operator for 1D problems [B, C, L]
  SpectralConv2d  - Fourier integral operator for 2D problems [B, C, H, W]
  FNOBlock        - One FNO layer: σ(spectral_path(v) + W_path(v))
  FNOBackbone     - Stacks n_layers FNOBlocks; the shared θ_F component

Tensor layout convention (Shared Knowledge #1):
  Channel-first: [B, C, L] for 1D, [B, C, H, W] for 2D.
  B=batch, C=channels (hidden_dim), L/H/W=spatial dimensions.

Config alignment (config.yaml):
  models.fno.hidden_dim: 64       -> hidden_dim parameter
  models.fno.n_modes: 16          -> n_modes parameter
  models.fno.n_layers: 4          -> n_layers parameter
  models.fno.n_dims: 2            -> n_dims parameter (1 or 2)
  models.fno.activation: "gelu"   -> activation parameter
  models.fno.target_params: 1e6   -> approximate parameter count target

Dependencies: torch, torch.nn, torch.fft. NO imports from other project files.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported activation functions
# ---------------------------------------------------------------------------

_ACTIVATION_MAP = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}


def _get_activation(name: str) -> nn.Module:
    """Instantiate an activation function by name.

    Args:
        name: Activation name (case-insensitive). One of 'gelu', 'relu',
            'tanh', 'silu', 'leaky_relu'. Default in config.yaml is 'gelu'.

    Returns:
        Instantiated activation module.

    Raises:
        ValueError: If name is not one of the supported activations.
    """
    name_lower: str = name.strip().lower()
    if name_lower not in _ACTIVATION_MAP:
        raise ValueError(
            f"Unsupported activation '{name}'. "
            f"Supported: {sorted(_ACTIVATION_MAP.keys())}."
        )
    return _ACTIVATION_MAP[name_lower]()


# ---------------------------------------------------------------------------
# SpectralConv1d
# ---------------------------------------------------------------------------


class SpectralConv1d(nn.Module):
    """Fourier integral operator for 1D spatial problems.

    Implements the spectral convolution path of an FNO block for 1D inputs:

        (K v)(x) = iFFT[ R(k) · FFT[v](k) ]_{k=0..n_modes-1}

    where R(k) are learnable complex weights applied to the first n_modes
    Fourier coefficients. Higher frequencies are zeroed out (truncation),
    acting as a low-pass filter that captures the dominant dynamics.

    Weight initialization follows the original FNO paper (Li et al., 2021):
    scale = 1 / (in_channels * out_channels), which prevents exploding
    activations in deep networks.

    Tensor layout: [B, C, L] — batch, channels, spatial length.

    Attributes:
        in_channels: Number of input channels (hidden_dim).
        out_channels: Number of output channels (hidden_dim).
        n_modes: Number of Fourier modes to retain (truncation threshold).
        weights: Complex-valued learnable weight tensor of shape
            [in_channels, out_channels, n_modes], dtype torch.cfloat.

    Example::

        conv = SpectralConv1d(in_channels=64, out_channels=64, n_modes=16)
        x = torch.randn(8, 64, 256)   # [B, C, L]
        out = conv(x)                  # [B, 64, 256]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int,
    ) -> None:
        """Initialise SpectralConv1d.

        Args:
            in_channels: Number of input feature channels. Typically
                hidden_dim from config.yaml (e.g., 64).
            out_channels: Number of output feature channels. Typically
                hidden_dim (same as in_channels within the backbone).
            n_modes: Number of Fourier modes to retain. From config.yaml
                models.fno.n_modes (e.g., 16). Must satisfy
                n_modes <= L // 2 + 1 where L is the spatial length.

        Raises:
            ValueError: If in_channels, out_channels, or n_modes <= 0.
        """
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}.")
        if n_modes <= 0:
            raise ValueError(f"n_modes must be positive, got {n_modes}.")

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.n_modes: int = n_modes

        # ── Complex-valued weight tensor ───────────────────────────────────
        # Shape: [in_channels, out_channels, n_modes]
        # dtype: torch.cfloat (complex64) — real and imaginary parts stored
        # together, halving memory vs. two separate real tensors.
        #
        # Initialization scale: 1 / (in_channels * out_channels) following
        # the original FNO paper to prevent exploding activations.
        scale: float = 1.0 / (in_channels * out_channels)
        self.weights: nn.Parameter = nn.Parameter(
            scale
            * torch.rand(in_channels, out_channels, n_modes, dtype=torch.cfloat)
        )

        _logger.debug(
            "SpectralConv1d: in_channels=%d, out_channels=%d, n_modes=%d. "
            "weights shape=%s.",
            in_channels,
            out_channels,
            n_modes,
            tuple(self.weights.shape),
        )

    def _compl_mul1d(self, a: Tensor, b: Tensor) -> Tensor:
        """Complex multiplication via einsum for 1D spectral convolution.

        Computes the batched matrix-vector product in Fourier space:
            out[b, o, k] = Σ_i a[b, i, k] * b[i, o, k]

        where b=batch, i=in_channels, o=out_channels, k=modes.

        Args:
            a: Input Fourier coefficients, shape [B, in_channels, n_modes],
                dtype torch.cfloat.
            b: Learnable weights, shape [in_channels, out_channels, n_modes],
                dtype torch.cfloat.

        Returns:
            Output Fourier coefficients, shape [B, out_channels, n_modes],
            dtype torch.cfloat.
        """
        return torch.einsum("bix,iox->box", a, b)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the 1D Fourier integral operator.

        Steps:
          1. rfft along the spatial dimension L.
          2. Truncate to first n_modes frequencies.
          3. Multiply truncated coefficients by learnable weights.
          4. Zero-pad back to L//2+1 frequencies.
          5. irfft to recover spatial domain signal of length L.

        Args:
            x: Input tensor of shape [B, in_channels, L].

        Returns:
            Output tensor of shape [B, out_channels, L].

        Raises:
            ValueError: If x has fewer than 3 dimensions.
            ValueError: If n_modes > L // 2 + 1 (more modes than available
                Fourier coefficients).
        """
        if x.dim() != 3:
            raise ValueError(
                f"SpectralConv1d expects 3D input [B, C, L], got {x.dim()}D "
                f"tensor with shape {tuple(x.shape)}."
            )

        batch_size: int = x.shape[0]
        spatial_len: int = x.shape[-1]  # L

        # ── Validate n_modes ───────────────────────────────────────────────
        n_freq: int = spatial_len // 2 + 1  # number of rfft output frequencies
        if self.n_modes > n_freq:
            raise ValueError(
                f"n_modes={self.n_modes} exceeds the number of available "
                f"Fourier frequencies n_freq={n_freq} for spatial_len={spatial_len}. "
                f"Reduce n_modes or increase the spatial resolution."
            )

        # ── Step 1: Real FFT along spatial dimension ───────────────────────
        # x_ft shape: [B, in_channels, L//2+1], dtype: cfloat
        x_ft: Tensor = torch.fft.rfft(x, dim=-1)

        # ── Step 2 & 3: Truncate and multiply with weights ─────────────────
        # Truncate to first n_modes frequencies, then apply learnable weights.
        # x_ft_trunc shape: [B, in_channels, n_modes]
        x_ft_trunc: Tensor = x_ft[:, :, : self.n_modes]

        # out_trunc shape: [B, out_channels, n_modes]
        out_trunc: Tensor = self._compl_mul1d(x_ft_trunc, self.weights)

        # ── Step 4: Zero-pad back to full frequency representation ─────────
        # Create zero tensor of shape [B, out_channels, L//2+1] and fill
        # the first n_modes slots with the computed output.
        out_ft: Tensor = torch.zeros(
            batch_size,
            self.out_channels,
            n_freq,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.n_modes] = out_trunc

        # ── Step 5: Inverse real FFT to recover spatial domain ─────────────
        # Pass n=spatial_len to handle odd-length signals correctly.
        # out shape: [B, out_channels, L]
        out: Tensor = torch.fft.irfft(out_ft, n=spatial_len, dim=-1)

        return out


# ---------------------------------------------------------------------------
# SpectralConv2d
# ---------------------------------------------------------------------------


class SpectralConv2d(nn.Module):
    """Fourier integral operator for 2D spatial problems.

    Extends SpectralConv1d to 2D by applying rfft2 and retaining modes in
    both spatial dimensions. Two weight tensors are used to capture both
    the top-left and bottom-left quadrants of the 2D Fourier space:

      - weights1: modes [:n_modes_x, :n_modes_y] — low positive x-frequencies
      - weights2: modes [-n_modes_x:, :n_modes_y] — low negative x-frequencies

    This two-quadrant approach is necessary because rfft2 returns the full
    frequency range along the first spatial dimension (both positive and
    negative frequencies), but only the non-redundant half along the last
    dimension. Capturing both quadrants ensures the operator can represent
    both symmetric and antisymmetric spatial patterns.

    Tensor layout: [B, C, H, W] — batch, channels, height, width.

    Attributes:
        in_channels: Number of input channels (hidden_dim).
        out_channels: Number of output channels (hidden_dim).
        n_modes_x: Number of Fourier modes to retain along height (H).
        n_modes_y: Number of Fourier modes to retain along width (W).
        weights1: Complex weights for top-left quadrant,
            shape [in_channels, out_channels, n_modes_x, n_modes_y].
        weights2: Complex weights for bottom-left quadrant,
            shape [in_channels, out_channels, n_modes_x, n_modes_y].

    Example::

        conv = SpectralConv2d(in_channels=64, out_channels=64,
                              n_modes_x=12, n_modes_y=12)
        x = torch.randn(8, 64, 64, 64)   # [B, C, H, W]
        out = conv(x)                      # [B, 64, 64, 64]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes_x: int,
        n_modes_y: int,
    ) -> None:
        """Initialise SpectralConv2d.

        Args:
            in_channels: Number of input feature channels. Typically
                hidden_dim from config.yaml (e.g., 64).
            out_channels: Number of output feature channels. Typically
                hidden_dim (same as in_channels within the backbone).
            n_modes_x: Number of Fourier modes to retain along the height
                (H) dimension. From config.yaml models.fno.n_modes (e.g., 16).
                Must satisfy n_modes_x <= H // 2 + 1.
            n_modes_y: Number of Fourier modes to retain along the width
                (W) dimension. From config.yaml models.fno.n_modes (e.g., 16).
                Must satisfy n_modes_y <= W // 2 + 1.

        Raises:
            ValueError: If any argument is non-positive.
        """
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}.")
        if n_modes_x <= 0:
            raise ValueError(f"n_modes_x must be positive, got {n_modes_x}.")
        if n_modes_y <= 0:
            raise ValueError(f"n_modes_y must be positive, got {n_modes_y}.")

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.n_modes_x: int = n_modes_x
        self.n_modes_y: int = n_modes_y

        # ── Complex-valued weight tensors ──────────────────────────────────
        # Two weight tensors for the two Fourier quadrants:
        #   weights1: top-left    [:n_modes_x, :n_modes_y]
        #   weights2: bottom-left [-n_modes_x:, :n_modes_y]
        #
        # Shape: [in_channels, out_channels, n_modes_x, n_modes_y]
        # dtype: torch.cfloat (complex64)
        #
        # Initialization scale: 1 / (in_channels * out_channels) following
        # the original FNO paper.
        scale: float = 1.0 / (in_channels * out_channels)

        self.weights1: nn.Parameter = nn.Parameter(
            scale
            * torch.rand(
                in_channels,
                out_channels,
                n_modes_x,
                n_modes_y,
                dtype=torch.cfloat,
            )
        )
        self.weights2: nn.Parameter = nn.Parameter(
            scale
            * torch.rand(
                in_channels,
                out_channels,
                n_modes_x,
                n_modes_y,
                dtype=torch.cfloat,
            )
        )

        _logger.debug(
            "SpectralConv2d: in_channels=%d, out_channels=%d, "
            "n_modes_x=%d, n_modes_y=%d. "
            "weights1 shape=%s, weights2 shape=%s.",
            in_channels,
            out_channels,
            n_modes_x,
            n_modes_y,
            tuple(self.weights1.shape),
            tuple(self.weights2.shape),
        )

    def _compl_mul2d(self, a: Tensor, b: Tensor) -> Tensor:
        """Complex multiplication via einsum for 2D spectral convolution.

        Computes the batched matrix-vector product in 2D Fourier space:
            out[b, o, x, y] = Σ_i a[b, i, x, y] * b[i, o, x, y]

        where b=batch, i=in_channels, o=out_channels, x=modes_x, y=modes_y.

        Args:
            a: Input Fourier coefficients, shape
                [B, in_channels, n_modes_x, n_modes_y], dtype torch.cfloat.
            b: Learnable weights, shape
                [in_channels, out_channels, n_modes_x, n_modes_y],
                dtype torch.cfloat.

        Returns:
            Output Fourier coefficients, shape
            [B, out_channels, n_modes_x, n_modes_y], dtype torch.cfloat.
        """
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the 2D Fourier integral operator.

        Steps:
          1. rfft2 along the two spatial dimensions (H, W).
          2. Apply weights1 to top-left Fourier block.
          3. Apply weights2 to bottom-left Fourier block.
          4. Accumulate into zero-padded output frequency tensor.
          5. irfft2 to recover spatial domain signal of shape (H, W).

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Output tensor of shape [B, out_channels, H, W].

        Raises:
            ValueError: If x has fewer than 4 dimensions.
            ValueError: If n_modes_x or n_modes_y exceed available frequencies.
        """
        if x.dim() != 4:
            raise ValueError(
                f"SpectralConv2d expects 4D input [B, C, H, W], got {x.dim()}D "
                f"tensor with shape {tuple(x.shape)}."
            )

        batch_size: int = x.shape[0]
        height: int = x.shape[-2]   # H
        width: int = x.shape[-1]    # W

        # ── Validate n_modes ───────────────────────────────────────────────
        n_freq_x: int = height       # rfft2 returns full range along H
        n_freq_y: int = width // 2 + 1  # rfft2 returns non-redundant half along W

        if self.n_modes_x > n_freq_x // 2:
            raise ValueError(
                f"n_modes_x={self.n_modes_x} exceeds half the available "
                f"x-frequencies ({n_freq_x // 2}) for height={height}. "
                f"Reduce n_modes_x or increase the spatial resolution."
            )
        if self.n_modes_y > n_freq_y:
            raise ValueError(
                f"n_modes_y={self.n_modes_y} exceeds the available "
                f"y-frequencies ({n_freq_y}) for width={width}. "
                f"Reduce n_modes_y or increase the spatial resolution."
            )

        # ── Step 1: Real 2D FFT along spatial dimensions ───────────────────
        # x_ft shape: [B, in_channels, H, W//2+1], dtype: cfloat
        x_ft: Tensor = torch.fft.rfft2(x, dim=(-2, -1))

        # ── Step 2 & 3: Apply weights to two Fourier quadrants ─────────────
        # Top-left block: positive low x-frequencies, low y-frequencies
        # x_ft[:, :, :n_modes_x, :n_modes_y] shape: [B, C_in, n_modes_x, n_modes_y]
        out_top_left: Tensor = self._compl_mul2d(
            x_ft[:, :, : self.n_modes_x, : self.n_modes_y],
            self.weights1,
        )  # [B, C_out, n_modes_x, n_modes_y]

        # Bottom-left block: negative low x-frequencies, low y-frequencies
        # x_ft[:, :, -n_modes_x:, :n_modes_y] shape: [B, C_in, n_modes_x, n_modes_y]
        out_bottom_left: Tensor = self._compl_mul2d(
            x_ft[:, :, -self.n_modes_x :, : self.n_modes_y],
            self.weights2,
        )  # [B, C_out, n_modes_x, n_modes_y]

        # ── Step 4: Accumulate into zero-padded output frequency tensor ────
        # Create zero tensor of shape [B, out_channels, H, W//2+1].
        out_ft: Tensor = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            n_freq_y,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Place top-left block at [:n_modes_x, :n_modes_y]
        out_ft[:, :, : self.n_modes_x, : self.n_modes_y] = out_top_left

        # Place bottom-left block at [-n_modes_x:, :n_modes_y]
        out_ft[:, :, -self.n_modes_x :, : self.n_modes_y] = out_bottom_left

        # ── Step 5: Inverse real 2D FFT to recover spatial domain ──────────
        # Pass s=(H, W) to handle odd-dimension signals correctly.
        # out shape: [B, out_channels, H, W]
        out: Tensor = torch.fft.irfft2(out_ft, s=(height, width), dim=(-2, -1))

        return out


# ---------------------------------------------------------------------------
# FNOBlock
# ---------------------------------------------------------------------------


class FNOBlock(nn.Module):
    """One layer of the Fourier Neural Operator.

    Implements the FNO block from equation (1) in the paper:

        F_t(x) = σ( A_t * v_t(x) + ∫ κ_t(x,y) v_t(y) dy + b_t(x) )

    where:
      - The integral term is approximated by the spectral convolution path
        (SpectralConv1d or SpectralConv2d).
      - A_t * v_t(x) + b_t(x) is the pointwise linear (W) path, implemented
        as a 1×1 convolution (kernel_size=1) which includes a bias term.
      - σ is the activation function (GELU by default, per config.yaml).

    The two paths are summed before applying the activation:
        output = σ(spectral_path(v) + W_path(v))

    This residual-like structure allows the spectral path to capture
    non-local (long-range) interactions while the W path handles local
    pointwise transformations.

    Attributes:
        hidden_dim: Number of input and output channels.
        n_modes: Number of Fourier modes retained in the spectral path.
        n_dims: Spatial dimensionality (1 or 2).
        _spectral_conv: SpectralConv1d or SpectralConv2d instance.
        _w: Pointwise linear transform (nn.Conv1d or nn.Conv2d, kernel=1).
        _activation: Activation function module (nn.GELU by default).

    Example::

        # 2D FNO block
        block = FNOBlock(hidden_dim=64, n_modes=16, n_dims=2)
        v = torch.randn(8, 64, 64, 64)   # [B, C, H, W]
        out = block(v)                    # [B, 64, 64, 64]

        # 1D FNO block
        block_1d = FNOBlock(hidden_dim=64, n_modes=16, n_dims=1)
        v_1d = torch.randn(8, 64, 256)   # [B, C, L]
        out_1d = block_1d(v_1d)          # [B, 64, 256]
    """

    def __init__(
        self,
        hidden_dim: int,
        n_modes: int,
        n_dims: int = 2,
        activation: str = "gelu",
    ) -> None:
        """Initialise FNOBlock.

        Args:
            hidden_dim: Number of input and output channels. From config.yaml
                models.fno.hidden_dim (e.g., 64). Both the spectral path and
                the W path map hidden_dim -> hidden_dim.
            n_modes: Number of Fourier modes to retain. From config.yaml
                models.fno.n_modes (e.g., 16). For 2D, the same n_modes is
                used for both spatial dimensions (n_modes_x = n_modes_y).
            n_dims: Spatial dimensionality. 1 for 1D problems (Burgers,
                Advection), 2 for 2D problems (NS, RD, Gray-Scott, Heat).
                From config.yaml models.{model}.n_dims. Default 2.
            activation: Activation function name. From config.yaml
                models.fno.activation (default 'gelu'). One of 'gelu',
                'relu', 'tanh', 'silu', 'leaky_relu'.

        Raises:
            ValueError: If hidden_dim or n_modes <= 0.
            ValueError: If n_dims is not 1 or 2.
            ValueError: If activation is not a supported name.
        """
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if n_modes <= 0:
            raise ValueError(f"n_modes must be positive, got {n_modes}.")
        if n_dims not in (1, 2):
            raise ValueError(
                f"n_dims must be 1 or 2, got {n_dims}. "
                f"3D problems are not currently supported."
            )

        self.hidden_dim: int = hidden_dim
        self.n_modes: int = n_modes
        self.n_dims: int = n_dims

        # ── Spectral convolution path ──────────────────────────────────────
        # Dispatches to 1D or 2D spectral conv based on n_dims.
        # For 2D, n_modes is used for both spatial dimensions.
        if n_dims == 1:
            self._spectral_conv: nn.Module = SpectralConv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                n_modes=n_modes,
            )
        else:  # n_dims == 2
            self._spectral_conv = SpectralConv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                n_modes_x=n_modes,
                n_modes_y=n_modes,
            )

        # ── Pointwise linear (W) path ──────────────────────────────────────
        # kernel_size=1 makes this a pointwise operation applied independently
        # at each spatial location. The bias term serves as b_t(x) in the
        # paper's equation. bias=True is the default for nn.Conv1d/Conv2d.
        if n_dims == 1:
            self._w: nn.Module = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=1,
                bias=True,
            )
        else:  # n_dims == 2
            self._w = nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=1,
                bias=True,
            )

        # ── Activation function ────────────────────────────────────────────
        # GELU is the default per config.yaml models.fno.activation: "gelu".
        self._activation: nn.Module = _get_activation(activation)

        _logger.debug(
            "FNOBlock: hidden_dim=%d, n_modes=%d, n_dims=%d, activation='%s'.",
            hidden_dim,
            n_modes,
            n_dims,
            activation,
        )

    def forward(self, v: Tensor) -> Tensor:
        """Apply one FNO block.

        Computes: σ(spectral_path(v) + W_path(v))

        The spectral path captures non-local (long-range) spatial interactions
        via Fourier convolution. The W path provides a local pointwise
        transformation. Their sum is passed through the activation function.

        Args:
            v: Hidden representation tensor.
                Shape [B, hidden_dim, L] for 1D (n_dims=1).
                Shape [B, hidden_dim, H, W] for 2D (n_dims=2).

        Returns:
            Transformed hidden representation with the same shape as input.

        Raises:
            ValueError: If v has the wrong number of dimensions for n_dims.
        """
        expected_ndim: int = 3 if self.n_dims == 1 else 4
        if v.dim() != expected_ndim:
            raise ValueError(
                f"FNOBlock (n_dims={self.n_dims}) expects {expected_ndim}D input, "
                f"got {v.dim()}D tensor with shape {tuple(v.shape)}."
            )

        # ── Spectral path: non-local Fourier convolution ───────────────────
        spectral_out: Tensor = self._spectral_conv(v)

        # ── W path: local pointwise linear transform ───────────────────────
        w_out: Tensor = self._w(v)

        # ── Sum and activate ───────────────────────────────────────────────
        # The bias from _w serves as b_t(x) in the paper's equation.
        return self._activation(spectral_out + w_out)


# ---------------------------------------------------------------------------
# FNOBackbone
# ---------------------------------------------------------------------------


class FNOBackbone(nn.Module):
    """Shared FNO backbone (θ_F) frozen during fine-tuning.

    Stacks n_layers FNOBlock instances to form the shared backbone of the
    adapter-based multi