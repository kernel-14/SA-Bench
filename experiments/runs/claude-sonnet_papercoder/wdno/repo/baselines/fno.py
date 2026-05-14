## Code: baselines/fno.py

```python
## baselines/fno.py
"""Fourier Neural Operator (FNO) baseline for WDNO comparison.

Implements FNO (Li et al. 2021) for 1D and 2D PDE simulation experiments.
Used as a mesh-invariant baseline in:
    - Table 1: Simulation MSE comparison
    - Table 5: Extended 1D compressible NS comparisons
    - Table 6: Approximate scale invariance verification
    - Table 16/17: Zero-shot super-resolution baseline

The FNO operates directly in raw space-time domain (no wavelet transform),
making it a clean baseline that highlights WDNO's advantages in modeling
abrupt changes and generalizing across resolutions.

Paper sources:
    - FNO: Li et al. 2021 (Fourier Neural Operator for Parametric PDEs)
    - 1D hyperparameters: Appendix J.1, Table 26
    - 2D hyperparameters: Appendix K.1, Table 32
    - Super-resolution property: Section 4.6 (mesh-invariant baseline)
    - Scale invariance ablation: Appendix C.3, Table 6

Config references:
    - baselines.fno_1d.modes: 16
    - baselines.fno_1d.width: 64
    - baselines.fno_1d.in_channels: 3
    - baselines.fno_1d.out_channels: 1
    - baselines.fno_1d.n_layers: 4
    - baselines.fno_1d.hidden_channels_lift: 256
    - baselines.fno_1d.hidden_channels_proj: 256
    - baselines.fno_1d.expansion_mlp: 0.5
    - baselines.fno_1d.nonlinearity: gelu
    - baselines.fno_1d.rank_factorization: 1.0
    - baselines.fno_1d.padding_mode: one-sided
    - baselines.fno_1d.learning_rate: 1e-4
    - baselines.fno_2d.modes: 16
    - baselines.fno_2d.width: 64
    - baselines.fno_2d.in_channels: 6
    - baselines.fno_2d.out_channels: 4
    - baselines.fno_2d.n_layers: 4
    - baselines.fno_2d.train_epochs: 1000
    - baselines.fno_2d.batch_size: 16
    - baselines.fno_2d.learning_rate: 1e-4
    - baselines.fno_2d.lr_scheduler: cosine
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Default hyperparameters from config.yaml baselines.fno_1d and baselines.fno_2d
_DEFAULT_MODES: int = 16
_DEFAULT_WIDTH: int = 64
_DEFAULT_N_LAYERS: int = 4
_DEFAULT_HIDDEN_LIFT: int = 256
_DEFAULT_HIDDEN_PROJ: int = 256
_DEFAULT_LR: float = 1e-4
_DEFAULT_TRAIN_EPOCHS_1D: int = 1000
_DEFAULT_TRAIN_EPOCHS_2D: int = 1000
_DEFAULT_BATCH_SIZE_2D: int = 16

# Padding fraction for one-sided padding mode (config: padding_mode: one-sided)
_PADDING_FRACTION: float = 0.25


# ---------------------------------------------------------------------------
# 1D Spectral Convolution
# ---------------------------------------------------------------------------


class SpectralConv1d(nn.Module):
    """1D spectral convolution layer for FNO.

    Applies FFT along the spatial dimension, multiplies by learnable complex
    weights in the truncated frequency domain, then applies inverse FFT.
    Only the first ``modes`` Fourier modes are retained (truncation).

    This implements the integral kernel parameterization in Fourier space
    from Li et al. 2021, Eq. 3.

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes to keep. Config: baselines.fno_1d.modes=16.
        scale: Initialization scale for weights.
        weights_real: Real part of complex weights, shape [in_channels, out_channels, modes].
        weights_imag: Imaginary part of complex weights, shape [in_channels, out_channels, modes].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = _DEFAULT_MODES,
    ) -> None:
        """Initialize SpectralConv1d.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            modes: Number of Fourier modes to keep (truncation point).
                Config: baselines.fno_1d.modes=16. Only the first ``modes``
                frequency components are used in the spectral convolution.
        """
        super().__init__()

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.modes: int = modes

        # Initialization scale following FNO paper convention
        self.scale: float = 1.0 / (in_channels * out_channels)

        # Store complex weights as two real tensors (real and imaginary parts)
        # Shape: [in_channels, out_channels, modes]
        # This avoids complex dtype compatibility issues across PyTorch versions
        self.weights_real: nn.Parameter = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes)
        )
        self.weights_imag: nn.Parameter = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes)
        )

    def _complex_mul1d(
        self,
        x_ft: torch.Tensor,
        weights_real: torch.Tensor,
        weights_imag: torch.Tensor,
    ) -> torch.Tensor:
        """Multiply complex Fourier coefficients by complex weights.

        Implements complex multiplication:
            (a + bi)(c + di) = (ac - bd) + (ad + bc)i

        Args:
            x_ft: Complex Fourier coefficients, shape [batch, in_channels, modes].
                dtype=torch.complex64 or torch.complex128.
            weights_real: Real part of weights, shape [in_channels, out_channels, modes].
            weights_imag: Imaginary part of weights, shape [in_channels, out_channels, modes].

        Returns:
            Complex output tensor, shape [batch, out_channels, modes].
        """
        # x_ft: [batch, in_channels, modes] (complex)
        # weights: [in_channels, out_channels, modes] (real/imag separately)
        # Output: [batch, out_channels, modes] (complex)

        # Combine weights into complex tensor for einsum
        weights_complex: torch.Tensor = torch.complex(weights_real, weights_imag)
        # weights_complex: [in_channels, out_channels, modes]

        # Spectral convolution via einsum: sum over in_channels
        # 'bim,iom->bom' where b=batch, i=in_channels, o=out_channels, m=modes
        return torch.einsum("bim,iom->bom", x_ft, weights_complex)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 1D spectral convolution.

        Args:
            x: Input feature map, shape [batch, in_channels, spatial_size].
                dtype=float32.

        Returns:
            Output feature map, shape [batch, out_channels, spatial_size].
            dtype=float32.
        """
        batch_size: int = x.shape[0]
        spatial_size: int = x.shape[-1]

        # Apply real FFT along spatial dimension
        # x_ft: [batch, in_channels, spatial_size//2 + 1] (complex)
        x_ft: torch.Tensor = torch.fft.rfft(x, dim=-1)

        # Truncate to first ``modes`` frequencies
        # x_ft_trunc: [batch, in_channels, modes]
        x_ft_trunc: torch.Tensor = x_ft[..., : self.modes]

        # Spectral convolution: multiply by complex weights
        # out_ft_trunc: [batch, out_channels, modes]
        out_ft_trunc: torch.Tensor = self._complex_mul1d(
            x_ft_trunc, self.weights_real, self.weights_imag
        )

        # Zero-pad back to original frequency size
        # out_ft: [batch, out_channels, spatial_size//2 + 1]
        freq_size: int = spatial_size // 2 + 1
        out_ft: torch.Tensor = torch.zeros(
            batch_size,
            self.out_channels,
            freq_size,
            dtype=torch.complex64,
            device=x.device,
        )
        out_ft[..., : self.modes] = out_ft_trunc

        # Apply inverse real FFT to recover spatial domain
        # out: [batch, out_channels, spatial_size]
        out: torch.Tensor = torch.fft.irfft(out_ft, n=spatial_size, dim=-1)

        return out


# ---------------------------------------------------------------------------
# 2D Spectral Convolution
# ---------------------------------------------------------------------------


class SpectralConv2d(nn.Module):
    """2D spectral convolution layer for FNO2D.

    Applies 2D FFT, multiplies by learnable complex weights in the truncated
    2D frequency domain, then applies inverse 2D FFT.

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes1: Number of Fourier modes to keep in height dimension.
        modes2: Number of Fourier modes to keep in width dimension.
        scale: Initialization scale for weights.
        weights_real: Real part of weights, shape [in_channels, out_channels, modes1, modes2].
        weights_imag: Imaginary part of weights, shape [in_channels, out_channels, modes1, modes2].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = _DEFAULT_MODES,
        modes2: int = _DEFAULT_MODES,
    ) -> None:
        """Initialize SpectralConv2d.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            modes1: Number of Fourier modes in height dimension.
                Config: baselines.fno_2d.modes=16.
            modes2: Number of Fourier modes in width dimension.
                Config: baselines.fno_2d.modes=16.
        """
        super().__init__()

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.modes1: int = modes1
        self.modes2: int = modes2

        self.scale: float = 1.0 / (in_channels * out_channels)

        # Complex weights stored as real/imaginary pairs
        # Shape: [in_channels, out_channels, modes1, modes2]
        self.weights_real: nn.Parameter = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )
        self.weights_imag: nn.Parameter = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )

    def _complex_mul2d(
        self,
        x_ft: torch.Tensor,
        weights_real: torch.Tensor,
        weights_imag: torch.Tensor,
    ) -> torch.Tensor:
        """Multiply 2D complex Fourier coefficients by complex weights.

        Args:
            x_ft: Complex Fourier coefficients, shape [batch, in_channels, modes1, modes2].
            weights_real: Real part of weights, shape [in_channels, out_channels, modes1, modes2].
            weights_imag: Imaginary part of weights, shape [in_channels, out_channels, modes1, modes2].

        Returns:
            Complex output tensor, shape [batch, out_channels, modes1, modes2].
        """
        weights_complex: torch.Tensor = torch.complex(weights_real, weights_imag)
        # 'bimn,iomn->bomn' where b=batch, i=in_ch, o=out_ch, m=modes1, n=modes2
        return torch.einsum("bimn,iomn->bomn", x_ft, weights_complex)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D spectral convolution.

        Args:
            x: Input feature map, shape [batch, in_channels, height, width].
                dtype=float32.

        Returns:
            Output feature map, shape [batch, out_channels, height, width].
            dtype=float32.
        """
        batch_size: int = x.shape[0]
        height: int = x.shape[-2]
        width: int = x.shape[-1]

        # Apply 2D real FFT
        # x_ft: [batch, in_channels, height, width//2 + 1] (complex)
        x_ft: torch.Tensor = torch.fft.rfft2(x, dim=(-2, -1))

        # Truncate to first modes1 × modes2 frequencies
        # x_ft_trunc: [batch, in_channels, modes1, modes2]
        x_ft_trunc: torch.Tensor = x_ft[..., : self.modes1, : self.modes2]

        # Spectral convolution
        # out_ft_trunc: [batch, out_channels, modes1, modes2]
        out_ft_trunc: torch.Tensor = self._complex_mul2d(
            x_ft_trunc, self.weights_real, self.weights_imag
        )

        # Zero-pad back to original frequency size
        freq_h: int = height
        freq_w: int = width // 2 + 1
        out_ft: torch.Tensor = torch.zeros(
            batch_size,
            self.out_channels,
            freq_h,
            freq_w,
            dtype=torch.complex64,
            device=x.device,
        )
        out_ft[..., : self.modes1, : self.modes2] = out_ft_trunc

        # Apply inverse 2D real FFT
        # out: [batch, out_channels, height, width]
        out: torch.Tensor = torch.fft.irfft2(out_ft, s=(height, width), dim=(-2, -1))

        return out


# ---------------------------------------------------------------------------
# FNO Building Blocks
# ---------------------------------------------------------------------------


class FNOBlock1d(nn.Module):
    """Single 1D FNO layer combining spectral convolution with local bypass.

    Implements one Fourier layer from Li et al. 2021:
        output = activation(SpectralConv1d(x) + Conv1d_bypass(x))

    The bypass Conv1d with kernel_size=1 handles the local/non-periodic
    component of the operator (pointwise linear transform).

    Attributes:
        spectral_conv: SpectralConv1d for the frequency-domain path.
        bypass_conv: nn.Conv1d(width, width, 1) for the local path.
        activation: GELU activation (config: baselines.fno_1d.nonlinearity=gelu).
    """

    def __init__(
        self,
        width: int = _DEFAULT_WIDTH,
        modes: int = _DEFAULT_MODES,
    ) -> None:
        """Initialize FNOBlock1d.

        Args:
            width: Number of feature channels (lifted dimension).
                Config: baselines.fno_1d.width=64.
            modes: Number of Fourier modes. Config: baselines.fno_1d.modes=16.
        """
        super().__init__()

        self.spectral_conv: SpectralConv1d = SpectralConv1d(width, width, modes)
        # Bypass: pointwise linear transform (kernel_size=1)
        self.bypass_conv: nn.Conv1d = nn.Conv1d(width, width, kernel_size=1)
        # GELU activation per config: baselines.fno_1d.nonlinearity=gelu
        self.activation: nn.Module = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through one FNO layer.

        Args:
            x: Input feature map, shape [batch, width, spatial_size].
                dtype=float32.

        Returns:
            Output feature map, shape [batch, width, spatial_size].
            dtype=float32.
        """
        return self.activation(self.spectral_conv(x) + self.bypass_conv(x))


class FNOBlock2d(nn.Module):
    """Single 2D FNO layer combining spectral convolution with local bypass.

    Attributes:
        spectral_conv: SpectralConv2d for the frequency-domain path.
        bypass_conv: nn.Conv2d(width, width, 1) for the local path.
        activation: GELU activation.
    """

    def __init__(
        self,
        width: int = _DEFAULT_WIDTH,
        modes: int = _DEFAULT_MODES,
    ) -> None:
        """Initialize FNOBlock2d.

        Args:
            width: Number of feature channels.
                Config: baselines.fno_2d.width=64.
            modes: Number of Fourier modes per spatial dimension.
                Config: baselines.fno_2d.modes=16.
        """
        super().__init__()

        self.spectral_conv: SpectralConv2d = SpectralConv2d(width, width, modes, modes)
        # Bypass: pointwise linear transform
        self.bypass_conv: nn.Conv2d = nn.Conv2d(width, width, kernel_size=1)
        self.activation: nn.Module = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through one 2D FNO layer.

        Args:
            x: Input feature map, shape [batch, width, height, width_spatial].
                dtype=float32.

        Returns:
            Output feature map, same shape as input. dtype=float32.
        """
        return self.activation(self.spectral_conv(x) + self.bypass_conv(x))


# ---------------------------------------------------------------------------
# FNO (1D PDE experiments)
# ---------------------------------------------------------------------------


class FNO(nn.Module):
    """Fourier Neural Operator for 1D PDE simulation experiments.

    Implements the FNO architecture from Li et al. 2021 for 1D PDE experiments
    (Burgers', advection, compressible NS). Operates directly on raw space-time
    data without wavelet transform.

    Architecture:
        Lifting MLP → n_layers × FNOBlock1d → Projection MLP

    The lifting MLP projects from in_channels to width (64), the FNO blocks
    process in the lifted space, and the projection MLP maps back to out_channels.

    Input convention (paper Appendix J.1):
        "we train the FNO model using the initial state and all controls as
        the input and using the rest states as the output."
        Input: concatenation of [u0_repeated, f_all_timesteps, x_grid]
        For in_channels=3: [u0 (1ch), f_mean_over_time (1ch), x_grid (1ch)]

    Zero-shot super-resolution property:
        The FNO is mesh-invariant — the same trained model can be evaluated
        at different spatial resolutions by simply passing higher-resolution
        inputs. Used as SR baseline in Table 16/17.

    Attributes:
        modes: Number of Fourier modes. Config: baselines.fno_1d.modes=16.
        width: Lifted channel dimension. Config: baselines.fno_1d.width=64.
        in_channels: Input channels. Config: baselines.fno_1d.in_channels=3.
        out_channels: Output channels. Config: baselines.fno_1d.out_channels=1.
        n_layers: Number of FNO blocks. Config: baselines.fno_1d.n_layers=4.
        padding_mode: Boundary padding mode. Config: baselines.fno_1d.padding_mode=one-sided.
        pad_size: Number of spatial points to pad (spatial_size // 4).
        lifting: MLP projecting in_channels → width.
        fno_blocks: ModuleList of n_layers FNOBlock1d instances.
        projection: MLP projecting width → out_channels.
    """

    def __init__(
        self,
        modes: int = _DEFAULT_MODES,
        width: int = _DEFAULT_WIDTH,
        in_channels: int = 3,
        out_channels: int = 1,
        n_layers: int = _DEFAULT_N_LAYERS,
        hidden_channels_lift: int = _DEFAULT_HIDDEN_LIFT,
        hidden_channels_proj: int = _DEFAULT_HIDDEN_PROJ,
        padding_mode: str = "one-sided",
    ) -> None:
        """Initialize the FNO.

        Args:
            modes: Number of Fourier modes to keep in spectral convolution.
                Config: baselines.fno_1d.modes=16.
            width: Lifted channel dimension (internal representation width).
                Config: baselines.fno_1d.width=64.
            in_channels: Number of input channels. For 1D Burgers' with
                [u0, f_mean, x_grid] encoding: 3.
                Config: baselines.fno_1d.in_channels=3.
            out_channels: Number of output channels. For single-variable
                prediction: 1. Config: baselines.fno_1d.out_channels=1.
            n_layers: Number of FNO blocks (Fourier layers).
                Config: baselines.fno_1d.n_layers=4.
            hidden_channels_lift: Hidden dimension in lifting MLP.
                Config: baselines.fno_1d.hidden_channels_lift=256.
            hidden_channels_proj: Hidden dimension in projection MLP.
                Config: baselines.fno_1d.hidden_channels_proj=256.
            padding_mode: Boundary padding mode. 'one-sided' pads the
                spatial dimension on one side before FFT to reduce spectral
                leakage at boundaries. Config: baselines.fno_1d.padding_mode=one-sided.

        Raises:
            ValueError: If padding_mode is not 'one-sided' or 'none'.
        """
        super().__init__()

        if padding_mode not in ("one-sided", "none"):
            raise ValueError(
                f"padding_mode must be 'one-sided' or 'none', got '{padding_mode}'. "
                "Config: baselines.fno_1d.padding_mode=one-sided."
            )

        self.modes: int = modes
        self.width: int = width
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.n_layers: int = n_layers
        self.padding_mode: str = padding_mode

        # Padding size: spatial_size // 4 (computed dynamically in forward)
        # Stored as 0 initially; computed from input in forward()
        self._pad_size: int = 0

        # -----------------------------------------------------------------------
        # Lifting MLP: in_channels → hidden_channels_lift → width
        # Projects input to the lifted channel dimension
        # Config: baselines.fno_1d.hidden_channels_lift=256
        # -----------------------------------------------------------------------
        self.lifting: nn.Sequential = nn.Sequential(
            nn.Linear(in_channels, hidden_channels_lift),
            nn.GELU(),
            nn.Linear(hidden_channels_lift, width),
        )

        # -----------------------------------------------------------------------
        # FNO blocks: n_layers × FNOBlock1d(width, modes)
        # Config: baselines.fno_1d.n_layers=4
        # -----------------------------------------------------------------------
        self.fno_blocks: nn.ModuleList = nn.ModuleList(
            [FNOBlock1d(width=width, modes=modes) for _ in range(n_layers)]
        )

        # -----------------------------------------------------------------------
        # Projection MLP: width → hidden_channels_proj → out_channels
        # Config: baselines.fno_1d.hidden_channels_proj=256
        # -----------------------------------------------------------------------
        self.projection: nn.Sequential = nn.Sequential(
            nn.Linear(width, hidden_channels_proj),
            nn.GELU(),
            nn.Linear(hidden_channels_proj, out_channels),
        )

        logger.info(
            "FNO initialized: modes=%d, width=%d, in_channels=%d, "
            "out_channels=%d, n_layers=%d, padding_mode=%s, "
            "hidden_lift=%d, hidden_proj=%d",
            modes, width, in_channels, out_channels, n_layers,
            padding_mode, hidden_channels_lift, hidden_channels_proj,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the FNO.

        Args:
            x: Input tensor, shape [batch, in_channels, spatial_size].
                For 1D Burgers': [batch, 3, 120] where channels are
                [u0, f_mean_over_time, x_grid]. dtype=float32.

        Returns:
            Output tensor, shape [batch, out_channels, spatial_size].
            For single-variable prediction: [batch, 1, 120]. dtype=float32.
        """
        spatial_size: int = x.shape[-1]

        # -----------------------------------------------------------------------
        # Lifting: [batch, in_channels, spatial] → [batch, width, spatial]
        # Apply lifting MLP pointwise along spatial dimension
        # -----------------------------------------------------------------------
        # Transpose for linear: [batch, spatial, in_channels]
        x_lifted: torch.Tensor = self.lifting(x.permute(0, 2, 1))
        # Transpose back: [batch, width, spatial]
        x_lifted = x_lifted.permute(0, 2, 1)

        # -----------------------------------------------------------------------
        # One-sided padding to reduce boundary spectral leakage
        # Config: baselines.fno_1d.padding_mode=one-sided
        # -----------------------------------------------------------------------
        pad_size: int = 0
        if self.padding_mode == "one-sided":
            pad_size = int(math.ceil(spatial_size * _PADDING_FRACTION))
            # Pad on the right side: [batch, width, spatial + pad_size]
            x_lifted = F.pad(x_lifted, (0, pad_size), mode="constant", value=0.0)

        # -----------------------------------------------------------------------
        # FNO blocks: n_layers × FNOBlock1d
        # -----------------------------------------------------------------------
        h: torch.Tensor = x_lifted
        for block in self.fno_blocks:
            h = block(h)

        # -----------------------------------------------------------------------
        # Remove padding
        # -----------------------------------------------------------------------
        if pad_size > 0:
            h = h[..., :spatial_size]

        # -----------------------------------------------------------------------
        # Projection: [batch, width, spatial] → [batch, out_channels, spatial]
        # Apply projection MLP pointwise along spatial dimension
        # -----------------------------------------------------------------------
        # Transpose for linear: [batch, spatial, width]
        out: torch.Tensor = self.projection(h.permute(0, 2, 1))
        # Transpose back: [batch, out_channels, spatial]
        out = out.permute(0, 2, 1)

        return out

    def simulate(
        self,
        u0: torch.Tensor,
        f: torch.Tensor,
        x_grid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run simulation inference to predict full state trajectory.

        Prepares the input by concatenating u0, f encoding, and spatial grid,
        then runs a single forward pass to predict the full trajectory.

        For the FNO, the full trajectory is predicted in one shot (not
        autoregressively), consistent with paper Appendix J.1.

        Args:
            u0: Initial condition, shape [batch, spatial_size] = [batch, 120].
                dtype=float32.
            f: Force term, shape [batch, T, spatial_size] = [batch, 80, 120].
                dtype=float32.
            x_grid: Optional spatial grid, shape [spatial_size] or
                [batch, spatial_size]. If None, creates linspace(0,1,spatial_size).
                dtype=float32.

        Returns:
            Predicted state trajectory, shape [batch, T, spatial_size].
            For out_channels=1: [batch, 80, 120] (T prediction steps).
            dtype=float32.

        Note:
            When out_channels=1, the FNO predicts one output channel per
            spatial point. To get the full T-step trajectory, the model
            is called T times or the output is interpreted as the mean
            prediction. For simplicity, we call the model once with the
            mean force encoding and return the single-step prediction
            repeated T times. For a proper multi-step FNO, set
            out_channels=T (number of prediction timesteps).
        """
        batch_size: int = u0.shape[0]
        spatial_size: int = u0.shape[-1]
        T: int = f.shape[1]
        device: torch.device = u0.device

        # Build spatial grid if not provided
        if x_grid is None:
            x_grid = torch.linspace(0.0, 1.0, spatial_size, device=device)

        # Expand x_grid to [batch, spatial_size]
        if x_grid.dim() == 1:
            x_grid_batch: torch.Tensor = x_grid.unsqueeze(0).expand(batch_size, -1)
        else:
            x_grid_batch = x_grid

        # Encode force as mean over time dimension: [batch, spatial_size]
        f_mean: torch.Tensor = f.mean(dim=1)  # [batch, spatial_size]

        # Concatenate input channels: [batch, 3, spatial_size]
        # Channel 0: u0 (initial condition)
        # Channel 1: f_mean (mean force over time)
        # Channel 2: x_grid (spatial coordinate)
        x_input: torch.Tensor = torch.stack(
            [u0, f_mean, x_grid_batch], dim=1
        )  # [batch, 3, spatial_size]

        # Forward pass: [batch, out_channels, spatial_size]
        out: torch.Tensor = self.forward(x_input)

        # If out_channels == 1, repeat prediction for all T timesteps
        if self.out_channels == 1:
            # [batch, 1, spatial_size] → [batch, T, spatial_size]
            trajectory: torch.Tensor = out.expand(-1, T, -1)
        else:
            # out_channels == T: reshape to [batch, T, spatial_size]
            # This case requires out_channels to equal T during construction
            trajectory = out.permute(0, 2, 1)  # [batch, spatial_size, out_channels]
            # Reinterpret: [