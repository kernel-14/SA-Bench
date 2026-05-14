```python
## baselines/wno.py
"""Wavelet Neural Operator (WNO) baseline for WDNO comparison.

Implements WNO (Tripura & Chakraborty 2022) for 1D and 2D PDE simulation
experiments. WNO replaces FNO's Fourier layers with wavelet layers, providing
better localization for capturing abrupt changes. Used as a direct comparison
baseline to WDNO in Table 1, Table 5, Table 16, and Table 17.

Key distinction from WDNO: WNO learns a deterministic operator mapping
(no diffusion), while WDNO learns a generative distribution in wavelet space.

Paper sources:
    - WNO: Tripura & Chakraborty 2022 (Wavelet Neural Operator)
    - 1D hyperparameters: Appendix J.2, Table 27
    - 2D hyperparameters: Appendix K.2, Table 33
    - SR limitation: Section 4.6 ("Due to WNO's implementation, it can only
      perform spatiotemporal super-resolution simultaneously, making it
      unsuitable for 2D super-resolution experiments.")
    - Results: Table 1 (MSE), Table 16/17 (SR), Table 5 (extended NS)

Config references:
    - baselines.wno_1d.wavelet_type: sym4 (Burgers), bior2.4 (compressible NS)
    - baselines.wno_1d.level: 5
    - baselines.wno_1d.width: 40
    - baselines.wno_1d.n_layers: 4
    - baselines.wno_1d.optimizer: adam
    - baselines.wno_1d.learning_rate: 1e-3
    - baselines.wno_1d.train_epochs: 1000
    - baselines.wno_1d.batch_size: 100
    - baselines.wno_1d.lr_scheduler: steplr
    - baselines.wno_2d.wavelet_type: bior1.3
    - baselines.wno_2d.level: 2
    - baselines.wno_2d.width: 8
    - baselines.wno_2d.n_layers: 3
    - baselines.wno_2d.optimizer: adam
    - baselines.wno_2d.learning_rate: 0.05
    - baselines.wno_2d.train_epochs: 500
    - baselines.wno_2d.batch_size: 50
    - baselines.wno_2d.lr_scheduler: steplr
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyperparameters from config.yaml baselines.wno_1d / wno_2d
# ---------------------------------------------------------------------------

# 1D WNO defaults (config: baselines.wno_1d)
_DEFAULT_WAVELET_1D: str = "sym4"
_DEFAULT_LEVEL_1D: int = 5
_DEFAULT_WIDTH_1D: int = 40
_DEFAULT_N_LAYERS_1D: int = 4
_DEFAULT_LR_1D: float = 1e-3
_DEFAULT_EPOCHS_1D: int = 1000
_DEFAULT_BATCH_SIZE_1D: int = 100
_DEFAULT_STEPLR_STEP_SIZE: int = 100
_DEFAULT_STEPLR_GAMMA: float = 0.5

# 2D WNO defaults (config: baselines.wno_2d)
_DEFAULT_WAVELET_2D: str = "bior1.3"
_DEFAULT_LEVEL_2D: int = 2
_DEFAULT_WIDTH_2D: int = 8
_DEFAULT_N_LAYERS_2D: int = 3
_DEFAULT_LR_2D: float = 0.05
_DEFAULT_EPOCHS_2D: int = 500
_DEFAULT_BATCH_SIZE_2D: int = 50

# Padding modes matching WDNO config
_PADDING_MODE_1D: str = "periodization"
_PADDING_MODE_2D: str = "zero"

# Hidden dimension for projection MLP
_PROJECTION_HIDDEN_DIM: int = 128

# Number of detail subbands per level for 2D DWT (LH, HL, HH)
_NUM_2D_SUBBANDS: int = 3

# Fixed ordering of 3D DWT detail subband keys (ptwt convention)
_PTWT_DETAIL_KEYS: Tuple[str, ...] = (
    "aad",
    "ada",
    "add",
    "daa",
    "dad",
    "dda",
    "ddd",
)
_NUM_3D_SUBBANDS: int = len(_PTWT_DETAIL_KEYS)  # 7


# ---------------------------------------------------------------------------
# WaveletLayer: 1D experiments (2D DWT via pytorch_wavelets)
# ---------------------------------------------------------------------------


class WaveletLayer(nn.Module):
    """Wavelet layer for 1D PDE experiments using pytorch_wavelets 2D DWT.

    Operates on feature maps of shape [B, C, T, X] (treating the time-space
    data as a 2D image). Applies a multi-level 2D DWT, performs learned
    channel-mixing on each coefficient band, then reconstructs via IDWT.

    The bypass path (local linear transform) is a 1×1 Conv2d that handles
    the non-wavelet-representable component, analogous to FNO's W operator.

    Architecture per layer:
        output = IDWT(W_low * DWT_low(x), W_high_l * DWT_high_l(x) for l in levels)
               + Conv2d_bypass(x)

    Attributes:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        wavelet: Wavelet basis string (e.g., 'sym4', 'bior2.4').
        level: Number of DWT decomposition levels (5 for 1D WNO).
        dwt: DWTForward instance from pytorch_wavelets.
        idwt: DWTInverse instance from pytorch_wavelets.
        weights_low: Learnable weights for low-frequency coefficients.
            Shape [in_channels, out_channels].
        weights_high: ParameterList of level tensors for high-frequency
            coefficients. Each tensor has shape [in_channels, out_channels, 3]
            (3 subbands: LH, HL, HH per level).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        wavelet: str = _DEFAULT_WAVELET_1D,
        level: int = _DEFAULT_LEVEL_1D,
        padding_mode: str = _PADDING_MODE_1D,
    ) -> None:
        """Initialize the WaveletLayer for 1D PDE experiments.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            wavelet: Wavelet basis string. Config: baselines.wno_1d.wavelet_type.
                'sym4' for Burgers', 'bior2.4' for compressible NS.
            level: Number of DWT decomposition levels.
                Config: baselines.wno_1d.level=5.
            padding_mode: DWT padding mode. 'periodization' for 1D PDE
                experiments (consistent with WDNO config).

        Raises:
            ImportError: If pytorch_wavelets is not installed.
        """
        super().__init__()

        try:
            from pytorch_wavelets import DWTForward, DWTInverse
        except ImportError as exc:
            raise ImportError(
                "pytorch_wavelets is required for WNO 1D experiments. "
                "Install with: pip install pytorch-wavelets"
            ) from exc

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.wavelet: str = wavelet
        self.level: int = level
        self.padding_mode: str = padding_mode

        # DWT forward and inverse transforms
        # J=level applies level decomposition levels
        self.dwt: nn.Module = DWTForward(
            J=level,
            wave=wavelet,
            mode=padding_mode,
        )
        self.idwt: nn.Module = DWTInverse(
            wave=wavelet,
            mode=padding_mode,
        )

        # Learnable weights for low-frequency (approximation) coefficients
        # Applied as channel mixing: [B, C_in, T_c, X_c] → [B, C_out, T_c, X_c]
        # Shape [in_channels, out_channels] for einsum 'bci...,co->boi...'
        scale: float = 1.0 / (in_channels * out_channels)
        self.weights_low: nn.Parameter = nn.Parameter(
            scale * torch.randn(in_channels, out_channels)
        )

        # Learnable weights for high-frequency (detail) coefficients
        # One weight tensor per decomposition level
        # Each tensor shape: [in_channels, out_channels, 3] (3 subbands: LH, HL, HH)
        self.weights_high: nn.ParameterList = nn.ParameterList([
            nn.Parameter(scale * torch.randn(in_channels, out_channels, _NUM_2D_SUBBANDS))
            for _ in range(level)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply wavelet layer to feature map.

        Args:
            x: Input feature map of shape [B, C_in, T, X]. dtype=float32.
                The (T, X) dimensions are treated as a 2D spatial domain
                for the DWT.

        Returns:
            Output feature map of shape [B, C_out, T, X]. dtype=float32.
            The spatial dimensions are preserved by the DWT/IDWT pair.
        """
        # Move DWT modules to same device as input
        device: torch.device = x.device
        self.dwt = self.dwt.to(device)
        self.idwt = self.idwt.to(device)

        # Apply 2D DWT: x [B, C_in, T, X] → (Yl, Yh_list)
        # Yl: [B, C_in, T/2^J, X/2^J]
        # Yh_list[j]: [B, C_in, 3, T/2^(j+1), X/2^(j+1)] for j=0..J-1
        Yl, Yh_list = self.dwt(x)

        # --- Transform low-frequency coefficients ---
        # Yl: [B, C_in, T_c, X_c]
        # weights_low: [C_in, C_out]
        # Output: [B, C_out, T_c, X_c]
        Yl_out: torch.Tensor = torch.einsum(
            "bcij,co->boij", Yl, self.weights_low
        )

        # --- Transform high-frequency coefficients for each level ---
        Yh_out_list: List[torch.Tensor] = []
        for j, Yh_j in enumerate(Yh_list):
            # Yh_j: [B, C_in, 3, T_j, X_j]
            # weights_high[j]: [C_in, C_out, 3]
            # Output: [B, C_out, 3, T_j, X_j]
            # Einsum: 'bcnij,con->bonij' where n=3 subbands
            Yh_j_out: torch.Tensor = torch.einsum(
                "bcnij,con->bonij", Yh_j, self.weights_high[j]
            )
            Yh_out_list.append(Yh_j_out)

        # --- Reconstruct via IDWT ---
        # IDWT expects (Yl, Yh_list) format
        out: torch.Tensor = self.idwt((Yl_out, Yh_out_list))

        # Crop to original spatial size (DWT may add padding)
        if out.shape[-2] != x.shape[-2] or out.shape[-1] != x.shape[-1]:
            out = out[..., : x.shape[-2], : x.shape[-1]]

        return out


# ---------------------------------------------------------------------------
# WaveletLayer3D: 2D experiments (3D DWT via ptwt)
# ---------------------------------------------------------------------------


class WaveletLayer3D(nn.Module):
    """Wavelet layer for 2D PDE experiments using ptwt 3D DWT.

    Operates on feature maps of shape [B, C, T, H, W]. Applies a multi-level
    3D DWT, performs learned channel-mixing on each coefficient band, then
    reconstructs via inverse 3D DWT.

    Attributes:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        wavelet: Wavelet basis string (e.g., 'bior1.3').
        level: Number of DWT decomposition levels (2 for 2D WNO).
        padding_mode: DWT padding mode ('zero' for 2D experiments).
        weights_low: Learnable weights for low-frequency coefficients.
            Shape [in_channels, out_channels].
        weights_high: ParameterList of level tensors for high-frequency
            coefficients. Each tensor has shape [in_channels, out_channels, 7]
            (7 subbands per level for 3D DWT).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        wavelet: str = _DEFAULT_WAVELET_2D,
        level: int = _DEFAULT_LEVEL_2D,
        padding_mode: str = _PADDING_MODE_2D,
    ) -> None:
        """Initialize the WaveletLayer3D for 2D PDE experiments.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            wavelet: Wavelet basis string. Config: baselines.wno_2d.wavelet_type='bior1.3'.
            level: Number of DWT decomposition levels.
                Config: baselines.wno_2d.level=2.
            padding_mode: DWT padding mode. 'zero' for 2D PDE experiments.

        Raises:
            ImportError: If ptwt is not installed.
        """
        super().__init__()

        try:
            import ptwt  # noqa: F401 — verify import works
        except ImportError as exc:
            raise ImportError(
                "ptwt (pytorch-wavelet-toolbox) is required for WNO 2D experiments. "
                "Install with: pip install ptwt"
            ) from exc

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.wavelet: str = wavelet
        self.level: int = level
        self.padding_mode: str = padding_mode

        # Learnable weights for low-frequency (approximation) coefficients
        # Shape [in_channels, out_channels] for channel mixing
        scale: float = 1.0 / (in_channels * out_channels)
        self.weights_low: nn.Parameter = nn.Parameter(
            scale * torch.randn(in_channels, out_channels)
        )

        # Learnable weights for high-frequency (detail) coefficients
        # One weight tensor per decomposition level
        # Each tensor shape: [in_channels, out_channels, 7] (7 subbands for 3D DWT)
        self.weights_high: nn.ParameterList = nn.ParameterList([
            nn.Parameter(
                scale * torch.randn(in_channels, out_channels, _NUM_3D_SUBBANDS)
            )
            for _ in range(level)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 3D wavelet layer to feature map.

        Args:
            x: Input feature map of shape [B, C_in, T, H, W]. dtype=float32.

        Returns:
            Output feature map of shape [B, C_out, T, H, W]. dtype=float32.
        """
        import ptwt

        B: int = x.shape[0]
        C: int = x.shape[1]
        T: int = x.shape[2]
        H: int = x.shape[3]
        W: int = x.shape[4]

        # ptwt requires [B, C, D, H, W] format — our data is already in this format
        # Apply 3D DWT: returns [cA, detail_dict_level_1, detail_dict_level_2, ...]
        # cA: [B, C, T_c, H_c, W_c]
        # detail_dict: {key: [B, C, T_c, H_c, W_c]} for each of 7 subband keys
        coeffs = ptwt.wavedec3(
            x,
            wavelet=self.wavelet,
            level=self.level,
            mode=self.padding_mode,
        )

        # coeffs[0]: low-frequency approximation [B, C, T_c, H_c, W_c]
        # coeffs[1..level]: detail dicts at each level

        # --- Transform low-frequency coefficients ---
        cA: torch.Tensor = coeffs[0]  # [B, C_in, T_c, H_c, W_c]
        # weights_low: [C_in, C_out]
        # Output: [B, C_out, T_c, H_c, W_c]
        cA_out: torch.Tensor = torch.einsum(
            "bctij,co->botij", cA, self.weights_low
        )

        # --- Transform high-frequency coefficients for each level ---
        detail_out_list: List[Dict[str, torch.Tensor]] = []
        for j in range(self.level):
            detail_dict: Dict[str, torch.Tensor] = coeffs[j + 1]
            detail_out_dict: Dict[str, torch.Tensor] = {}

            for k, key in enumerate(_PTWT_DETAIL_KEYS):
                if key in detail_dict:
                    # detail_dict[key]: [B, C_in, T_c, H_c, W_c]
                    # weights_high[j]: [C_in, C_out, 7]
                    # We use the k-th subband weight
                    w_k: torch.Tensor = self.weights_high[j][:, :, k]  # [C_in, C_out]
                    detail_out_dict[key] = torch.einsum(
                        "bctij,co->botij", detail_dict[key], w_k
                    )
                else:
                    logger.warning(
                        "WaveletLayer3D: key '%s' not found in detail_dict at level %d. "
                        "Available keys: %s",
                        key, j, list(detail_dict.keys()),
                    )

            detail_out_list.append(detail_out_dict)

        # --- Reconstruct via inverse 3D DWT ---
        # ptwt expects [cA, detail_dict_1, detail_dict_2, ...]
        ptwt_coeffs_out = [cA_out] + detail_out_list
        out: torch.Tensor = ptwt.waverec3(ptwt_coeffs_out, wavelet=self.wavelet)

        # Crop to original spatial size (padding may add extra elements)
        if (out.shape[2] != T or out.shape[3] != H or out.shape[4] != W):
            out = out[:, :, :T, :H, :W]

        return out


# ---------------------------------------------------------------------------
# WNO: 1D PDE experiments
# ---------------------------------------------------------------------------


class WNO(nn.Module):
    """Wavelet Neural Operator for 1D PDE simulation experiments.

    Implements the WNO architecture from Tripura & Chakraborty 2022 for
    1D PDE experiments (Burgers', advection, compressible NS). Operates
    directly on raw space-time data [B, C, T, X] without a diffusion model.

    Architecture:
        Lifting (Linear) → n_layers × (WaveletLayer + Conv2d_bypass + GELU)
        → Projection MLP

    The lifting layer projects from input channels to the internal width.
    Each WNO block combines a wavelet-domain transform with a local bypass.
    The projection MLP maps from width to output channels.

    Input convention (paper Appendix J.2):
        "we train the FNO model using the initial state and all controls as
        the input and using the rest states as the output."
        Input: [u_0 repeated, f, x_grid, t_grid] → [B, 4, T, X]
        Output: predicted u[1:T] → [B, 1, T, X]

    Zero-shot super-resolution:
        WNO is mesh-invariant — the same trained model can be evaluated at
        different spatial resolutions. Used as SR baseline in Table 16.
        However, WNO can only perform spatiotemporal SR simultaneously
        (paper Section 4.6), limiting its 2D SR applicability.

    Attributes:
        wavelet: Wavelet basis string. Config: baselines.wno_1d.wavelet_type.
        level: DWT decomposition levels. Config: baselines.wno_1d.level=5.
        width: Internal channel dimension. Config: baselines.wno_1d.width=40.
        n_layers: Number of WNO blocks. Config: baselines.wno_1d.n_layers=4.
        in_channels: Input channels (including grid coordinates).
        out_channels: Output channels (1 for single-variable prediction).
        lifting: Linear layer projecting in_channels → width.
        wavelet_layers: ModuleList of n_layers WaveletLayer instances.
        bypass_convs: ModuleList of n_layers Conv2d(width, width, 1) bypass layers.
        activation: GELU activation.
        projection: Sequential MLP projecting width → out_channels.
    """

    def __init__(
        self,
        wavelet: str = _DEFAULT_WAVELET_1D,
        level: int = _DEFAULT_LEVEL_1D,
        width: int = _DEFAULT_WIDTH_1D,
        n_layers: int = _DEFAULT_N_LAYERS_1D,
        in_channels: int = 4,
        out_channels: int = 1,
        padding_mode: str = _PADDING_MODE_1D,
    ) -> None:
        """Initialize the WNO for 1D PDE experiments.

        Args:
            wavelet: Wavelet basis string. Config: baselines.wno_1d.wavelet_type.
                'sym4' for Burgers' (default), 'bior2.4' for compressible NS
                (per paper Appendix J.2 note).
            level: Number of DWT decomposition levels.
                Config: baselines.wno_1d.level=5.
            width: Internal channel dimension (lifted representation).
                Config: baselines.wno_1d.width=40.
            n_layers: Number of WNO blocks (wavelet layers).
                Config: baselines.wno_1d.n_layers=4.
            in_channels: Number of input channels. Default 4 includes:
                u_0 (1ch), f (1ch), x_grid (1ch), t_grid (1ch).
                Set to 2 if grid coordinates are not appended.
            out_channels: Number of output channels. 1 for single-variable
                prediction (u state). Config: baselines.wno_1d.out_channels
                (not explicitly stated; inferred as 1).
            padding_mode: DWT padding mode. 'periodization' for 1D PDEs.
        """
        super().__init__()

        self.wavelet: str = wavelet
        self.level: int = level
        self.width: int = width
        self.n_layers: int = n_layers
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.padding_mode: str = padding_mode

        # -----------------------------------------------------------------------
        # Lifting: projects in_channels → width
        # Applied pointwise (1×1 conv equivalent via Linear on channel dim)
        # -----------------------------------------------------------------------
        self.lifting: nn.Conv2d = nn.Conv2d(
            in_channels, width, kernel_size=1
        )

        # -----------------------------------------------------------------------
        # WNO blocks: n_layers × (WaveletLayer + Conv2d_bypass)
        # Config: baselines.wno_1d.n_layers=4
        # -----------------------------------------------------------------------
        self.wavelet_layers: nn.ModuleList = nn.ModuleList([
            WaveletLayer(
                in_channels=width,
                out_channels=width,
                wavelet=wavelet,
                level=level,
                padding_mode=padding_mode,
            )
            for _ in range(n_layers)
        ])

        # Bypass path: local linear transform (1×1 conv)
        # Handles non-wavelet-representable components (analogous to FNO's W)
        self.bypass_convs: nn.ModuleList = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1)
            for _ in range(n_layers)
        ])

        # Activation function (GELU per FNO/WNO convention)
        self.activation: nn.Module = nn.GELU()

        # -----------------------------------------------------------------------
        # Projection MLP: width → hidden → out_channels
        # Applied pointwise along spatial dimensions
        # -----------------------------------------------------------------------
        self.projection: nn.Sequential = nn.Sequential(
            nn.Linear(width, _PROJECTION_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(_PROJECTION_HIDDEN_DIM, out_channels),
        )

        logger.info(
            "WNO initialized: wavelet=%s, level=%d, width=%d, n_layers=%d, "
            "in_channels=%d, out_channels=%d, padding_mode=%s",
            wavelet, level, width, n_layers, in_channels, out_channels, padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the WNO.

        Args:
            x: Input tensor of shape [B, in_channels, T, X]. dtype=float32.
                Channels typically include [u_0, f, x_grid, t_grid].

        Returns:
            Output tensor of shape [B, out_channels, T, X]. dtype=float32.
            For simulation: predicted state trajectory u[0:T].
        """
        # -----------------------------------------------------------------------
        # Lifting: [B, in_channels, T, X] → [B, width, T, X]
        # -----------------------------------------------------------------------
        h: torch.Tensor = self.lifting(x)

        # -----------------------------------------------------------------------
        # WNO blocks: n_layers × (WaveletLayer + bypass + GELU)
        # -----------------------------------------------------------------------
        for i in range(self.n_layers):
            h_wavelet: torch.Tensor = self.wavelet_layers[i](h)
            h_bypass: torch.Tensor = self.bypass_convs[i](h)
            h = self.activation(h_wavelet + h_bypass)

        # -----------------------------------------------------------------------
        # Projection: [B, width, T, X] → [B, out_channels, T, X]
        # Apply MLP pointwise: permute to [B, T, X, width] → MLP → permute back
        # -----------------------------------------------------------------------
        # [B, width, T, X] → [B, T, X, width]
        h_perm: torch.Tensor = h.permute(0, 2, 3, 1)
        # Apply projection MLP: [B, T, X, width] → [B, T, X, out_channels]
        out_perm: torch.Tensor = self.projection(h_perm)
        # [B, T, X, out_channels] → [B, out_channels, T, X]
        out: torch.Tensor = out_perm.permute(0, 3, 1, 2)

        return out

    def simulate(
        self,
        u0: torch.Tensor,
        f: torch.Tensor,
        append_grid: bool = True,
    ) -> torch.Tensor:
        """Run simulation inference to predict full state trajectory.

        Prepares the input by concatenating u_0 (repeated across time),
        force f, and optionally spatial/temporal grid coordinates, then
        runs a single forward pass.

        Args:
            u0: Initial condition, shape [B, X] = [B, 120]. dtype=float32.
            f: Force term, shape [B, T, X] = [B, 80, 120]. dtype=float32.
            append_grid: If True, append x_grid and t_grid as additional
                input channels (increases in_channels by 2). Default True
                following WNO paper convention.

        Returns:
            Predicted state trajectory, shape [B, T, X] = [B, 80, 120].
            dtype=float32. Excludes the initial condition (t=0).
        """
        B: int = u0.shape[0]
        X: int = u0.shape[-1]
        T: int = f.shape[1]
        device: torch.device = u0.device

        # Repeat u_0 across time dimension: [B, X] → [B, T, X]
        u0_repeated: torch.Tensor = u0.unsqueeze(1).expand(-1, T, -1)
        # [B, T, X]

        # Stack channels: [B, 2, T, X] = [u_0_repeated, f]
        channels: List[torch.Tensor] = [
            u0_repeated,  # [B, T, X]
            f,            # [B, T, X]
        ]

        if append_grid:
            # Spatial grid: x in [0, 1], shape [B, T, X]
            x_grid: torch.Tensor = torch.linspace(
                0.0, 1.0, X, device=device
            ).view(1, 1, X).expand(B, T, X)

            # Temporal grid: t in [0, 1], shape [B, T, X]
            t_grid: torch.Tensor = torch.linspace(
                0.0, 1.0, T, device=device
            ).view(1, T, 1).expand(B, T, X)

            channels.extend([x_grid, t_grid])

        # Stack along channel dimension: [B, n_channels, T, X]
        x_input: torch.Tensor = torch.stack(channels, dim=1)
        # x_input: [B, 2 or 4, T, X]

        # Forward pass: [B, n_channels, T, X] → [B, out_channels, T, X]
        out: torch.Tensor = self.forward(x_input)

        # Return predicted trajectory: [B, out_channels, T, X]
        # If out_channels == 1, squeeze channel dim: [B, T, X]
        if self.out_channels == 1:
            return out.squeeze(1)  # [B, T, X]
        else:
            return out  