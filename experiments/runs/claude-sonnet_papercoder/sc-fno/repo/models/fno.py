## Code: models/fno.py

```python
## models/fno.py
"""Fourier Neural Operator (FNO) backbone for SC-FNO experiments.

Implements the shared FNO architecture used by all four model variants:
  - FNO:          data loss L_u only
  - SC-FNO:       L_u + sensitivity loss L_s
  - FNO-PINN:     L_u + equation loss L_Eq
  - SC-FNO-PINN:  L_u + L_s + L_Eq

The architecture is identical across all variants — only the loss configuration
differs. The variant tag (stored as self.variant) is read by the Trainer to
determine which losses to activate.

Three operational modes are supported, driven by cfg['equation']:
  - Mode A (dim=1): ODEs (ODE1, ODE2) — 1D temporal sequences
  - Mode B (dim=2, spatiotemporal): 1D PDEs (PDE1, PDE2, PDE4) — (x,t) grids
  - Mode C (dim=2, spatial): PDE3 (Navier-Stokes) — (x,y) spatial grids

Hyperparameters from Table C.7 of the SC-FNO paper:
  - width = 20 (number of channels in hidden layers)
  - n_fourier_layers = 4
  - modes = 8 for all dimensions
  - learning_rate = 0.001

References:
    - Li et al. (2021): "Fourier Neural Operator for Parametric Partial
      Differential Equations" (https://arxiv.org/abs/2010.08895)
    - Paper Table C.7: Hyperparameters for FNOs
    - Paper Section 2.4: Implementation Details
    - config.yaml: model.width=20, model.n_fourier_layers=4
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from models.spectral_conv import SpectralConv1d, SpectralConv2d


# ---------------------------------------------------------------------------
# FNO building blocks — defined here per the design (sc_fno.py imports them)
# ---------------------------------------------------------------------------

class FNOBlock1d(nn.Module):
    """Single Fourier layer for 1D sequences (ODEs and 1D temporal inputs).

    Implements one FNO layer:
        v_{t+1}(x) = σ(W·v_t(x) + F⁻¹(R_φ · F(v_t))(x))

    where W is a pointwise linear (skip connection) and R_φ is the learnable
    spectral convolution. The activation σ is GELU.

    Attributes:
        spectral_conv: SpectralConv1d layer applying the Fourier integral operator.
        linear: Pointwise Conv1d (kernel_size=1) as the skip connection W.
        activation: GELU activation function.

    Example:
        >>> block = FNOBlock1d(width=20, modes=8)
        >>> x = torch.randn(4, 20, 100)  # [B, C, L]
        >>> out = block(x)
        >>> out.shape  # [4, 20, 100]
    """

    def __init__(self, width: int, modes: int) -> None:
        """Initializes FNOBlock1d.

        Args:
            width: Number of feature channels (FNO hidden dimension).
                   From config.yaml model.width = 20.
            modes: Number of Fourier modes for the spectral convolution.
                   From config.yaml modes_t = 8 (Table C.7).
        """
        super().__init__()

        self.spectral_conv: SpectralConv1d = SpectralConv1d(
            in_channels=width,
            out_channels=width,
            modes=modes,
        )

        # Pointwise linear skip connection: Conv1d with kernel_size=1 is
        # equivalent to a linear layer applied independently at each position.
        self.linear: nn.Conv1d = nn.Conv1d(
            in_channels=width,
            out_channels=width,
            kernel_size=1,
        )

        self.activation: nn.Module = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies one FNO layer: σ(spectral_conv(x) + linear(x)).

        Args:
            x: Input tensor of shape [B, width, L].

        Returns:
            Output tensor of shape [B, width, L]. Same shape as input.
        """
        # Spectral path: F⁻¹(R_φ · F(v_t))
        x_spectral: torch.Tensor = self.spectral_conv(x)  # [B, width, L]

        # Skip connection: pointwise linear W·v_t
        x_linear: torch.Tensor = self.linear(x)  # [B, width, L]

        # Sum and apply activation.
        return self.activation(x_spectral + x_linear)


class FNOBlock2d(nn.Module):
    """Single Fourier layer for 2D grids (1D PDEs and PDE3).

    Implements one FNO layer for 2D inputs:
        v_{t+1}(x) = σ(W·v_t(x) + F⁻¹(R_φ · F(v_t))(x))

    where W is a pointwise Conv2d (kernel_size=1) and R_φ is the 2D spectral
    convolution. The activation σ is GELU.

    Attributes:
        spectral_conv: SpectralConv2d layer applying the 2D Fourier integral operator.
        linear: Pointwise Conv2d (kernel_size=1) as the skip connection W.
        activation: GELU activation function.

    Example:
        >>> block = FNOBlock2d(width=20, modes1=8, modes2=8)
        >>> x = torch.randn(4, 20, 64, 64)  # [B, C, H, W]
        >>> out = block(x)
        >>> out.shape  # [4, 20, 64, 64]
    """

    def __init__(self, width: int, modes1: int, modes2: int) -> None:
        """Initializes FNOBlock2d.

        Args:
            width: Number of feature channels (FNO hidden dimension).
                   From config.yaml model.width = 20.
            modes1: Number of Fourier modes in the first spatial dimension.
                    From config.yaml modes_x = 8 (Table C.7).
            modes2: Number of Fourier modes in the second spatial dimension.
                    From config.yaml modes_t = 8 or modes_y = 8 (Table C.7).
        """
        super().__init__()

        self.spectral_conv: SpectralConv2d = SpectralConv2d(
            in_channels=width,
            out_channels=width,
            modes1=modes1,
            modes2=modes2,
        )

        # Pointwise linear skip connection: Conv2d with kernel_size=1.
        self.linear: nn.Conv2d = nn.Conv2d(
            in_channels=width,
            out_channels=width,
            kernel_size=1,
        )

        self.activation: nn.Module = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies one 2D FNO layer: σ(spectral_conv(x) + linear(x)).

        Args:
            x: Input tensor of shape [B, width, H, W].

        Returns:
            Output tensor of shape [B, width, H, W]. Same shape as input.
        """
        # Spectral path: F⁻¹(R_φ · F(v_t))
        x_spectral: torch.Tensor = self.spectral_conv(x)  # [B, width, H, W]

        # Skip connection: pointwise linear W·v_t
        x_linear: torch.Tensor = self.linear(x)  # [B, width, H, W]

        # Sum and apply activation.
        return self.activation(x_spectral + x_linear)


# ---------------------------------------------------------------------------
# Main FNO class
# ---------------------------------------------------------------------------

class FNO(nn.Module):
    """Fourier Neural Operator backbone shared by all SC-FNO model variants.

    The architecture is identical across FNO, SC-FNO, FNO-PINN, and SC-FNO-PINN.
    The variant tag (self.variant) is set by the build_model factory in sc_fno.py
    and read by the Trainer to determine which losses to activate.

    Architecture:
        1. Lifting layer: Linear(d_in → width)
        2. n_fourier_layers FNO blocks (FNOBlock1d or FNOBlock2d)
        3. Projection: Linear(width → 128) → GELU → Linear(128 → d_out=1)

    Input construction (_build_input):
        Parameters p are broadcast to the full spatial-temporal grid and
        concatenated with coordinate grids and initial conditions before
        the lifting layer. This is the same for all four model variants.

    Attributes:
        cfg: The equation-specific configuration sub-dict.
        equation: Equation identifier string (e.g., 'ode1', 'pde2', 'pde3').
        dim: Spatial-temporal dimensionality of the FNO (1 or 2).
        width: Number of hidden channels (20 from Table C.7).
        n_layers: Number of Fourier layers (4 from Table C.7).
        n_params: Number of physical parameters for this equation.
        M: Number of input time steps (from config discretization).
        N: Total number of time steps (from config discretization).
        modes1: Fourier modes for first dimension (8 from Table C.7).
        modes2: Fourier modes for second dimension (8 from Table C.7).
        d_in: Number of input channels to the lifting layer.
        d_out: Number of output channels (always 1 for scalar fields).
        normalize_params: Whether to normalize parameters to [0,1].
        param_ranges: List of [a, b] bounds for parameter normalization.
        is_pde3: True if equation is 'pde3' (single output time step).
        lifting: nn.Linear(d_in → width).
        fno_blocks: nn.ModuleList of FNOBlock1d or FNOBlock2d instances.
        projection: nn.Sequential projection to output channels.
        variant: Model variant tag ('fno', 'sc_fno', 'fno_pinn', 'sc_fno_pinn').

    Example:
        >>> from utils.config_loader import ConfigLoader
        >>> cfg_loader = ConfigLoader('config.yaml')
        >>> cfg = cfg_loader.cfg
        >>> # Build FNO for PDE1
        >>> pde1_cfg = cfg['pde1']
        >>> pde1_cfg['equation'] = 'pde1'
        >>> pde1_cfg['n_params'] = 5
        >>> model = FNO(pde1_cfg)
        >>> model.count_parameters()  # Should be ~107,897
        >>> params = torch.randn(4, 5)
        >>> u0 = torch.randn(4, 5, 20)   # [B, M, Sx]
        >>> coords = torch.randn(4, 25, 20, 2)  # [B, T_out, Sx, 2]
        >>> u_pred = model(params, u0, coords)
        >>> u_pred.shape  # [4, 25, 20]
    """

    def __init__(self, cfg: dict) -> None:
        """Initializes the FNO from an equation-specific configuration dict.

        Reads all hyperparameters from cfg — no hardcoded values. Builds the
        lifting layer, FNO blocks, and projection layer based on the equation
        type and discretization settings.

        Args:
            cfg: Equation-specific configuration sub-dict. Must contain:
                 - 'equation': str, e.g. 'ode1', 'pde1', 'pde3'
                 - 'n_params': int, number of physical parameters
                 - 'model.dim': int, 1 for ODEs, 2 for PDEs
                 - 'model.width': int (default 20, Table C.7)
                 - 'model.n_fourier_layers': int (default 4, Table C.7) —
                   falls back to global config key if not in equation sub-dict
                 - 'model.modes_t': int (default 8, Table C.7)
                 - 'model.modes_x': int (default 8, Table C.7)
                 - 'model.modes_y': int (default 8, Table C.7, PDE3 only)
                 - 'discretization.M': int, input time steps
                 - 'discretization.N': int, total time steps
                 - 'params': dict mapping param_name -> [a, b] (for normalization)
                 Optional:
                 - 'model.normalize_params': bool (default True)
                 - 'model.activation': str (default 'gelu')

        Raises:
            ValueError: If equation type is unrecognized or required config
                        keys are missing.
        """
        super().__init__()

        self.cfg: dict = cfg

        # ------------------------------------------------------------------
        # Read equation identifier and core settings.
        # ------------------------------------------------------------------
        self.equation: str = str(cfg.get("equation", "pde1")).lower()
        self.n_params: int = int(cfg.get("n_params", 5))

        # ------------------------------------------------------------------
        # Read model architecture hyperparameters (Table C.7).
        # Support both flat cfg (e.g., cfg['model']['width']) and nested
        # access for equation sub-configs that embed model settings.
        # ------------------------------------------------------------------
        model_cfg: dict = cfg.get("model", {})

        self.dim: int = int(model_cfg.get("dim", 2))
        self.width: int = int(model_cfg.get("width", 20))
        self.n_layers: int = int(model_cfg.get("n_fourier_layers", 4))

        # Fourier modes per dimension (Table C.7: all 8).
        self.modes1: int = int(model_cfg.get("modes_t", model_cfg.get("modes_x", 8)))
        self.modes2: int = int(model_cfg.get("modes_x", model_cfg.get("modes_y", 8)))

        # For PDE3, modes are spatial (x, y) not temporal.
        if self.equation == "pde3":
            self.modes1 = int(model_cfg.get("modes_x", 8))
            self.modes2 = int(model_cfg.get("modes_y", 8))

        # ------------------------------------------------------------------
        # Read discretization parameters.
        # ------------------------------------------------------------------
        disc_cfg: dict = cfg.get("discretization", {})
        self.M: int = int(disc_cfg.get("M", 5))
        self.N: int = int(disc_cfg.get("N", 30))

        # PDE3 flag: single output time step (vorticity at t=3 only).
        self.is_pde3: bool = (self.equation == "pde3") or bool(
            disc_cfg.get("output_single_timestep", False)
        )

        # ------------------------------------------------------------------
        # Parameter normalization settings.
        # Normalizing parameters to [0,1] improves training stability when
        # parameters span very different ranges (e.g., γ∈[20,60] for ODE2).
        # ------------------------------------------------------------------
        self.normalize_params: bool = bool(
            model_cfg.get("normalize_params", True)
        )

        # Extract parameter ranges for normalization.
        params_cfg: dict = cfg.get("params", {})
        self.param_ranges: List[List[float]] = [
            [float(v[0]), float(v[1])] for v in params_cfg.values()
        ]

        # Guard: if param_ranges is empty or mismatched, disable normalization.
        if len(self.param_ranges) != self.n_params:
            self.normalize_params = False

        # ------------------------------------------------------------------
        # Compute d_in: number of input channels to the lifting layer.
        # ------------------------------------------------------------------
        self.d_in: int = self._compute_d_in()
        self.d_out: int = 1  # Scalar solution field (u or ω).

        # ------------------------------------------------------------------
        # Build the three components of the FNO architecture.
        # ------------------------------------------------------------------

        # 1. Lifting layer: projects d_in input channels to width hidden channels.
        self.lifting: nn.Linear = nn.Linear(self.d_in, self.width)

        # 2. FNO blocks: n_layers Fourier layers.
        if self.dim == 1:
            # ODEs: 1D temporal FNO blocks.
            self.fno_blocks: nn.ModuleList = nn.ModuleList([
                FNOBlock1d(width=self.width, modes=self.modes1)
                for _ in range(self.n_layers)
            ])
        else:
            # PDEs: 2D spatiotemporal or spatial FNO blocks.
            self.fno_blocks = nn.ModuleList([
                FNOBlock2d(
                    width=self.width,
                    modes1=self.modes1,
                    modes2=self.modes2,
                )
                for _ in range(self.n_layers)
            ])

        # 3. Projection: maps width hidden channels to d_out output channels.
        # The intermediate dimension 128 follows the standard FNO convention
        # from Li et al. (2021) — not specified in the SC-FNO paper.
        self.projection: nn.Sequential = nn.Sequential(
            nn.Linear(self.width, 128),
            nn.GELU(),
            nn.Linear(128, self.d_out),
        )

        # ------------------------------------------------------------------
        # Variant tag — set by build_model factory in sc_fno.py.
        # Default to 'fno' (data loss only).
        # ------------------------------------------------------------------
        self.variant: str = "fno"

        # ------------------------------------------------------------------
        # Print architecture summary for verification against Table C.7.
        # ------------------------------------------------------------------
        n_learnable: int = self.count_parameters()
        print(
            f"[FNO] Initialized for equation='{self.equation}' | "
            f"dim={self.dim} | width={self.width} | n_layers={self.n_layers} | "
            f"modes=({self.modes1},{self.modes2}) | d_in={self.d_in} | "
            f"n_params={self.n_params} | learnable_params={n_learnable:,}"
        )

    # ------------------------------------------------------------------
    # Architecture helpers
    # ------------------------------------------------------------------

    def _compute_d_in(self) -> int:
        """Computes the number of input channels to the lifting layer.

        The input to the FNO is a concatenation of:
          - Initial condition u0 (1 channel, broadcast over grid)
          - Physical parameters p (n_params channels, broadcast over grid)
          - Coordinate grid (d_coord channels: 1 for ODEs, 2 for PDEs)

        Mode A (ODEs, dim=1):
            d_in = 1 (u0) + n_params + 1 (time coord) = n_params + 2

        Mode B (1D PDEs, dim=2, spatiotemporal):
            d_in = 1 (u0 at t=0) + n_params + 2 (x-coord + t-coord) = n_params + 3

        Mode C (PDE3, dim=2, spatial):
            d_in = 1 (initial vorticity) + n_params + 2 (x-coord + y-coord) = n_params + 3

        Returns:
            Integer number of input channels d_in.
        """
        if self.dim == 1:
            # ODEs: u0 (1) + params (n_params) + time coord (1)
            return self.n_params + 2
        else:
            # PDEs: u0 (1) + params (n_params) + spatial coords (2)
            return self.n_params + 3

    def _normalize_params(self, params: torch.Tensor) -> torch.Tensor:
        """Normalizes physical parameters to [0, 1] using stored ranges.

        Applies the transformation: p_norm = (p - a) / (b - a)
        where [a, b] are the parameter bounds from config.yaml.

        This improves training stability when parameters span very different
        scales (e.g., γ∈[20,60] for ODE2 vs ω∈[0.01,0.1] for PDE2).

        Args:
            params: Physical parameter tensor, shape [B, n_params].
                    May have requires_grad=True for sensitivity loss computation.

        Returns:
            Normalized parameter tensor, shape [B, n_params], values in [0,1].
            Retains the gradient graph if params.requires_grad=True.
        """
        device: torch.device = params.device
        dtype: torch.dtype = params.dtype

        # Build normalization tensors from stored ranges.
        # Shape: [1, n_params] for broadcasting over batch dimension.
        lo_vals: List[float] = [r[0] for r in self.param_ranges]
        hi_vals: List[float] = [r[1] for r in self.param_ranges]

        lo: torch.Tensor = torch.tensor(
            lo_vals, dtype=dtype, device=device
        ).unsqueeze(0)  # [1, n_params]

        hi: torch.Tensor = torch.tensor(
            hi_vals, dtype=dtype, device=device
        ).unsqueeze(0)  # [1, n_params]

        # Compute range, guarding against zero-range parameters.
        range_vals: torch.Tensor = hi - lo  # [1, n_params]
        # Replace zero ranges with 1.0 to avoid division by zero.
        range_vals = torch.where(
            range_vals.abs() < 1e-12,
            torch.ones_like(range_vals),
            range_vals,
        )

        # Normalize: p_norm = (p - lo) / (hi - lo)
        params_norm: torch.Tensor = (params - lo) / range_vals

        return params_norm

    def _build_input(
        self,
        params: torch.Tensor,
        u0: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Constructs the FNO input tensor by broadcasting and concatenating.

        Broadcasts physical parameters and initial conditions to match the
        spatial-temporal grid shape, then concatenates with coordinate grids
        to form the full input tensor of shape [B, d_in, grid...].

        The output is in channel-first format ([B, d_in, grid...]) ready for
        the FNO blocks. The lifting layer is applied in forward() after
        permuting to channel-last.

        Args:
            params: Physical parameter tensor, shape [B, n_params].
                    May have requires_grad=True for sensitivity loss.
            u0: Initial condition tensor. Shape depends on equation:
                - ODEs:    [B, M] — first M time steps
                - 1D PDEs: [B, M, Sx] — first M time steps at all spatial points
                - PDE3:    [B, Sx, Sy] — initial vorticity field
            coords: Coordinate grid tensor. Shape depends on equation:
                - ODEs:    [T_out, 1] — time coordinates for output steps
                - 1D PDEs: [T_out, Sx, 2] — (x, t) coordinates
                - PDE3:    [Sx, Sy, 2] — (x, y) spatial coordinates
                Note: coords does NOT have a batch dimension — it is shared
                across all samples in a batch and is broadcast here.

        Returns:
            Input tensor in channel-first format:
            - ODEs:    [B, d_in, T_out]
            - 1D PDEs: [B, d_in, Sx, T_out]
            - PDE3:    [B, d_in, Sx, Sy]

        Note:
            The output covers only the OUTPUT time steps [M:N], not the full
            [0:N] range. The FNO predicts u:[M:N] given u:[0:M] and p.
        """
        B: int = params.shape[0]
        device: torch.device = params.device
        dtype: torch.dtype = params.dtype

        # Optionally normalize parameters to [0, 1].
        if self.normalize_params and len(self.param_ranges) == self.n_params:
            params_proc: torch.Tensor = self._normalize_params(params)
        else:
            params_proc = params

        # Ensure coords is on the correct device and dtype.
        coords_dev: torch.Tensor = coords.to(device=device, dtype=dtype)

        if self.dim == 1:
            # ------------------------------------------------------------------
            # Mode A: ODEs — 1D temporal sequences.
            # coords shape: [T_out, 1]
            # Output: [B, d_in, T_out]
            # ------------------------------------------------------------------
            T_out: int = coords_dev.shape[0]  # Number of output time steps.

            # Broadcast params over time: [B, n_params] → [B, T_out, n_params]
            params_expanded: torch.Tensor = params_proc.unsqueeze(1).expand(
                B, T_out, self.n_params
            )  # [B, T_out, n_params]

            # Extract initial value from u0 (first time step or scalar).
            # u0 shape: [B, M] for ODEs.
            # We use the initial value u0[:, 0:1] broadcast over all T_out.
            if u0.ndim == 1:
                # Scalar u0 per sample: [B] → [B, T_out, 1]
                u0_channel: torch.Tensor = u0.unsqueeze(-1).unsqueeze(-1).expand(
                    B, T_out, 1
                )
            elif u0.ndim == 2:
                # u0 shape [B, M]: use first time step as the constant channel.
                u0_channel = u0[:, 0:1].unsqueeze(1).expand(
                    B, T_out, 1
                )  # [B, T_out, 1]
            else:
                # Fallback: flatten and use first element.
                u0_channel = u0.reshape(B, -1)[:, 0:1].unsqueeze(1).expand(
                    B, T_out, 1
                )

            # Broadcast coords over batch: [T_out, 1] → [B, T_out, 1]
            coords_expanded: torch.Tensor = coords_dev.unsqueeze(0).expand(
                B, T_out, 1
            )  # [B, T_out, 1]

            # Concatenate along the channel dimension (last dim).
            # Order: [u0, params, coords] → [B, T_out, d_in]
            x: torch.Tensor = torch.cat(
                [u0_channel, params_expanded, coords_expanded], dim=-1
            )  # [B, T_out, d_in]

            # Permute to channel-first: [B, T_out, d_in] → [B, d_in, T_out]
            x = x.permute(0, 2, 1)  # [B, d_in, T_out]

        elif self.is_pde3:
            # ------------------------------------------------------------------
            # Mode C: PDE3 — 2D spatial grid (x, y).
            # coords shape: [Sx, Sy, 2]
            # u0 shape: [B, Sx, Sy]
            # Output: [B, d_in, Sx, Sy]
            # ------------------------------------------------------------------
            Sx: int = coords_dev.shape[0]
            Sy: int = coords_dev.shape[1]

            # Broadcast params over spatial grid: [B, n_params] → [B, Sx, Sy, n_params]
            params_expanded = params_proc.unsqueeze(1).unsqueeze(1).expand(
                B, Sx, Sy, self.n_params
            )  # [B, Sx, Sy, n_params]

            # Reshape u0 to channel format: [B, Sx, Sy] → [B, Sx, Sy, 1]
            if u0.ndim == 3:
                u0_channel = u0.unsqueeze(-1)  # [B, Sx, Sy, 1]
            elif u0.ndim == 2:
                # Fallback: u0 is [B, Sx*Sy], reshape to [B, Sx, Sy, 1].
                u0_channel = u0.reshape(B, Sx, Sy).unsqueeze(-1)
            else:
                u0_channel = u0.reshape(B, Sx, Sy).unsqueeze(-1)

            # Broadcast coords over batch: [Sx, Sy, 2] → [B, Sx, Sy, 2]
            coords_expanded = coords_dev.unsqueeze(0).expand(
                B, Sx, Sy, 2
            )  # [B, Sx, Sy, 2]

            # Concatenate: [B, Sx, Sy, d_in]
            x = torch.cat(
                [u0_channel, params_expanded, coords_expanded], dim=-1
            )  # [B, Sx, Sy, d_in]

            # Permute to channel-first: [B, Sx, Sy, d_in] → [B, d_in, Sx, Sy]
            x = x.permute(0, 3, 1, 2)  # [B, d_in, Sx, Sy]

        else:
            # ------------------------------------------------------------------
            # Mode B: 1D PDEs — 2D spatiotemporal grid (x, t).
            # coords shape: [T_out, Sx, 2]
            # u0 shape: [B, M, Sx]
            # Output: [B, d_in, Sx, T_out]
            # ------------------------------------------------------------------
            T_out = coords_dev.shape[0]
            Sx = coords_dev.shape[1]

            # Broadcast params over spatiotemporal grid:
            # [B, n_params] → [B, Sx, T_out, n_params]
            params_expanded = (
                params_proc
                .unsqueeze(1)
                .unsqueeze(1)
                .expand(B, Sx, T_out, self.n_params)
            )  # [B, Sx, T_out, n_params]

            # Extract initial spatial field from u0.
            # u0 shape: [B, M, Sx] — use the first time step u0[:, 0, :] as
            # the constant spatial channel broadcast over all T_out.
            if u0.ndim == 3:
                # u0 shape [B, M, Sx]: use first time step.
                u0_spatial: torch.Tensor = u0[:, 0, :]  # [B, Sx]
            elif u0.ndim == 2:
                # u0 shape [B, Sx]: already the initial spatial field.
                u0_spatial = u0
            else:
                # Fallback.
                u0_spatial = u0.reshape(B, Sx)

            # Broadcast u0_spatial over T_out:
            # [B, Sx] → [B, Sx, T_out, 1]
            u0_channel = (
                u0_spatial