# models/fno.py
"""Fourier Neural Operator (FNO) backbone shared by all SC‑FNO variants.

The architecture follows Li et al. (2021) and the hyperparameters in Table C.7.
It consists of a lifting layer, a stack of Fourier layers (spectral + local
convolution) and a final point‑wise projection.  Physical parameters are
embedded via a `ParameterEmbedding` module and concatenated with the input
function and grid coordinates before the lifting step.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from config import Config
from .parameter_embedding import ParameterEmbedding


# ---------------------------------------------------------------------------
# Helper: build activation from string
# ---------------------------------------------------------------------------
def _get_activation(name: str) -> nn.Module:
    if name.lower() == "gelu":
        return nn.GELU()
    if name.lower() == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


# ===========================================================================
# Spectral convolution for 1D and 2D problems
# ===========================================================================

class SpectralConv(nn.Module):
    """Fourier‑mode truncation + learnable complex weight multiplication.

    Supports 1D (over time or space) and 2D (space+time or space+space)
    operations using `torch.fft.rfftn`/`irfftn`.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (must equal in_channels for
                      the residual connection used in the paper).
        modes: List of integers specifying how many low‑frequency modes are
               retained in each dimension.  Length = 1 for 1D, 2 for 2D.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: List[int]) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.ndim = len(modes)

        # Create complex weights for the truncated modes.
        # For 1D: shape (in, out, modes[0])
        # For 2D: shape (in, out, modes[0], modes[1])
        weight_shape = (in_channels, out_channels) + tuple(modes)
        # Weights are complex and initialised as scaled uniform random.
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(*weight_shape, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution to input tensor.

        Args:
            x: (batch, channels, d1, ...) real tensor.

        Returns:
            Output of same spatial shape.
        """
        batch, c, *spatial = x.shape

        # ----- 1D case -----
        if self.ndim == 1:
            # RFFT along the last dimension
            x_ft = fft.rfft(x, dim=-1)                # (B, C, L//2+1)
            # Truncate to the first modes[0] frequencies
            out_ft = torch.zeros(
                batch, self.out_channels, x_ft.shape[-1],
                dtype=torch.cfloat, device=x.device
            )
            # Multiply each channel pair by the complex weight
            # For simplicity, we use a full mat‑mul on the channel axis.
            # We'll implement as a batched mat‑mul via einsum.
            # Effective formula: out_ft[:, :, :self.modes[0]] = sum_j x_ft[:, j, :self.modes[0]] * W[j, i, :]
            # Using torch.einsum:
            out_ft[:, :, :self.modes[0]] = torch.einsum(
                "b i k, i o k -> b o k",
                x_ft[:, :, :self.modes[0]],
                self.weights
            )
            # Inverse RFFT
            x = fft.irfft(out_ft, n=spatial[0], dim=-1)  # (B, O, L)

        # ----- 2D case -----
        elif self.ndim == 2:
            # RFFT2 over the last two dimensions
            x_ft = fft.rfft2(x, dim=(-2, -1))            # (B, C, H, W//2+1)
            # Output shape; only the first modes[0] rows and modes[1] columns are used
            out_ft = torch.zeros(
                batch, self.out_channels, *x_ft.shape[-2:],
                dtype=torch.cfloat, device=x.device
            )
            # Truncate to modes
            out_ft[:, :, :self.modes[0], :self.modes[1]] = torch.einsum(
                "b i x y, i o x y -> b o x y",
                x_ft[:, :, :self.modes[0], :self.modes[1]],
                self.weights
            )
            # Inverse RFFT2
            x = fft.irfft2(out_ft, s=spatial, dim=(-2, -1))  # (B, O, H, W)

        else:
            raise RuntimeError(f"SpectralConv expects ndim 1 or 2, got {self.ndim}")

        return x


# ===========================================================================
# Fourier layer (spectral + local conv, residual + activation)
# ===========================================================================

class FourierLayer(nn.Module):
    """Single Fourier layer of the FNO.

    It contains a spectral branch (SpectralConv) and a local branch
    (pointwise 1x1 conv in the frequency domain? Actually original uses a
    local convolution in the spatial domain).  Here we use a small
    convolution (kernel=3 in 1D, 3x3 in 2D) that acts on the input before
    summation.  A residual connection is added and GELU is applied.

    Args:
        in_channels: Input channels (= `width`).
        modes: Fourier modes to retain.
        ndim: Spatial/temporal dimensionality of the data (1 or 2).
        activation: String name of the activation function.
    """

    def __init__(self,
                 in_channels: int,
                 modes: List[int],
                 ndim: int,
                 activation: str = "gelu"):
        super().__init__()
        self.ndim = ndim

        # Spectral branch
        self.spectral = SpectralConv(in_channels, in_channels, modes)

        # Local convolution branch
        if ndim == 1:
            self.local_conv = nn.Conv1d(in_channels, in_channels, 3, padding=1)
        else:
            self.local_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

        self.act = _get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the Fourier layer.

        Args:
            x: (B, C, *spatial_dims)

        Returns:
            (B, C, *spatial_dims)
        """
        identity = x

        # Spectral + local convolution
        out = self.spectral(x) + self.local_conv(x)
        # Residual connection
        out = identity + out
        return self.act(out)


# ===========================================================================
# Main FNO model
# ===========================================================================

class FNO(nn.Module):
    """Fourier Neural Operator with parameter conditioning.

    Constructed from a `Config` object.  The forward pass expects a batch of
    initial conditions `x`, parameter vectors `p`, and optionally a
    coordinate grid `grid`.  If `grid` is ``None`` it is generated from the
    domain information in the config.

    The output is the full solution field (same spatial/temporal shape as the
    input grid).  The final layer projects to a single channel.

    Notes:
        - For the zoned Burgers' equation (``pde2_zoned``), the parameter
          embedding must produce a spatially varying feature map.  This is
          assumed to be handled by the `ParameterEmbedding` module.
        - The model does **not** compute losses or Jacobians – those are
          the responsibility of the training loop.
    """

    def __init__(self, config: Config) -> None:
        """Initialise the FNO from a configuration object.

        Args:
            config: Frozen `Config` object containing at least the
                    ``model`` section and the selected ``equation``.
        """
        super().__init__()

        # ---- basic parameters --------------------------------------------------
        eq = config.equation
        eq_params = config.sol_params          # per‑equation dict
        model_cfg = config.model_params       # 'model' section

        # ---- Determine active modes and spatial dimensionality ---------------
        spatial_dims = eq_params.get("spatial_dims", 0)
        modes_t = model_cfg["modes"]["t"]     # always defined, even if unused
        modes_x = model_cfg["modes"]["x"]
        modes_y = model_cfg["modes"]["y"]     # only for PDE3

        self.ndim: int                # 1 for ODE, 2 for PDEs
        self.modes: List[int]         # list of relevant modes
        self.input_spatial_shape: Tuple[int, ...]

        if eq in ("ode1", "ode2"):
            # ODE: time is the only dimension
            self.ndim = 1
            self.modes = [modes_t]
            Nt = eq_params["N_time"]
            self.input_spatial_shape = (Nt,)
        elif eq == "pde3":
            # Navier‑Stokes: 2D spatial, no time in output
            self.ndim = 2
            self.modes = [modes_x, modes_y]
            Sx = eq_params["S_x"]
            Sy = eq_params["S_y"]
            self.input_spatial_shape = (Sx, Sy)
        else:  # PDE1, PDE2, PDE4, and pde2_zoned (all 1D space + time)
            self.ndim = 2
            self.modes = [modes_x, modes_t]   # space first, time second
            Sx = eq_params["S_x"]
            Nt = eq_params["N_time"]
            self.input_spatial_shape = (Sx, Nt)

        self.out_shape = self.input_spatial_shape   # output same as input grid

        # ---- hidden width ------------------------------------------------------
        self.width = model_cfg["width"]           # e.g., 20

        # ---- activation --------------------------------------------------------
        act_name = model_cfg.get("activation", "gelu")
        self.activation = _get_activation(act_name)

        # ---- parameter embedding -----------------------------------------------
        # This module maps the raw parameter vector (size P) to a feature map
        # of shape (B, width, *input_spatial_shape).
        self.param_embedding = ParameterEmbedding(config)

        # ---- grid coordinate channels: 1 for 1D, 2 for 2D --------------------
        self.grid_channels = self.ndim   # coordinates per grid cell

        # ---- lifting layer (1x1 conv) ------------------------------------------
        # Input channels = 1 (solution) + grid_channels + width (from embedding)
        in_ch = 1 + self.grid_channels + self.width
        if self.ndim == 1:
            self.lifting = nn.Conv1d(in_channels=in_ch,
                                     out_channels=self.width,
                                     kernel_size=1)
        else:
            self.lifting = nn.Conv2d(in_channels=in_ch,
                                     out_channels=self.width,
                                     kernel_size=1)

        # ---- Fourier layers ----------------------------------------------------
        num_layers = model_cfg["n_layers"]
        self.fourier_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.fourier_layers.append(
                FourierLayer(in_channels=self.width,
                             modes=self.modes,
                             ndim=self.ndim,
                             activation=act_name)
            )

        # ---- projection back to 1 channel --------------------------------------
        # Two point‑wise convs with GELU in between (standard FNO).
        if self.ndim == 1:
            self.projection = nn.Sequential(
                nn.Conv1d(self.width, 128, 1),
                self.activation,
                nn.Conv1d(128, 1, 1),
            )
        else:
            self.projection = nn.Sequential(
                nn.Conv2d(self.width, 128, 1),
                self.activation,
                nn.Conv2d(128, 1, 1),
            )

        # ---- store domain ranges for grid generation (optional) ----------------
        # Used when `grid` is None in forward.
        if eq in ("ode1", "ode2"):
            t_start, t_end = eq_params["temporal_domain"]
            self._domain = [(t_start, t_end)]
        elif eq == "pde3":
            # spatial domain: [x0, x1, y0, y1]
            dom = eq_params["spatial_domain"]
            self._domain = [(dom[0], dom[1]), (dom[2], dom[3])]
        else:
            # space + time
            t_start, t_end = eq_params["temporal_domain"]
            x_start, x_end = eq_params["spatial_domain"]
            self._domain = [(x_start, x_end), (t_start, t_end)]

    # --------------------------------------------------------------------------
    # helper: create a coordinate grid from stored domain and shape
    # --------------------------------------------------------------------------
    def _create_grid(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate a normalised coordinate grid for the forward pass.

        Args:
            batch_size: Number of samples in the batch.
            device: Target device.

        Returns:
            Tensor of shape (batch_size, grid_channels, *spatial_shape)
        """
        spatial = self.input_spatial_shape
        grids = []
        for d in range(self.ndim):
            start, end = self._domain[d]
            coords = torch.linspace(start, end, spatial[d],
                                    device=device, dtype=torch.float32)
            # Expand to match spatial shape
            # We want coords broadcastable to (*spatial,).  We'll use
            # meshgrid and then tile.
            # Simpler: create a meshgrid of all dimensions and gather.
            # We'll build a full meshgrid then stack along channel dim.
            if self.ndim == 1:
                grids.append(coords)
            else:
                # For 2D, we need a meshgrid.
                # If self.ndim==2, spatial is (d1, d2). We'll create meshgrids
                # but here we only add one dimension at a time; we'll assemble later.
                grids.append(coords)

        # Build full grid using torch.meshgrid (indexing='ij')
        if self.ndim == 1:
            grid = grids[0].unsqueeze(0).unsqueeze(0)   # (1, 1, N)
            grid = grid.expand(batch_size, 1, -1)
        else:  # ndim == 2
            # Create meshgrid
            c0, c1 = grids
            mg = torch.meshgrid(c0, c1, indexing='ij')
            # Stack as channels: (1, 2, d0, d1)
            grid = torch.stack(mg, dim=0).unsqueeze(0)
            grid = grid.expand(batch_size, 2, -1, -1)

        return grid

    # --------------------------------------------------------------------------
    # public forward
    # --------------------------------------------------------------------------
    def forward(self,
                x: torch.Tensor,
                p: torch.Tensor,
                grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute the FNO forward pass.

        Args:
            x:    Input function (initial condition or first M time steps).
                  Shape ``(batch, 1, *spatial_shape)``.
            p:    Physical parameter vector, shape ``(batch, num_params)``.
            grid: Coordinate grid, shape ``(batch, grid_channels, *spatial_shape)``.
                  If ``None``, a default grid is generated from the config.

        Returns:
            The full solution field ``u(x,t)`` (or ``u(x,y)`` for Navier‑Stokes),
            shape ``(batch, 1, *spatial_shape)``.
        """
        batch_size = x.shape[0]
        device = x.device

        # Generate grid if not provided
        if grid is None:
            grid = self._create_grid(batch_size, device)

        # 1. Parameter embedding
        p_emb = self.param_embedding(p, grid)

        # 2. Concatenate input, grid, embedding
        combined = torch.cat([x, grid, p_emb], dim=1)

        # 3. Lifting
        h = self.lifting(combined)

        # 4. Fourier layers
        for layer in self.fourier_layers:
            h = layer(h)

        # 5. Projection to 1 channel
        out = self.projection(h)

        return out

