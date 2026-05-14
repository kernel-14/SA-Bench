## control_surrogate.py
"""
Differentiable surrogate models used for guiding WDNO control generation.

The surrogate network approximates the mapping from physical initial condition
and control force to the control objective:

* 1D Burgers : (u0, f) -> final state u(T)   (shape (B, 120))
* 2D fluid   : (initial_density, peripheral_force) -> bucket exit percentage (scalar)

During control inference, this surrogate provides a differentiable estimate of
the objective, enabling gradient‑based steering via the term λ ∇_W I in Eq. (4).

The module is designed to work with the configuration provided by
``config.yaml`` (via the ``Config`` object).  It is independent of the
wavelet‑transform and diffusion modules.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Helper networks (not exposed outside the module)
# ------------------------------------------------------------------------------

class _SurrogateUNet2D(nn.Module):
    """
    A compact 2D convolutional network that consumes a (B, 2, T, X) tensor
    (stacked u0 and f) and produces the predicted final state u(T) of shape
    (B, X).  The time dimension is collapsed via strided convolutions.
    """

    def __init__(
        self,
        spatial_size: int,                # spatial grid size (e.g., 120)
        base_dim: int = 64,
        in_channels: int = 2,
    ) -> None:
        """
        Args:
            spatial_size: length of the spatial domain (X).
            base_dim:   base number of channels.
            in_channels: number of input channels (2: u0 and f).
        """
        super().__init__()
        self.spatial_size = spatial_size

        self.input = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)
        self.down1 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=3,
                      stride=(2, 1), padding=1),
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=3,
                      stride=(2, 1), padding=1),
            nn.BatchNorm2d(base_dim * 4),
            nn.ReLU(inplace=True),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(base_dim * 4, base_dim * 8, kernel_size=3,
                      stride=(2, 1), padding=1),
            nn.BatchNorm2d(base_dim * 8),
            nn.ReLU(inplace=True),
        )
        # Adaptive pooling reduces temporal dimension to 1, keeps spatial size
        self.adaptive = nn.AdaptiveAvgPool2d((1, spatial_size))
        self.final_conv = nn.Conv2d(base_dim * 8, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, T, X) input tensor.

        Returns:
            (B, X) predicted final state.
        """
        x = self.input(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)                # (B, 512, T', X')
        x = self.adaptive(x)             # (B, 512, 1, spatial_size)
        x = self.final_conv(x)           # (B, 1, 1, spatial_size)
        return x.squeeze(1).squeeze(1)   # (B, spatial_size)


class _SurrogateUNet3D(nn.Module):
    """
    A compact 3D convolutional network that takes a (B, C, T, H, W) tensor
    (stacked initial density and control force) and outputs a scalar prediction
    for the bucket‑exit percentage.
    """

    def __init__(
        self,
        in_channels: int,                # e.g., 2 (density + force) or more
        base_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        """
        Args:
            in_channels: number of input channels (after concatenation).
            base_dim:    base number of channels.
            hidden_dim:  size of the linear embedding before the final layer.
        """
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_dim),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool3d((2, 2, 2))   # halves T, H, W

        self.conv2 = nn.Sequential(
            nn.Conv3d(base_dim, base_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_dim * 2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))  # global average pool

        self.fc = nn.Sequential(
            nn.Linear(base_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) input tensor.

        Returns:
            (B,) scalar metric.
        """
        x = self.conv1(x)            # (B, base_dim, T, H, W)
        x = self.pool1(x)            # (B, base_dim, T/2, H/2, W/2)
        x = self.conv2(x)            # (B, 2*base_dim, T/2, H/2, W/2)
        x = self.pool2(x)            # (B, 2*base_dim, 1, 1, 1)
        x = x.view(x.size(0), -1)    # (B, 2*base_dim)
        return self.fc(x).squeeze(-1)   # (B,)


# ------------------------------------------------------------------------------
# Public class
# ------------------------------------------------------------------------------

class ControlSurrogate:
    """
    Trains and evaluates a surrogate model for control guidance.

    The model maps (initial condition, control force) → objective component
    (final state for 1D, scalar metric for 2D).  During control inference,
    ``predict_metric`` is called with gradients enabled, allowing the
    WDNO algorithm to compute ``∇_W I``.

    Args:
        config: Full configuration dictionary (returned by
                ``Config.get_all_configs()``).
    """

    def __init__(self, config: Dict[str, object]) -> None:
        self.config = config
        self.experiment: str = config["experiment"]
        self.device = torch.device(config.get("device", "cuda"))

        self.model = self._build_model()
        self.model.to(self.device)

        # Optimizer (only used during training)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config["control"]["surrogate"]["lr"],
        )
        self.loss_fn = nn.MSELoss()

    # --------------------------------------------------------------------------
    # Model construction
    # --------------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        """Internal factory that selects and instantiates the correct surrogate."""
        if "1d" in self.experiment:
            # Spatial resolution is fixed per experiment; read from data config if needed
            # For simplicity, we assume the dataset provides a consistent spatial size.
            # We'll determine spatial size later from the first training batch, but for
            # model creation we need a default.  A lazy approach: we can postpone
            # creation until the first call.  However, we need the model to create
            # the optimizer.  Better to infer from data_config.
            data_cfg = self.config["data"]
            # In 1D experiments, burgers/advection/cfd have spatial_res = 120
            spatial_size = 120
            if self.experiment == "advection_1d":
                # Advection might have 1024 originally, but we resized to 120.
                spatial_size = 120
            base_dim = 64
            model = _SurrogateUNet2D(
                spatial_size=spatial_size,
                base_dim=base_dim,
                in_channels=2,
            )
        elif "2d" in self.experiment or "era5" in self.experiment:
            # For 2D fluid or ERA5, the surrogate expects a 3D volume.
            # in_channels: initial density (1) + control force channels.
            # We'll assume the force is a single channel (scalar force at each pixel)
            # for simplicity, but can be extended.
            in_channels = 2   # density + force (single channel)
            # Possibly increase if multi‑channel force.
            # For bucket‑exit percentage, we may want more capacity.
            model = _SurrogateUNet3D(
                in_channels=in_channels,
                base_dim=32,
                hidden_dim=64,
            )
        else:
            raise ValueError(f"Unknown experiment type: {self.experiment}")
        return model

    # --------------------------------------------------------------------------
    # Forward and predict
    # --------------------------------------------------------------------------

    def _forward(self, f: torch.Tensor, u0: torch.Tensor) -> torch.Tensor:
        """
        Prepare input tensors, pass through the surrogate, and return the
        raw prediction (final state for 1D, scalar for 2D).

        Args:
            f: control force tensor.  Expected shapes:
               - 1D experiments: (B, T, X) e.g., (B, 80, 120)
               - 2D experiments: (B, T, C_f, H, W) or (B, T, H, W)
                 (C_f may be 1, omitted)
            u0: initial condition tensor.
               - 1D: (B, X) e.g., (B, 120)
               - 2D: (B, H, W)

        Returns:
            Prediction tensor:
            - 1D: (B, X)  (predicted final state)
            - 2D/ERA5: (B,) scalar
        """
        if "1d" in self.experiment:
            if f.dim() != 3:
                raise ValueError(f"For 1D experiments, f must be 3‑D [B,T,X], got {f.shape}")
            T = f.size(1)
            if u0.dim() != 2:
                raise ValueError(f"u0 must be 2‑D [B,X], got {u0.shape}")
            # Expand initial condition along the time axis
            u0_exp = u0.unsqueeze(1).expand(-1, T, -1)  # (B, T, X)
            # Stack as two channels: (u0, f)
            x = torch.stack([u0_exp, f], dim=1)         # (B, 2, T, X)
            return self.model(x)                         # (B, X)

        else:  # 2D or ERA5
            # Detect shape of f
            if f.dim() == 4:
                # (B, T, H, W) – single channel force, no explicit control channels
                T = f.size(1)
                u0_exp = u0.unsqueeze(1).expand(-1, T, -1, -1)   # (B, T, H, W)
                # Stack as channels: (density, force)
                x = torch.stack([u0_exp, f], dim=1)               # (B, 2, T, H, W)
            elif f.dim() == 5:
                # (B, T, C_f, H, W) – force already has channels
                T = f.size(1)
                C_f = f.size(2)
                # Expand initial density to (B, T, 1, H, W)
                u0_exp = u0.unsqueeze(1).unsqueeze(2).expand(-1, T, 1, -1, -1)
                # Concatenate along channel dimension (dim=2)
                x = torch.cat([u0_exp, f], dim=2)                # (B, T, 1+C_f, H, W)
                # Permute to (B, 1+C_f, T, H, W) for 3D conv
                x = x.permute(0, 2, 1, 3, 4).contiguous()       # (B, 1+C_f, T, H, W)
            else:
                raise ValueError(
                    f"Unsupported force shape for 2D/ERA5: {f.shape}. "
                    f"Expected 4‑D (B,T,H,W) or 5‑D (B,T,C,H,W)."
                )
            return self.model(x)  # (B,) scalar

    def predict_metric(self, f: torch.Tensor, u0: torch.Tensor) -> torch.Tensor:
        """
        Differentiable forward pass used during control inference.

        The returned tensor represents the objective component contributed by
        the final state approximation (1D) or the full scalar objective (2D).

        This method **must** be called inside a ``torch.enable_grad()`` context
        so that gradients can flow back to ``f`` (and further to the wavelet
        coefficients).

        Args:
            f:  control force in physical space.
            u0: initial condition in physical space.

        Returns:
            Predicted objective component.
        """
        self.model.eval()
        # We do NOT use torch.no_grad() – we need autograd.
        return self._forward(f, u0)

    # --------------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------------

    def train(self, dataloader: DataLoader, epochs: Optional[int] = None) -> None:
        """
        Train the surrogate model on a dataset of (u0, f, target) tuples.

        Args:
            dataloader: PyTorch DataLoader that yields (u0, f, target) tensors.
                - 1D: target shape (B, X)  (final state u(T))
                - 2D: target shape (B,)    (scalar metric)
            epochs: Number of training epochs. If ``None``, uses the value
                    from ``config.yaml`` (``control.surrogate.epochs``).
        """
        if epochs is None:
            epochs = self.config["control"]["surrogate"]["epochs"]

        self.model.train()
        total_batches = len(dataloader)

        for epoch in range(epochs):
            epoch_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Surrogate epoch {epoch+1}/{epochs}")
            for u0, f, target in pbar:
                u0 = u0.to(self.device)
                f = f.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()
                pred = self._forward(f, u0)
                loss = self.loss_fn(pred, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            logger.info(
                f"Surrogate epoch {epoch+1}/{epochs} complete, "
                f"avg loss = {epoch_loss/total_batches:.6f}"
            )

    # --------------------------------------------------------------------------
    # Persistence
    # --------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save surrogate model state dict to `path`."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load surrogate model weights from `path`."""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )

    def eval(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()

