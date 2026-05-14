## training/losses.py
"""
Loss functions for the multi-physics neural operator pretraining framework
described in:
  "Towards Universal Neural Operators through Multiphysics Pretraining"

Implements three loss functions:
  - NMAELoss: Range-normalized mean absolute error, equation (3) in the paper.
  - MSELoss: Mean squared error, primary training loss per config.yaml.
  - RelativeL2Loss: Relative L2 norm, standard FNO literature alternative.

Design contract (Data structures and interfaces):
  NMAELoss(eps: float = 1e-8)
    forward(pred: Tensor, target: Tensor) -> Tensor  # scalar, percentage units
  MSELoss()
    forward(pred: Tensor, target: Tensor) -> Tensor  # scalar
  RelativeL2Loss(eps: float = 1e-8)
    forward(pred: Tensor, target: Tensor) -> Tensor  # scalar

Tensor layout convention (Shared Knowledge):
  All spatial fields are channel-first: [B, C, H, W] for 2D or [B, C, L] for 1D.
  B = batch, C = channels, H/W/L = spatial dimensions.

Config alignment:
  training.loss: "mse"          -> MSELoss is the default training loss.
  evaluation.nmae_eps: 1.0e-8   -> passed to NMAELoss(eps=1e-8).

NO imports from other project files. torch only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# NMAELoss — equation (3) from the paper
# ---------------------------------------------------------------------------


class NMAELoss(nn.Module):
    """Range-normalized mean absolute error loss.

    Implements equation (3) from the paper exactly:

        NMAE(θ) = (1/|D_test|) * Σ_{(a,u)∈D_test}
                  ||G_θ(a) - u||_{1,G} / (max_G(u) - min_G(u) + ε)

    where:
      - ``||·||_{1,G}`` is the **mean** L1 norm over the spatial grid G
        (sum of absolute errors divided by the number of grid points).
      - ``max_G(u)`` and ``min_G(u)`` are the per-sample max and min of the
        **target** field over all spatial and channel dimensions.
      - The outer mean is over the batch.
      - The result is expressed as a **percentage** (multiplied by 100).

    This loss is differentiable with respect to ``pred`` via ``torch.abs``,
    which uses the subgradient (sign function) at zero. It can therefore be
    used as a training loss as well as an evaluation metric.

    From config.yaml: ``evaluation.nmae_eps: 1.0e-8`` should be passed as
    ``eps`` when instantiating this class.

    Attributes:
        eps: Small constant added to the denominator to prevent division by
            zero when the target field is constant (value range = 0).

    Example::

        loss_fn = NMAELoss(eps=1e-8)
        pred = torch.randn(16, 1, 64, 64)   # [B, C, H, W]
        target = torch.randn(16, 1, 64, 64)
        nmae_pct = loss_fn(pred, target)     # scalar tensor, percentage
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialise NMAELoss.

        Args:
            eps: Small positive constant added to the value-range denominator
                to prevent division by zero. Default ``1e-8`` matches
                ``evaluation.nmae_eps`` in config.yaml.
        """
        super().__init__()
        self.eps: float = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the range-normalized mean absolute error.

        Args:
            pred: Model predictions with shape ``[B, C, ...]`` where ``...``
                represents one or more spatial dimensions (e.g. ``[B, C, L]``
                for 1D or ``[B, C, H, W]`` for 2D).
            target: Ground-truth values with the same shape as ``pred``.

        Returns:
            Scalar tensor containing the NMAE in percentage units.
            Differentiable with respect to ``pred``.

        Raises:
            ValueError: If ``pred`` and ``target`` have different shapes.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape, "
                f"got pred={pred.shape} and target={target.shape}."
            )

        batch_size: int = pred.shape[0]

        # ── Step 1: Flatten all non-batch dims to [B, N] ──────────────────
        # N = C * H * W (2D) or C * L (1D) — the full grid G per sample.
        pred_flat: Tensor = pred.reshape(batch_size, -1)    # [B, N]
        target_flat: Tensor = target.reshape(batch_size, -1)  # [B, N]

        # ── Step 2: Per-sample mean absolute error over the grid ──────────
        # ||G_θ(a) - u||_{1,G} interpreted as mean L1 norm:
        #   sum(|pred - target|) / N  (mean over grid points)
        abs_error: Tensor = torch.abs(pred_flat - target_flat)  # [B, N]
        l1_per_sample: Tensor = abs_error.mean(dim=1)           # [B]

        # ── Step 3: Per-sample value range of the target ──────────────────
        # max_G(u) - min_G(u) computed over all grid points for each sample.
        # Uses the TARGET only, not the prediction (consistent with paper).
        target_max: Tensor = target_flat.max(dim=1).values   # [B]
        target_min: Tensor = target_flat.min(dim=1).values   # [B]
        value_range: Tensor = target_max - target_min + self.eps  # [B]

        # ── Step 4: Per-sample normalized error ───────────────────────────
        nmae_per_sample: Tensor = l1_per_sample / value_range  # [B]

        # ── Step 5: Mean over batch, convert to percentage ─────────────────
        # Paper reports NMAE as a percentage (e.g. 0.0120 in Table 1 means
        # 0.0120 %, which corresponds to a raw ratio of 0.000120).
        nmae: Tensor = nmae_per_sample.mean() * 100.0  # scalar

        return nmae


# ---------------------------------------------------------------------------
# MSELoss — primary training loss per config.yaml
# ---------------------------------------------------------------------------


class MSELoss(nn.Module):
    """Mean squared error loss.

    A thin wrapper around ``torch.nn.MSELoss(reduction='mean')`` that
    computes the mean over all elements (batch × channels × spatial dims).
    This is the primary training loss specified in config.yaml
    (``training.loss: "mse"``) and is also reported as a metric in
    Tables 1 and 2 of the paper.

    Using a wrapper class (rather than ``nn.MSELoss`` directly) keeps the
    interface consistent with ``NMAELoss`` and ``RelativeL2Loss``, allowing
    all three to be used interchangeably in training loops.

    Example::

        loss_fn = MSELoss()
        pred = torch.randn(16, 1, 64, 64)
        target = torch.randn(16, 1, 64, 64)
        mse = loss_fn(pred, target)  # scalar tensor
    """

    def __init__(self) -> None:
        """Initialise MSELoss with mean reduction."""
        super().__init__()
        self._loss: nn.MSELoss = nn.MSELoss(reduction="mean")

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute mean squared error.

        Args:
            pred: Model predictions with shape ``[B, C, ...]``.
            target: Ground-truth values with the same shape as ``pred``.

        Returns:
            Scalar tensor containing the MSE averaged over all elements.
            Differentiable with respect to ``pred``.

        Raises:
            ValueError: If ``pred`` and ``target`` have different shapes.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape, "
                f"got pred={pred.shape} and target={target.shape}."
            )
        return self._loss(pred, target)


# ---------------------------------------------------------------------------
# RelativeL2Loss — standard FNO literature alternative
# ---------------------------------------------------------------------------


class RelativeL2Loss(nn.Module):
    """Relative L2 norm loss.

    Computes the per-sample relative L2 error:

        rel_l2 = ||pred - target||_2 / (||target||_2 + ε)

    and returns the mean over the batch. This is the standard training loss
    used in the original FNO paper (Li et al., 2021) and is included here
    as an alternative to MSE. It is not explicitly defined in the paper
    being reproduced but is a common choice in the neural operator literature.

    The normalization by the target's L2 norm makes this loss scale-invariant
    and often leads to more stable training when target magnitudes vary across
    samples or physics problems.

    Attributes:
        eps: Small constant added to the target norm denominator to prevent
            division by zero when the target is identically zero.

    Example::

        loss_fn = RelativeL2Loss(eps=1e-8)
        pred = torch.randn(16, 2, 64, 64)   # [B, C, H, W]
        target = torch.randn(16, 2, 64, 64)
        rel_l2 = loss_fn(pred, target)       # scalar tensor
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialise RelativeL2Loss.

        Args:
            eps: Small positive constant added to the target L2 norm
                denominator to prevent division by zero. Default ``1e-8``.
        """
        super().__init__()
        self.eps: float = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the mean relative L2 error over the batch.

        Args:
            pred: Model predictions with shape ``[B, C, ...]`` where ``...``
                represents one or more spatial dimensions.
            target: Ground-truth values with the same shape as ``pred``.

        Returns:
            Scalar tensor containing the mean relative L2 error.
            Differentiable with respect to ``pred``.

        Raises:
            ValueError: If ``pred`` and ``target`` have different shapes.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape, "
                f"got pred={pred.shape} and target={target.shape}."
            )

        batch_size: int = pred.shape[0]

        # ── Flatten all non-batch dims to [B, N] ──────────────────────────
        pred_flat: Tensor = pred.reshape(batch_size, -1)      # [B, N]
        target_flat: Tensor = target.reshape(batch_size, -1)  # [B, N]

        # ── Per-sample L2 norm of the error ───────────────────────────────
        # ||pred - target||_2 for each sample in the batch.
        diff: Tensor = pred_flat - target_flat                          # [B, N]
        error_norm: Tensor = torch.norm(diff, p=2, dim=1)              # [B]

        # ── Per-sample L2 norm of the target ──────────────────────────────
        # ||target||_2 + eps to avoid division by zero.
        target_norm: Tensor = torch.norm(target_flat, p=2, dim=1) + self.eps  # [B]

        # ── Per-sample relative L2 error ──────────────────────────────────
        rel_l2_per_sample: Tensor = error_norm / target_norm  # [B]

        # ── Mean over batch ────────────────────────────────────────────────
        return rel_l2_per_sample.mean()  # scalar


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def get_loss_fn(
    loss_name: str,
    eps: float = 1e-8,
) -> nn.Module:
    """Instantiate a loss function by name.

    Convenience factory used by training loops to select the loss function
    specified in config.yaml (``training.loss``).

    Supported names (case-insensitive):
      - ``"mse"``        -> :class:`MSELoss`
      - ``"nmae"``       -> :class:`NMAELoss`
      - ``"relative_l2"`` or ``"rel_l2"`` -> :class:`RelativeL2Loss`

    Args:
        loss_name: Name of the loss function. Must be one of the supported
            names listed above. Case-insensitive.
        eps: Epsilon value passed to :class:`NMAELoss` and
            :class:`RelativeL2Loss`. Ignored for :class:`MSELoss`.
            Default ``1e-8`` matches ``evaluation.nmae_eps`` in config.yaml.

    Returns:
        Instantiated loss function module.

    Raises:
        ValueError: If ``loss_name`` is not one of the supported names.

    Example::

        # From config.yaml: training.loss: "mse"
        loss_fn = get_loss_fn("mse")

        # From config.yaml: evaluation.nmae_eps: 1.0e-8
        nmae_fn = get_loss_fn("nmae", eps=1e-8)
    """
    name_lower: str = loss_name.strip().lower()

    if name_lower == "mse":
        return MSELoss()
    elif name_lower == "nmae":
        return NMAELoss(eps=eps)
    elif name_lower in {"relative_l2", "rel_l2"}:
        return RelativeL2Loss(eps=eps)
    else:
        raise ValueError(
            f"Unknown loss function '{loss_name}'. "
            f"Supported: 'mse', 'nmae', 'relative_l2' (or 'rel_l2')."
        )
