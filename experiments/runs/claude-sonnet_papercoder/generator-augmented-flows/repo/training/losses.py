## training/losses.py
"""Distance functions for the iCT-GC consistency model training loss.

This module implements ``DistanceFunction``, which computes the per-sample
distance D(x, y) used in the consistency loss:

    L(θ) = λ(σ_ti) · D(sg(f_θ(x̃_ti, σ_ti)), f_θ(x̃_{ti+1}, σ_{ti+1}))

The ``DistanceFunction`` class handles only the D(·,·) part. The λ weighting
and .mean() over the batch are applied in ``trainer.py``.

Three distance modes are supported:

1. **pseudo_huber** (default): D(x,y) = sqrt(||x-y||² + c²) - c
   - Behaves like L2 for small differences, L1 for large differences.
   - Primary distance function used in iCT (Song & Dhariwal, 2024).
   - Config: ``distance_fn: pseudo_huber``, ``pseudo_huber_c: 0.00054``

2. **l2**: D(x,y) = ||x-y||²
   - Squared L2 norm. Corresponds to α=2 in the theoretical analysis
     (Theorem 1) where the quadratic loss gives equal gradients for CT and CD.
   - Used for theoretical analysis and ablations.

3. **lpips**: D(x,y) = LPIPS(x, y) with VGG backbone.
   - Perceptual distance. Requires images in [-1, 1] range and C=3.
   - LPIPS parameters are frozen (no gradients through the perceptual network).

Configuration values from config.yaml (defaults section):
    distance_fn:    pseudo_huber
    pseudo_huber_c: 0.00054

Typical usage in trainer.py::

    distance_fn = DistanceFunction(mode='pseudo_huber', c=0.00054)
    distance_fn = distance_fn.to(device)

    # Inside training step:
    f_upper = model(x_tilde_i1, sigma_i1)           # (B, C, H, W)
    f_lower = model(x_tilde_i, sigma_i).detach()    # (B, C, H, W), stop-gradient
    lam = schedule.get_lambda(sigma_i, sigma_i1)    # (B,)
    dist = distance_fn(f_upper, f_lower)             # (B,)
    loss = (lam * dist).mean()                       # scalar
"""

from typing import Optional

import torch
import torch.nn as nn


class DistanceFunction(nn.Module):
    """Per-sample distance function D(x, y) for the consistency training loss.

    Subclasses ``nn.Module`` to support device management via ``.to(device)``
    (required for the LPIPS mode which contains a neural network). For
    ``pseudo_huber`` and ``l2`` modes, the ``nn.Module`` base class is
    harmless overhead with no learnable parameters.

    The ``forward`` method returns a 1-D tensor of shape ``(B,)`` containing
    one scalar distance per sample. The caller (``trainer.py``) is responsible
    for multiplying by the loss weight λ(σ_i) and calling ``.mean()``.

    Attributes:
        mode: Distance metric identifier. One of ``'pseudo_huber'``,
            ``'l2'``, or ``'lpips'``.
        c: Pseudo-Huber smoothing constant. Only used when
            ``mode='pseudo_huber'``. Default 0.00054 from EDM2 conventions
            and config.yaml ``pseudo_huber_c`` field.
        lpips_net: LPIPS perceptual network (VGG backbone). Only instantiated
            when ``mode='lpips'``. Frozen (no gradient computation through
            its parameters). ``None`` for other modes.
    """

    def __init__(
        self,
        mode: str = "pseudo_huber",
        c: float = 0.00054,
    ) -> None:
        """Initialise the distance function.

        Args:
            mode: Distance metric to use. Must be one of:
                - ``'pseudo_huber'``: D(x,y) = sqrt(||x-y||² + c²) - c.
                  Primary mode used in iCT experiments. Config default.
                - ``'l2'``: D(x,y) = ||x-y||². Squared Euclidean norm.
                  Used for theoretical analysis (α=2 case in Theorem 1).
                - ``'lpips'``: Perceptual distance via LPIPS with VGG
                  backbone. Requires ``lpips`` package to be installed.
                  Images must be in [-1, 1] range with C=3.
            c: Pseudo-Huber smoothing constant. Controls the transition
                between L2 behaviour (small differences) and L1 behaviour
                (large differences). Only used when ``mode='pseudo_huber'``.
                Default 0.00054 matches config.yaml ``pseudo_huber_c`` and
                EDM2 conventions. Must be strictly positive.

        Raises:
            ValueError: If ``mode`` is not one of the supported values.
            ValueError: If ``c <= 0`` when ``mode='pseudo_huber'``.
            ImportError: If ``mode='lpips'`` and the ``lpips`` package is
                not installed.
        """
        super().__init__()

        _VALID_MODES = ("pseudo_huber", "l2", "lpips")
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown distance mode '{mode}'. "
                f"Must be one of: {_VALID_MODES}. "
                "Config default: distance_fn = pseudo_huber."
            )

        if mode == "pseudo_huber" and c <= 0.0:
            raise ValueError(
                f"Pseudo-Huber constant c must be strictly positive, "
                f"got c={c}. Config default: pseudo_huber_c = 0.00054."
            )

        self.mode: str = mode
        self.c: float = float(c)

        # LPIPS perceptual network — only instantiated for 'lpips' mode
        self.lpips_net: Optional[nn.Module] = None

        if mode == "lpips":
            try:
                import lpips as lpips_lib
            except ImportError as exc:
                raise ImportError(
                    "The 'lpips' package is required for mode='lpips'. "
                    "Install it with: pip install lpips==0.1.4"
                ) from exc

            # Instantiate LPIPS with VGG backbone (standard choice for iCT)
            # The LPIPS network is registered as a submodule so .to(device)
            # moves it correctly alongside the DistanceFunction.
            lpips_network = lpips_lib.LPIPS(net="vgg")

            # Freeze all LPIPS parameters — gradients flow through x and y
            # (the consistency model outputs), not through the perceptual net.
            for param in lpips_network.parameters():
                param.requires_grad = False

            # Set to eval mode: disables dropout and uses running statistics
            # for any BatchNorm layers. Must stay in eval mode throughout
            # training (do not call distance_fn.train() from the trainer).
            lpips_network.eval()

            # Register as a submodule for proper device management
            self.lpips_net = lpips_network

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample distances between two batches of images.

        Dispatches to the appropriate distance implementation based on
        ``self.mode``. Returns a 1-D tensor of per-sample distances.

        The stop-gradient operation (``sg(·)`` in the paper) is applied
        **before** calling this function in ``trainer.py`` via
        ``f_lower.detach()``. This function does not apply any detach.

        Args:
            x: First image batch of shape ``(B, C, H, W)``. Typically the
               online model output ``f_θ(x̃_{ti+1}, σ_{ti+1})`` (upper noise
               level, gradient flows through this).
            y: Second image batch of shape ``(B, C, H, W)``. Typically the
               stop-gradient output ``sg(f_θ(x̃_ti, σ_ti))`` (lower noise
               level, already detached by the caller).

        Returns:
            Float tensor of shape ``(B,)`` containing per-sample distances.
            All values are non-negative. The caller multiplies by λ(σ_i)
            and calls ``.mean()`` to obtain the scalar training loss.

        Raises:
            ValueError: If ``x`` and ``y`` have different shapes.
            RuntimeError: If ``x`` and ``y`` are on different devices.
        """
        if x.shape != y.shape:
            raise ValueError(
                f"x and y must have the same shape, "
                f"got x.shape={x.shape} and y.shape={y.shape}."
            )

        if self.mode == "pseudo_huber":
            return self._pseudo_huber(x, y)
        elif self.mode == "l2":
            return self._l2(x, y)
        elif self.mode == "lpips":
            return self._lpips(x, y)
        else:
            # This branch is unreachable due to __init__ validation,
            # but included for defensive completeness.
            raise ValueError(
                f"Unknown distance mode '{self.mode}'. "
                "This should not happen — check __init__ validation."
            )

    def _pseudo_huber(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pseudo-Huber distance per sample.

        Formula:
            D(x, y) = sqrt(||x - y||² + c²) - c

        where ||·||² is the squared Euclidean norm summed over all
        non-batch dimensions (C, H, W).

        Properties:
        - D(x, x) = sqrt(c²) - c = 0  (zero at equality)
        - For small ||x-y||: D ≈ ||x-y||²/(2c)  (L2-like, smooth near zero)
        - For large ||x-y||: D ≈ ||x-y|| - c    (L1-like, robust to outliers)
        - Gradient at x=y: well-defined (1/(2c)), no singularity

        Numerical stability:
        - The argument of sqrt is sq + c² ≥ c² > 0 always (since c > 0).
        - No risk of sqrt(0) or division by zero.

        Args:
            x: Image tensor of shape ``(B, C, H, W)``.
            y: Image tensor of shape ``(B, C, H, W)``.

        Returns:
            Float tensor of shape ``(B,)`` with non-negative values.
        """
        # Element-wise difference: (B, C, H, W)
        diff: torch.Tensor = x - y

        # Squared Euclidean norm summed over C, H, W dimensions → (B,)
        # This is ||x - y||² for each sample in the batch
        sq: torch.Tensor = (diff ** 2).sum(dim=[1, 2, 3])

        # Pseudo-Huber: sqrt(||x-y||² + c²) - c
        # c² is a Python float; PyTorch handles float-to-tensor promotion
        c_sq: float = self.c ** 2
        return torch.sqrt(sq + c_sq) - self.c

    def _l2(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute squared L2 distance per sample.

        Formula:
            D(x, y) = ||x - y||²

        where ||·||² is the squared Euclidean norm summed over all
        non-batch dimensions (C, H, W).

        This corresponds to the α=2 case in Theorem 1 of the paper, where
        the quadratic loss gives equal limiting gradients for consistency
        training and consistency distillation. Used for theoretical analysis
        and ablations.

        Args:
            x: Image tensor of shape ``(B, C, H, W)``.
            y: Image tensor of shape ``(B, C, H, W)``.

        Returns:
            Float tensor of shape ``(B,)`` with non-negative values.
        """
        # Element-wise difference: (B, C, H, W)
        diff: torch.Tensor = x - y

        # Squared Euclidean norm summed over C, H, W dimensions → (B,)
        return (diff ** 2).sum(dim=[1, 2, 3])

    def _lpips(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LPIPS perceptual distance per sample.

        Uses a VGG backbone to compute perceptual similarity between image
        pairs. The LPIPS network parameters are frozen — gradients flow
        through x and y (the consistency model outputs) but not through
        the VGG feature extractor.

        Input requirements:
        - Values in [-1, 1] range (satisfied by the consistency model's
          output range, which matches the data normalisation).
        - C=3 channels (RGB). All supported datasets are RGB.
        - Spatial size ≥ 16×16 (VGG requirement). CIFAR-10 (32×32) is
          borderline but works in practice.

        The LPIPS network returns shape (B, 1, 1, 1); we squeeze to (B,).

        Args:
            x: Image tensor of shape ``(B, 3, H, W)`` in ``[-1, 1]``.
            y: Image tensor of shape ``(B, 3, H, W)`` in ``[-1, 1]``.

        Returns:
            Float tensor of shape ``(B,)`` with non-negative perceptual
            distance values.

        Raises:
            RuntimeError: If ``self.lpips_net`` is None (should not happen
                if mode='lpips' was set in __init__).
        """
        if self.lpips_net is None:
            raise RuntimeError(
                "lpips_net is None but mode='lpips'. "
                "This indicates a bug in DistanceFunction.__init__."
            )

        # Ensure LPIPS network stays in eval mode throughout training.
        # This is a defensive check in case the caller accidentally called
        # distance_fn.train() on the parent DistanceFunction module.
        self.lpips_net.eval()

        # LPIPS forward pass: returns (B, 1, 1, 1)
        # No gradient flows through lpips_net (parameters are frozen).
        # Gradient flows through x and y via the feature difference computation.
        lpips_output: torch.Tensor = self.lpips_net(x, y)

        # Squeeze spatial and channel singleton dimensions: (B, 1, 1, 1) → (B,)
        batch_size: int = x.shape[0]
        return lpips_output.view(batch_size)

    def __repr__(self) -> str:
        """Return a human-readable summary of the distance function."""
        if self.mode == "pseudo_huber":
            return (
                f"DistanceFunction(mode='pseudo_huber', c={self.c})\n"
                f"  Formula: D(x,y) = sqrt(||x-y||² + {self.c}²) - {self.c}"
            )
        elif self.mode == "l2":
            return (
                "DistanceFunction(mode='l2')\n"
                "  Formula: D(x,y) = ||x-y||²"
            )
        elif self.mode == "lpips":
            return (
                "DistanceFunction(mode='lpips', net='vgg')\n"
                "  Formula: D(x,y) = LPIPS_VGG(x, y)  [frozen]"
            )
        else:
            return f"DistanceFunction(mode='{self.mode}')"
