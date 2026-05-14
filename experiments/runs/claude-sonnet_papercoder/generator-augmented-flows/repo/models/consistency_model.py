## models/consistency_model.py
"""Consistency model with EDM-style preconditioning for iCT-GC training.

This module implements ``ConsistencyModel``, which wraps a ``SongUNet``
backbone with the preconditioning functions from Karras et al. (2022) adapted
for consistency training (Song et al., 2023; Song & Dhariwal, 2024).

The parametrization enforces the boundary condition f_θ(x_0, σ_0) = x_0,
which is required for consistency training to be well-posed:

    f_θ(x_t, σ_t) = c_skip(σ_t) · x_t + c_out(σ_t) · F_θ(c_in(σ_t) · x_t, c_noise(σ_t))

where:
    c_skip(σ) = σ_d² / (σ_d² + (σ - σ_0)²)          → 1 as σ → σ_0
    c_out(σ)  = σ_d · (σ - σ_0) / sqrt(σ_d² + σ²)   → 0 as σ → σ_0
    c_in(σ)   = 1 / sqrt(σ_d² + σ²)
    c_noise(σ) = log(σ) / 4

Config values used (from config.yaml defaults):
    sigma_min:  0.002   (σ_0, boundary condition noise level)
    sigma_data: 0.5     (σ_d, data distribution standard deviation)
    sigma_max:  80.0    (σ_T, used in generate() for single-step sampling)

Typical usage in the training loop (Algorithm 1)::

    # Endpoint prediction via EMA (stop-gradient)
    ema.apply_shadow(model)
    with torch.no_grad():
        x_hat = model(x_ti, sigma_i)   # f_ema(x_ti, σ_ti)
    ema.restore(model)

    # Consistency loss computation
    f_upper = model(x_tilde_i1, sigma_i1)
    f_lower = model(x_tilde_i,  sigma_i).detach()   # sg(·)
    loss = lambda_weight * distance_fn(f_upper, f_lower).mean()

Single-step generation::

    z = torch.randn(batch_size, 3, 32, 32, device=device)
    x_0_hat = model.generate(z, sigma_T=80.0)   # one NFE
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from models.song_unet import SongUNet


class ConsistencyModel(nn.Module):
    """Consistency model with EDM preconditioning wrapping a SongUNet backbone.

    Implements the parametrization from Song et al. (2023) / Karras et al.
    (2022) that enforces the boundary condition f_θ(x_0, σ_0) = x_0 via
    the c_skip / c_out decomposition.

    This class is used in three contexts:
    1. **Training**: Called with EMA weights for stop-gradient endpoint
       prediction, and with online weights for the consistency loss.
    2. **Evaluation**: Called via ``generate()`` for single-step synthesis.
    3. **Ablations**: Reused across IC, OT, and GC coupling strategies.

    The class is a standard ``nn.Module`` — its parameters are managed by
    PyTorch and can be swapped by the ``EMA`` utility for endpoint prediction.

    Attributes:
        net: The ``SongUNet`` backbone F_θ. Contains all learnable parameters.
        sigma_min: Minimum noise level σ_0 (boundary condition). Default 0.002.
        sigma_data: Data distribution standard deviation σ_d. Default 0.5.
    """

    def __init__(
        self,
        net: SongUNet,
        sigma_min: float = 0.002,
        sigma_data: float = 0.5,
    ) -> None:
        """Initialise the consistency model with preconditioning constants.

        Args:
            net: Instantiated ``SongUNet`` backbone. All learnable parameters
                reside in this module. The ``ConsistencyModel`` registers it
                as a submodule so its parameters appear in
                ``model.parameters()`` and are saved in ``state_dict()``.
            sigma_min: Minimum noise level σ_0 used in ``c_skip`` and
                ``c_out``. Must match ``sigma_min`` in config.yaml (default
                0.002). Enforces the boundary condition: as σ → σ_min,
                c_skip → 1 and c_out → 0, so f_θ(x_0, σ_0) = x_0.
            sigma_data: Standard deviation of the data distribution σ_d.
                Used in all four preconditioning functions. Default 0.5
                matches the EDM convention (Karras et al., 2022) and the
                ``sigma_data: 0.5`` entry in config.yaml.

        Raises:
            ValueError: If ``sigma_min <= 0`` or ``sigma_data <= 0``.
            TypeError: If ``net`` is not a ``SongUNet`` instance.
        """
        super().__init__()

        if not isinstance(net, SongUNet):
            raise TypeError(
                f"Expected 'net' to be a SongUNet instance, "
                f"got {type(net).__name__}."
            )
        if sigma_min <= 0.0:
            raise ValueError(
                f"sigma_min must be positive, got {sigma_min}. "
                "Config default: sigma_min = 0.002."
            )
        if sigma_data <= 0.0:
            raise ValueError(
                f"sigma_data must be positive, got {sigma_data}. "
                "Config default: sigma_data = 0.5."
            )

        # Register SongUNet as a submodule so its parameters are tracked
        self.net: SongUNet = net

        # Fixed preconditioning constants (not learned, not nn.Parameter)
        self.sigma_min: float = float(sigma_min)
        self.sigma_data: float = float(sigma_data)

    # ------------------------------------------------------------------
    # Preconditioning functions
    # ------------------------------------------------------------------

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the skip-connection scaling coefficient c_skip(σ).

        Formula (Karras et al. 2022, adapted for consistency models):
            c_skip(σ) = σ_d² / (σ_d² + (σ - σ_0)²)

        Properties:
        - c_skip(σ_0) = 1.0  (boundary condition: full skip at σ_min)
        - c_skip(∞) → 0.0    (no skip at high noise)
        - Monotonically decreasing from 1 to 0

        The denominator σ_d² + (σ - σ_0)² is always strictly positive
        (sum of a positive constant and a non-negative term), so no
        numerical instability arises.

        Args:
            sigma: Noise level tensor of any broadcastable shape. Typically
                ``(B,)`` or ``(B, 1, 1, 1)`` depending on the calling context.
                Values should be in ``[sigma_min, sigma_max]``.

        Returns:
            Tensor of the same shape as ``sigma`` with values in ``(0, 1]``.
        """
        sigma_d_sq: float = self.sigma_data ** 2
        sigma_min_t: torch.Tensor = torch.tensor(
            self.sigma_min, dtype=sigma.dtype, device=sigma.device
        )
        numerator: torch.Tensor = torch.tensor(
            sigma_d_sq, dtype=sigma.dtype, device=sigma.device
        )
        denominator: torch.Tensor = sigma_d_sq + (sigma - sigma_min_t) ** 2
        return numerator / denominator

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the output scaling coefficient c_out(σ).

        Formula (Karras et al. 2022, adapted for consistency models):
            c_out(σ) = σ_d · (σ - σ_0) / sqrt(σ_d² + σ²)

        Properties:
        - c_out(σ_0) = 0.0   (boundary condition: zero output at σ_min)
        - c_out(∞) → σ_d     (approaches data std at high noise)
        - Together with c_skip, ensures f_θ(x_0, σ_0) = x_0

        The denominator sqrt(σ_d² + σ²) is always strictly positive since
        σ_d > 0, so no division-by-zero risk.

        Args:
            sigma: Noise level tensor of any broadcastable shape. Typically
                ``(B,)`` or ``(B, 1, 1, 1)`` depending on the calling context.

        Returns:
            Tensor of the same shape as ``sigma`` with non-negative values.
        """
        sigma_d: float = self.sigma_data
        sigma_min_t: torch.Tensor = torch.tensor(
            self.sigma_min, dtype=sigma.dtype, device=sigma.device
        )
        numerator: torch.Tensor = sigma_d * (sigma - sigma_min_t)
        denominator: torch.Tensor = torch.sqrt(
            torch.tensor(sigma_d ** 2, dtype=sigma.dtype, device=sigma.device)
            + sigma ** 2
        )
        return numerator / denominator

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the input scaling coefficient c_in(σ).

        Formula (Karras et al. 2022):
            c_in(σ) = 1 / sqrt(σ_d² + σ²)

        This normalises the network input so that ``c_in(σ) · x_t`` has
        approximately unit variance when ``x_t = x★ + σ·z`` has variance
        ``σ_d² + σ²`` (assuming x★ and z are independent with stds σ_d and 1).

        The denominator is always strictly positive (same as c_out).

        Args:
            sigma: Noise level tensor of any broadcastable shape. Typically
                ``(B,)`` or ``(B, 1, 1, 1)`` depending on the calling context.

        Returns:
            Tensor of the same shape as ``sigma`` with positive values.
        """
        denominator: torch.Tensor = torch.sqrt(
            torch.tensor(
                self.sigma_data ** 2, dtype=sigma.dtype, device=sigma.device
            )
            + sigma ** 2
        )
        return 1.0 / denominator

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the noise conditioning signal c_noise(σ).

        Formula (Karras et al. 2022):
            c_noise(σ) = log(σ) / 4

        Maps the noise level to a scalar conditioning signal for the network's
        time embedding. The division by 4 normalises the range:
        - For σ ∈ [0.002, 80]: log(σ) ∈ [-6.2, 4.4]
        - After scaling: c_noise ∈ [-1.55, 1.1]

        This scalar (or batch of scalars) is passed to ``SongUNet.forward``
        as the ``sigma`` argument, which internally converts it to a
        sinusoidal time embedding via ``PositionalEmbedding``.

        Since all sigma values in the schedule are strictly positive
        (sigma_min = 0.002 > 0), log(sigma) is always well-defined.

        Args:
            sigma: Noise level tensor of any shape. Values must be strictly
                positive. Typically shape ``(B,)`` for passing to the UNet.

        Returns:
            Tensor of the same shape as ``sigma``.
        """
        return torch.log(sigma) / 4.0

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the consistency model to a noisy image.

        Implements the full preconditioning pipeline:
            F_out = net(c_in(σ) · x, c_noise(σ))
            f_θ(x, σ) = c_skip(σ) · x + c_out(σ) · F_out

        The sigma tensor is reshaped to ``(B, 1, 1, 1)`` for broadcasting
        with image tensors ``(B, C, H, W)``, while the raw ``(B,)`` sigma
        is passed to the UNet for time conditioning.

        Boundary condition verification:
            When σ = σ_min = 0.002:
                c_skip(σ_min) = 1.0
                c_out(σ_min)  = 0.0
                → f_θ(x_0, σ_0) = 1.0 · x_0 + 0.0 · F_out = x_0  ✓

        Args:
            x: Noisy image tensor of shape ``(B, C, H, W)`` in ``[-1, 1]``.
                Represents ``x_t = x★ + σ_t · z`` at noise level σ.
            sigma: Per-sample noise levels of shape ``(B,)``. All values
                must be strictly positive and within ``[sigma_min, sigma_max]``.
                Within a batch, different samples can have different noise
                levels (due to per-sample timestep sampling in Algorithm 1).

        Returns:
            Denoised image estimate of shape ``(B, C, H, W)``. Values are
            approximately in ``[-1, 1]`` but not explicitly clipped.
            At σ = σ_min, the output equals the input exactly.

        Raises:
            RuntimeError: If ``x`` and ``sigma`` are on different devices.
        """
        batch_size: int = x.shape[0]

        # Ensure sigma is 1-D (B,) for UNet time conditioning
        sigma_1d: torch.Tensor = sigma.reshape(batch_size)

        # Reshape sigma to (B, 1, 1, 1) for broadcasting with image tensors
        sigma_4d: torch.Tensor = sigma_1d.reshape(batch_size, 1, 1, 1)

        # Step 1: Compute preconditioned input for the network
        # c_in(σ) normalises x_t to approximately unit variance
        x_in: torch.Tensor = self.c_in(sigma_4d) * x

        # Step 2: Compute noise conditioning signal for the UNet
        # c_noise maps σ → scalar in ~[-1.55, 1.1] for time embedding
        sigma_cond: torch.Tensor = self.c_noise(sigma_1d)

        # Step 3: Raw network forward pass
        # SongUNet.forward(x, sigma) expects:
        #   x:     (B, C, H, W) — preconditioned input
        #   sigma: (B,)         — noise conditioning scalar
        f_out: torch.Tensor = self.net(x_in, sigma_cond)

        # Step 4: Apply skip connection and output scaling
        # c_skip(σ) · x + c_out(σ) · F_out
        # Both c_skip and c_out are broadcast via (B, 1, 1, 1) sigma_4d
        output: torch.Tensor = (
            self.c_skip(sigma_4d) * x
            + self.c_out(sigma_4d) * f_out
        )

        return output

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        z: torch.Tensor,
        sigma_T: float = 80.0,
    ) -> torch.Tensor:
        """Generate images from noise in a single forward pass (one NFE).

        This is the key property of consistency models: they map directly
        from noise to data in one neural function evaluation, unlike diffusion
        models that require many steps.

        The generation procedure:
            x_T = σ_T · z                    (scale noise to σ_T level)
            x̂_0 = f_θ(x_T, σ_T)             (single-step denoising)

        where σ_T = sigma_max = 80.0 from config.yaml. At this noise level,
        p_T ≈ N(0, σ_T² · I), so σ_T · z ≈ x_T for z ~ N(0, I).

        The output is the consistency model's estimate of the clean data
        point x★ corresponding to the noisy input x_T.

        Args:
            z: Standard Gaussian noise tensor of shape ``(B, C, H, W)``.
               Typically sampled as ``torch.randn(B, C, H, W)``.
               Must be on the same device as the model parameters.
            sigma_T: Maximum noise level for generation. Should match
               ``sigma_max`` from config.yaml (default 80.0). This is the
               noise level at which the diffusion process starts.

        Returns:
            Generated image tensor of shape ``(B, C, H, W)``. Values are
            approximately in ``[-1, 1]`` (not explicitly clipped here;
            the evaluator clips before metric computation).

        Example::

            model.eval()
            with torch.no_grad():
                z = torch.randn(16, 3, 32, 32, device='cuda')
                images = model.generate(z, sigma_T=80.0)
                # images: (16, 3, 32, 32), values ≈ [-1, 1]
        """
        batch_size: int = z.shape[0]

        # Construct noisy input at the maximum noise level
        # x_T = σ_T · z, where z ~ N(0, I)
        x_T: torch.Tensor = sigma_T * z

        # Create per-sample sigma tensor: all samples at σ_T
        sigma: torch.Tensor = torch.full(
            (batch_size,),
            fill_value=sigma_T,
            dtype=z.dtype,
            device=z.device,
        )

        # Single-step denoising: f_θ(x_T, σ_T) → x̂_0
        return self.forward(x_T, sigma)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a human-readable summary of the consistency model."""
        num_params: int = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return (
            f"ConsistencyModel(\n"
            f"  net={self.net.__class__.__name__}("
            f"img_resolution={self.net.img_resolution}, "
            f"model_channels={self.net.model_channels}),\n"
            f"  sigma_min={self.sigma_min},\n"
            f"  sigma_data={self.sigma_data},\n"
            f"  trainable_params={num_params:,}\n"
            f")"
        )
