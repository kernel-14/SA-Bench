## training/schedules.py
"""Noise and timestep scheduling for iCT-GC consistency model training.

This module implements ``NoiseSchedule``, which encapsulates all scheduling
mathematics from Karras et al. (2022) and Song & Dhariwal (2024) that govern
how consistency model training progresses over time.

The module is a pure computation layer with no neural network components.
It handles:

1. **Noise schedule** (Karras et al. 2022): Maps discrete timestep indices to
   sigma values using a power-law interpolation in sigma^(1/rho) space.

2. **Progressive discretization** (Song & Dhariwal 2024): Exponentially
   increases the number of timesteps N during training, starting coarse and
   refining over time for training stability.

3. **Timestep sampling distribution** (Song & Dhariwal 2024): A log-normal
   distribution over sigma values that concentrates training on intermediate
   noise levels where consistency is hardest to enforce.

4. **Loss weighting**: The lambda(sigma_i) = 1 / (sigma_{i+1} - sigma_i)
   weighting that emphasizes consistency at low noise levels.

Config values used (from config.yaml defaults section):
    sigma_min: 0.002    (sigma_0, boundary condition noise level)
    sigma_max: 80.0     (sigma_T, maximum noise level)
    rho:       7.0      (noise schedule exponent)
    P_mean:   -1.1      (log-normal timestep distribution mean)
    P_std:     2.0      (log-normal timestep distribution std)
    s0:        10       (initial number of discretization intervals)
    s1:        1280     (final number of discretization intervals)

Typical usage in the training loop::

    schedule = NoiseSchedule(
        sigma_min=0.002, sigma_max=80.0, rho=7.0,
        P_mean=-1.1, P_std=2.0
    )

    # At each training step k:
    N = schedule.get_N(k, K=100000, s0=10, s1=1280)
    sigmas = schedule.get_sigmas(N)                          # (N+1,)
    indices = schedule.sample_timestep_indices(sigmas, B)    # (B,) LongTensor
    sigma_i  = sigmas[indices]                               # (B,)
    sigma_i1 = sigmas[indices + 1]                           # (B,)
    lam = schedule.get_lambda(sigma_i, sigma_i1)             # (B,)
    loss = (lam * distance_fn(f_upper, f_lower.detach())).mean()
"""

import math
from typing import Optional

import torch

# scipy.special.erf is imported as a fallback for CPU-only environments.
# The primary implementation uses torch.special.erf for GPU compatibility.
try:
    from scipy.special import erf as scipy_erf
    _SCIPY_AVAILABLE: bool = True
except ImportError:
    _SCIPY_AVAILABLE = False


class NoiseSchedule:
    """Encapsulates all scheduling logic for iCT-GC consistency model training.

    This class is a pure computation utility — it holds no ``nn.Module``
    components and performs no gradient computation. All methods return
    ``torch.Tensor`` objects (or Python ``int`` for ``get_N``).

    The class is stateless after ``__init__``: calling any method multiple
    times with the same arguments returns identical results. This makes it
    safe to share a single ``NoiseSchedule`` instance across the entire
    training run.

    Attributes:
        sigma_min: Minimum noise level σ_0. Default 0.002. Corresponds to
            the boundary condition: p_0 ≈ p★ (data distribution).
        sigma_max: Maximum noise level σ_T. Default 80.0. Corresponds to
            p_T ≈ N(0, σ_T² · I) (pure noise).
        rho: Noise schedule exponent. Default 7.0. Controls the density of
            timesteps near the data end (higher rho → more timesteps at low
            noise). Value 7 is the EDM default.
        P_mean: Mean of the log-normal timestep sampling distribution.
            Default -1.1. Concentrates training around σ ≈ exp(-1.1) ≈ 0.33.
        P_std: Standard deviation of the log-normal timestep sampling
            distribution. Default 2.0. Controls the spread of training
            across noise levels.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        P_mean: float = -1.1,
        P_std: float = 2.0,
    ) -> None:
        """Initialise the noise schedule with EDM and iCT hyperparameters.

        All parameter values are taken directly from config.yaml (defaults
        section). No computation is performed at initialisation time.

        Args:
            sigma_min: Minimum noise level σ_0. Must be strictly positive.
                Config default: 0.002. Chosen so that p_0 ≈ p★.
            sigma_max: Maximum noise level σ_T. Must be greater than
                sigma_min. Config default: 80.0. Chosen so that
                p_T ≈ N(0, σ_T² · I).
            rho: Exponent for the power-law noise schedule. Must be positive.
                Config default: 7.0 (EDM convention). Higher values
                concentrate more timesteps near the data end.
            P_mean: Mean of the log-normal distribution used for timestep
                sampling. Config default: -1.1. In log-sigma space, this
                corresponds to σ ≈ 0.33.
            P_std: Standard deviation of the log-normal distribution used
                for timestep sampling. Config default: 2.0. Controls how
                broadly training is spread across noise levels.

        Raises:
            ValueError: If ``sigma_min <= 0``, ``sigma_max <= sigma_min``,
                ``rho <= 0``, or ``P_std <= 0``.
        """
        if sigma_min <= 0.0:
            raise ValueError(
                f"sigma_min must be strictly positive, got {sigma_min}. "
                "Config default: sigma_min = 0.002."
            )
        if sigma_max <= sigma_min:
            raise ValueError(
                f"sigma_max ({sigma_max}) must be greater than "
                f"sigma_min ({sigma_min}). "
                "Config defaults: sigma_min=0.002, sigma_max=80.0."
            )
        if rho <= 0.0:
            raise ValueError(
                f"rho must be strictly positive, got {rho}. "
                "Config default: rho = 7.0."
            )
        if P_std <= 0.0:
            raise ValueError(
                f"P_std must be strictly positive, got {P_std}. "
                "Config default: P_std = 2.0."
            )

        self.sigma_min: float = float(sigma_min)
        self.sigma_max: float = float(sigma_max)
        self.rho: float = float(rho)
        self.P_mean: float = float(P_mean)
        self.P_std: float = float(P_std)

    # ------------------------------------------------------------------
    # Noise schedule
    # ------------------------------------------------------------------

    def get_sigmas(self, N: int) -> torch.Tensor:
        """Compute the noise schedule sigma values for N discretization steps.

        Implements the Karras et al. (2022) power-law schedule:
            sigma_i = (sigma_min^(1/rho) + (i/N) * (sigma_max^(1/rho)
                       - sigma_min^(1/rho)))^rho
            for i in {0, 1, ..., N}

        The schedule interpolates linearly in sigma^(1/rho) space, which
        with rho=7 produces a schedule heavily weighted toward lower noise
        levels (more timesteps near the data end). This is intentional:
        consistency at low noise is most important for generation quality.

        Properties of the returned schedule:
        - sigmas[0] = sigma_min = 0.002  (boundary, close to data)
        - sigmas[N] = sigma_max = 80.0   (maximum noise, close to pure Gaussian)
        - Monotonically increasing: sigmas[i] < sigmas[i+1] for all i
        - Shape: (N+1,) float32 tensor

        Args:
            N: Number of discretization intervals. Must be a positive integer.
                Determined by ``get_N(k, K, s0, s1)`` during training.
                Ranges from s0=10 (early training) to s1=1280 (late training).

        Returns:
            Float32 tensor of shape ``(N+1,)`` containing sigma values in
            ascending order from sigma_min to sigma_max.

        Raises:
            ValueError: If ``N < 1``.

        Example::

            schedule = NoiseSchedule(sigma_min=0.002, sigma_max=80.0, rho=7.0)
            sigmas = schedule.get_sigmas(N=10)
            # sigmas[0]  ≈ 0.002  (sigma_min)
            # sigmas[10] ≈ 80.0   (sigma_max)
            # sigmas.shape == (11,)
        """
        if N < 1:
            raise ValueError(
                f"N must be at least 1, got {N}. "
                "N represents the number of discretization intervals."
            )

        # Precompute the endpoints in sigma^(1/rho) space
        inv_rho: float = 1.0 / self.rho
        sigma_min_inv_rho: float = self.sigma_min ** inv_rho
        sigma_max_inv_rho: float = self.sigma_max ** inv_rho

        # Create index tensor i ∈ {0, 1, ..., N} — shape (N+1,)
        # Using float32 for the interpolation arithmetic
        i: torch.Tensor = torch.arange(
            start=0,
            end=N + 1,
            dtype=torch.float32,
        )

        # Linear interpolation in sigma^(1/rho) space:
        # interp_i = sigma_min^(1/rho) + (i/N) * (sigma_max^(1/rho) - sigma_min^(1/rho))
        interp: torch.Tensor = (
            sigma_min_inv_rho
            + (i / float(N)) * (sigma_max_inv_rho - sigma_min_inv_rho)
        )

        # Raise to the power rho to recover sigma values:
        # sigma_i = interp_i^rho
        sigmas: torch.Tensor = interp ** self.rho

        return sigmas

    # ------------------------------------------------------------------
    # Progressive discretization schedule
    # ------------------------------------------------------------------

    def get_N(
        self,
        k: int,
        K: int,
        s0: int = 10,
        s1: int = 1280,
    ) -> int:
        """Compute the number of discretization intervals at training step k.

        Implements the exponential progressive discretization schedule from
        Song & Dhariwal (2024):
            K' = floor(K / (log2(s1/s0) + 1))
            N(k) = min(s0 * 2^floor(k / K'), s1) + 1

        The schedule doubles N every K' steps, starting from s0 and capping
        at s1. The +1 ensures there are N+1 sigma values (N intervals).

        Behavior over training (with K=100000, s0=10, s1=1280):
            K' = floor(100000 / (log2(128) + 1)) = floor(100000 / 8) = 12500
            k=0:       N = min(10 * 2^0, 1280) + 1 = 11
            k=12500:   N = min(10 * 2^1, 1280) + 1 = 21
            k=25000:   N = min(10 * 2^2, 1280) + 1 = 41
            k=37500:   N = min(10 * 2^3, 1280) + 1 = 81
            k=50000:   N = min(10 * 2^4, 1280) + 1 = 161
            k=62500:   N = min(10 * 2^5, 1280) + 1 = 321
            k=75000:   N = min(10 * 2^6, 1280) + 1 = 641
            k=87500:   N = min(10 * 2^7, 1280) + 1 = 1281
            k≥87500:   N = 1281 (capped at s1+1)

        The progressive increase stabilises early training by starting with
        a coarse discretization and gradually refining it.

        Args:
            k: Current training step (0-indexed). Must be non-negative.
            K: Total number of training steps. E.g. 100000 for CIFAR-10,
                150000 for CelebA/LSUN/ImageNet. From config.yaml.
            s0: Initial number of discretization intervals. Config default: 10.
                The model starts with N = s0 + 1 = 11 sigma values.
            s1: Final (maximum) number of discretization intervals. Config
                default: 1280. The model ends with N = s1 + 1 = 1281 sigma
                values.

        Returns:
            Integer N(k) representing the number of discretization intervals
            at step k. Always in the range [s0+1, s1+1].

        Raises:
            ValueError: If ``k < 0``, ``K <= 0``, ``s0 < 1``, or
                ``s1 < s0``.

        Example::

            schedule = NoiseSchedule()
            # CIFAR-10: K=100000, s0=10, s1=1280
            assert schedule.get_N(0,      100000) == 11
            assert schedule.get_N(12500,  100000) == 21
            assert schedule.get_N(87500,  100000) == 1281
            assert schedule.get_N(100000, 100000) == 1281
        """
        if k < 0:
            raise ValueError(f"Training step k must be non-negative, got {k}.")
        if K <= 0:
            raise ValueError(
                f"Total training steps K must be positive, got {K}."
            )
        if s0 < 1:
            raise ValueError(
                f"s0 must be at least 1, got {s0}. Config default: s0=10."
            )
        if s1 < s0:
            raise ValueError(
                f"s1 ({s1}) must be >= s0 ({s0}). "
                "Config defaults: s0=10, s1=1280."
            )

        # Compute K': the number of steps between each doubling of N
        # K' = floor(K / (log2(s1/s0) + 1))
        # With K=100000, s0=10, s1=1280: log2(128)=7, K'=floor(100000/8)=12500
        log2_ratio: float = math.log2(float(s1) / float(s0))
        K_prime: int = math.floor(float(K) / (log2_ratio + 1.0))

        # Guard against K_prime=0 (would cause division by zero)
        # This can happen if K is very small relative to the ratio
        if K_prime < 1:
            K_prime = 1

        # Compute the doubling exponent: floor(k / K')
        exponent: int = math.floor(float(k) / float(K_prime))

        # Compute N_raw = s0 * 2^exponent, then clamp to s1
        N_raw: int = s0 * (2 ** exponent)
        N_clamped: int = min(N_raw, s1)

        # Add 1: N intervals require N+1 sigma values
        return N_clamped + 1

    # ------------------------------------------------------------------
    # Timestep sampling distribution
    # ------------------------------------------------------------------

    def get_timestep_weights(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Compute the probability distribution over timestep intervals.

        Implements the discrete approximation to the log-normal timestep
        distribution from Song & Dhariwal (2024) / Karras et al. (2022):

            p(sigma_i) ∝ erf((log(sigma_{i+1}) - P_mean) / (sqrt(2) * P_std))
                        - erf((log(sigma_i)   - P_mean) / (sqrt(2) * P_std))
            for i in {0, 1, ..., N-1}

        This is the CDF difference of a log-normal distribution evaluated
        at consecutive sigma values, giving the probability mass in each
        interval [sigma_i, sigma_{i+1}].

        With P_mean=-1.1 and P_std=2.0, the distribution concentrates
        training around sigma ≈ exp(-1.1) ≈ 0.33, with broad coverage
        across the full range [sigma_min, sigma_max].

        The weights are normalised to sum to 1 and clamped to be non-negative
        (numerical safety for edge cases where erf differences could be
        slightly negative due to floating-point precision).

        Args:
            sigmas: Noise schedule tensor of shape ``(N+1,)`` as returned by
                ``get_sigmas(N)``. Must be strictly positive and monotonically
                increasing. Typically on CPU or GPU depending on the training
                device.

        Returns:
            Float32 tensor of shape ``(N,)`` containing normalised probability
            weights summing to 1.0. On the same device as ``sigmas``.

        Raises:
            ValueError: If ``sigmas`` has fewer than 2 elements (need at
                least one interval).

        Example::

            schedule = NoiseSchedule(P_mean=-1.1, P_std=2.0)
            sigmas = schedule.get_sigmas(N=10)
            weights = schedule.get_timestep_weights(sigmas)
            # weights.shape == (10,)
            # weights.sum() ≈ 1.0
            # weights are highest around sigma ≈ 0.33
        """
        if sigmas.numel() < 2:
            raise ValueError(
                f"sigmas must have at least 2 elements to define one interval, "
                f"got {sigmas.numel()} elements."
            )

        # Extract lower and upper sigma boundaries for each interval
        # sigmas_lower: sigma_i   for i in {0, ..., N-1}, shape (N,)
        # sigmas_upper: sigma_{i+1} for i in {0, ..., N-1}, shape (N,)
        sigmas_lower: torch.Tensor = sigmas[:-1]  # shape (N,)
        sigmas_upper: torch.Tensor = sigmas[1:]   # shape (N,)

        # Compute the erf arguments:
        # arg = (log(sigma) - P_mean) / (sqrt(2) * P_std)
        sqrt2_P_std: float = math.sqrt(2.0) * self.P_std

        # log(sigma) is safe since all sigma values are strictly positive
        # (sigma_min = 0.002 > 0 by construction)
        arg_upper: torch.Tensor = (
            torch.log(sigmas_upper) - self.P_mean
        ) / sqrt2_P_std

        arg_lower: torch.Tensor = (
            torch.log(sigmas_lower) - self.P_mean
        ) / sqrt2_P_std

        # Apply erf using torch.special.erf for GPU compatibility
        # torch.special.erf is available in PyTorch >= 1.8
        erf_upper: torch.Tensor = torch.special.erf(arg_upper)
        erf_lower: torch.Tensor = torch.special.erf(arg_lower)

        # Compute probability mass in each interval: erf(upper) - erf(lower)
        # Since erf is monotonically increasing and sigmas is monotonically
        # increasing, these differences should be non-negative. We clamp for
        # numerical safety.
        weights: torch.Tensor = (erf_upper - erf_lower).clamp(min=0.0)

        # Normalise to sum to 1
        weight_sum: torch.Tensor = weights.sum()

        # Guard against degenerate case where all weights are zero
        # (should not happen with valid sigma schedules and P_mean/P_std)
        if weight_sum.item() <= 0.0:
            # Fall back to uniform distribution
            weights = torch.ones_like(weights)
            weight_sum = weights.sum()

        weights = weights / weight_sum

        return weights

    def sample_timestep_indices(
        self,
        sigmas: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Sample per-sample timestep indices from the training distribution.

        Draws ``batch_size`` independent samples from the discrete probability
        distribution defined by ``get_timestep_weights(sigmas)``. Each sample
        is an index ``i ∈ {0, ..., N-1}`` identifying a timestep interval
        ``[sigma_i, sigma_{i+1}]``.

        The returned indices are used in the training loop to construct
        per-sample noisy images:
            sigma_i  = sigmas[indices]       # lower noise level
            sigma_i1 = sigmas[indices + 1]   # upper noise level
            x_ti  = x_star + sigma_i.view(B,1,1,1)  * z
            x_ti1 = x_star + sigma_i1.view(B,1,1,1) * z

        Args:
            sigmas: Noise schedule tensor of shape ``(N+1,)`` as returned by
                ``get_sigmas(N)``. Used to compute the sampling weights.
                The returned indices are on the same device as ``sigmas``.
            batch_size: Number of indices to sample. Typically equals the
                training batch size (e.g. 512 for CIFAR-10).

        Returns:
            ``LongTensor`` of shape ``(batch_size,)`` with values in
            ``{0, 1, ..., N-1}``. On the same device as ``sigmas``.

        Raises:
            ValueError: If ``batch_size < 1``.

        Example::

            schedule = NoiseSchedule()
            sigmas = schedule.get_sigmas(N=100)
            indices = schedule.sample_timestep_indices(sigmas, batch_size=512)
            # indices.shape == (512,)
            # indices.dtype == torch.int64
            # all(0 <= indices < 100)
            sigma_i  = sigmas[indices]       # (512,)
            sigma_i1 = sigmas[indices + 1]   # (512,)
        """
        if batch_size < 1:
            raise ValueError(
                f"batch_size must be at least 1, got {batch_size}."
            )

        # Compute normalised probability weights over N intervals
        # weights: (N,) float32 tensor on same device as sigmas
        weights: torch.Tensor = self.get_timestep_weights(sigmas)

        # Sample batch_size indices with replacement from the distribution
        # torch.multinomial expects a 1-D or 2-D weight tensor.
        # With replacement=True, each draw is independent.
        indices: torch.Tensor = torch.multinomial(
            input=weights,
            num_samples=batch_size,
            replacement=True,
        )

        # indices is already a LongTensor on the same device as weights/sigmas
        return indices

    # ------------------------------------------------------------------
    # Loss weighting
    # ------------------------------------------------------------------

    def get_lambda(
        self,
        sigma_i: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the per-sample consistency loss weighting lambda(sigma_i).

        Implements the loss weighting from Appendix D of the paper:
            lambda(sigma_i) = 1 / (sigma_{i+1} - sigma_i)

        This weighting emphasizes consistency at low noise levels (small
        sigma gaps → large lambda), which is where the consistency property
        is most important for generation quality. Combined with the noise
        schedule (which has smaller gaps at low sigma due to rho=7), this
        creates a strong training signal near the data distribution.

        The denominator (sigma_next - sigma_i) is always strictly positive
        because the noise schedule is monotonically increasing by construction.
        No numerical instability arises.

        Args:
            sigma_i: Per-sample lower noise levels of shape ``(B,)`` or
                broadcastable. Typically ``sigmas[indices]`` where ``indices``
                is the output of ``sample_timestep_indices``.
            sigma_next: Per-sample upper noise levels of shape ``(B,)`` or
                broadcastable. Typically ``sigmas[indices + 1]``.
                Must satisfy ``sigma_next > sigma_i`` element-wise.

        Returns:
            Float tensor of the same shape as the inputs containing the
            per-sample loss weights. Values are strictly positive.

        Example::

            schedule = NoiseSchedule()
            sigmas = schedule.get_sigmas(N=100)
            indices = schedule.sample_timestep_indices(sigmas, batch_size=4)
            sigma_i  = sigmas[indices]
            sigma_i1 = sigmas[indices + 1]
            lam = schedule.get_lambda(sigma_i, sigma_i1)
            # lam.shape == (4,)
            # lam values are large for small sigma gaps (low noise)
            # Usage: loss = (lam * distance_fn(f_upper, f_lower.detach())).mean()
        """
        # Compute the sigma gap for each sample in the batch
        # sigma_gap > 0 always since the schedule is strictly increasing
        sigma_gap: torch.Tensor = sigma_next - sigma_i

        # Lambda = 1 / gap: large for small gaps (low noise), small for large gaps
        return 1.0 / sigma_gap

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a human-readable summary of the noise schedule configuration."""
        return (
            f"NoiseSchedule(\n"
            f"  sigma_min={self.sigma_min},\n"
            f"  sigma_max={self.sigma_max},\n"
            f"  rho={self.rho},\n"
            f"  P_mean={self.P_mean},\n"
            f"  P_std={self.P_std}\n"
            f")"
        )
