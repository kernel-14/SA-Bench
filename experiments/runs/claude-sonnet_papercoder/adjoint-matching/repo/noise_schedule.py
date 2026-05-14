## noise_schedule.py
"""Noise schedule mathematics for Adjoint Matching experiments.

This module implements all mathematical formulas for the memoryless noise
schedule from Section 4.3, Proposition 1, and Table 1 of the paper:

    "Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models
     with Memoryless Stochastic Optimal Control"

The reference flow uses α_t = t, β_t = 1 - t (Flow Matching convention).
The memoryless noise schedule is σ(t) = sqrt(2 * η_t) where:
    η_t = β_t * (α̇_t/α_t * β_t - β̇_t) = (1-t)/t

With the practical offset from Appendix G.1:
    σ(t) = sqrt(2 * (1-t+h) / (t+h))   where h = 1/K = 0.025

This file has NO dependencies on other project files. It only uses
Python stdlib (math, typing) and PyTorch for the tensor method.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch


class NoiseSchedule:
    """Encodes all mathematical formulas for the memoryless noise schedule.

    Implements the unified SDE framework from equations (10)-(11) of the paper
    with the specific choice α_t = t, β_t = 1 - t (Flow Matching convention,
    Section 2.1 and Table 1).

    The key formulas are:
        κ_t = α̇_t / α_t = 1/t
        η_t = β_t * (α̇_t/α_t * β_t - β̇_t) = (1-t)/t
        σ_memoryless(t) = sqrt(2 * η_t) = sqrt(2*(1-t)/t)

    With practical offset (Appendix G.1):
        σ_memoryless(t) = sqrt(2*(1-t+h)/(t+h))

    The base drift under the memoryless schedule (σ²/2 = η_t):
        b(x, t) = κ_t * x + 2*η_t * s(x, t)
                = 2 * v_base(x, t) - κ_t * x

    Attributes:
        h: Step size offset used in the practical sigma formula (Appendix G.1).
            Equals 1/K = 0.025 for K=40 timesteps. Sourced from
            config.yaml noise_schedule.offset and sampling.h.

    Example:
        >>> ns = NoiseSchedule(h=0.025)
        >>> ns.sigma_memoryless(0.025)   # First step: large sigma
        6.324555320336759
        >>> ns.sigma_memoryless(0.975)   # Last step: small sigma
        0.31622776601683794
        >>> ns.get_timesteps(K=40)[:3]
        [0.025, 0.05, 0.075]
    """

    def __init__(self, h: float = 0.025) -> None:
        """Initialize the noise schedule with the step size offset.

        Args:
            h: Step size h = 1/K used as the practical offset in the sigma
               formula (Appendix G.1). Default 0.025 corresponds to K=40
               timesteps as specified in config.yaml (sampling.h: 0.025 and
               noise_schedule.offset: 0.025).

               The offset prevents division by zero at t=0 and allows slightly
               more deviation near t=1 for faster fine-tuning convergence.

        Raises:
            ValueError: If h <= 0 or h >= 1.
        """
        if h <= 0.0 or h >= 1.0:
            raise ValueError(
                f"Step size h must be in (0, 1), got h={h}. "
                f"For K=40 timesteps, use h=0.025."
            )
        self.h: float = h

    # ------------------------------------------------------------------
    # Reference flow coefficients: α_t = t, β_t = 1 - t
    # ------------------------------------------------------------------

    def alpha_fn(self, t: float) -> float:
        """Return α_t = t (data interpolation coefficient).

        The reference flow is X̄_t = β_t * X̄_0 + α_t * X̄_1.
        Boundary conditions: α_0 = 0, α_1 = 1.

        Args:
            t: Continuous time in [0, 1].

        Returns:
            α_t = t.
        """
        return float(t)

    def beta_fn(self, t: float) -> float:
        """Return β_t = 1 - t (noise interpolation coefficient).

        The reference flow is X̄_t = β_t * X̄_0 + α_t * X̄_1.
        Boundary conditions: β_0 = 1, β_1 = 0.

        Args:
            t: Continuous time in [0, 1].

        Returns:
            β_t = 1 - t.
        """
        return 1.0 - float(t)

    def alpha_dot_fn(self, t: float) -> float:
        """Return α̇_t = d/dt α_t = 1 (time derivative of α_t = t).

        Args:
            t: Continuous time in [0, 1]. Unused since derivative is constant.

        Returns:
            α̇_t = 1.0 (constant).
        """
        return 1.0

    def beta_dot_fn(self, t: float) -> float:
        """Return β̇_t = d/dt β_t = -1 (time derivative of β_t = 1 - t).

        Args:
            t: Continuous time in [0, 1]. Unused since derivative is constant.

        Returns:
            β̇_t = -1.0 (constant).
        """
        return -1.0

    # ------------------------------------------------------------------
    # Unified SDE coefficients (equations 10-11)
    # ------------------------------------------------------------------

    def kappa(self, t: float) -> float:
        """Return κ_t = α̇_t / α_t = 1/t (linear drift coefficient).

        From equation (11): b(x, t) = κ_t * x + (σ²/2 + η_t) * s(x, t).
        For α_t = t: κ_t = 1/t.

        Note: This is undefined at t=0. In practice, timesteps start at
        t=h=0.025 (config.yaml sampling.t_start: 0.025), so t=0 is never
        encountered during training.

        At t=0.025 (first step): κ = 40.0 (large, compensated by large σ).
        At t=0.975 (last step): κ ≈ 1.026 (near 1.0).

        Args:
            t: Continuous time in (0, 1]. Must be strictly positive.

        Returns:
            κ_t = 1/t.

        Raises:
            ZeroDivisionError: If t == 0.0 exactly.
        """
        return 1.0 / float(t)

    def eta(self, t: float) -> float:
        """Return η_t = β_t * (α̇_t/α_t * β_t - β̇_t) = (1-t)/t.

        From equation (11) and Table 1 (Flow Matching row):
            η_t = β_t * (κ_t * β_t - β̇_t)
                = (1-t) * ((1-t)/t - (-1))
                = (1-t) * ((1-t)/t + 1)
                = (1-t) * (1/t)
                = (1-t)/t

        This coefficient determines the score function contribution to the
        base drift and the memoryless noise schedule magnitude.

        Properties:
            - η_t ≥ 0 for all t ∈ (0, 1] (required for valid SDE, Lemma 1)
            - η_t → ∞ as t → 0 (large mixing near noise)
            - η_t → 0 as t → 1 (stabilizing near data)

        Verification:
            t=0.025: η = 0.975/0.025 = 39.0
            t=0.5:   η = 0.5/0.5 = 1.0
            t=0.975: η = 0.025/0.975 ≈ 0.0256

        Args:
            t: Continuous time in (0, 1]. Must be strictly positive.

        Returns:
            η_t = (1-t)/t ≥ 0.

        Raises:
            ZeroDivisionError: If t == 0.0 exactly.
        """
        t_f: float = float(t)
        return (1.0 - t_f) / t_f

    # ------------------------------------------------------------------
    # Memoryless noise schedule (Proposition 1, Table 1, Appendix G.1)
    # ------------------------------------------------------------------

    def sigma_memoryless(
        self,
        t: float,
        h: Optional[float] = None,
    ) -> float:
        """Return the memoryless noise schedule σ(t) with practical offset.

        Implements the memoryless noise schedule from Proposition 1 and Table 1:
            σ(t) = sqrt(2 * η_t) = sqrt(2*(1-t)/t)

        With the practical offset from Appendix G.1 to avoid division by zero
        at t=0 and to allow slightly more deviation near t=1:
            σ(t) = sqrt(2 * (1-t+h) / (t+h))

        This is the ONLY sigma formula used throughout the project for the
        memoryless schedule. The offset h is sourced from config.yaml:
            noise_schedule.offset: 0.025
            sampling.h: 0.025

        Properties:
            - Monotonically decreasing in t (large noise early, small late)
            - σ(t) → ∞ as t → 0 (enormous mixing near noise X_0)
            - σ(t) → 0 as t → 1 (stabilizing near data X_1)
            - Always positive for t ∈ [0, 1] with h > 0

        Verification (with h=0.025):
            t=0.025: σ = sqrt(2 * 1.0 / 0.05) = sqrt(40) ≈ 6.325
            t=0.5:   σ = sqrt(2 * 0.525 / 0.525) = sqrt(2) ≈ 1.414
            t=0.975: σ = sqrt(2 * 0.05 / 1.0) = sqrt(0.1) ≈ 0.316

        Args:
            t: Continuous time in [0, 1].
            h: Optional override for the offset. If None, uses self.h.
               Useful for ablation studies testing different offsets.

        Returns:
            σ(t) = sqrt(2*(1-t+h_eff)/(t+h_eff)) > 0.
        """
        h_eff: float = h if h is not None else self.h
        numerator: float = 2.0 * (1.0 - float(t) + h_eff)
        denominator: float = float(t) + h_eff
        return math.sqrt(numerator / denominator)

    def sigma_memoryless_tensor(
        self,
        t_tensor: torch.Tensor,
        h: Optional[float] = None,
    ) -> torch.Tensor:
        """Batched tensor version of sigma_memoryless for vectorized computation.

        Computes σ(t) = sqrt(2*(1-t+h)/(t+h)) element-wise on a tensor.
        Used in losses.py when computing the Adjoint Matching loss over
        multiple timesteps simultaneously.

        The computation is performed on the same device and dtype as t_tensor.
        For numerical safety, the argument to sqrt is clamped to a small
        positive value to prevent NaN from floating-point edge cases.

        Args:
            t_tensor: Float tensor of timestep values in [0, 1]. Can be any
                shape (scalar, 1D, 2D, etc.). Must be a floating-point dtype.
            h: Optional override for the offset. If None, uses self.h.

        Returns:
            Tensor of same shape and device as t_tensor containing σ(t) values.
            All values are strictly positive.

        Example:
            >>> ns = NoiseSchedule(h=0.025)
            >>> t = torch.tensor([0.025, 0.5, 0.975])
            >>> ns.sigma_memoryless_tensor(t)
            tensor([6.3246, 1.4142, 0.3162])
        """
        h_eff: float = h if h is not None else self.h
        h_tensor: torch.Tensor = torch.tensor(
            h_eff, dtype=t_tensor.dtype, device=t_tensor.device
        )
        numerator: torch.Tensor = 2.0 * (1.0 - t_tensor + h_tensor)
        denominator: torch.Tensor = t_tensor + h_tensor
        # Clamp for numerical safety (argument to sqrt must be positive)
        ratio: torch.Tensor = torch.clamp(numerator / denominator, min=1e-8)
        return torch.sqrt(ratio)

    # ------------------------------------------------------------------
    # Timestep grid
    # ------------------------------------------------------------------

    def get_timesteps(self, K: Optional[int] = None) -> List[float]:
        """Return the list of K discretization timesteps [h, 2h, ..., 1.0].

        Generates the uniform timestep grid used in Algorithm 1 (equation 40).
        The grid starts at h (not 0) because t=0 is the initial state X_0,
        and the SDE steps are applied at t ∈ {h, 2h, ..., Kh=1.0}.

        From config.yaml (Appendix G.1):
            K: 40
            t_start: 0.025
            t_end: 1.0
            Timesteps: {0, 0.025, 0.05, ..., 0.975}

        The trajectory has K+1 states: X_0, X_h, X_{2h}, ..., X_{Kh=1}.
        This method returns the K transition timesteps (not including t=0).

        Floating-point rounding is applied to prevent accumulation errors
        (e.g., 0.025 * 3 = 0.07500000000000001 → rounded to 0.075).

        Args:
            K: Number of timesteps. If None, inferred as round(1.0 / self.h).
               For h=0.025: K = round(1.0/0.025) = 40.

        Returns:
            List of K float values: [h, 2h, 3h, ..., K*h=1.0].
            Length is exactly K. Last element is exactly 1.0.

        Example:
            >>> ns = NoiseSchedule(h=0.025)
            >>> ts = ns.get_timesteps(K=40)
            >>> len(ts)
            40
            >>> ts[0]
            0.025
            >>> ts[-1]
            1.0
            >>> ts[:4]
            [0.025, 0.05, 0.075, 0.1]
        """
        if K is None:
            K = round(1.0 / self.h)

        timesteps: List[float] = []
        for i in range(1, K + 1):
            # Round to 10 decimal places to prevent floating-point accumulation
            t_val: float = round(i * self.h, 10)
            timesteps.append(t_val)

        # Ensure the last timestep is exactly 1.0 (boundary condition α_1 = 1)
        if timesteps:
            timesteps[-1] = 1.0

        return timesteps

    # ------------------------------------------------------------------
    # Base drift coefficients
    # ------------------------------------------------------------------

    def base_drift_coeff(self, t: float) -> Tuple[float, float]:
        """Return (κ_t, 2*η_t) — coefficients for the base drift.

        Under the memoryless noise schedule (σ²/2 = η_t), the base drift
        from equation (11) simplifies to:
            b(x, t) = κ_t * x + (σ²/2 + η_t) * s(x, t)
                    = κ_t * x + 2*η_t * s(x, t)

        This is equivalent to (using equation 8 for v_base):
            b(x, t) = 2 * v_base(x, t) - κ_t * x

        The returned coefficients are used in:
        - trajectory_sampler.py: drift = 2*v_theta(X_t,t) - kappa_t*X_t
          (Algorithm 1, equation 40)
        - lean_adjoint.py: VJP of b(x,t) = 2*v_base(x,t) - kappa_t*x
          (equation 41)

        Verification:
            t=0.025: (κ, 2η) = (40.0, 78.0)
            t=0.5:   (κ, 2η) = (2.0, 2.0)
            t=0.975: (κ, 2η) ≈ (1.026, 0.051)

        Args:
            t: Continuous time in (0, 1]. Must be strictly positive.

        Returns:
            Tuple (κ_t, 2*η_t) where:
                κ_t = 1/t (coefficient for the linear x term)
                2*η_t = 2*(1-t)/t (coefficient for the score function term)
            Both values are non-negative for t ∈ (0, 1].

        Raises:
            ZeroDivisionError: If t == 0.0 exactly.
        """
        kappa_t: float = self.kappa(t)
        two_eta_t: float = 2.0 * self.eta(t)
        return kappa_t, two_eta_t

    # ------------------------------------------------------------------
    # Convenience / validation methods
    # ------------------------------------------------------------------

    def verify_memoryless_condition(self, t: float) -> float:
        """Verify that σ²/2 ≈ η_t (memoryless condition, without offset).

        The exact memoryless condition is σ(t)² = 2*η_t (Proposition 1).
        With the practical offset, σ(t)² = 2*(1-t+h)/(t+h) ≠ 2*(1-t)/t
        exactly, but they converge as h → 0.

        This method returns the ratio σ²/(2*η_t) for diagnostic purposes.
        A value close to 1.0 indicates the offset is small relative to t.

        Args:
            t: Continuous time in (0, 1].

        Returns:
            Ratio σ(t)² / (2*η_t). Equals 1.0 exactly when h=0.
            With h=0.025: ratio ≈ 1.0 for t >> h, deviates near t=0.
        """
        sigma_sq: float = self.sigma_memoryless(t) ** 2
        two_eta: float = 2.0 * self.eta(t)
        if two_eta == 0.0:
            return float("inf")
        return sigma_sq / two_eta

    def __repr__(self) -> str:
        """Human-readable representation of the noise schedule."""
        return (
            f"NoiseSchedule("
            f"h={self.h}, "
            f"alpha_type='linear' (alpha_t=t), "
            f"beta_type='linear' (beta_t=1-t), "
            f"sigma_type='memoryless' (sigma(t)=sqrt(2*(1-t+h)/(t+h)))"
            f")"
        )
