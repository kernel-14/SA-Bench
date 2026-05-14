## metrics.py
"""Metrics module for the randomized midpoint diffusion sampler.

Provides static utility methods for:
  1. KL divergence between two diagonal Gaussians (kl_gaussian).
  2. TV distance upper bound via Pinsker's inequality (tv_from_kl).
  3. Fitting the theoretical convergence rate C * log^4(T) / T^3 to
     empirical KL data (fit_theoretical_rate).
  4. Computing iteration complexity for all methods compared in the paper
     (compute_iteration_complexity).
  5. Computing TV distance bounds for all methods (compute_tv_bound).

All methods are @staticmethod — the class is a pure namespace with no
instance state.  All tensor computations use float64 for numerical stability.

Mathematical background
-----------------------
KL divergence between diagonal Gaussians N(mu1, Sigma1) and N(mu2, Sigma2):

    KL(p || q) = 0.5 * [ sum(s1/s2) + sum((mu2-mu1)^2/s2) - d
                          + sum(log(s2) - log(s1)) ]

where s1 = diag(Sigma1), s2 = diag(Sigma2) are 1-D tensors.

Pinsker's inequality:  TV(p, q) <= sqrt(KL(p||q) / 2).

Theoretical convergence rate (Theorem 1, paper):
    KL ~ C * log^4(T) / T^3   (for fixed d, L, exact score).

Iteration complexity formulas (Section 1.1, omitting log factors):
    ours    : min(d, d^(2/3)*L^(1/3), d^(1/3)*L) * eps^(-2/3)
    benton  : d * eps^(-2)
    li_yan  : d * eps^(-1)
    li_cai  : d^(5/4) * eps^(-1/2)
    li_jiao : d^(1/3) * L * eps^(-2/3)
"""

import math
from typing import List, Tuple

import numpy as np
import torch

# Minimum clamp value for diagonal covariance entries before inversion.
# Matches config.yaml: numerics.sigma_min_clamp: 1.0e-12
_SIGMA_MIN_CLAMP: float = 1.0e-12

# Supported method names for complexity / TV bound computations
_SUPPORTED_METHODS: Tuple[str, ...] = (
    "ours",
    "benton",
    "li_yan",
    "li_cai",
    "li_jiao",
)


class Metrics:
    """Static utility class for divergence computation and rate fitting.

    All methods are @staticmethod.  No instance state is maintained.
    Instantiation is allowed but unnecessary — call methods directly on
    the class, e.g. ``Metrics.kl_gaussian(...)``.
    """

    # ------------------------------------------------------------------
    # 1. KL divergence between diagonal Gaussians
    # ------------------------------------------------------------------

    @staticmethod
    def kl_gaussian(
        mu1: torch.Tensor,
        Sigma1_diag: torch.Tensor,
        mu2: torch.Tensor,
        Sigma2_diag: torch.Tensor,
        sigma_min_clamp: float = _SIGMA_MIN_CLAMP,
    ) -> float:
        """Compute KL(N(mu1, Sigma1) || N(mu2, Sigma2)) for diagonal covariances.

        Uses the closed-form formula for diagonal Gaussians:

            KL = 0.5 * [ sum(s1/s2) + sum((mu2-mu1)^2/s2) - d
                         + sum(log(s2) - log(s1)) ]

        where s1 = diag(Sigma1), s2 = diag(Sigma2).

        Both covariance diagonals are clamped from below at sigma_min_clamp
        before any division or logarithm to ensure numerical stability.

        Args:
            mu1: Mean of the first distribution.  Shape (d,).
            Sigma1_diag: Diagonal entries of the first covariance.  Shape (d,).
                All entries should be non-negative.
            mu2: Mean of the second distribution.  Shape (d,).
            Sigma2_diag: Diagonal entries of the second covariance.  Shape (d,).
                All entries should be positive (the reference distribution must
                have full support).
            sigma_min_clamp: Minimum value for clamping covariance entries
                before inversion and log.  Default 1e-12.

        Returns:
            KL divergence as a Python float.  Clamped to >= 0 to handle
            tiny negative values from floating-point rounding.

        Raises:
            ValueError: If the shapes of mu1, Sigma1_diag, mu2, Sigma2_diag
                are not all equal.
        """
        # --- shape validation ---
        d: int = mu1.shape[0]
        if mu2.shape[0] != d:
            raise ValueError(
                f"mu1 and mu2 must have the same shape. "
                f"Got mu1.shape={mu1.shape}, mu2.shape={mu2.shape}"
            )
        if Sigma1_diag.shape[0] != d:
            raise ValueError(
                f"Sigma1_diag must have shape ({d},), got {Sigma1_diag.shape}"
            )
        if Sigma2_diag.shape[0] != d:
            raise ValueError(
                f"Sigma2_diag must have shape ({d},), got {Sigma2_diag.shape}"
            )

        # --- cast to float64 for numerical stability ---
        s1: torch.Tensor = Sigma1_diag.to(dtype=torch.float64)
        s2: torch.Tensor = Sigma2_diag.to(dtype=torch.float64)
        m1: torch.Tensor = mu1.to(dtype=torch.float64)
        m2: torch.Tensor = mu2.to(dtype=torch.float64)

        # --- clamp from below before division / log ---
        s1_clamped: torch.Tensor = s1.clamp(min=sigma_min_clamp)
        s2_clamped: torch.Tensor = s2.clamp(min=sigma_min_clamp)

        # --- KL formula terms ---
        # Term 1: sum(s1 / s2)
        term_trace: torch.Tensor = (s1_clamped / s2_clamped).sum()

        # Term 2: sum((mu2 - mu1)^2 / s2)
        diff_mu: torch.Tensor = m2 - m1
        term_mean: torch.Tensor = ((diff_mu ** 2) / s2_clamped).sum()

        # Term 3: -d
        term_dim: float = float(d)

        # Term 4: sum(log(s2) - log(s1))
        # Use separate logs to avoid intermediate overflow/underflow
        term_logdet: torch.Tensor = (
            torch.log(s2_clamped) - torch.log(s1_clamped)
        ).sum()

        # Combine
        kl_tensor: torch.Tensor = 0.5 * (
            term_trace + term_mean - term_dim + term_logdet
        )

        kl_value: float = float(kl_tensor.item())

        # Clamp to >= 0 to handle tiny negative values from floating-point errors
        return max(kl_value, 0.0)

    # ------------------------------------------------------------------
    # 2. TV distance from KL via Pinsker's inequality
    # ------------------------------------------------------------------

    @staticmethod
    def tv_from_kl(kl: float) -> float:
        """Compute TV distance upper bound from KL divergence via Pinsker.

        Pinsker's inequality:  TV(p, q) <= sqrt(KL(p || q) / 2).

        Args:
            kl: KL divergence value.  Should be >= 0.  Negative values
                (numerical artifacts) are clamped to 0 before taking sqrt.

        Returns:
            Upper bound on TV distance as a Python float in [0, 1].
        """
        kl_safe: float = max(float(kl), 0.0)
        return math.sqrt(kl_safe / 2.0)

    # ------------------------------------------------------------------
    # 3. Fit theoretical convergence rate
    # ------------------------------------------------------------------

    @staticmethod
    def fit_theoretical_rate(
        T_values: List[int],
        kl_values: List[float],
    ) -> Tuple[float, np.ndarray]:
        """Fit the constant C in the theoretical rate C * log^4(T) / T^3.

        The paper predicts (Theorem 1, Appendix A):
            KL ~ C * log^4(T) / T^3

        Taking logarithms:
            log(KL) = log(C) + 4*log(log(T)) - 3*log(T)

        The exponents 4 and -3 are fixed by theory.  Only the constant C
        is estimated by averaging:
            log(C) = mean_i [ log(KL_i) - 4*log(log(T_i)) + 3*log(T_i) ]

        Data points with KL <= 0 (numerical underflow) are excluded from
        the fit.

        Args:
            T_values: List of iteration counts.  All values must be >= 2
                (so that log(T) > 0 and log(log(T)) is defined).
            kl_values: List of empirical KL divergence values corresponding
                to each T.  Must have the same length as T_values.

        Returns:
            Tuple (C_fit, fitted_kl_array) where:
                C_fit: Fitted constant as a Python float.  Returns 1.0 if
                    no valid data points are available.
                fitted_kl_array: numpy array of shape (len(T_values),)
                    with values C_fit * log(T)^4 / T^3 for each T in
                    T_values.

        Raises:
            ValueError: If T_values and kl_values have different lengths.
            ValueError: If T_values is empty.
        """
        if len(T_values) != len(kl_values):
            raise ValueError(
                f"T_values and kl_values must have the same length. "
                f"Got len(T_values)={len(T_values)}, "
                f"len(kl_values)={len(kl_values)}"
            )
        if len(T_values) == 0:
            raise ValueError("T_values must not be empty.")

        # --- collect valid data points (KL > 0 and T >= 2) ---
        log_C_estimates: List[float] = []
        for T_val, kl_val in zip(T_values, kl_values):
            T_f: float = float(T_val)
            if T_f < 2.0:
                # log(log(T)) undefined for T < e; skip
                continue
            log_T: float = math.log(T_f)
            if log_T <= 0.0:
                # log(T) <= 0 means T <= 1; skip
                continue
            log_log_T: float = math.log(log_T)
            if kl_val <= 0.0:
                # Cannot take log of non-positive KL; skip
                continue
            log_kl: float = math.log(float(kl_val))
            # log(C) = log(KL) - 4*log(log(T)) + 3*log(T)
            log_C_i: float = log_kl - 4.0 * log_log_T + 3.0 * log_T
            log_C_estimates.append(log_C_i)

        # --- estimate C ---
        if len(log_C_estimates) == 0:
            # No valid data; return C=1 as fallback
            C_fit: float = 1.0
        else:
            log_C_mean: float = float(np.mean(log_C_estimates))
            C_fit = math.exp(log_C_mean)

        # --- compute fitted curve for all T values ---
        T_arr: np.ndarray = np.array(T_values, dtype=np.float64)
        # Avoid log(0) or log(log(T)) issues for T < 2
        T_arr_safe: np.ndarray = np.maximum(T_arr, 2.0)
        log_T_arr: np.ndarray = np.log(T_arr_safe)
        log_T_arr_safe: np.ndarray = np.maximum(log_T_arr, 1e-15)
        fitted_kl_array: np.ndarray = C_fit * (log_T_arr_safe ** 4) / (T_arr_safe ** 3)

        return C_fit, fitted_kl_array

    # ------------------------------------------------------------------
    # 4. Iteration complexity for all methods
    # ------------------------------------------------------------------

    @staticmethod
    def compute_iteration_complexity(
        method: str,
        d: int,
        L: float,
        eps: float,
    ) -> float:
        """Return the iteration complexity T for a given method.

        All formulas omit logarithmic factors (Õ notation), consistent
        with how Figure 1 is drawn in the paper.

        Supported methods and their formulas (Section 1.1):
            'ours'    : min(d, d^(2/3)*L^(1/3), d^(1/3)*L) * eps^(-2/3)
            'benton'  : d * eps^(-2)
            'li_yan'  : d * eps^(-1)
            'li_cai'  : d^(5/4) * eps^(-1/2)
            'li_jiao' : d^(1/3) * L * eps^(-2/3)

        Special cases:
            - 'ours' with L = inf: min reduces to d, giving d * eps^(-2/3).
            - 'li_jiao' with L = inf: returns float('inf') (no finite bound
              without smoothness).

        Args:
            method: One of 'ours', 'benton', 'li_yan', 'li_cai', 'li_jiao'.
            d: Data dimension.  Must be >= 1.
            L: Non-uniform Lipschitz constant.  Use a large value (e.g.,
               1e12) as a proxy for L = infinity.
            eps: Target accuracy in TV distance.  Must be in (0, 1].

        Returns:
            Iteration complexity T as a Python float.  Returns float('inf')
            for methods that have no finite bound (e.g., li_jiao with L=inf).

        Raises:
            ValueError: If method is not one of the supported strings.
            ValueError: If d < 1 or eps <= 0.
        """
        if method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Supported methods: {_SUPPORTED_METHODS}"
            )
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")

        d_f: float = float(d)
        L_f: float = float(L)
        eps_f: float = float(eps)

        if method == "ours":
            # min(d, d^(2/3)*L^(1/3), d^(1/3)*L) * eps^(-2/3)
            # For L = inf: d^(1/3)*L -> inf, d^(2/3)*L^(1/3) -> inf,
            # so min reduces to d.
            term1: float = d_f
            if math.isinf(L_f) or L_f > 1e30:
                # L = infinity: min is just d
                complexity_factor: float = d_f
            else:
                term2: float = (d_f ** (2.0 / 3.0)) * (L_f ** (1.0 / 3.0))
                term3: float = (d_f ** (1.0 / 3.0)) * L_f
                complexity_factor = min(term1, term2, term3)
            return complexity_factor * (eps_f ** (-2.0 / 3.0))

        elif method == "benton":
            # d * eps^(-2)
            return d_f * (eps_f ** (-2.0))

        elif method == "li_yan":
            # d * eps^(-1)
            return d_f * (eps_f ** (-1.0))

        elif method == "li_cai":
            # d^(5/4) * eps^(-1/2)
            return (d_f ** (5.0 / 4.0)) * (eps_f ** (-0.5))

        elif method == "li_jiao":
            # d^(1/3) * L * eps^(-2/3)
            if math.isinf(L_f) or L_f > 1e30:
                return float("inf")
            return (d_f ** (1.0 / 3.0)) * L_f * (eps_f ** (-2.0 / 3.0))

        # Should never reach here due to the check above
        raise ValueError(f"Unhandled method '{method}'")  # pragma: no cover

    # ------------------------------------------------------------------
    # 5. TV distance bound for all methods
    # ------------------------------------------------------------------

    @staticmethod
    def compute_tv_bound(
        method: str,
        d: int,
        L: float,
        T: float,
    ) -> float:
        """Return the TV distance bound achieved after T iterations.

        Derived by inverting the iteration complexity formulas.  If
        T ~ f(d, L) * eps^(-alpha), then eps ~ (f(d, L) / T)^(1/alpha).

        Formulas (omitting log factors):
            'ours'    : (min(d, d^(2/3)*L^(1/3), d^(1/3)*L) / T)^(3/2)
            'benton'  : (d / T)^(1/2)
            'li_yan'  : d / T
            'li_cai'  : (d^(5/4) / T)^2
            'li_jiao' : (d^(1/3) * L / T)^(3/2)

        Special cases:
            - 'ours' with L = inf: min reduces to d, giving (d/T)^(3/2).
            - 'li_jiao' with L = inf: returns 1.0 (trivial bound).

        All results are clamped to [0, 1] since TV distance is bounded.

        Args:
            method: One of 'ours', 'benton', 'li_yan', 'li_cai', 'li_jiao'.
            d: Data dimension.  Must be >= 1.
            L: Non-uniform Lipschitz constant.  Use a large value (e.g.,
               1e12) as a proxy for L = infinity.
            T: Number of iterations.  Must be > 0.

        Returns:
            TV distance bound as a Python float in [0, 1].

        Raises:
            ValueError: If method is not one of the supported strings.
            ValueError: If d < 1 or T <= 0.
        """
        if method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Supported methods: {_SUPPORTED_METHODS}"
            )
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        if T <= 0.0:
            raise ValueError(f"T must be > 0, got {T}")

        d_f: float = float(d)
        L_f: float = float(L)
        T_f: float = float(T)

        if method == "ours":
            # (min(d, d^(2/3)*L^(1/3), d^(1/3)*L) / T)^(3/2)
            if math.isinf(L_f) or L_f > 1e30:
                complexity_factor: float = d_f
            else:
                term1: float = d_f
                term2: float = (d_f ** (2.0 / 3.0)) * (L_f ** (1.0 / 3.0))
                term3: float = (d_f ** (1.0 / 3.0)) * L_f
                complexity_factor = min(term1, term2, term3)
            tv: float = (complexity_factor / T_f) ** (3.0 / 2.0)

        elif method == "benton":
            # (d / T)^(1/2)
            tv = (d_f / T_f) ** 0.5

        elif method == "li_yan":
            # d / T
            tv = d_f / T_f

        elif method == "li_cai":
            # (d^(5/4) / T)^2
            tv = ((d_f ** (5.0 / 4.0)) / T_f) ** 2.0

        elif method == "li_jiao":
            # (d^(1/3) * L / T)^(3/2)
            if math.isinf(L_f) or L_f > 1e30:
                return 1.0
            tv = ((d_f ** (1.0 / 3.0)) * L_f / T_f) ** (3.0 / 2.0)

        else:
            raise ValueError(f"Unhandled method '{method}'")  # pragma: no cover

        # Clamp to [0, 1]
        return min(max(tv, 0.0), 1.0)
