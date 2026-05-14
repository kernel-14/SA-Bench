## score.py
"""Score module for the randomized midpoint diffusion sampler.

Implements the GaussianTarget class representing the d-dimensional Gaussian
target distribution N(0, Sigma) with diagonal covariance, and its exact
score function.

Mathematical background
-----------------------
Target distribution:
    X_0 ~ N(0, Sigma),   Sigma = diag(sigma_1^2, ..., sigma_d^2)

where the first k_active entries sigma_i^2 ~ Unif(0, 10) and the remaining
d - k_active entries are sigma_i^2 = 0.

Marginal at noise level tau in (0, 1):
    X_tau = sqrt(1-tau) * X_0 + sqrt(tau) * Z,   Z ~ N(0, I_d)
    X_tau ~ N(0, Sigma_tau),   Sigma_tau = (1-tau)*Sigma + tau*I_d

So (Sigma_tau)_ii = (1-tau)*sigma_i^2 + tau, which is always strictly
positive for tau > 0 since the tau*I_d term dominates even when sigma_i^2 = 0.

Exact score function (Definition 1 in paper, Appendix C.1):
    s_tau*(x) = nabla log p_{X_tau}(x) = -Sigma_tau^{-1} x

Since Sigma_tau is diagonal:
    s_tau*(x)_i = -x_i / ((1-tau)*sigma_i^2 + tau)

Score Jacobian (constant for Gaussian, independent of x):
    J_tau(x) = d s_tau*(x) / dx = -Sigma_tau^{-1} = -diag(1/((1-tau)*sigma_i^2 + tau))

This linearity is the key property enabling GaussianTracker to propagate
distributions analytically.
"""

from typing import Dict, Optional

import torch


# Minimum clamp value for diagonal covariance entries before inversion.
# Matches config.yaml: numerics.sigma_min_clamp: 1.0e-12
_SIGMA_MIN_CLAMP: float = 1.0e-12


class GaussianTarget:
    """d-dimensional Gaussian target distribution with diagonal covariance.

    Represents the target distribution p_data = N(0, Sigma) where Sigma is
    a diagonal matrix. Provides the exact score function, marginal covariance
    at any noise level tau, and sampling utilities.

    Attributes:
        d: Data dimension.
        k_active: Number of non-degenerate dimensions. First k_active diagonal
            entries of Sigma are drawn from Unif(0, 10); remaining d - k_active
            entries are zero.
        seed: Random seed used to construct the covariance diagonal.
        device: PyTorch device string ('cpu' or 'cuda').
        Sigma: Diagonal entries of the target covariance, shape (d,),
            dtype float64. First k_active entries in [0, 10), rest are 0.
    """

    def __init__(
        self,
        d: int,
        k_active: int,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        """Initialise the Gaussian target distribution.

        Args:
            d: Data dimension. Must be >= 1.
            k_active: Number of non-degenerate dimensions. Must satisfy
                1 <= k_active <= d.
            seed: Random seed for constructing the covariance diagonal.
                Default 42.
            device: PyTorch device string. Must be 'cpu' or 'cuda'.
                Default 'cpu'.

        Raises:
            TypeError: If d, k_active, or seed are not integers.
            ValueError: If d < 1, k_active < 1, k_active > d, or device
                is not 'cpu' or 'cuda'.
        """
        # --- type checks ---
        if not isinstance(d, int):
            raise TypeError(f"d must be int, got {type(d).__name__}")
        if not isinstance(k_active, int):
            raise TypeError(f"k_active must be int, got {type(k_active).__name__}")
        if not isinstance(seed, int):
            raise TypeError(f"seed must be int, got {type(seed).__name__}")
        if not isinstance(device, str):
            raise TypeError(f"device must be str, got {type(device).__name__}")

        # --- value checks ---
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        if k_active < 1:
            raise ValueError(f"k_active must be >= 1, got {k_active}")
        if k_active > d:
            raise ValueError(
                f"k_active ({k_active}) must be <= d ({d})"
            )
        if device not in ("cpu", "cuda"):
            raise ValueError(
                f"device must be 'cpu' or 'cuda', got '{device}'"
            )

        self.d: int = d
        self.k_active: int = k_active
        self.seed: int = seed
        self.device: str = device

        # Manual cache for get_marginal_cov_inv results.
        # Keys are float tau values; values are (d,) float64 tensors.
        # lru_cache cannot be applied directly to instance methods because
        # self is not hashable by default, so we use a plain dict.
        self._cache: Dict[float, torch.Tensor] = {}

        # Build the diagonal covariance vector
        self.Sigma: torch.Tensor = self._build_covariance()

    # ------------------------------------------------------------------
    # Private construction helpers
    # ------------------------------------------------------------------

    def _build_covariance(self) -> torch.Tensor:
        """Build the diagonal covariance vector Sigma.

        Sets the random seed, draws k_active values from Unif(0, 10) for
        the first k_active diagonal entries, and leaves the remaining
        d - k_active entries at zero.

        Returns:
            Tensor of shape (d,) with dtype float64 on self.device.
            First k_active entries are in [0, 10); remaining entries are 0.
        """
        # Set seed for reproducibility
        torch.manual_seed(self.seed)

        # Initialise all entries to zero
        sigma_diag: torch.Tensor = torch.zeros(
            self.d, dtype=torch.float64, device=self.device
        )

        # Draw k_active values from Unif(0, 10)
        # torch.rand gives values in [0, 1); multiply by 10 gives [0, 10)
        active_values: torch.Tensor = (
            torch.rand(self.k_active, dtype=torch.float64, device=self.device) * 10.0
        )
        sigma_diag[: self.k_active] = active_values

        return sigma_diag

    # ------------------------------------------------------------------
    # Marginal covariance methods
    # ------------------------------------------------------------------

    def get_marginal_cov(self, tau: float) -> torch.Tensor:
        """Return the diagonal of the marginal covariance Sigma_tau.

        Sigma_tau = (1 - tau) * Sigma + tau * I_d

        So (Sigma_tau)_ii = (1 - tau) * sigma_i^2 + tau.

        For tau = 0: returns Sigma (pure data distribution).
        For tau = 1: returns ones(d) (pure noise, standard Gaussian).
        For tau in (0, 1): all entries are strictly positive since
            (Sigma_tau)_ii >= tau > 0.

        Args:
            tau: Noise level in [0, 1].

        Returns:
            Tensor of shape (d,) with dtype float64 on self.device.
            Entry i equals (1 - tau) * sigma_i^2 + tau.
        """
        tau_f: float = float(tau)
        one_minus_tau: float = 1.0 - tau_f

        # Element-wise: (1 - tau) * Sigma + tau * 1
        sigma_tau_diag: torch.Tensor = (
            one_minus_tau * self.Sigma
            + tau_f * torch.ones(self.d, dtype=torch.float64, device=self.device)
        )
        return sigma_tau_diag

    def get_marginal_cov_inv(self, tau: float) -> torch.Tensor:
        """Return the diagonal of the inverse marginal covariance Sigma_tau^{-1}.

        Sigma_tau^{-1} = diag(1 / ((1-tau)*sigma_i^2 + tau))

        Results are cached by tau value to avoid redundant computation.
        Diagonal entries are clamped from below at _SIGMA_MIN_CLAMP = 1e-12
        before inversion for numerical safety.

        Args:
            tau: Noise level in (0, 1]. Must be > 0 for the inverse to exist.
                For tau very close to 0, the clamp ensures numerical stability.

        Returns:
            Tensor of shape (d,) with dtype float64 on self.device.
            Entry i equals 1 / max((1-tau)*sigma_i^2 + tau, 1e-12).
        """
        tau_f: float = float(tau)

        # Check cache first
        if tau_f in self._cache:
            return self._cache[tau_f]

        # Compute marginal covariance diagonal
        sigma_tau_diag: torch.Tensor = self.get_marginal_cov(tau_f)

        # Clamp from below for numerical safety before inversion
        sigma_tau_clamped: torch.Tensor = sigma_tau_diag.clamp(min=_SIGMA_MIN_CLAMP)

        # Element-wise inversion
        sigma_tau_inv: torch.Tensor = 1.0 / sigma_tau_clamped

        # Cache and return
        self._cache[tau_f] = sigma_tau_inv
        return sigma_tau_inv

    # ------------------------------------------------------------------
    # Score function methods
    # ------------------------------------------------------------------

    def score(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        """Compute the exact score function s_tau*(x) = -Sigma_tau^{-1} x.

        Since Sigma_tau is diagonal, this reduces to element-wise
        multiplication:
            s_tau*(x)_i = -x_i / ((1-tau)*sigma_i^2 + tau)

        Supports both single samples and batches:
            - x shape (d,)       -> result shape (d,)
            - x shape (batch, d) -> result shape (batch, d)

        Args:
            x: Input tensor. Shape (d,) for a single sample or (batch, d)
                for a batch. Must have dtype compatible with float64.
            tau: Noise level in (0, 1].

        Returns:
            Score tensor with the same shape as x, dtype float64,
            on self.device.
        """
        # Ensure float64 dtype for numerical consistency
        if x.dtype != torch.float64:
            x = x.to(dtype=torch.float64)

        # Ensure correct device
        if x.device != torch.device(self.device):
            x = x.to(device=self.device)

        # Get inverse covariance diagonal, shape (d,)
        sigma_tau_inv: torch.Tensor = self.get_marginal_cov_inv(float(tau))

        # Element-wise multiplication; broadcasting handles batch dimension
        # For x shape (d,):       -sigma_tau_inv * x  -> shape (d,)
        # For x shape (batch, d): -sigma_tau_inv * x  -> shape (batch, d)
        return -sigma_tau_inv * x

    def get_score_jacobian(self, tau: float) -> torch.Tensor:
        """Return the Jacobian of the score function J_tau = d s_tau*(x) / dx.

        For the Gaussian case, the Jacobian is constant (independent of x):
            J_tau = -Sigma_tau^{-1} = -diag(1/((1-tau)*sigma_i^2 + tau))

        This is a diagonal matrix of shape (d, d).

        Args:
            tau: Noise level in (0, 1].

        Returns:
            Tensor of shape (d, d) with dtype float64 on self.device.
            This is a diagonal matrix with entries -1/((1-tau)*sigma_i^2 + tau).
        """
        # Get inverse covariance diagonal, shape (d,)
        sigma_tau_inv: torch.Tensor = self.get_marginal_cov_inv(float(tau))

        # Construct diagonal matrix: J_tau = -diag(Sigma_tau_inv)
        jacobian: torch.Tensor = -torch.diag(sigma_tau_inv)
        return jacobian

    # ------------------------------------------------------------------
    # Sampling method
    # ------------------------------------------------------------------

    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample from the target distribution X_0 ~ N(0, Sigma).

        Uses the reparameterisation X_0 = sqrt(Sigma) * Z where Z ~ N(0, I_d).
        For degenerate dimensions (sigma_i^2 = 0), the corresponding entries
        of the samples are always 0.

        Args:
            n_samples: Number of samples to draw. Must be >= 1.

        Returns:
            Tensor of shape (n_samples, d) with dtype float64 on self.device.
            Each row is an independent sample from N(0, Sigma).

        Raises:
            ValueError: If n_samples < 1.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        # Standard deviation: sqrt(sigma_i^2), clamped to handle zero entries
        # sqrt(0) = 0, so degenerate dimensions produce zero samples
        std: torch.Tensor = torch.sqrt(self.Sigma.clamp(min=0.0))  # shape (d,)

        # Draw standard Gaussian samples
        z: torch.Tensor = torch.randn(
            n_samples, self.d, dtype=torch.float64, device=self.device
        )

        # Scale by standard deviation; broadcasting: (n_samples, d) * (d,)
        return z * std

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear the internal cache for get_marginal_cov_inv.

        Should be called between different T runs to prevent memory
        accumulation, since tau values are sampled from continuous
        distributions and the cache may grow unboundedly.
        """
        self._cache.clear()

    # ------------------------------------------------------------------
    # Diagnostic / utility methods
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the target distribution.

        Returns:
            Multi-line string with key distribution statistics.
        """
        sigma_active: torch.Tensor = self.Sigma[: self.k_active]
        sigma_min: float = float(sigma_active.min()) if self.k_active > 0 else 0.0
        sigma_max: float = float(sigma_active.max()) if self.k_active > 0 else 0.0
        sigma_mean: float = float(sigma_active.mean()) if self.k_active > 0 else 0.0

        lines = [
            "GaussianTarget Summary",
            f"  Data dimension d       : {self.d}",
            f"  Active dimensions k    : {self.k_active}",
            f"  Degenerate dimensions  : {self.d - self.k_active}",
            f"  Random seed            : {self.seed}",
            f"  Device                 : {self.device}",
            f"  Sigma (active) min     : {sigma_min:.6f}",
            f"  Sigma (active) max     : {sigma_max:.6f}",
            f"  Sigma (active) mean    : {sigma_mean:.6f}",
            f"  Sigma dtype            : {self.Sigma.dtype}",
            f"  Cache size             : {len(self._cache)} entries",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"GaussianTarget(d={self.d}, k_active={self.k_active}, "
            f"seed={self.seed}, device={self.device!r})"
        )
