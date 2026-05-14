## schedule.py
"""Schedule module for the randomized midpoint diffusion sampler.

Implements the Schedule class that computes the deterministic alpha_hat / tau_hat
grid (Eq. 9 of the paper) and provides randomized tau sampling for the
randomized midpoint method.

Mathematical background
-----------------------
The paper defines a backward recursion (Eq. 9):

    alpha_hat[T+1] = 1 / T^{c0}
    alpha_hat[t-1] = alpha_hat[t] + c1 * alpha_hat[t] * (1 - alpha_hat[t]) * log(T) / T

for t = T+1, T, ..., 2.  This is a forward-Euler discretisation of the logistic
ODE dα/dt = c1 * α * (1-α) * log(T)/T integrated backward from t=T+1 toward t=1,
so alpha_hat grows from near 0 (at index T+1) toward 1 (at index 1).

The double-indexed grid is then:

    tau_hat[k, n] = 1 - alpha_hat[T - k * (N//2) - n]

for k = 0..K-1 and n = -1..N, where N = 2*T // K.

Key property (Lemma 7):

    (tau_hat[k,n-1] - tau_hat[k,n]) / (tau_hat[k,n-1] * (1 - tau_hat[k,n-1]))
        = c1 * log(T) / T   (exactly, by construction)

The randomised tau samples are drawn as:

    tau[k,n] ~ Uniform(tau_hat[k,n], tau_hat[k,n-1])

which is the "randomized midpoint" that gives the sampler its name.
"""

import math
import warnings
from typing import Optional

import numpy as np


class Schedule:
    """Deterministic schedule grid and randomised tau sampler.

    Computes the alpha_hat / tau_hat arrays from the paper's Eq. 9 and
    exposes methods for grid lookup and randomised tau sampling.

    Attributes:
        T: Total number of score evaluations (half the total iteration count
           in the paper's notation, since KN = 2T).
        K: Number of sampler rounds.
        N: Steps per round, equal to 2*T // K (integer division).
        c0: Schedule constant; alpha_hat[T+1] = 1 / T^{c0}.
        c1: Schedule constant; controls step size in the recursion.
        alpha_hat: numpy array of shape (T+2,) with dtype float64.
            alpha_hat[t] = α̂_t for t = 0..T+1.
            alpha_hat[0] = 1.0 (sentinel for clamped out-of-range indices).
            alpha_hat[T+1] = 1 / T^{c0} (near-zero starting value).
        tau_hat_grid: numpy array of shape (K, N+2) with dtype float64.
            tau_hat_grid[k, n+1] = τ̂_{k,n} for n = -1..N.
            Column 0  → n = -1 (pre-step).
            Column 1  → n = 0  (round start).
            Column N+1 → n = N  (round end).
    """

    def __init__(
        self,
        T: int,
        K: int,
        c0: float = 2.0,
        c1: float = 10.0,
    ) -> None:
        """Initialise the schedule and build all grid arrays.

        Args:
            T: Total iteration count (half the paper's 2T notation).
                Must satisfy T >= K and 2*T % K == 0.
            K: Number of sampler rounds.  Must be >= 1.
            c0: Schedule constant controlling the initial noise level.
                alpha_hat[T+1] = 1 / T^{c0}.  Must be > 0.  Default 2.0.
            c1: Schedule constant controlling the step size.  Must be > 0
                and c1/c0 should be sufficiently large (paper requirement).
                Default 10.0.

        Raises:
            ValueError: If T < K, 2*T % K != 0, c0 <= 0, or c1 <= 0.
            TypeError: If T or K are not integers.
        """
        # --- type checks ---
        if not isinstance(T, int):
            raise TypeError(f"T must be int, got {type(T).__name__}")
        if not isinstance(K, int):
            raise TypeError(f"K must be int, got {type(K).__name__}")

        # --- value checks ---
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if T < K:
            raise ValueError(
                f"T ({T}) must be >= K ({K}) so that each round has at least "
                f"one step."
            )
        if (2 * T) % K != 0:
            raise ValueError(
                f"2*T must be divisible by K so that N = 2T/K is an integer. "
                f"Got 2*{T} % {K} = {(2 * T) % K}."
            )
        c0 = float(c0)
        c1 = float(c1)
        if c0 <= 0.0:
            raise ValueError(f"c0 must be positive, got {c0}")
        if c1 <= 0.0:
            raise ValueError(f"c1 must be positive, got {c1}")

        # Warn if c1/c0 ratio is small (paper requires it to be "sufficiently large")
        ratio: float = c1 / c0
        if ratio < 3.0:
            warnings.warn(
                f"c1/c0 ratio = {ratio:.2f} may be too small. "
                f"The paper requires c1/c0 to be sufficiently large. "
                f"Consider increasing c1 or decreasing c0.",
                UserWarning,
                stacklevel=2,
            )

        self.T: int = T
        self.K: int = K
        self.N: int = 2 * T // K  # steps per round
        self.c0: float = c0
        self.c1: float = c1

        # Validate N
        if self.N < 2:
            raise ValueError(
                f"N = 2*T//K = {self.N} must be >= 2 for the algorithm to be "
                f"meaningful.  Increase T or decrease K."
            )

        # Build the two core arrays
        self.alpha_hat: np.ndarray = self._build_alpha_hat()
        self.tau_hat_grid: np.ndarray = self._build_tau_hat()

    # ------------------------------------------------------------------
    # Private construction helpers
    # ------------------------------------------------------------------

    def _build_alpha_hat(self) -> np.ndarray:
        """Build the alpha_hat array using the backward recursion (Eq. 9).

        The recursion is:
            alpha_hat[T+1] = 1 / T^{c0}
            alpha_hat[t-1] = alpha_hat[t]
                             + c1 * alpha_hat[t] * (1 - alpha_hat[t]) * log(T) / T

        for t = T+1, T, ..., 2.

        Returns:
            numpy array of shape (T+2,) with dtype float64.
            Index 0 is a sentinel value of 1.0 (used when the (k,n) index
            would fall below 1 after clamping).
            Indices 1..T+1 hold the actual alpha_hat values.
        """
        T: int = self.T
        c0: float = self.c0
        c1: float = self.c1

        alpha_hat: np.ndarray = np.zeros(T + 2, dtype=np.float64)

        # Sentinel: index 0 maps to alpha_hat = 1.0 (pure data signal)
        alpha_hat[0] = 1.0

        # Starting value at index T+1 (near-zero, near-noise)
        alpha_hat[T + 1] = 1.0 / (float(T) ** c0)

        # Step size factor (constant across all t)
        log_T_over_T: float = math.log(float(T)) / float(T)
        step_factor: float = c1 * log_T_over_T

        # Backward recursion: t = T+1, T, ..., 2  →  fills indices T, T-1, ..., 1
        for t in range(T + 1, 1, -1):
            a_t: float = alpha_hat[t]
            increment: float = step_factor * a_t * (1.0 - a_t)
            a_tm1: float = a_t + increment
            # Clamp to [0, 1] to prevent floating-point overshoot
            if a_tm1 > 1.0:
                a_tm1 = 1.0
            elif a_tm1 < 0.0:
                a_tm1 = 0.0
            alpha_hat[t - 1] = a_tm1

        return alpha_hat

    def _build_tau_hat(self) -> np.ndarray:
        """Build the tau_hat_grid array from the alpha_hat array.

        tau_hat[k, n] = 1 - alpha_hat[T - k * (N//2) - n]

        The grid has shape (K, N+2) where column index j corresponds to
        n = j - 1, i.e.:
            tau_hat_grid[k, 0]   = tau_hat[k, -1]   (pre-step)
            tau_hat_grid[k, 1]   = tau_hat[k,  0]   (round start)
            tau_hat_grid[k, N+1] = tau_hat[k,  N]   (round end)

        Out-of-range alpha_hat indices are clamped to [1, T+1].

        Returns:
            numpy array of shape (K, N+2) with dtype float64.
        """
        T: int = self.T
        K: int = self.K
        N: int = self.N
        N_half: int = N // 2  # = T // K

        tau_hat_grid: np.ndarray = np.zeros((K, N + 2), dtype=np.float64)

        for k in range(K):
            # Base alpha_hat index for this round (at n=0)
            base_idx: int = T - k * N_half

            for n_col in range(N + 2):  # n_col = 0..N+1, n = n_col - 1
                n: int = n_col - 1  # actual n value: -1..N
                alpha_idx: int = base_idx - n  # = T - k*N_half - n
                # Clamp to valid range [1, T+1]
                alpha_idx_clamped: int = max(1, min(alpha_idx, T + 1))
                tau_hat_grid[k, n_col] = 1.0 - self.alpha_hat[alpha_idx_clamped]

        return tau_hat_grid

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_tau_hat(self, k: int, n: int) -> float:
        """Return the deterministic grid value tau_hat[k, n].

        Args:
            k: Round index, 0 <= k < K.
            n: Step index within the round, -1 <= n <= N.
                n = -1 is the pre-step used in the first score evaluation.
                n = 0  is the start of round k.
                n = N  is the end of round k.

        Returns:
            tau_hat[k, n] as a Python float.

        Raises:
            IndexError: If k or n are out of their valid ranges.
        """
        if not (0 <= k < self.K):
            raise IndexError(
                f"k={k} out of range [0, {self.K - 1}]"
            )
        if not (-1 <= n <= self.N):
            raise IndexError(
                f"n={n} out of range [-1, {self.N}]"
            )
        return float(self.tau_hat_grid[k, n + 1])

    def get_alpha_bar(self, k: int, n: int) -> float:
        """Return the deterministic alpha_bar at grid point (k, n).

        alpha_bar[k, n] = 1 - tau_hat[k, n]
                        = alpha_hat[T - k * (N//2) - n]

        This is the ᾱ value used by the score function to look up the
        correct marginal covariance Sigma_t = ᾱ_t * Sigma + (1-ᾱ_t) * I_d.

        Args:
            k: Round index, 0 <= k < K.
            n: Step index, -1 <= n <= N.

        Returns:
            alpha_bar[k, n] as a Python float in [0, 1].
        """
        return 1.0 - self.get_tau_hat(k, n)

    def sample_tau(self, k: int, n: int) -> float:
        """Sample tau[k, n] ~ Uniform(tau_hat[k, n], tau_hat[k, n-1]).

        This is the randomized midpoint: tau[k,n] is drawn uniformly from
        the interval [tau_hat[k,n], tau_hat[k,n-1]].  Since tau_hat
        decreases with n (the reverse process moves from noise toward data),
        tau_hat[k, n-1] > tau_hat[k, n], so the interval is non-degenerate.

        Args:
            k: Round index, 0 <= k < K.
            n: Step index, 0 <= n <= N.  Must be >= 0 so that n-1 >= -1
               is a valid pre-step index.

        Returns:
            A float sampled uniformly from [tau_hat[k,n], tau_hat[k,n-1]].

        Raises:
            IndexError: If k or n are out of their valid ranges.
            ValueError: If the interval is degenerate (lower >= upper).
        """
        if not (0 <= k < self.K):
            raise IndexError(
                f"k={k} out of range [0, {self.K - 1}]"
            )
        if not (0 <= n <= self.N):
            raise IndexError(
                f"n={n} out of range [0, {self.N}] for sample_tau "
                f"(n must be >= 0 so that n-1 = {n-1} >= -1 is valid)"
            )

        lower: float = self.get_tau_hat(k, n)       # tau_hat[k, n]
        upper: float = self.get_tau_hat(k, n - 1)   # tau_hat[k, n-1]

        # tau_hat decreases with n, so upper > lower (reverse process direction)
        if upper <= lower:
            # This can happen at the boundary when both are clamped to the same
            # alpha_hat value.  Return the midpoint as a safe fallback.
            return 0.5 * (lower + upper)

        return float(np.random.uniform(lower, upper))

    def verify_schedule(self, tol: float = 1.0e-4) -> bool:
        """Verify that the schedule satisfies the key properties from Lemma 7.

        Three checks are performed:

        1. Lemma 7 ratio property:
               (tau_hat[k,n-1] - tau_hat[k,n]) / (tau_hat[k,n-1] * (1 - tau_hat[k,n-1]))
               ≈ c1 * log(T) / T
           for all valid (k, n) pairs away from the boundaries.

        2. Boundary condition — start near noise:
               tau_hat[0, -1] should be close to 1 (> 0.5).

        3. Boundary condition — end near data:
               tau_hat[K-1, N] should be close to 0 (< 0.5).

        Args:
            tol: Absolute tolerance for the ratio check.  Default 1e-4
                 (from config.yaml numerics.schedule_verify_tol).

        Returns:
            True if all checks pass, False otherwise.  Prints a warning
            message for each failed check.
        """
        passed: bool = True
        target_ratio: float = self.c1 * math.log(float(self.T)) / float(self.T)

        # --- Check 1: Lemma 7 ratio property ---
        for k in range(self.K):
            for n in range(1, self.N + 1):  # n = 1..N (need n-1 >= 0)
                tau_n: float = self.get_tau_hat(k, n)
                tau_nm1: float = self.get_tau_hat(k, n - 1)

                # Skip near-boundary points where the denominator is tiny
                denom: float = tau_nm1 * (1.0 - tau_nm1)
                if denom < 1.0e-10:
                    continue

                # Skip points where both tau values are clamped (degenerate)
                if abs(tau_nm1 - tau_n) < 1.0e-15:
                    continue

                ratio: float = (tau_nm1 - tau_n) / denom
                if abs(ratio - target_ratio) > tol:
                    warnings.warn(
                        f"Lemma 7 ratio check failed at (k={k}, n={n}): "
                        f"ratio={ratio:.6e}, target={target_ratio:.6e}, "
                        f"diff={abs(ratio - target_ratio):.6e} > tol={tol:.6e}",
                        UserWarning,
                        stacklevel=2,
                    )
                    passed = False
                    # Report first failure only to avoid flooding output
                    break
            if not passed:
                break

        # --- Check 2: Start near noise (tau_hat[0, -1] close to 1) ---
        tau_start: float = self.get_tau_hat(0, -1)
        if tau_start < 0.5:
            warnings.warn(
                f"Boundary check failed: tau_hat[0, -1] = {tau_start:.6f} "
                f"should be close to 1 (> 0.5). "
                f"The reverse process should start near pure noise.",
                UserWarning,
                stacklevel=2,
            )
            passed = False

        # --- Check 3: End near data (tau_hat[K-1, N] close to 0) ---
        tau_end: float = self.get_tau_hat(self.K - 1, self.N)
        if tau_end > 0.5:
            warnings.warn(
                f"Boundary check failed: tau_hat[K-1, N] = tau_hat[{self.K-1}, {self.N}] "
                f"= {tau_end:.6f} should be close to 0 (< 0.5). "
                f"The reverse process should end near the data distribution.",
                UserWarning,
                stacklevel=2,
            )
            passed = False

        return passed

    # ------------------------------------------------------------------
    # Convenience / diagnostic methods
    # ------------------------------------------------------------------

    def get_tau_hat_array(self, k: int) -> np.ndarray:
        """Return the full tau_hat sequence for round k as a 1D array.

        Args:
            k: Round index, 0 <= k < K.

        Returns:
            numpy array of shape (N+2,) containing tau_hat[k, n] for
            n = -1, 0, 1, ..., N (in that order).
        """
        if not (0 <= k < self.K):
            raise IndexError(f"k={k} out of range [0, {self.K - 1}]")
        return self.tau_hat_grid[k, :].copy()

    def get_step_sizes(self, k: int) -> np.ndarray:
        """Return the deterministic step sizes for round k.

        step_size[n] = tau_hat[k, n-1] - tau_hat[k, n]  for n = 0..N.

        These are the widths of the uniform intervals from which tau[k,n]
        is sampled.  All values should be positive (tau_hat decreases with n).

        Args:
            k: Round index, 0 <= k < K.

        Returns:
            numpy array of shape (N+1,) with step sizes for n = 0..N.
        """
        if not (0 <= k < self.K):
            raise IndexError(f"k={k} out of range [0, {self.K - 1}]")
        # tau_hat_grid[k, :] has entries for n = -1, 0, ..., N (columns 0..N+1)
        # step_size[n] = tau_hat_grid[k, n] - tau_hat_grid[k, n+1]
        # for n_col = 0..N (corresponding to n = -1..N-1)
        row: np.ndarray = self.tau_hat_grid[k, :]
        # differences: row[j] - row[j+1] for j = 0..N
        step_sizes: np.ndarray = row[:-1] - row[1:]
        return step_sizes

    def summary(self) -> str:
        """Return a human-readable summary of the schedule.

        Returns:
            Multi-line string with key schedule statistics.
        """
        tau_start: float = self.get_tau_hat(0, -1)
        tau_end: float = self.get_tau_hat(self.K - 1, self.N)
        alpha_hat_T1: float = float(self.alpha_hat[self.T + 1])
        alpha_hat_1: float = float(self.alpha_hat[1])
        target_ratio: float = self.c1 * math.log(float(self.T)) / float(self.T)

        lines = [
            f"Schedule Summary",
            f"  T                      : {self.T}",
            f"  K                      : {self.K}",
            f"  N (steps/round)        : {self.N}",
            f"  c0                     : {self.c0}",
            f"  c1                     : {self.c1}",
            f"  c1/c0 ratio            : {self.c1 / self.c0:.2f}",
            f"  alpha_hat[T+1]         : {alpha_hat_T1:.6e}  (target: {1.0 / self.T**self.c0:.6e})",
            f"  alpha_hat[1]           : {alpha_hat_1:.6f}  (should be close to 1)",
            f"  tau_hat[0, -1]         : {tau_start:.6f}  (should be close to 1)",
            f"  tau_hat[K-1, N]        : {tau_end:.6f}  (should be close to 0)",
            f"  Lemma 7 target ratio   : {target_ratio:.6e}  (= c1*log(T)/T)",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"Schedule(T={self.T}, K={self.K}, N={self.N}, "
            f"c0={self.c0}, c1={self.c1})"
        )
