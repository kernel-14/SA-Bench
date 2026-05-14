## gaussian_tracker.py
"""Gaussian distribution tracker for the randomized midpoint diffusion sampler.

Analytically propagates the Gaussian distribution through the sampler's
update steps. Since the score function s_tau*(x) = -Sigma_tau^{-1} x is
linear in x, every update step is an affine transformation of Y_{k,0},
and the output distribution remains Gaussian throughout. This enables
exact KL computation without Monte Carlo sampling.

Mathematical background
-----------------------
The sampler's update rule (Eq. 10 in the paper) in normalized space
Ỹ_{k,n} = Y_{k,n} / sqrt(1 - tau_{k,n}) is:

    Ỹ_{k,n} = Ỹ_{k,0}
        + s_{tau_{k,0}}(Y_{k,0}) / (2(1-tau_{k,0})^(3/2)) * (tau_{k,0} - tau_hat_{k,0})
        + sum_{i=1}^{n-1} s_{tau_{k,i}}(Y_{k,i}) / (2(1-tau_{k,i})^(3/2)) * (tau_hat_{k,i-1} - tau_hat_{k,i})
        + s_{tau_{k,n-1}}(Y_{k,n-1}) / (2(1-tau_{k,n-1})^(3/2)) * (tau_hat_{k,n-1} - tau_{k,n})

Since s_tau*(x) = -Sigma_tau^{-1} x and Sigma_tau is diagonal, substituting
Y_{k,i} = sqrt(1-tau_{k,i}) * Ỹ_{k,i} gives:

    s_{tau_{k,i}}(Y_{k,i}) / (2(1-tau_{k,i})^(3/2))
    = -Sigma_{tau_{k,i}}^{-1} * Ỹ_{k,i} / (2(1-tau_{k,i}))
    = -c_i * Ỹ_{k,i}

where c_i = Sigma_{tau_{k,i}}^{-1} / (2*(1-tau_{k,i})) is a d-dimensional
diagonal coefficient vector.

Since all operations are element-wise (diagonal covariances), Ỹ_{k,n} is
a diagonal linear map of Ỹ_{k,0}:

    Ỹ_{k,n} = m_n * Ỹ_{k,0}   (element-wise)

where m_n is a d-dimensional vector satisfying the two-step recurrence:

    m_0 = ones(d)
    m_1 = (1 + A_0 + C_0) * m_0
    m_n = m_{n-1} * (1 + B_{n-1} + C_{n-1}) - C_{n-2} * m_{n-2}   for n >= 2

with:
    A_0    = -c_0 * (tau_{k,0} - tau_hat_{k,0})          [first term coefficient]
    B_i    = -c_i * (tau_hat_{k,i-1} - tau_hat_{k,i})    [middle term coefficient, i >= 1]
    C_i    = -c_i * (tau_hat_{k,i} - tau_{k,i+1})        [last term coefficient at step i+1]

The noise injection step:
    Y_{k+1} = sqrt((1-tau_{k+1,0})/(1-tau_{k,N})) * Y_{k,N}
             + sqrt((tau_{k+1,0}-tau_{k,N})/(1-tau_{k,N})) * Z_k

updates the distribution as:
    mean_{k+1} = scale_y * mean_{k,N}
    cov_{k+1}  = scale_y^2 * cov_{k,N} + scale_z^2 * I_d

where scale_y = sqrt((1-tau_{k+1,0})/(1-tau_{k,N})) and
      scale_z = sqrt((tau_{k+1,0}-tau_{k,N})/(1-tau_{k,N})).
"""

import math
from typing import Dict, Tuple

import torch

from schedule import Schedule
from score import GaussianTarget

# Minimum clamp for tau denominators to avoid division by zero.
# Matches config.yaml: numerics.sigma_min_clamp: 1.0e-12
_TAU_MIN_CLAMP: float = 1.0e-12


class GaussianTracker:
    """Analytically propagates a Gaussian distribution through the sampler.

    Since the score function is linear (diagonal), all intermediate
    distributions Y_{k,n} remain Gaussian. This class tracks the mean
    and diagonal covariance exactly, enabling closed-form KL computation.

    All covariances are stored as 1D tensors (diagonal entries only),
    exploiting the diagonal structure throughout.

    Attributes:
        d: Data dimension.
        device: PyTorch device string ('cpu' or 'cuda').
    """

    def __init__(self, d: int, device: str = "cpu") -> None:
        """Initialise the Gaussian tracker.

        Args:
            d: Data dimension. Must be >= 1.
            device: PyTorch device string. Must be 'cpu' or 'cuda'.
                Default 'cpu'.

        Raises:
            TypeError: If d is not an integer.
            ValueError: If d < 1 or device is not 'cpu' or 'cuda'.
        """
        if not isinstance(d, int):
            raise TypeError(f"d must be int, got {type(d).__name__}")
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        if device not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda', got '{device}'")

        self.d: int = d
        self.device: str = device

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_initial_distribution(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the initial distribution Y_0 ~ N(0, I_d).

        The sampler is initialized with Y_0 ~ N(0, I_d) (standard Gaussian
        noise), corresponding to the starting point of the reverse process.

        Returns:
            Tuple (mean, cov_diag) where:
                mean: Zero vector of shape (d,), dtype float64.
                cov_diag: Ones vector of shape (d,), dtype float64.
                    Represents the diagonal of I_d.
        """
        mean: torch.Tensor = torch.zeros(
            self.d, dtype=torch.float64, device=self.device
        )
        cov_diag: torch.Tensor = torch.ones(
            self.d, dtype=torch.float64, device=self.device
        )
        return mean, cov_diag

    def propagate_round(
        self,
        mean_k: torch.Tensor,
        cov_k: torch.Tensor,
        schedule: Schedule,
        target: GaussianTarget,
        k: int,
        tau_samples: Dict[int, float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate the distribution through one full round of N ODE steps.

        Computes the linear map m_N (d-dimensional diagonal coefficient)
        such that Ỹ_{k,N} = m_N * Ỹ_{k,0} element-wise, then converts
        back to the un-normalized space Y_{k,N} = sqrt(1-tau_{k,N}) * Ỹ_{k,N}.

        The distribution update is:
            mean_{k,N} = sqrt(1-tau_{k,N}) * m_N * (mean_k / sqrt(1-tau_{k,0}))
            cov_{k,N}  = (1-tau_{k,N}) * m_N^2 * (cov_k / (1-tau_{k,0}))

        Args:
            mean_k: Mean of Y_{k,0}, shape (d,), dtype float64.
            cov_k: Diagonal covariance of Y_{k,0}, shape (d,), dtype float64.
            schedule: Schedule object providing tau_hat grid values.
            target: GaussianTarget providing score Jacobian coefficients.
            k: Round index, 0 <= k < K.
            tau_samples: Dict mapping step index n (0..N) to the sampled
                tau_{k,n} value. Must contain keys 0, 1, ..., N.

        Returns:
            Tuple (mean_kN, cov_kN) where:
                mean_kN: Mean of Y_{k,N}, shape (d,), dtype float64.
                cov_kN: Diagonal covariance of Y_{k,N}, shape (d,), dtype float64.

        Raises:
            KeyError: If tau_samples is missing required keys.
            ValueError: If k is out of range.
        """
        N: int = schedule.N

        # Validate inputs
        if not (0 <= k < schedule.K):
            raise ValueError(
                f"k={k} out of range [0, {schedule.K - 1}]"
            )
        for n_req in range(N + 1):
            if n_req not in tau_samples:
                raise KeyError(
                    f"tau_samples missing key n={n_req}. "
                    f"Required keys: 0..{N}"
                )

        # Compute the full linear map m_N via the two-step recurrence
        m_N: torch.Tensor = self.compute_round_matrix(
            schedule, target, k, tau_samples
        )

        # tau values at start and end of round
        tau_k0: float = float(tau_samples[0])
        tau_kN: float = float(tau_samples[N])

        # Denominators for normalization (clamped for safety)
        one_minus_tau_k0: float = max(1.0 - tau_k0, _TAU_MIN_CLAMP)
        one_minus_tau_kN: float = max(1.0 - tau_kN, _TAU_MIN_CLAMP)

        # Convert mean from Y_{k,0} space to Ỹ_{k,0} space, apply m_N,
        # then convert back to Y_{k,N} space.
        #
        # Ỹ_{k,0} = Y_{k,0} / sqrt(1 - tau_{k,0})
        # Ỹ_{k,N} = m_N * Ỹ_{k,0}
        # Y_{k,N} = sqrt(1 - tau_{k,N}) * Ỹ_{k,N}
        #
        # So: Y_{k,N} = sqrt(1-tau_{k,N}) * m_N * Y_{k,0} / sqrt(1-tau_{k,0})
        #             = (sqrt(1-tau_{k,N}) / sqrt(1-tau_{k,0})) * m_N * Y_{k,0}

        scale_norm: float = math.sqrt(one_minus_tau_kN / one_minus_tau_k0)

        # Mean update: mean_{k,N} = scale_norm * m_N * mean_k
        mean_kN: torch.Tensor = scale_norm * m_N * mean_k

        # Covariance update: cov_{k,N} = scale_norm^2 * m_N^2 * cov_k
        cov_kN: torch.Tensor = (scale_norm ** 2) * (m_N ** 2) * cov_k

        return mean_kN, cov_kN

    def apply_noise_injection(
        self,
        mean_kN: torch.Tensor,
        cov_kN: torch.Tensor,
        tau_kN: float,
        tau_k1_0: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the noise injection step at the end of round k.

        The noise injection (Eq. 10, step 3 in the paper) is:
            Y_{k+1} = sqrt((1-tau_{k+1,0})/(1-tau_{k,N})) * Y_{k,N}
                     + sqrt((tau_{k+1,0}-tau_{k,N})/(1-tau_{k,N})) * Z_k

        where Z_k ~ N(0, I_d) is independent of Y_{k,N}.

        This updates the distribution as:
            mean_{k+1} = scale_y * mean_{k,N}
            cov_{k+1}  = scale_y^2 * cov_{k,N} + scale_z^2 * I_d

        where:
            scale_y = sqrt((1-tau_{k+1,0}) / (1-tau_{k,N}))
            scale_z = sqrt((tau_{k+1,0} - tau_{k,N}) / (1-tau_{k,N}))

        Args:
            mean_kN: Mean of Y_{k,N}, shape (d,), dtype float64.
            cov_kN: Diagonal covariance of Y_{k,N}, shape (d,), dtype float64.
            tau_kN: Sampled tau_{k,N} (end of round k). Must be in (0, 1).
            tau_k1_0: Deterministic tau_{k+1,0} (start of round k+1).
                Must satisfy tau_k1_0 > tau_kN for the noise injection to
                add positive variance.

        Returns:
            Tuple (mean_k1, cov_k1) where:
                mean_k1: Mean of Y_{k+1}, shape (d,), dtype float64.
                cov_k1: Diagonal covariance of Y_{k+1}, shape (d,), dtype float64.

        Raises:
            ValueError: If tau_kN >= tau_k1_0 (degenerate noise injection).
        """
        tau_kN_f: float = float(tau_kN)
        tau_k1_0_f: float = float(tau_k1_0)

        # Denominator: 1 - tau_{k,N}
        one_minus_tau_kN: float = max(1.0 - tau_kN_f, _TAU_MIN_CLAMP)

        # Numerators
        one_minus_tau_k1_0: float = max(1.0 - tau_k1_0_f, _TAU_MIN_CLAMP)
        delta_tau: float = tau_k1_0_f - tau_kN_f

        if delta_tau < 0.0:
            raise ValueError(
                f"tau_k1_0 ({tau_k1_0_f:.6e}) must be >= tau_kN ({tau_kN_f:.6e}). "
                f"The noise injection requires tau_{k+1,0} > tau_{k,N}."
            )

        # Scale factors
        scale_y: float = math.sqrt(one_minus_tau_k1_0 / one_minus_tau_kN)
        scale_z_sq: float = max(delta_tau / one_minus_tau_kN, 0.0)

        # Mean update: mean_{k+1} = scale_y * mean_{k,N}
        mean_k1: torch.Tensor = scale_y * mean_kN

        # Covariance update: cov_{k+1} = scale_y^2 * cov_{k,N} + scale_z^2 * I_d
        cov_k1: torch.Tensor = (scale_y ** 2) * cov_kN + scale_z_sq * torch.ones(
            self.d, dtype=torch.float64, device=self.device
        )

        return mean_k1, cov_k1

    def compute_round_matrix(
        self,
        schedule: Schedule,
        target: GaussianTarget,
        k: int,
        tau_samples: Dict[int, float],
    ) -> torch.Tensor:
        """Compute the diagonal linear map m_N for round k.

        Computes the d-dimensional vector m_N such that:
            Ỹ_{k,N} = m_N * Ỹ_{k,0}   (element-wise)

        using the two-step recurrence:
            m_0 = ones(d)
            m_1 = (1 + A_0 + C_0) * m_0
            m_n = m_{n-1} * (1 + B_{n-1} + C_{n-1}) - C_{n-2} * m_{n-2}  for n >= 2

        where (all operations are element-wise on d-dimensional vectors):
            c_i    = Sigma_{tau_{k,i}}^{-1} / (2 * (1 - tau_{k,i}))
            A_0    = -c_0 * (tau_{k,0} - tau_hat_{k,0})
            B_i    = -c_i * (tau_hat_{k,i-1} - tau_hat_{k,i})   for i >= 1
            C_i    = -c_i * (tau_hat_{k,i} - tau_{k,i+1})       for i >= 0

        Note on C_i: the "last term" at step n uses score at tau_{k,n-1}
        with interval (tau_hat_{k,n-1} - tau_{k,n}), so C_{n-1} uses
        c_{n-1} and interval (tau_hat_{k,n-1} - tau_{k,n}).

        Args:
            schedule: Schedule object providing tau_hat grid values.
            target: GaussianTarget providing inverse marginal covariances.
            k: Round index, 0 <= k < K.
            tau_samples: Dict mapping n (0..N) to sampled tau_{k,n}.

        Returns:
            Tensor of shape (d,) with dtype float64 on self.device.
            This is the diagonal of the linear map from Ỹ_{k,0} to Ỹ_{k,N}.
        """
        N: int = schedule.N

        # --- Precompute score coefficients c_i for i = 0..N-1 ---
        # c_i = Sigma_{tau_{k,i}}^{-1} / (2 * (1 - tau_{k,i}))
        # Shape: (d,) for each i
        c_coeffs: Dict[int, torch.Tensor] = {}
        for i in range(N):
            tau_ki: float = float(tau_samples[i])
            one_minus_tau_ki: float = max(1.0 - tau_ki, _TAU_MIN_CLAMP)
            # Sigma_{tau_{k,i}}^{-1}: shape (d,)
            sigma_inv: torch.Tensor = target.get_marginal_cov_inv(tau_ki)
            # c_i = sigma_inv / (2 * (1 - tau_{k,i}))
            c_coeffs[i] = sigma_inv / (2.0 * one_minus_tau_ki)

        # --- Compute A_0 ---
        # A_0 = -c_0 * (tau_{k,0} - tau_hat_{k,0})
        tau_k0: float = float(tau_samples[0])
        tau_hat_k0: float = schedule.get_tau_hat(k, 0)
        A_0: torch.Tensor = -c_coeffs[0] * (tau_k0 - tau_hat_k0)

        # --- Compute C_i for i = 0..N-1 ---
        # C_i = -c_i * (tau_hat_{k,i} - tau_{k,i+1})
        # The last term at step n uses C_{n-1}: score at tau_{k,n-1},
        # interval (tau_hat_{k,n-1} - tau_{k,n}).
        C_coeffs: Dict[int, torch.Tensor] = {}
        for i in range(N):
            tau_hat_ki: float = schedule.get_tau_hat(k, i)
            tau_ki_plus1: float = float(tau_samples[i + 1])
            interval: float = tau_hat_ki - tau_ki_plus1
            C_coeffs[i] = -c_coeffs[i] * interval

        # --- Compute B_i for i = 1..N-1 ---
        # B_i = -c_i * (tau_hat_{k,i-1} - tau_hat_{k,i})
        B_coeffs: Dict[int, torch.Tensor] = {}
        for i in range(1, N):
            tau_hat_ki_minus1: float = schedule.get_tau_hat(k, i - 1)
            tau_hat_ki: float = schedule.get_tau_hat(k, i)
            interval: float = tau_hat_ki_minus1 - tau_hat_ki
            B_coeffs[i] = -c_coeffs[i] * interval

        # --- Two-step recurrence for m_n ---
        # m_0 = ones(d)
        m_prev_prev: torch.Tensor = torch.ones(
            self.d, dtype=torch.float64, device=self.device
        )  # m_{n-2}, initialized to m_0

        # m_1 = (1 + A_0 + C_0) * m_0
        ones_d: torch.Tensor = torch.ones(
            self.d, dtype=torch.float64, device=self.device
        )
        m_prev: torch.Tensor = (ones_d + A_0 + C_coeffs[0]) * m_prev_prev  # m_1

        if N == 1:
            # Only one step: return m_1
            return m_prev

        # For n = 2..N:
        # m_n = m_{n-1} * (1 + B_{n-1} + C_{n-1}) - C_{n-2} * m_{n-2}
        m_curr: torch.Tensor = m_prev  # will be overwritten in loop

        for n in range(2, N + 1):
            # B_{n-1}: middle term coefficient (i = n-1 >= 1)
            B_nm1: torch.Tensor = B_coeffs[n - 1]

            # C_{n-1}: last term coefficient at step n (i = n-1)
            C_nm1: torch.Tensor = C_coeffs[n - 1]

            # C_{n-2}: last term coefficient at step n-1 (i = n-2)
            C_nm2: torch.Tensor = C_coeffs[n - 2]

            # Recurrence: m_n = m_{n-1} * (1 + B_{n-1} + C_{n-1}) - C_{n-2} * m_{n-2}
            m_curr = m_prev * (ones_d + B_nm1 + C_nm1) - C_nm2 * m_prev_prev

            # Shift: m_{n-2} <- m_{n-1}, m_{n-1} <- m_n
            m_prev_prev = m_prev
            m_prev = m_curr

        return m_curr

    # ------------------------------------------------------------------
    # Convenience / diagnostic methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"GaussianTracker(d={self.d}, device={self.device!r})"

    def summary(self) -> str:
        """Return a human-readable summary of the tracker.

        Returns:
            Multi-line string with tracker configuration.
        """
        lines = [
            "GaussianTracker Summary",
            f"  Data dimension d       : {self.d}",
            f"  Device                 : {self.device}",
            f"  Covariance storage     : diagonal (1D tensor of shape ({self.d},))",
            f"  Propagation method     : analytical (two-step recurrence)",
        ]
        return "\n".join(lines)
