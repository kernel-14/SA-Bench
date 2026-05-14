## sampler.py
"""Sampler module for the randomized midpoint diffusion sampler.

Implements the Sampler class that orchestrates the full K-round randomized
midpoint sampling procedure described in Section 2.2 of the paper.

Since the target is a Gaussian with exact (linear) score functions, the
entire procedure reduces to tracking a Gaussian distribution analytically
through affine transformations — no actual particle sampling is needed.
The GaussianTracker handles the analytical propagation; this class
coordinates the rounds, tau sampling, and noise injection.

Mathematical background
-----------------------
The sampler runs K rounds, each consisting of N = 2T/K steps.

Initialization:  Y_0 ~ N(0, I_d)

For k = 0, ..., K-1:
  1. Sample tau_{k,n} ~ Unif(tau_hat[k,n], tau_hat[k,n-1]) for n = 0..N
  2. Propagate Y_{k,0} through N ODE steps to get Y_{k,N}
  3. Inject noise: Y_{k+1} = sqrt((1-tau_{k+1,0})/(1-tau_{k,N})) * Y_{k,N}
                             + sqrt((tau_{k+1,0}-tau_{k,N})/(1-tau_{k,N})) * Z_k

The reference distribution for KL evaluation is:
  q_K = N(0, Sigma_{tau_{K,0}}) where tau_{K,0} = tau_hat[K-1, N]
  (the deterministic endpoint of the last round, near the data distribution)
"""

import math
from typing import Dict, Tuple

import torch
from tqdm import tqdm

from gaussian_tracker import GaussianTracker
from schedule import Schedule
from score import GaussianTarget

# Minimum clamp for tau denominators to avoid division by zero.
# Matches config.yaml: numerics.sigma_min_clamp: 1.0e-12
_TAU_MIN_CLAMP: float = 1.0e-12


class Sampler:
    """Orchestrates the full K-round randomized midpoint sampling procedure.

    Coordinates tau sampling, ODE propagation via GaussianTracker, and
    noise injection across all K rounds. Since the score is linear, all
    intermediate distributions remain Gaussian and are tracked analytically.

    Attributes:
        K: Number of sampler rounds. Fixed at 10 per paper (Appendix A).
        device: PyTorch device string ('cpu' or 'cuda').
    """

    def __init__(self, K: int, device: str = "cpu") -> None:
        """Initialise the sampler.

        Args:
            K: Number of sampler rounds. Must be >= 1.
            device: PyTorch device string. Must be 'cpu' or 'cuda'.
                Default 'cpu'.

        Raises:
            TypeError: If K is not an integer.
            ValueError: If K < 1 or device is not 'cpu' or 'cuda'.
        """
        if not isinstance(K, int):
            raise TypeError(f"K must be int, got {type(K).__name__}")
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if device not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda', got '{device}'")

        self.K: int = K
        self.device: str = device

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        T: int,
        schedule: Schedule,
        target: GaussianTarget,
        tracker: GaussianTracker,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute the full K-round randomized midpoint sampling procedure.

        Runs the sampler analytically by tracking the Gaussian distribution
        through all K rounds of ODE discretization and noise injection.
        Returns the mean and diagonal covariance of the final output
        distribution p_{Y_K}.

        Algorithm:
            1. Initialize: mean_0 = 0, cov_0 = I_d  (Y_0 ~ N(0, I_d))
            2. For k = 0..K-1:
               a. Sample tau_{k,n} ~ Unif(tau_hat[k,n], tau_hat[k,n-1])
                  for n = 0, 1, ..., N
               b. Propagate distribution through N ODE steps via tracker
               c. Apply noise injection to get distribution of Y_{k+1}
            3. Return (mean_K, cov_K_diag)

        Args:
            T: Total iteration count. Must match schedule.T.
            schedule: Schedule object providing tau_hat grid and sampling.
                Must have schedule.K == self.K.
            target: GaussianTarget providing score function and dimension.
            tracker: GaussianTracker for analytical distribution propagation.

        Returns:
            Tuple (mean_K, cov_K_diag) where:
                mean_K: Mean of p_{Y_K}, shape (d,), dtype float64.
                cov_K_diag: Diagonal covariance of p_{Y_K}, shape (d,),
                    dtype float64.

        Raises:
            ValueError: If schedule.K != self.K or schedule.T != T.
            ValueError: If tracker.d != target.d.
        """
        # --- Validate consistency between components ---
        if schedule.K != self.K:
            raise ValueError(
                f"schedule.K ({schedule.K}) must equal self.K ({self.K})"
            )
        if schedule.T != T:
            raise ValueError(
                f"schedule.T ({schedule.T}) must equal T ({T})"
            )
        if tracker.d != target.d:
            raise ValueError(
                f"tracker.d ({tracker.d}) must equal target.d ({target.d})"
            )

        d: int = target.d
        N: int = schedule.N

        # --- Step 1: Initialize distribution Y_0 ~ N(0, I_d) ---
        mean_k: torch.Tensor
        cov_k_diag: torch.Tensor
        mean_k, cov_k_diag = tracker.get_initial_distribution()

        # Ensure correct device and dtype
        mean_k = mean_k.to(dtype=torch.float64, device=self.device)
        cov_k_diag = cov_k_diag.to(dtype=torch.float64, device=self.device)

        # --- Step 2: Main loop over K rounds ---
        for k in tqdm(range(self.K), desc=f"Sampler rounds (T={T})", leave=False):

            # --- Step 2a: Sample tau_{k,n} for n = 0, 1, ..., N ---
            # tau_{k,n} ~ Unif(tau_hat[k,n], tau_hat[k,n-1])
            # Keys: n = 0, 1, ..., N
            tau_samples: Dict[int, float] = {}
            for n in range(N + 1):
                tau_samples[n] = schedule.sample_tau(k, n)

            # --- Step 2b: Propagate distribution through N ODE steps ---
            mean_kN: torch.Tensor
            cov_kN_diag: torch.Tensor
            mean_kN, cov_kN_diag = tracker.propagate_round(
                mean_k=mean_k,
                cov_k=cov_k_diag,
                schedule=schedule,
                target=target,
                k=k,
                tau_samples=tau_samples,
            )

            # --- Step 2c: Determine tau values for noise injection ---
            # tau_{k,N}: randomized tau at the end of round k (step N)
            tau_kN: float = float(tau_samples[N])

            # tau_{k+1,0}: deterministic starting point of the next round.
            # For k < K-1: use tau_hat[k+1, 0] (start of round k+1).
            # For k = K-1 (last round): use tau_hat[K-1, N] (end of last
            # round), which is the deterministic endpoint corresponding to
            # alpha_hat[1] (near the data distribution).
            if k < self.K - 1:
                tau_k1_0: float = schedule.get_tau_hat(k + 1, 0)
            else:
                # Last round: tau_{K,0} = tau_hat[K-1, N] (near data)
                tau_k1_0 = schedule.get_tau_hat(self.K - 1, N)

            # Safety check: tau_{k+1,0} should be >= tau_{k,N} for valid
            # noise injection. If not (numerical edge case), clamp.
            if tau_k1_0 < tau_kN:
                # This can happen at the last round boundary due to floating
                # point; use tau_kN as a safe fallback (zero noise injection).
                tau_k1_0 = tau_kN

            # --- Step 2d: Apply noise injection ---
            # Y_{k+1} = sqrt((1-tau_{k+1,0})/(1-tau_{k,N})) * Y_{k,N}
            #          + sqrt((tau_{k+1,0}-tau_{k,N})/(1-tau_{k,N})) * Z_k
            mean_k, cov_k_diag = tracker.apply_noise_injection(
                mean_kN=mean_kN,
                cov_kN=cov_kN_diag,
                tau_kN=tau_kN,
                tau_k1_0=tau_k1_0,
            )

        # --- Step 3: Return final distribution p_{Y_K} ---
        return mean_k, cov_k_diag

    def get_reference_distribution(
        self,
        schedule: Schedule,
        target: GaussianTarget,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the reference distribution q_K for KL divergence evaluation.

        The reference distribution is q_K = X_{tau_{K,0}} ~ N(0, Sigma_{tau_{K,0}})
        where tau_{K,0} = tau_hat[K-1, N] is the deterministic endpoint of
        the last round's interval.

        Since tau_{K,0} = 1 - alpha_hat[1] is very small (≈ 1/T^{c0}),
        Sigma_{tau_{K,0}} ≈ Sigma (the data covariance), so q_K ≈ p_data.

        More precisely:
            Sigma_{tau_{K,0}} = (1 - tau_{K,0}) * Sigma + tau_{K,0} * I_d

        This is the distribution that the sampler aims to approximate with
        its output p_{Y_K}.

        Args:
            schedule: Schedule object providing tau_hat grid values.
                Must have schedule.K == self.K.
            target: GaussianTarget providing marginal covariance computation.

        Returns:
            Tuple (mu_ref, cov_ref_diag) where:
                mu_ref: Zero vector of shape (d,), dtype float64.
                    q_K has zero mean since the target has zero mean.
                cov_ref_diag: Diagonal covariance of q_K, shape (d,),
                    dtype float64. Equals (1-tau_{K,0})*Sigma + tau_{K,0}*I_d.

        Raises:
            ValueError: If schedule.K != self.K.
        """
        if schedule.K != self.K:
            raise ValueError(
                f"schedule.K ({schedule.K}) must equal self.K ({self.K})"
            )

        d: int = target.d
        N: int = schedule.N

        # tau_{K,0} = tau_hat[K-1, N]: deterministic endpoint of last round
        # This is 1 - alpha_hat[1], which is very small (near 0 = clean data)
        tau_K0: float = schedule.get_tau_hat(self.K - 1, N)

        # Reference distribution covariance diagonal:
        # Sigma_{tau_{K,0}} = (1 - tau_{K,0}) * Sigma + tau_{K,0} * I_d
        cov_ref_diag: torch.Tensor = target.get_marginal_cov(tau_K0)

        # Ensure correct device and dtype
        cov_ref_diag = cov_ref_diag.to(dtype=torch.float64, device=self.device)

        # Zero mean (target has zero mean, so all marginals have zero mean)
        mu_ref: torch.Tensor = torch.zeros(
            d, dtype=torch.float64, device=self.device
        )

        return mu_ref, cov_ref_diag

    # ------------------------------------------------------------------
    # Diagnostic / utility methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"Sampler(K={self.K}, device={self.device!r})"

    def summary(self) -> str:
        """Return a human-readable summary of the sampler configuration.

        Returns:
            Multi-line string with sampler configuration details.
        """
        lines = [
            "Sampler Summary",
            f"  Number of rounds K     : {self.K}",
            f"  Device                 : {self.device}",
            f"  Propagation method     : analytical Gaussian tracking",
            f"  Score type             : exact (linear, Gaussian target)",
            f"  Distribution storage   : diagonal covariance (1D tensor)",
        ]
        return "\n".join(lines)
