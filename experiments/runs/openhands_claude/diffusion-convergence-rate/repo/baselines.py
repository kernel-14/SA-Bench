"""
Baseline samplers for comparison with the randomized midpoint sampler.

Implements simplified versions of the baseline methods discussed in the paper:
  - DDPM (Ho et al., 2020): standard denoising diffusion probabilistic model
  - Probability flow ODE with uniform step size (Chen et al., 2022)
  - Accelerated sampler (Li and Cai, 2024 style)

These are used for empirical comparison in the experiments.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from score_functions import ScoreFunction
from forward_process import LearningRateSchedule


@dataclass
class BaselineSamplerResult:
    """Result from a baseline sampler."""
    samples: np.ndarray
    score_evals: int
    method_name: str


class DDPMSampler:
    """
    DDPM reverse process sampler (Ho et al., 2020).

    Reverse process:
      Y_{t-1} = 1/sqrt(alpha_t) * (Y_t + (1-alpha_t)/sqrt(1-alpha_bar_t) * s_t*(Y_t))
              + sqrt(beta_t) * Z_t

    where beta_t = 1 - alpha_t, Z_t ~ N(0, I_d).

    This corresponds to the SDE-based reverse process, not the ODE.
    """

    def __init__(
        self,
        score_fn: ScoreFunction,
        T: int,
        beta_schedule: str = "linear",
        beta_min: float = 1e-4,
        beta_max: float = 0.02,
        seed: Optional[int] = None,
    ):
        """
        Args:
            score_fn: score function
            T: number of diffusion steps
            beta_schedule: 'linear' or 'cosine'
            beta_min: minimum beta value
            beta_max: maximum beta value
            seed: random seed
        """
        self.score_fn = score_fn
        self.T = T
        self.rng = np.random.default_rng(seed)
        self.betas = self._make_beta_schedule(T, beta_schedule, beta_min, beta_max)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

    def _make_beta_schedule(
        self,
        T: int,
        schedule: str,
        beta_min: float,
        beta_max: float,
    ) -> np.ndarray:
        """Create beta schedule."""
        if schedule == "linear":
            return np.linspace(beta_min, beta_max, T)
        elif schedule == "cosine":
            # Cosine schedule (Nichol and Dhariwal, 2021)
            s = 0.008
            t = np.linspace(0, T, T + 1)
            f = np.cos((t / T + s) / (1 + s) * np.pi / 2)**2
            alpha_bars = f / f[0]
            betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
            return np.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def sample(self, d: int, n_samples: int = 1) -> BaselineSamplerResult:
        """
        Run DDPM reverse process.

        Args:
            d: data dimension
            n_samples: number of samples

        Returns:
            BaselineSamplerResult
        """
        all_samples = []

        for _ in range(n_samples):
            # Initialize from Gaussian noise
            Y = self.rng.standard_normal(d)

            for t in range(self.T - 1, -1, -1):
                alpha_t = self.alphas[t]
                alpha_bar_t = self.alpha_bars[t]
                beta_t = self.betas[t]

                # tau = 1 - alpha_bar_t
                tau = 1.0 - alpha_bar_t

                # Score evaluation
                s = self.score_fn.score(Y, tau)

                # DDPM reverse step
                mean = (1.0 / np.sqrt(alpha_t)) * (Y + beta_t / np.sqrt(1.0 - alpha_bar_t) * s)

                if t > 0:
                    # Add noise (not at last step)
                    alpha_bar_prev = self.alpha_bars[t - 1]
                    variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                    Y = mean + np.sqrt(variance) * self.rng.standard_normal(d)
                else:
                    Y = mean

            all_samples.append(Y)

        samples = np.stack(all_samples) if n_samples > 1 else all_samples[0]
        return BaselineSamplerResult(
            samples=samples,
            score_evals=self.T * n_samples,
            method_name="DDPM",
        )


class ProbabilityFlowODESampler:
    """
    Probability flow ODE sampler with uniform step size.

    Discretizes the ODE:
      d(Y/sqrt(1-tau))/dtau = -s_tau*(Y) / (2(1-tau)^{3/2})

    using a simple Euler method with uniform step size.

    This corresponds to the basic ODE sampler without randomization.
    """

    def __init__(
        self,
        score_fn: ScoreFunction,
        T: int,
        tau_start: float = 0.999,
        tau_end: float = 1e-4,
        seed: Optional[int] = None,
    ):
        """
        Args:
            score_fn: score function
            T: number of ODE steps
            tau_start: starting noise level (close to 1)
            tau_end: ending noise level (close to 0)
            seed: random seed
        """
        self.score_fn = score_fn
        self.T = T
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.rng = np.random.default_rng(seed)
        self.taus = np.linspace(tau_start, tau_end, T + 1)

    def sample(self, d: int, n_samples: int = 1) -> BaselineSamplerResult:
        """
        Run probability flow ODE sampler.

        Args:
            d: data dimension
            n_samples: number of samples

        Returns:
            BaselineSamplerResult
        """
        all_samples = []

        for _ in range(n_samples):
            # Initialize from Gaussian noise
            Y = self.rng.standard_normal(d)

            for i in range(self.T):
                tau = self.taus[i]
                tau_next = self.taus[i + 1]
                dtau = tau_next - tau  # negative (decreasing tau)

                # Score evaluation
                s = self.score_fn.score(Y, tau)

                # ODE step in normalized coordinates
                # d(Y/sqrt(1-tau))/dtau = -s/(2(1-tau)^{3/2})
                y_norm = Y / np.sqrt(1.0 - tau)
                dy_norm = -s / (2.0 * (1.0 - tau)**1.5)
                y_norm_next = y_norm + dtau * dy_norm
                Y = y_norm_next * np.sqrt(1.0 - tau_next)

            all_samples.append(Y)

        samples = np.stack(all_samples) if n_samples > 1 else all_samples[0]
        return BaselineSamplerResult(
            samples=samples,
            score_evals=self.T * n_samples,
            method_name="ProbabilityFlowODE",
        )


class UniformMidpointSampler:
    """
    Midpoint sampler with uniform (non-randomized) step sizes.

    This is the deterministic version of the randomized midpoint sampler,
    used to illustrate the benefit of randomization.

    Uses the same ODE discretization but with fixed tau_{k,n} = tau_hat_{k,n}
    (no randomization).
    """

    def __init__(
        self,
        score_fn: ScoreFunction,
        T: int,
        K: int,
        c0: float = 5.0,
        c1: float = 50.0,
        seed: Optional[int] = None,
    ):
        self.score_fn = score_fn
        self.T = T
        self.K = K
        self.N = 2 * T // K
        self.schedule = LearningRateSchedule(T, K, c0, c1)
        self.rng = np.random.default_rng(seed)

    def sample(self, d: int, n_samples: int = 1) -> BaselineSamplerResult:
        """Run uniform midpoint sampler."""
        all_samples = []

        for _ in range(n_samples):
            Y_k = self.rng.standard_normal(d)

            for k in range(self.K):
                # Use deterministic tau values (midpoints of intervals)
                tau_hat_arr = np.array([self.schedule.tau_hat(k, n) for n in range(-1, self.N + 1)])
                # Use midpoints instead of random samples
                tau_arr = np.array([
                    0.5 * (tau_hat_arr[n] + tau_hat_arr[n + 1])
                    for n in range(self.N + 1)
                ])

                Y_kN = self._compute_round(Y_k, tau_hat_arr, tau_arr)

                tau_kN = tau_arr[self.N]
                if k < self.K - 1:
                    tau_k1_0 = self.schedule.tau_hat(k + 1, 0)
                    scale_signal = np.sqrt((1.0 - tau_k1_0) / (1.0 - tau_kN))
                    scale_noise = np.sqrt((tau_k1_0 - tau_kN) / (1.0 - tau_kN))
                    Z_k = self.rng.standard_normal(d)
                    Y_k = scale_signal * Y_kN + scale_noise * Z_k
                else:
                    Y_k = Y_kN

            all_samples.append(Y_k)

        samples = np.stack(all_samples) if n_samples > 1 else all_samples[0]
        return BaselineSamplerResult(
            samples=samples,
            score_evals=self.T * n_samples,
            method_name="UniformMidpoint",
        )

    def _compute_round(
        self,
        Y_k0: np.ndarray,
        tau_hat_arr: np.ndarray,
        tau_arr: np.ndarray,
    ) -> np.ndarray:
        """Compute one round with deterministic tau values."""
        N = self.N
        tau_k0 = tau_arr[0]
        tau_hat_k0 = tau_hat_arr[1]

        s0 = self.score_fn.score(Y_k0, tau_k0)
        base_norm = Y_k0 / np.sqrt(1.0 - tau_k0)
        base_norm = base_norm + s0 / (2.0 * (1.0 - tau_k0)**1.5) * (tau_k0 - tau_hat_k0)

        running_sum = np.zeros_like(Y_k0)
        Y_prev = Y_k0
        s_prev = s0
        tau_prev = tau_k0
        tau_hat_prev = tau_hat_k0

        Y_kN = Y_k0

        for n in range(1, N + 1):
            tau_kn = tau_arr[n]
            tau_hat_kn_minus1 = tau_hat_arr[n]
            tau_hat_kn = tau_hat_arr[n + 1] if n + 1 < len(tau_hat_arr) else tau_hat_arr[-1]

            if n >= 2:
                tau_hat_prev_prev = tau_hat_arr[n - 1]
                running_sum = running_sum + s_prev / (2.0 * (1.0 - tau_prev)**1.5) * (tau_hat_prev_prev - tau_hat_prev)

            last_term = s_prev / (2.0 * (1.0 - tau_prev)**1.5) * (tau_hat_prev - tau_kn)
            y_kn_norm = base_norm + running_sum + last_term
            Y_kn = y_kn_norm * np.sqrt(1.0 - tau_kn)

            if n < N:
                s_kn = self.score_fn.score(Y_kn, tau_kn)
                Y_prev = Y_kn
                s_prev = s_kn
                tau_prev = tau_kn
                tau_hat_prev = tau_hat_kn_minus1
            else:
                Y_kN = Y_kn

        return Y_kN


class LangevinSampler:
    """
    Unadjusted Langevin algorithm (ULA) for comparison.

    Update: Y_{t+1} = Y_t + eta * s_tau*(Y_t) + sqrt(2*eta) * Z_t

    This is a simple baseline that does not use the ODE structure.
    """

    def __init__(
        self,
        score_fn: ScoreFunction,
        T: int,
        tau: float = 0.5,
        step_size: float = 0.01,
        seed: Optional[int] = None,
    ):
        """
        Args:
            score_fn: score function at fixed noise level tau
            T: number of Langevin steps
            tau: noise level (fixed)
            step_size: Langevin step size eta
            seed: random seed
        """
        self.score_fn = score_fn
        self.T = T
        self.tau = tau
        self.step_size = step_size
        self.rng = np.random.default_rng(seed)

    def sample(self, d: int, n_samples: int = 1) -> BaselineSamplerResult:
        """Run Langevin sampler."""
        all_samples = []
        eta = self.step_size

        for _ in range(n_samples):
            Y = self.rng.standard_normal(d)

            for _ in range(self.T):
                s = self.score_fn.score(Y, self.tau)
                Z = self.rng.standard_normal(d)
                Y = Y + eta * s + np.sqrt(2 * eta) * Z

            all_samples.append(Y)

        samples = np.stack(all_samples) if n_samples > 1 else all_samples[0]
        return BaselineSamplerResult(
            samples=samples,
            score_evals=self.T * n_samples,
            method_name="Langevin",
        )
