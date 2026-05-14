r"""Parallel sampling algorithm from Appendix E.1 of:
"Instance-dependent Convergence Theory for Diffusion Models"
by Yuchen Jiao and Gen Li (2025).

The parallel sampler uses N processors with M parallel iterations per round.
Total parallel rounds = MK, total score evals per processor = M*K.

Key equations:
  - Parallel update (Eq. 93): each processor uses outputs from the previous
    parallel iteration across all processors.
  - Noise injection (Eq. 94): same as sequential case.

The parallelization works because the ODE integration can be done using
Gauss-Seidel style iterations across all time steps simultaneously.
"""

import torch
from typing import List, Tuple

from score_function import GaussianScoreFunction
from sampler import build_alpha_hat_schedule


class ParallelDiffusionSampler:
    r"""Parallel implementation of the randomized midpoint sampler.

    Uses N processors, each responsible for one time step n = 1, ..., N.
    Runs M parallel iterations per round (total MK parallel rounds).
    """

    def __init__(
        self,
        score_fn: GaussianScoreFunction,
        T: int,
        K: int,
        M: int = 5,
        c_0: float = 15.0,
        c_1: float = 75.0,
    ):
        """Initialize the parallel sampler.

        Args:
            score_fn: The (exact) score function.
            T: Total iteration budget. Sequential equivalent is KN = 2T.
            K: Number of rounds.
            M: Number of parallel iterations per round.
            c_0: Schedule constant.
            c_1: Schedule constant.
        """
        self.score_fn = score_fn
        self.d = score_fn.d
        self.T = T
        self.K = K
        self.N = 2 * T // K
        self.M = M
        self.c_0 = c_0
        self.c_1 = c_1
        self.alpha_hat = build_alpha_hat_schedule(T, c_0, c_1)

    def _sample_single_alpha_bar(self, t: int) -> torch.Tensor:
        u = torch.rand(1)
        return self.alpha_hat[t] + u * (self.alpha_hat[t - 1] - self.alpha_hat[t])

    def _run_round(
        self, Y_k: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Run one round with M parallel iterations.

        Each processor n updates Y_{m,k,n} using Y_{m-1,k,i} for all i < n.

        For m=0, all processors start with Y_{0,k,n} = Y_k.

        The update (Eq. 93):
          Y_{m,k,n} / sqrt(1 - τ_{k,n})
            = Y_k / sqrt(1 - τ_{k,0})
            + s(Y_k) / (2*(1-τ_{k,0})^{3/2}) * (τ_{k,0} - τ̂_{k,0})
            + Σ_{i=1}^{n-1} s(Y_{m-1,k,i}) / (2*(1-τ_{k,i})^{3/2}) * (τ̂_{k,i-1} - τ̂_{k,i})
            + s(Y_{m-1,k,n-1}) / (2*(1-τ_{k,n-1})^{3/2}) * (τ̂_{k,n-1} - τ_{k,n})
        """
        batch_size = Y_k.shape[0]
        t_base = self.T - k * self.N // 2

        alpha_bars = [
            self._sample_single_alpha_bar(t_base + 1 - n)
            for n in range(self.N + 1)
        ]
        tau = torch.stack([1.0 - ab for ab in alpha_bars])  # (N+1,)

        alpha_hats = [
            self.alpha_hat[t_base + 1 - n] for n in range(self.N + 2)
        ]
        tau_hat = torch.stack([1.0 - ah for ah in alpha_hats])  # (N+2,)

        Y_scaled_base = Y_k / torch.sqrt(1.0 - tau[0])
        alpha0 = 1.0 - tau[0]
        s0 = self.score_fn.score(Y_k, alpha0)
        initial_term = s0 / (2.0 * alpha0 ** 1.5) * (tau[0] - tau_hat[1])

        Y_all = [Y_k.clone() for _ in range(self.N + 1)]

        for m in range(self.M):
            Y_prev = Y_all.copy()

            for n in range(1, self.N + 1):
                Y_scaled = Y_scaled_base + initial_term

                for i in range(1, n):
                    alpha_i = 1.0 - tau[i]
                    s_i = self.score_fn.score(Y_prev[i], alpha_i)
                    Y_scaled = Y_scaled + (
                        s_i / (2.0 * alpha_i ** 1.5)
                        * (tau_hat[i] - tau_hat[i + 1])
                    )

                alpha_nm1 = 1.0 - tau[n - 1]
                s_nm1 = self.score_fn.score(Y_prev[n - 1], alpha_nm1)
                Y_scaled = Y_scaled + (
                    s_nm1 / (2.0 * alpha_nm1 ** 1.5)
                    * (tau_hat[n] - tau[n])
                )

                Y_all[n] = Y_scaled * torch.sqrt(1.0 - tau[n])

        return Y_all[self.N], tau[self.N]

    def sample(self, batch_size: int = 1) -> torch.Tensor:
        """Run the full parallel sampling process over K rounds."""
        Y_curr = torch.randn(batch_size, self.d)

        for k in range(self.K):
            Y_final, tau_kN = self._run_round(Y_curr, k)

            t_base_next = self.T - (k + 1) * self.N // 2
            alpha_bar_next0 = self._sample_single_alpha_bar(t_base_next + 1)
            tau_next0 = 1.0 - alpha_bar_next0

            denom = 1.0 - tau_kN
            scale = torch.sqrt((1.0 - tau_next0) / denom)
            noise_var = torch.clamp(tau_next0 - tau_kN, min=0.0)
            noise_scale = torch.sqrt(noise_var / denom)

            Z_k = torch.randn(batch_size, self.d)
            Y_curr = scale * Y_final + noise_scale * Z_k

        return Y_curr

    def compute_output_distribution(
        self, num_samples: int = 10000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = self.sample(batch_size=num_samples)
        mean = samples.mean(dim=0)
        centered = samples - mean
        cov_diag = (centered * centered).mean(dim=0)
        return mean, cov_diag

    def compute_kl_divergence(self, num_samples: int = 10000) -> float:
        mean_y, cov_diag_y = self.compute_output_distribution(num_samples)

        t_base_last = self.T - (self.K - 1) * self.N // 2
        alpha_K0 = self._sample_single_alpha_bar(t_base_last + 1)
        tau_K0 = (1.0 - alpha_K0).item()

        kl = self.score_fn.kl_divergence_closed_form(mean_y, cov_diag_y, tau_K0)
        return kl.item()
