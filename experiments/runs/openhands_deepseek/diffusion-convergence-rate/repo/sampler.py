r"""Randomized midpoint sampler for diffusion models.

Implements the sampler described in Section 2.2 of:
"Instance-dependent Convergence Theory for Diffusion Models"
by Yuchen Jiao and Gen Li (2025).

The sampler discretizes the probability flow ODE using a randomized schedule
and midpoint integration. It operates over K rounds with N steps each,
for a total of T = KN/2 iterations.

Key equations:
  - Schedule (Eq. 8): alpha_hat_{t-1} = alpha_hat_t + c1 * alpha_hat_t * (1 - alpha_hat_t) * log(T) / T
  - Randomized alpha_bar (Eq. 9): alpha_bar_t ~ Unif(alpha_hat_t, alpha_hat_{t-1})
  - Sampler update (Eq. 10): discretized ODE integration
  - Noise injection (Eq. 11): Y_{k+1} from Y_{k,N}
"""

import torch
from typing import List, Tuple

from score_function import GaussianScoreFunction


def build_alpha_hat_schedule(T: int, c_0: float, c_1: float) -> torch.Tensor:
    r"""Build the deterministic alpha_hat schedule (Eq. 8).

    alpha_hat_{T+1} = 1 / T^{c_0}
    alpha_hat_{t-1} = alpha_hat_t + c_1 * alpha_hat_t * (1 - alpha_hat_t) * log(T) / T

    Indices go from 0 to T+1 (inclusive), where alpha_hat[t] = alpha_hat_t.
    alpha_hat[0] is unused; alpha_hat[T+1] is the initial small value.

    Returns:
        alpha_hat: Tensor of shape (T + 2,). alpha_hat[t] for t in [0, T+1].
    """
    alpha_hat = torch.zeros(T + 2)
    alpha_hat[T + 1] = 1.0 / (T ** c_0)

    log_T = torch.log(torch.tensor(T, dtype=torch.float32))
    for t_idx in range(T + 1, 0, -1):
        alpha_hat[t_idx - 1] = (
            alpha_hat[t_idx]
            + c_1 * alpha_hat[t_idx] * (1.0 - alpha_hat[t_idx]) * log_T / T
        )
        alpha_hat[t_idx] = torch.clamp(alpha_hat[t_idx], 0.0, 1.0 - 1e-15)
    alpha_hat[0] = torch.clamp(alpha_hat[0], 0.0, 1.0 - 1e-15)
    return alpha_hat


class DiffusionSampler:
    r"""Randomized midpoint sampler for the diffusion model.

    Implements Algorithm described in Section 2.2 (Eqs. 8-11).
    """

    def __init__(
        self,
        score_fn: GaussianScoreFunction,
        T: int,
        K: int,
        c_0: float = 15.0,
        c_1: float = 75.0,
    ):
        self.score_fn = score_fn
        self.d = score_fn.d
        self.T = T
        self.K = K
        self.N = 2 * T // K
        self.c_0 = c_0
        self.c_1 = c_1
        self.alpha_hat = build_alpha_hat_schedule(T, c_0, c_1)

    def _run_round(
        self, Y_k: torch.Tensor, k: int, alpha_bar_k0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Run one round of the sampler (N ODE steps).

        t_base = T - k*N/2

        Returns:
            Y_{k,N}: Output after N steps, shape (batch_size, d).
            tau_{k,N}: The tau value at step N (for noise injection to next round).
        """
        batch_size = Y_k.shape[0]
        t_base = self.T - k * self.N // 2

        alpha_bars = [alpha_bar_k0] + [
            self._sample_single_alpha_bar(t_base + 1 - n)
            for n in range(1, self.N + 1)
        ]
        tau = torch.stack([1.0 - ab for ab in alpha_bars])

        alpha_hats = [
            self.alpha_hat[t_base + 1 - n] for n in range(self.N + 2)
        ]
        tau_hat = torch.stack([1.0 - ah for ah in alpha_hats])

        intermediates = [Y_k.clone()]
        Y_curr = Y_k

        for n in range(1, self.N + 1):
            Y_scaled = Y_k / torch.sqrt(1.0 - tau[0])
            alpha0 = 1.0 - tau[0]
            s0 = self.score_fn.score(Y_k, alpha0)
            Y_scaled = Y_scaled + (
                s0 / (2.0 * alpha0 ** 1.5) * (tau[0] - tau_hat[1])
            )
            for i in range(1, n):
                alpha_i = 1.0 - tau[i]
                s_i = self.score_fn.score(intermediates[i], alpha_i)
                Y_scaled = Y_scaled + (
                    s_i / (2.0 * alpha_i ** 1.5)
                    * (tau_hat[i] - tau_hat[i + 1])
                )
            alpha_nm1 = 1.0 - tau[n - 1]
            s_nm1 = self.score_fn.score(intermediates[n - 1], alpha_nm1)
            Y_scaled = Y_scaled + (
                s_nm1 / (2.0 * alpha_nm1 ** 1.5)
                * (tau_hat[n] - tau[n])
            )
            Y_curr = Y_scaled * torch.sqrt(1.0 - tau[n])
            intermediates.append(Y_curr.clone())

        return Y_curr, tau[self.N]

    def _sample_single_alpha_bar(self, t: int) -> torch.Tensor:
        u = torch.rand(1)
        return self.alpha_hat[t] + u * (self.alpha_hat[t - 1] - self.alpha_hat[t])

    def sample(self, batch_size: int = 1) -> torch.Tensor:
        r"""Run the full sampling process over K rounds.

        Each round:
          1. ODE integration from tau_{k,0} to tau_{k,N} producing Y_{k,N}
          2. Noise injection: Y_{k+1} ~ N(scale * Y_{k,N}, noise_var)
             The alpha_bar_{k+1,0} used here is passed as alpha_bar_{k+1,0}
             to the next round for consistency.

        Returns:
            Y_K: Final sample after K rounds, shape (batch_size, d).
        """
        Y_curr = torch.randn(batch_size, self.d)

        alpha_bar_next0 = self._sample_single_alpha_bar(self.T + 1)

        for k in range(self.K):
            alpha_bar_k0 = alpha_bar_next0
            Y_final, tau_kN = self._run_round(Y_curr, k, alpha_bar_k0)

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
        """Estimate mean and diagonal covariance of Y_K via Monte Carlo.

        Returns:
            mean: (d,) tensor.
            cov_diag: (d,) tensor (diagonal of covariance).
        """
        samples = self.sample(batch_size=num_samples)
        mean = samples.mean(dim=0)
        centered = samples - mean
        cov_diag = (centered * centered).mean(dim=0)
        return mean, cov_diag

    def compute_kl_divergence(self, num_samples: int = 10000) -> float:
        """Compute KL(p_Y_K || q_K) using closed form for Gaussians.

        q_K is the distribution of X_{tau_{K,0}}.
        """
        mean_y, cov_diag_y = self.compute_output_distribution(num_samples)

        t_base_last = self.T - (self.K - 1) * self.N // 2
        alpha_K0 = self._sample_single_alpha_bar(t_base_last + 1)
        tau_K0 = (1.0 - alpha_K0).item()

        kl = self.score_fn.kl_divergence_closed_form(mean_y, cov_diag_y, tau_K0)
        return kl.item()
