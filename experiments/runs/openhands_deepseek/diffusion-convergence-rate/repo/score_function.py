"""Exact score functions for Gaussian target distributions.

For a Gaussian target X_0 ~ N(0, Sigma), the forward process at time t is:
    X_t = sqrt(alpha_bar_t) * X_0 + sqrt(1 - alpha_bar_t) * Z,  Z ~ N(0, I_d)

The marginal distribution of X_t is N(0, Sigma_t) where:
    Sigma_t = alpha_bar_t * Sigma + (1 - alpha_bar_t) * I_d

The score function is:
    s_t^*(x) = -Sigma_t^{-1} * x

This matches Example 1 in the paper (Appendix C.1).
"""

import torch
import torch.nn as nn


class GaussianScoreFunction:
    """Exact score function for a Gaussian target distribution.

    The target distribution p_data is a d-dimensional Gaussian with zero mean
    and diagonal covariance matrix Sigma. The first k diagonal entries are
    uniformly distributed in [0, sigma_max]; the remaining d-k are zero.

    The score at time t (with noise level 1 - alpha_bar_t) is:
        s_t^*(x) = -Sigma_t^{-1} * x
    where Sigma_t = alpha_bar_t * Sigma + (1 - alpha_bar_t) * I_d.
    """

    def __init__(self, d: int, k: int, sigma_max: float = 10.0, seed: int = 42):
        """Initialize with target distribution parameters.

        Args:
            d: Data dimension.
            k: Number of non-zero variance components.
            sigma_max: Maximum variance for non-zero components.
            seed: Random seed for reproducibility.
        """
        self.d = d
        self.k = k
        self.sigma_max = sigma_max

        generator = torch.Generator()
        generator.manual_seed(seed)

        # Create diagonal variances: first k ~ Uniform[0, sigma_max], rest = 0.
        variances = torch.zeros(d)
        if k > 0:
            variances[:k] = torch.rand(k, generator=generator) * sigma_max
        self.sigma_diag = variances  # Diagonal of Sigma (d x d)

    def get_sigma_t_inv(self, alpha_bar_t: float) -> torch.Tensor:
        """Compute Sigma_t^{-1} = (alpha_bar_t * Sigma + (1 - alpha_bar_t) * I)^{-1}.

        Since Sigma is diagonal, Sigma_t is diagonal with entries:
            sigma_t_i = alpha_bar_t * sigma_i + (1 - alpha_bar_t)
        So Sigma_t^{-1} has entries 1 / sigma_t_i.
        """
        diag = alpha_bar_t * self.sigma_diag + (1.0 - alpha_bar_t)
        inv_diag = 1.0 / diag
        return inv_diag  # Shape: (d,)

    def score(self, x: torch.Tensor, alpha_bar_t: float) -> torch.Tensor:
        """Compute s_t^*(x) = -Sigma_t^{-1} * x.

        Args:
            x: Input tensor of shape (..., d).
            alpha_bar_t: Noise level parameter.

        Returns:
            Score vector of same shape as x.
        """
        inv_diag = self.get_sigma_t_inv(alpha_bar_t).to(x.device)
        # For diagonal Sigma_t: s(x) = -inv_diag * x (elementwise)
        return -inv_diag * x

    def score_batch(
        self, x: torch.Tensor, alpha_bar_t: torch.Tensor
    ) -> torch.Tensor:
        """Compute score for a batch with possibly different alpha_bar_t per sample.

        Args:
            x: Input tensor of shape (batch, d).
            alpha_bar_t: Noise level of shape (batch,).

        Returns:
            Score tensor of shape (batch, d).
        """
        batch_size = x.shape[0]
        inv_diag_list = []
        for i in range(batch_size):
            inv_diag = self.get_sigma_t_inv(alpha_bar_t[i].item())
            inv_diag_list.append(inv_diag)
        inv_diag = torch.stack(inv_diag_list, dim=0).to(x.device)
        return -inv_diag * x

    def get_distribution_params(self, alpha_bar_t: float):
        """Get mean and covariance of X_t for the given alpha_bar_t.

        Returns:
            mean: Zero vector of shape (d,).
            cov_diag: Diagonal of the covariance matrix, shape (d,).
        """
        mean = torch.zeros(self.d)
        cov_diag = alpha_bar_t * self.sigma_diag + (1.0 - alpha_bar_t)
        return mean, cov_diag

    def kl_divergence_closed_form(
        self, mean_y: torch.Tensor, cov_diag_y: torch.Tensor, tau: float
    ) -> torch.Tensor:
        """Compute KL(p_Y || q) where q is the distribution of X_tau.

        Both are Gaussians: Y ~ N(mean_y, diag(cov_diag_y)) and
        X_tau ~ N(0, diag(cov_diag_q)).

        Args:
            mean_y: Mean of the sampler output distribution, shape (d,).
            cov_diag_y: Diagonal covariance of the sampler output.
            tau: The noise level tau = 1 - alpha_bar.

        Returns:
            KL divergence scalar.
        """
        _, cov_diag_q = self.get_distribution_params(1.0 - tau)
        d = self.d
        # KL(N(μ_y, Σ_y) || N(0, Σ_q))
        # = 0.5 * [tr(Σ_q^{-1} Σ_y) + μ_y^T Σ_q^{-1} μ_y - d + log(det(Σ_q)/det(Σ_y))]
        inv_cov_q = 1.0 / cov_diag_q
        trace_term = torch.sum(inv_cov_q * cov_diag_y)
        quad_term = torch.sum(inv_cov_q * mean_y * mean_y)
        log_det_q = torch.sum(torch.log(cov_diag_q))
        log_det_y = torch.sum(torch.log(cov_diag_y))
        kl = 0.5 * (trace_term + quad_term - d + log_det_q - log_det_y)
        return kl
