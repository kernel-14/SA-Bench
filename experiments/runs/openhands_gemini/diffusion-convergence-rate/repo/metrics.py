
import torch
import numpy as np
from scipy.stats import multivariate_normal

class Metrics:
    """
    Provides methods for calculating metrics like KL Divergence and Total Variation Distance.
    Specifically for Gaussian distributions, as used in the numerical experiments.
    """

    @staticmethod
    def kl_divergence_gaussian(mean1: torch.Tensor, cov1: torch.Tensor,
                               mean2: torch.Tensor, cov2: torch.Tensor) -> torch.Tensor:
        """
        Calculates the KL divergence between two multivariate Gaussian distributions.
        KL(N(mean1, cov1) || N(mean2, cov2))
        Formula: 0.5 * (tr(cov2^-1 @ cov1) + (mean2 - mean1)^T @ cov2^-1 @ (mean2 - mean1) - d + log(det(cov2)/det(cov1)))

        Args:
            mean1 (torch.Tensor): Mean vector of the first Gaussian distribution (d,).
            cov1 (torch.Tensor): Covariance matrix of the first Gaussian distribution (d, d).
            mean2 (torch.Tensor): Mean vector of the second Gaussian distribution (d,).
            cov2 (torch.Tensor): Covariance matrix of the second Gaussian distribution (d, d).

        Returns:
            torch.Tensor: The KL divergence.
        """
        d = mean1.shape[0]

        # Ensure covariance matrices are numerically stable for inverse and determinant
        epsilon = 1e-6 * torch.eye(d)
        stable_cov1 = cov1 + epsilon
        stable_cov2 = cov2 + epsilon

        # Compute inverses
        try:
            inv_cov1 = torch.inverse(stable_cov1)
            inv_cov2 = torch.inverse(stable_cov2)
        except RuntimeError:
            print("Warning: Covariance matrix might be singular even with epsilon. Using pseudo-inverse.")
            inv_cov1 = torch.linalg.pinv(stable_cov1)
            inv_cov2 = torch.linalg.pinv(stable_cov2)


        # Term 1: tr(cov2^-1 @ cov1)
        term1 = torch.trace(torch.matmul(inv_cov2, stable_cov1))

        # Term 2: (mean2 - mean1)^T @ cov2^-1 @ (mean2 - mean1)
        mean_diff = mean2 - mean1
        term2 = torch.matmul(mean_diff.T, torch.matmul(inv_cov2, mean_diff))

        # Term 3: -d
        term3 = -d

        # Term 4: log(det(cov2)/det(cov1)) = log(det(cov2)) - log(det(cov1))
        _, logdet_cov1 = torch.linalg.slogdet(stable_cov1)
        _, logdet_cov2 = torch.linalg.slogdet(stable_cov2)
        term4 = logdet_cov2 - logdet_cov1
        
        return 0.5 * (term1 + term2 + term3 + term4)

    @staticmethod
    def tv_distance_from_kl(kl_div: torch.Tensor) -> torch.Tensor:
        """
        Estimates the Total Variation (TV) distance using Pinsker's inequality.
        TV(P, Q) <= sqrt(0.5 * KL(P || Q))

        Args:
            kl_div (torch.Tensor): The KL divergence between two distributions.

        Returns:
            torch.Tensor: An upper bound for the Total Variation distance.
        """
        # KL divergence must be non-negative. If it's negative due to numerical issues, clamp to 0.
        kl_div_clamped = torch.clamp(kl_div, min=0.0)
        return torch.sqrt(0.5 * kl_div_clamped)

    @staticmethod
    def estimate_kl_div_monte_carlo(samples_p: torch.Tensor, samples_q: torch.Tensor, log_prob_p_func, log_prob_q_func, num_samples: int = 1000) -> torch.Tensor:
        """
        Estimates KL(P || Q) using Monte Carlo samples.
        KL(P || Q) = E_x~P [log(p(x) / q(x))] = E_x~P [log p(x) - log q(x)]

        Args:
            samples_p (torch.Tensor): Samples drawn from distribution P.
            samples_q (torch.Tensor): Samples drawn from distribution Q.
            log_prob_p_func (callable): Function to compute log_prob for distribution P.
                                        Signature: log_prob_p_func(x) -> torch.Tensor (log_probs for x)
            log_prob_q_func (callable): Function to compute log_prob for distribution Q.
                                        Signature: log_prob_q_func(x) -> torch.Tensor (log_probs for x)
            num_samples (int): Number of samples to use for estimation.

        Returns:
            torch.Tensor: Monte Carlo estimate of KL divergence.
        """
        # For estimating KL(P || Q), we need samples from P.
        # If samples_p are available, use them directly. Otherwise, generate.
        if samples_p.shape[0] < num_samples:
            print(f"Warning: Not enough samples from P provided ({samples_p.shape[0]}). Using all available samples for P.")
            # We would need a way to sample more from P, but we only have `samples_p` as given.
            # This function assumes `samples_p` are *the* samples from P to be used.

        # Compute log probabilities for samples from P
        log_p_x = log_prob_p_func(samples_p)
        log_q_x = log_prob_q_func(samples_p)
        
        # Avoid -inf for log_q_x if samples are outside support of Q
        # If log_q_x is -inf, then log_p_x - log_q_x will be inf.
        # This is correct if Q assigns zero probability where P assigns non-zero.
        
        kl_divergence = (log_p_x - log_q_x).mean()
        return kl_divergence

