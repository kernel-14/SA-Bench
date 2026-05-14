
import torch
import numpy as np
from typing import Tuple

class GaussianTargetDistribution:
    """
    Represents the d-dimensional Gaussian target distribution described in Appendix A.

    The target distribution p_0 is a d-dimensional Gaussian distribution with zero mean
    and a diagonal covariance matrix. The first k diagonal entries are uniformly
    distributed within the interval [0, 10], while the remaining d - k diagonal entries
    are set to zero.
    """
    def __init__(self, d_dim: int, k_components: int, diag_var_range: Tuple[float, float], seed: int = 42):
        """
        Initializes the Gaussian target distribution.

        Args:
            d_dim (int): The dimension of the data.
            k_components (int): The number of non-zero diagonal entries in the covariance matrix.
            diag_var_range (Tuple[float, float]): The range [min_val, max_val] for uniform
                                                  distribution of the first k_components diagonal entries.
            seed (int): Random seed for reproducibility.
        """
        if k_components > d_dim:
            raise ValueError("k_components cannot be greater than d_dim.")

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.d_dim = d_dim
        self.k_components = k_components
        self.diag_var_range = diag_var_range

        # Create diagonal covariance matrix
        self.sigma_0 = torch.zeros(d_dim)
        if k_components > 0:
            min_val, max_val = diag_var_range
            self.sigma_0[:k_components] = torch.rand(k_components) * (max_val - min_val) + min_val
        
        # The paper states "remaining d - k diagonal entries are set to zero."
        # This implies that these dimensions have no variance, which would make the distribution singular.
        # For a practical simulation where these dimensions still participate in noise addition (even if fixed),
        # we might assume a very small variance or handle it differently.
        # However, for exact score function calculation as implied by the paper for the 'exact score function',
        # these zero variances would lead to infinite scores, or need careful handling.
        # Given the context of "KL divergence... has a closed-form expression", it likely means
        # that these zero-variance dimensions are implicitly handled (e.g., they don't contribute to KL)
        # or are interpreted as having a small, non-zero variance for computation.
        # For now, we will strictly follow the "set to zero" and handle potential issues in score function.
        # For a truly non-singular Gaussian, all diagonal entries must be positive.
        # Assuming the paper implies non-zero for practical purposes if it leads to singularities.
        # Let's assume for numerical stability, we might add a tiny epsilon if zero causes issues,
        # but for now strictly follow "set to zero".

        self.covariance_matrix = torch.diag(self.sigma_0)
        self.mean = torch.zeros(d_dim) # Zero mean as specified

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Samples from the target Gaussian distribution p_0.

        Args:
            num_samples (int): The number of samples to generate.

        Returns:
            torch.Tensor: Samples from the distribution, shape (num_samples, d_dim).
        """
        # torch.distributions.MultivariateNormal requires positive definite covariance.
        # If any sigma_0 is zero, it's singular. We need to handle this.
        # For now, let's sample directly if it's diagonal, acknowledging potential issues.
        if self.k_components < self.d_dim and torch.any(self.sigma_0 == 0):
            # If there are zero variances, treat them as fixed at their mean (0)
            samples = torch.randn(num_samples, self.d_dim) * torch.sqrt(self.sigma_0)
            # This implicitly sets dimensions with sigma_0 = 0 to 0.
            return samples
        else:
            distribution = torch.distributions.MultivariateNormal(self.mean, self.covariance_matrix)
            return distribution.sample((num_samples,))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability density of x under the target Gaussian distribution p_0.

        Args:
            x (torch.Tensor): Input data, shape (..., d_dim).

        Returns:
            torch.Tensor: Log probabilities, shape (...).
        """
        # Handle singular covariance if k_components < d_dim
        if self.k_components < self.d_dim:
            # For singular distributions, the log_prob is infinite for points not in the support.
            # And specific for points in support.
            # This is complex. For simulation, assuming non-singular distributions if not explicitly stated.
            # If 0 variance means fixed value, log_prob should be 0 for those dimensions.
            # Let's use a small epsilon for zero diagonal entries for numerical stability in log_prob calculation
            # if they are indeed meant to have *some* noise.
            # However, the paper implies exact score functions are used, which for a singular Gaussian
            # requires careful definition.
            # Given that Appendix A says "KL divergence between Y_k,0 and X_1 has a closed-form expression"
            # for Gaussian distribution, this implies a standard non-singular setup in the context of KL computation.
            # Thus, we'll ensure self.sigma_0 has no strict zeros for log_prob and score.
            # For now, let's assume if sigma_0[i] is 0, then x[..., i] must be 0 for it to be in support.
            # For simplicity for numerical experiments, let's just use the diagonal entries that are non-zero.
            
            # If using torch.distributions.MultivariateNormal for log_prob, it expects non-singular.
            # For now, return sum of log_probs for independent dimensions if sigma_0 is diagonal
            # For dimensions with sigma_0[i] == 0, log_prob should be -inf if x[...,i] != 0, else 0.
            
            # A common way to handle pseudo-inverse for singular covariance in log_prob/score
            # is to only consider the non-singular part.
            
            # For now, implement as if it's a non-singular Gaussian,
            # which means self.sigma_0 should not have zero entries if the simulation is to work for KL.
            # Given the problem's context, the "remaining d-k diagonal entries are set to zero" might imply
            # that these dimensions are simply removed (effectively reducing d) or have infinitesimal variance.
            
            # Let's adjust initialization to ensure non-zero for active dimensions if required by subsequent uses.
            # For the purpose of score function, we need (1 - alpha_bar_t + sigma_i^2) != 0.
            
            # Revisit: The simplest interpretation for the numerical experiment, to avoid singularities
            # and allow KL computation, is that the 'zero' diagonal entries are instead a very small epsilon,
            # or that the 'd' in relevant calculations is effectively 'k_components' when considering variance.
            # For now, let's define it for a standard Gaussian.
            
            # This is for p_0(x).
            # The exact definition of p_0 as a singular Gaussian makes direct log_prob tricky.
            # Let's assume that for the numerical experiments, the 'zero' means infinitesimal,
            # or implicitly handled by only considering the 'k_components' dimensions that have variance.
            # Let's use a very small epsilon for zero components for numerical stability
            # for calculations involving inverse covariance.
            
            # For accurate log_prob of a possibly singular Gaussian, one typically calculates it in the subspace
            # spanned by the non-zero eigenvectors of the covariance matrix.
            
            # For the purpose of this problem, which is reproducing numerical experiments for convergence rates,
            # and the hint "KL divergence... has a closed-form expression", it suggests standard Gaussian assumptions.
            # Let's assume we are working in the effective 'k_components' dimension for log_prob/score
            # or that "zero" implies an infinitesimal positive variance.
            
            # Let's take the approach that dimensions with zero variance are simply fixed at their mean (0).
            # If x has non-zero values in these fixed dimensions, log_prob is -infinity.
            # Otherwise, it's the log_prob of the k_components-dimensional Gaussian.
            
            # To simplify, let's assume we can compute it for the k_components active dimensions.
            if self.k_components == 0:
                if torch.any(x != 0):
                    return -torch.inf * torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype)
                else:
                    # Constant for all-zero input if no variance, effectively delta function.
                    return torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype) # Or a large constant

            # Check if values in zero-variance dimensions are actually zero
            if self.k_components < self.d_dim:
                if torch.any(x[..., self.k_components:] != 0):
                    return -torch.inf * torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype)

            # Compute log_prob for the active k_components dimensions
            active_x = x[..., :self.k_components]
            active_mean = self.mean[:self.k_components]
            active_sigma_0 = self.sigma_0[:self.k_components]

            # Standard Gaussian log_prob formula for independent dimensions
            term1 = -0.5 * self.k_components * np.log(2 * np.pi)
            term2 = -0.5 * torch.sum(torch.log(active_sigma_0))
            term3 = -0.5 * torch.sum((active_x - active_mean)**2 / active_sigma_0, dim=-1)
            
            return term1 + term2 + term3
        else:
            # Non-singular case
            distribution = torch.distributions.MultivariateNormal(self.mean, self.covariance_matrix)
            return distribution.log_prob(x)

