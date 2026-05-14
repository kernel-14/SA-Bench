
import torch
import numpy as np
from typing import Tuple

class ScoreFunction:
    """
    Implements the exact score function s_t^*(x) for a Gaussian target distribution.
    The formula is derived from Appendix C.1.
    """
    def __init__(self, d_dim: int, sigma_0: torch.Tensor):
        """
        Initializes the ScoreFunction.

        Args:
            d_dim (int): The dimension of the data.
            sigma_0 (torch.Tensor): Diagonal entries of the covariance matrix of X_0.
                                     Shape (d_dim,).
        """
        self.d_dim = d_dim
        self.sigma_0 = sigma_0 # This is the diagonal of the covariance matrix of X_0

        # For dimensions where sigma_0 is 0, special handling is needed for score.
        # The paper implies exact score is used, and it's for Gaussian.
        # Appendix C.1 shows s_t^*(x) = -Sigma_t^-1 x where Sigma_t is a diagonal matrix
        # with (Sigma_t)_i,i = overline_alpha_t * sigma_i^2 + 1 - overline_alpha_t.
        # If sigma_i^2 = 0, then (Sigma_t)_i,i = 1 - overline_alpha_t.
        # So it seems even zero initial variance dimensions will have non-zero variance at time t,
        # provided overline_alpha_t < 1. This avoids singularity for Sigma_t.

    def s_t_star(self, x: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
        """
        Calculates the exact score function s_t^*(x) = nabla log p_{X_t}(x).
        From Appendix C.1, for Gaussian X_0 ~ N(0, Sigma_0),
        X_t ~ N(0, overline_alpha_t * Sigma_0 + (1 - overline_alpha_t) * I_d).
        So, p_{X_t}(x) is N(0, Sigma_t) where Sigma_t = overline_alpha_t * Sigma_0 + (1 - overline_alpha_t) * I_d.
        The score function is -Sigma_t^-1 * x.

        Args:
            x (torch.Tensor): Current sample, shape (..., d_dim).
            alpha_bar_t (torch.Tensor): The product overline_alpha_t, shape (...,).

        Returns:
            torch.Tensor: The score function, shape (..., d_dim).
        """
        # Ensure alpha_bar_t is broadcastable with x
        if alpha_bar_t.dim() == 0:
            alpha_bar_t = alpha_bar_t.unsqueeze(0) # Make it 1D if scalar
        
        # Sigma_t_diag_elements is (Sigma_t)_i,i
        # Expand sigma_0 to be broadcastable if x has batch dimensions
        sigma_0_expanded = self.sigma_0.unsqueeze(0) if x.dim() > 1 and self.sigma_0.dim() == 1 else self.sigma_0
        
        # Calculate diagonal elements of Sigma_t
        # Sigma_t_diag_elements has shape (..., d_dim)
        sigma_t_diag_elements = alpha_bar_t.unsqueeze(-1) * sigma_0_expanded + (1 - alpha_bar_t.unsqueeze(-1))
        
        # Inverse of diagonal matrix is 1/diagonal elements
        # Ensure no division by zero; these should always be > 0 given 0 < alpha_bar_t < 1.
        # The smallest sigma_t_diag_elements can be is when sigma_0_i = 0 and alpha_bar_t is close to 1.
        # In that case, it is 1 - alpha_bar_t, which tends to 0.
        # However, the forward process (Eq 3) uses 1-alpha_t, implying alpha_t is strictly < 1.
        # And overline_alpha_t approaches 0 as t increases, meaning 1-overline_alpha_t approaches 1.
        # So sigma_t_diag_elements will always be positive if 0 < alpha_bar_t < 1.
        
        inv_sigma_t_diag_elements = 1.0 / sigma_t_diag_elements
        
        # Score function: -Sigma_t^-1 * x
        score = -inv_sigma_t_diag_elements * x
        
        return score

