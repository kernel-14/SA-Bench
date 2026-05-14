# posterior_calculator.py

import numpy as np
from scipy.stats import dirichlet
from typing import Tuple

class PosteriorCalculator:
    """
    Implements Bayesian posterior calculations for expected loss,
    quantile spacings (Dirichlet-distributed), and highest posterior density bounds.
    """

    def __init__(self) -> None:
        """
        Initialize the PosteriorCalculator utility class.
        """
        pass

    def calculate_posterior_dirichlet(self, losses: np.ndarray, num_samples: int = 1000) -> np.ndarray:
        """
        Generate Dirichlet-distributed random quantile spacings from calibration losses.
        
        Args:
            losses (np.ndarray): Sorted calibration losses (n-dimensional).
            num_samples (int): Number of Dirichlet samples to generate.

        Returns:
            np.ndarray: Array (num_samples, n+1) of quantile spacings.
        """
        # Validate inputs
        if not isinstance(losses, np.ndarray) or losses.ndim != 1:
            raise ValueError("Input 'losses' must be a 1D NumPy array.")
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("Input 'num_samples' must be a positive integer.")

        n = losses.shape[0]
        # Generate Dirichlet-distributed samples with (n+1) components
        spacing_samples = dirichlet.rvs([1] * (n + 1), size=num_samples, random_state=42)

        return spacing_samples

    def compute_expected_loss_dirichlet(self, u: np.ndarray, losses: np.ndarray) -> float:
        """
        Compute the posterior expected loss for given quantile spacings and ordered losses.

        Args:
            u (np.ndarray): Matrix (num_samples, n+1) of Dirichlet quantile spacings.
            losses (np.ndarray): Array (n+1) of calibration losses with the upper bound appended.

        Returns:
            float: The posterior expected loss (E[L|u, losses]).
        """
        # Validate inputs
        if not isinstance(u, np.ndarray) or u.ndim != 2:
            raise ValueError("Input 'u' must be a 2D NumPy array (num_samples, n+1).")
        if not isinstance(losses, np.ndarray) or losses.ndim != 1:
            raise ValueError("Input 'losses' must be a 1D NumPy array (n+1).")

        if u.shape[1] != losses.shape[0]:
            raise ValueError("The second dimension of 'u' must match the size of 'losses' (n+1).")

        # Compute element-wise product of spacings and losses
        weighted_losses = u * losses  # Shape: (num_samples, n+1)

        # Calculate the expected loss by averaging over the samples
        expected_loss = np.mean(np.sum(weighted_losses, axis=1))  # Expectation over L^+

        return expected_loss

    def calculate_hpd_bounds(self, samples: np.ndarray, beta: float = 0.95) -> float:
        """
        Calculate the one-sided highest posterior density (HPD) bound for a given confidence level.

        Args:
            samples (np.ndarray): Array of posterior expected loss realizations from Monte Carlo sampling.
            beta (float): Desired confidence level for the HPD interval (default: 0.95).

        Returns:
            float: The lower bound of the one-sided HPD interval (λ_hpd^β).
        """
        # Validate inputs
        if not isinstance(samples, np.ndarray) or samples.ndim != 1:
            raise ValueError("Input 'samples' must be a 1D NumPy array.")
        if not 0 < beta <= 1:
            raise ValueError("Parameter 'beta' must be a float in (0, 1].")

        # Sort samples in ascending order
        sorted_samples = np.sort(samples)

        # Compute the index corresponding to the (1-beta) quantile
        num_samples = len(sorted_samples)
        k = int(np.floor(beta * num_samples))  # Number of samples to include in HPD

        # Identify the narrowest range for the HPD interval
        min_width = float('inf')
        hpd_lower_bound = None
        for start_idx in range(num_samples - k + 1):
            width = sorted_samples[start_idx + k - 1] - sorted_samples[start_idx]
            if width < min_width:
                min_width = width
                hpd_lower_bound = sorted_samples[start_idx]

        return hpd_lower_bound
