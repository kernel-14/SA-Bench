# decision_rules.py

import numpy as np
from scipy.special import gammaln
from scipy.stats import dirichlet
from typing import Callable


class DecisionRules:
    """
    Implements decision rule methods for uncertainty quantification.
    - Split Conformal Prediction (split_conformal_prediction)
    - Conformal Risk Control (conformal_risk_control)
    - Bayesian HPD Interval (bayesian_hpd_interval)
    """

    def __init__(self) -> None:
        """
        Initialize the DecisionRules utility class.
        """
        pass

    def split_conformal_prediction(self, s: np.ndarray, alpha: float) -> float:
        """
        Split Conformal Prediction methodology.

        Args:
            s (np.ndarray): Array of scores (e.g., nonconformity measures or losses).
            alpha (float): Coverage level (1 - alpha is the target coverage).

        Returns:
            float: The threshold parameter lambda_scp.
        """

        # Validate inputs
        if not 0 < alpha < 1 or not isinstance(alpha, float):
            raise ValueError("alpha must be a float between 0 and 1.")
        if not isinstance(s, np.ndarray) or s.ndim != 1:
            raise ValueError("s must be a 1D numpy array.")

        # Sort the scores in ascending order
        sorted_scores = np.sort(s)

        # Determine the index for the (1 - alpha) quantile
        n = len(s)
        k_star = int(np.ceil((n + 1) * (1 - alpha)))

        # Compute lambda_scp based on the index
        if k_star <= n:
            lambda_scp = sorted_scores[k_star - 1]  # Adjust for zero-based indexing
        else:
            lambda_scp = float('inf')  # No valid threshold

        return lambda_scp

    def conformal_risk_control(
        self, losses: np.ndarray, alpha: float, B: float
    ) -> float:
        """
        Conformal Risk Control methodology.

        Args:
            losses (np.ndarray): Array of empirical losses for the calibration set.
            alpha (float): Target maximum risk.
            B (float): Upper bound for losses.

        Returns:
            float: The threshold parameter lambda_crc.
        """

        # Validate inputs
        if not 0 < alpha < 1 or not isinstance(alpha, float):
            raise ValueError("alpha must be a float between 0 and 1.")
        if not isinstance(losses, np.ndarray) or losses.ndim != 1:
            raise ValueError("losses must be a 1D numpy array.")
        if not isinstance(B, float) or B <= 0:
            raise ValueError("B must be a positive float.")

        n = len(losses)
        empirical_risks = np.cumsum(losses[np.argsort(losses)]) / n

        # Check for the compliance criterion
        for idx, emp_risk in enumerate(empirical_risks, start=1):
            lhs = emp_risk * (n / (n + 1)) + (B / (n + 1))
            if lhs <= alpha:
                return losses[np.argsort(losses)][idx - 1]

        return float('inf')  # Failure case (fallback, should not happen with valid inputs)

    def bayesian_hpd_interval(
        self, losses: np.ndarray, alpha: float, beta: float, num_samples: int = 1000
    ) -> float:
        """
        Compute Bayesian HPD intervals for expected loss.

        Args:
            losses (np.ndarray): Array of ordered loss values.
            alpha (float): Target maximum expected loss.
            beta (float): Desired confidence for posterior bound.
            num_samples (int): Number of Monte Carlo samples for Dirichlet distribution.

        Returns:
            float: The threshold parameter lambda_hpd.
        """

        # Validate inputs
        if not 0 < alpha < 1 or not isinstance(alpha, float):
            raise ValueError("alpha must be a float between 0 and 1.")
        if not 0 < beta < 1 or not isinstance(beta, float):
            raise ValueError("beta must be a float between 0 and 1.")
        if not isinstance(losses, np.ndarray) or losses.ndim != 1:
            raise ValueError("losses must be a 1D numpy array.")
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")

        n = len(losses)
        max_loss = 1.0  # Assuming maximum loss is 1 per the configuration
        ordered_losses = np.sort(losses)

        # Simulate samples from Dirichlet distribution to model quantile spacings
        spacing_samples = dirichlet.rvs([1] * (n + 1), size=num_samples)
        monte_carlo_losses = np.dot(spacing_samples, np.append(ordered_losses, max_loss))

        # Compute the highest HPD interval
        lower_bounds = np.quantile(monte_carlo_losses, beta)
