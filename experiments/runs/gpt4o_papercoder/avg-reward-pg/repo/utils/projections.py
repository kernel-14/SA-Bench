"""
utils/projections.py

This module contains utility functions for performing projection operations.
Specifically, it projects policies onto the simplex space, ensuring probabilistic constraints
are met (non-negative and summing to 1). This is a critical step in the projected policy gradient algorithm.
"""

import numpy as np
from typing import Callable

# Default projection threshold from the configuration
from utils.constants import DEFAULT_PROJECTION_THRESHOLD


class Projections:
    """
    Utility class that provides methods for projecting vectors and matrices onto probability simplices.
    """

    @staticmethod
    def _project_single(policy_row: np.ndarray, projection_threshold: float = DEFAULT_PROJECTION_THRESHOLD) -> np.ndarray:
        """
        Projects a single policy row onto the simplex.

        Args:
            policy_row (np.ndarray): A 1D numpy array representing probabilities for a single state.
            projection_threshold (float): Minimum value threshold to enforce numerical stability.

        Returns:
            np.ndarray: A 1D numpy array, projected onto the simplex.
        """
        # Sort the probabilities in descending order
        sorted_policy = np.sort(policy_row)[::-1]

        # Compute the cumulative sum minus 1
        cumulative_sum = np.cumsum(sorted_policy) - 1

        # Find the largest index k where sorted_policy[k] > (cumulative_sum[k] / (k + 1))
        k_candidates = np.arange(len(sorted_policy))
        threshold_violations = sorted_policy - cumulative_sum / (k_candidates + 1)
        k = np.where(threshold_violations > 0)[0][-1]  # Get the last valid index

        # Compute the threshold tau
        tau = cumulative_sum[k] / (k + 1)

        # Perform the projection
        projected_policy = np.maximum(policy_row - tau, 0)

        # Ensure numerical stability
        projected_policy[np.abs(projected_policy) < projection_threshold] = 0

        return projected_policy

    @staticmethod
    def project_to_simplex(policy: np.ndarray, projection_threshold: float = DEFAULT_PROJECTION_THRESHOLD) -> np.ndarray:
        """
        Projects a policy (either 1D or 2D) onto the simplex.

        Args:
            policy (np.ndarray): A 1D or 2D numpy array representing action probabilities.
                - 1D: policy distribution for a single state (shape = (actions,))
                - 2D: tabular policy across states (shape = (states, actions)).
            projection_threshold (float): Minimum value threshold to enforce numerical stability.

        Returns:
            np.ndarray: Projected policy onto the simplex (same shape as input).
        """
        # Validate policy dimensions
        if policy.ndim == 1:
            # Single state (1D vector)
            return Projections._project_single(policy, projection_threshold)
        elif policy.ndim == 2:
            # Tabular policy (2D matrix)
            projected_policy = np.zeros_like(policy)
            for i in range(policy.shape[0]):  # Process each state independently
                projected_policy[i] = Projections._project_single(
                    policy[i], projection_threshold
                )
            return projected_policy
        else:
            raise ValueError(
                f"Invalid policy dimensions: {policy.shape}. Expecting a 1D or 2D numpy array."
            )
