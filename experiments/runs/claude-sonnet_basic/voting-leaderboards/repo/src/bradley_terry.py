"""
Bradley-Terry model implementation for Chatbot Arena leaderboard.

This module implements the Bradley-Terry model used by Chatbot Arena to rank models.
The Bradley-Terry model assigns a coefficient (rating) to each model, and the probability
that model i is preferred over model j is given by:

    Pr(i preferred over j) = 1 / (1 + exp(-(Q_i - Q_j) / s))

where Q_i and Q_j are the ratings and s is a scaling factor.

Reference: Hunter (2004), "MM algorithms for generalized Bradley-Terry models"
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_bradley_terry_coefficients(
    wins: np.ndarray,
    n_iterations: int = 1000,
    tol: float = 1e-8,
    initial_ratings: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute Bradley-Terry coefficients using the MM (Minorization-Maximization) algorithm.

    The MM algorithm iteratively updates ratings as:
        Q_i^(t+1) = W_i / sum_j (n_ij / (Q_i^(t) + Q_j^(t)))

    where W_i is the total number of wins for model i, and n_ij is the total number
    of comparisons between models i and j.

    Args:
        wins: Matrix of shape (n_models, n_models) where wins[i, j] is the number
              of times model i was preferred over model j
        n_iterations: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        Array of Bradley-Terry coefficients (ratings) for each model
    """
    n_models = wins.shape[0]

    # Initialize ratings
    if initial_ratings is not None:
        ratings = initial_ratings.copy().astype(float)
    else:
        ratings = np.ones(n_models, dtype=float)

    # Total comparisons matrix (symmetric)
    comparisons = wins + wins.T

    # Total wins per model
    total_wins = wins.sum(axis=1).astype(float)

    for iteration in range(n_iterations):
        old_ratings = ratings.copy()

        # MM update
        for i in range(n_models):
            denominator = 0.0
            for j in range(n_models):
                if i != j and comparisons[i, j] > 0:
                    denominator += comparisons[i, j] / (ratings[i] + ratings[j])

            if denominator > 0:
                ratings[i] = total_wins[i] / denominator
            else:
                ratings[i] = 1.0  # No comparisons, keep default

        # Normalize to prevent numerical issues
        ratings = ratings / ratings.sum() * n_models

        # Check convergence
        if np.max(np.abs(ratings - old_ratings)) < tol:
            logger.debug(f"Bradley-Terry converged after {iteration + 1} iterations")
            break

    return ratings


def get_rankings(ratings: np.ndarray, model_names: list) -> list:
    """
    Get model rankings from Bradley-Terry ratings.

    Args:
        ratings: Array of Bradley-Terry coefficients
        model_names: List of model names

    Returns:
        List of (rank, model_name, rating) tuples sorted by rank (1 = best)
    """
    sorted_indices = np.argsort(ratings)[::-1]
    rankings = [
        (rank + 1, model_names[idx], ratings[idx])
        for rank, idx in enumerate(sorted_indices)
    ]
    return rankings


def win_probability(rating_i: float, rating_j: float, scale: float = 1.0) -> float:
    """
    Compute the probability that model i is preferred over model j.

    Uses the logistic function as described in Section 4.2.3:
        Pr(i preferred over j) = 1 / (1 + exp(-(Q_i - Q_j) / s))

    Args:
        rating_i: Bradley-Terry rating of model i
        rating_j: Bradley-Terry rating of model j
        scale: Scaling factor s (default: 1.0)

    Returns:
        Probability that model i is preferred over model j
    """
    return 1.0 / (1.0 + np.exp(-(rating_i - rating_j) / scale))


def compute_win_probabilities(ratings: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Compute pairwise win probabilities for all model pairs.

    Args:
        ratings: Array of Bradley-Terry ratings
        scale: Scaling factor

    Returns:
        Matrix of shape (n_models, n_models) where entry [i, j] is
        Pr(model i preferred over model j)
    """
    n = len(ratings)
    probs = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                probs[i, j] = win_probability(ratings[i], ratings[j], scale)
    return probs


def compute_benign_vote_distribution(
    ratings: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Compute the expected vote distribution for a benign user.

    For each model i, the probability that a benign user votes for it is the
    product of probabilities that it is preferred over each other model:
        Pr_B(i) = prod_j Pr_B(i preferred over j | true BT ratings)

    As described in Section 4.2.3.

    Args:
        ratings: Array of Bradley-Terry ratings
        scale: Scaling factor

    Returns:
        Array of vote probabilities for each model (normalized)
    """
    n = len(ratings)
    vote_probs = np.ones(n)

    for i in range(n):
        for j in range(n):
            if i != j:
                vote_probs[i] *= win_probability(ratings[i], ratings[j], scale)

    # Normalize
    total = vote_probs.sum()
    if total > 0:
        vote_probs = vote_probs / total

    return vote_probs


def ratings_to_elo(ratings: np.ndarray, base_elo: float = 1000.0, scale: float = 400.0) -> np.ndarray:
    """
    Convert Bradley-Terry ratings to Elo scores for display.

    Args:
        ratings: Array of Bradley-Terry ratings
        base_elo: Base Elo score
        scale: Elo scale factor

    Returns:
        Array of Elo scores
    """
    log_ratings = np.log(ratings)
    # Normalize so mean is base_elo
    log_ratings = log_ratings - np.mean(log_ratings)
    return base_elo + scale * log_ratings / np.log(10)
