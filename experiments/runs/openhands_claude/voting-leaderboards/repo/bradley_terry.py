"""
Bradley-Terry model for ranking LLMs from pairwise comparisons.

Implements the MM (Minorization-Maximization) algorithm from Hunter (2004),
which is the ranking method used by Chatbot Arena (Section 3.1).

The Bradley-Terry model assigns a strength parameter Q_i to each model i.
Given two models i and j, the probability that i is preferred over j is:

    Pr(i preferred over j) = 1 / (1 + exp(-(Q_i - Q_j) / s))

where s is a scaling factor (Eq. 5 in the paper).

The MM algorithm iteratively updates the strength parameters until convergence.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from config import BradleyTerryConfig
from data import VoteRecord


# ---------------------------------------------------------------------------
# Bradley-Terry MM algorithm (Hunter, 2004)
# ---------------------------------------------------------------------------

def fit_bradley_terry(
    models: List[str],
    votes: List[VoteRecord],
    config: Optional[BradleyTerryConfig] = None,
    exclude_ties: bool = False,
) -> Dict[str, float]:
    """
    Fit Bradley-Terry strength parameters using the MM algorithm.

    The MM update rule for model i is (Hunter, 2004, Eq. 2):

        Q_i^{new} = W_i / sum_{j != i} (n_{ij} / (Q_i + Q_j))

    where W_i is the number of wins for model i and n_{ij} is the total
    number of comparisons between i and j.

    Ties are handled by splitting each tie as half a win for each model
    (standard approach used in Chatbot Arena).

    Args:
        models: List of model names.
        votes: List of VoteRecord objects.
        config: Bradley-Terry configuration.
        exclude_ties: If True, ignore tie votes entirely.

    Returns:
        Dict mapping model name to strength parameter (log-scale rating).
    """
    if config is None:
        config = BradleyTerryConfig()

    n = len(models)
    model_idx = {m: i for i, m in enumerate(models)}

    # Count wins and comparisons
    wins = np.zeros(n, dtype=np.float64)
    comparisons = np.zeros((n, n), dtype=np.float64)

    for vote in votes:
        if vote.model_a not in model_idx or vote.model_b not in model_idx:
            continue

        i = model_idx[vote.model_a]
        j = model_idx[vote.model_b]

        if vote.winner == "model_a":
            wins[i] += 1.0
            comparisons[i, j] += 1.0
            comparisons[j, i] += 1.0
        elif vote.winner == "model_b":
            wins[j] += 1.0
            comparisons[i, j] += 1.0
            comparisons[j, i] += 1.0
        elif vote.winner == "tie" and not exclude_ties:
            # Split tie as 0.5 win for each model
            wins[i] += 0.5
            wins[j] += 0.5
            comparisons[i, j] += 1.0
            comparisons[j, i] += 1.0

    # Initialize strengths
    strengths = np.full(n, config.initial_rating, dtype=np.float64)

    # MM iterations
    for iteration in range(config.max_iter):
        strengths_old = strengths.copy()

        for i in range(n):
            if wins[i] == 0:
                strengths[i] = 1e-10
                continue

            denominator = 0.0
            for j in range(n):
                if i != j and comparisons[i, j] > 0:
                    denominator += comparisons[i, j] / (strengths[i] + strengths[j])

            if denominator > 0:
                strengths[i] = wins[i] / denominator
            else:
                strengths[i] = 1e-10

        # Normalize so that the geometric mean is 1
        log_mean = np.mean(np.log(strengths + 1e-300))
        strengths = strengths / np.exp(log_mean)

        # Check convergence
        max_change = np.max(np.abs(strengths - strengths_old))
        if max_change < config.tol:
            break

    return {models[i]: float(strengths[i]) for i in range(n)}


def strengths_to_elo(
    strengths: Dict[str, float],
    base: float = 10.0,
    scale: float = 400.0,
    anchor_model: Optional[str] = None,
    anchor_elo: float = 1000.0,
) -> Dict[str, float]:
    """
    Convert Bradley-Terry strength parameters to Elo-like ratings.

    Elo_i = anchor_elo + scale * log_base(strength_i / strength_anchor)

    Args:
        strengths: Dict of model -> BT strength.
        base: Logarithm base (10 for standard Elo).
        scale: Elo scale factor (400 for standard Elo).
        anchor_model: Model to use as anchor. If None, uses the model with
                      median strength.
        anchor_elo: Elo rating for the anchor model.

    Returns:
        Dict mapping model name to Elo rating.
    """
    models = list(strengths.keys())
    strength_values = np.array([strengths[m] for m in models])

    if anchor_model is None:
        median_idx = int(np.argsort(strength_values)[len(models) // 2])
        anchor_strength = strength_values[median_idx]
    else:
        anchor_strength = strengths[anchor_model]

    elo_ratings = {}
    for model in models:
        log_ratio = np.log(strengths[model] / anchor_strength) / np.log(base)
        elo_ratings[model] = anchor_elo + scale * log_ratio

    return elo_ratings


def get_rankings(strengths: Dict[str, float]) -> List[Tuple[int, str, float]]:
    """
    Convert strength parameters to a ranked list.

    Returns:
        List of (rank, model_name, strength) sorted by descending strength.
    """
    sorted_models = sorted(strengths.items(), key=lambda x: x[1], reverse=True)
    return [(i + 1, model, strength) for i, (model, strength) in enumerate(sorted_models)]


def get_rank(model: str, strengths: Dict[str, float]) -> int:
    """Get the rank of a specific model (1 = best)."""
    rankings = get_rankings(strengths)
    for rank, name, _ in rankings:
        if name == model:
            return rank
    raise ValueError(f"Model '{model}' not found in strengths dict.")


def bradley_terry_win_probability(
    model_i: str,
    model_j: str,
    strengths: Dict[str, float],
    scale: float = 1.0,
) -> float:
    """
    Compute the probability that model_i is preferred over model_j.

    Uses the logistic formulation from Eq. 5 in the paper:
        Pr(i preferred over j) = 1 / (1 + exp(-(Q_i - Q_j) / s))

    Args:
        model_i: First model.
        model_j: Second model.
        strengths: Bradley-Terry strength parameters.
        scale: Scaling factor s.

    Returns:
        Probability in [0, 1].
    """
    q_i = np.log(strengths[model_i] + 1e-300)
    q_j = np.log(strengths[model_j] + 1e-300)
    return float(1.0 / (1.0 + np.exp(-(q_i - q_j) / scale)))


def compute_vote_distribution(
    strengths: Dict[str, float],
    scale: float = 1.0,
) -> Dict[str, float]:
    """
    Compute the marginal probability that each model wins a random comparison.

    Pr_B(i) = product_{j != i} Pr(i preferred over j | true BT ratings)

    This is used in the malicious user detection (Eq. 6 in the paper).

    Args:
        strengths: Bradley-Terry strength parameters.
        scale: Scaling factor.

    Returns:
        Dict mapping model name to marginal win probability.
    """
    models = list(strengths.keys())
    probs = {}

    for model_i in models:
        log_prob = 0.0
        for model_j in models:
            if model_i != model_j:
                p = bradley_terry_win_probability(model_i, model_j, strengths, scale)
                log_prob += np.log(p + 1e-300)
        probs[model_i] = float(np.exp(log_prob))

    # Normalize
    total = sum(probs.values())
    if total > 0:
        probs = {m: p / total for m, p in probs.items()}

    return probs


def compute_pairwise_win_probability(
    strengths: Dict[str, float],
    scale: float = 1.0,
) -> Dict[Tuple[str, str], float]:
    """
    Compute win probabilities for all model pairs.

    Returns:
        Dict mapping (model_i, model_j) to Pr(i preferred over j).
    """
    models = list(strengths.keys())
    probs = {}
    for i, model_i in enumerate(models):
        for j, model_j in enumerate(models):
            if i != j:
                probs[(model_i, model_j)] = bradley_terry_win_probability(
                    model_i, model_j, strengths, scale
                )
    return probs
