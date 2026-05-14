"""
Adversarial voting simulation for estimating the number of votes needed to
manipulate the Chatbot Arena leaderboard.

This module implements the simulation described in Section 3 of the paper.
The simulation estimates the number of adversarial votes and interactions needed
to achieve the objectives:
  - Up(M, x): manipulate model M to rise x positions in the leaderboard
  - Down(M, x): manipulate model M to fall x positions in the leaderboard

Key assumptions from the paper (Section 3.1):
- Detection accuracy of 95%, with symmetric false positive and false negative rates of 5%
- Attacker remains passive when they fail to detect the target model
- Rankings are recalculated after every 1,000 interactions
- Historical voting data from Chatbot Arena (1,670,250 votes from 477,322 unique users)
"""

import numpy as np
import logging
from typing import Optional
from dataclasses import dataclass, field

from bradley_terry import compute_bradley_terry_coefficients, get_rankings

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for the adversarial voting simulation."""

    # Detection accuracy (default: 95% as in paper)
    detection_accuracy: float = 0.95

    # False positive rate (symmetric with false negative: 5%)
    false_positive_rate: float = 0.05

    # False negative rate (symmetric: 5%)
    false_negative_rate: float = 0.05

    # Number of interactions between leaderboard recalculations
    recalc_interval: int = 1000

    # Maximum number of interactions to simulate
    max_interactions: int = 500000

    # Random seed for reproducibility
    random_seed: int = 42

    # Strategy when target model not detected
    # Options: "do_nothing", "random_upvote", "vote_tie", "vote_tie_both_bad"
    non_detection_strategy: str = "do_nothing"

    # Attack direction: "up" (promote) or "down" (demote)
    attack_direction: str = "up"


@dataclass
class LeaderboardState:
    """State of the leaderboard at a given point in time."""

    model_names: list
    wins_matrix: np.ndarray  # wins[i, j] = times model i beat model j
    ratings: np.ndarray
    rankings: list  # List of (rank, model_name, rating)

    def get_rank(self, model_name: str) -> int:
        """Get the current rank of a model (1 = best)."""
        for rank, name, _ in self.rankings:
            if name == model_name:
                return rank
        raise ValueError(f"Model {model_name} not found in leaderboard")

    def get_model_index(self, model_name: str) -> int:
        """Get the index of a model in the model_names list."""
        return self.model_names.index(model_name)


def load_historical_data(data_path: str) -> tuple:
    """
    Load historical voting data from Chatbot Arena.

    The dataset includes 1,670,250 votes from 477,322 unique users,
    with 1,093,875 votes resulting in wins and 576,375 in ties.
    These votes cover 6,895 unique combinations of side-by-side model comparisons.

    Args:
        data_path: Path to the voting data file (CSV or JSON)

    Returns:
        Tuple of (model_names, wins_matrix, vote_counts)
    """
    import pandas as pd

    df = pd.read_csv(data_path)

    # Extract unique models
    model_names = sorted(
        list(set(df["model_a"].unique()) | set(df["model_b"].unique()))
    )
    n_models = len(model_names)
    model_idx = {name: i for i, name in enumerate(model_names)}

    # Build wins matrix
    wins_matrix = np.zeros((n_models, n_models))
    vote_counts = np.zeros(n_models)

    for _, row in df.iterrows():
        idx_a = model_idx[row["model_a"]]
        idx_b = model_idx[row["model_b"]]
        winner = row.get("winner", None)

        if winner == "model_a":
            wins_matrix[idx_a, idx_b] += 1
        elif winner == "model_b":
            wins_matrix[idx_b, idx_a] += 1
        elif winner in ["tie", "tie (bothbad)"]:
            # Ties count as 0.5 wins for each model
            wins_matrix[idx_a, idx_b] += 0.5
            wins_matrix[idx_b, idx_a] += 0.5

        vote_counts[idx_a] += 1
        vote_counts[idx_b] += 1

    return model_names, wins_matrix, vote_counts


def create_synthetic_leaderboard(
    n_models: int = 130,
    n_votes_range: tuple = (1000, 100000),
    random_seed: int = 42,
) -> LeaderboardState:
    """
    Create a synthetic leaderboard for testing when real data is not available.

    Creates a leaderboard with n_models models with varying vote counts,
    approximating the structure of the Chatbot Arena leaderboard.

    Args:
        n_models: Number of models in the leaderboard
        n_votes_range: Range of vote counts per model
        random_seed: Random seed

    Returns:
        LeaderboardState with synthetic data
    """
    rng = np.random.RandomState(random_seed)

    model_names = [f"model_{i:03d}" for i in range(n_models)]

    # Generate true ratings (log-normal distribution)
    true_ratings = np.exp(rng.randn(n_models) * 0.5)
    true_ratings = true_ratings / true_ratings.sum() * n_models

    # Generate vote counts (log-normal, higher-ranked models get more votes)
    rank_order = np.argsort(true_ratings)[::-1]
    vote_counts = np.zeros(n_models)
    for rank, idx in enumerate(rank_order):
        # Higher-ranked models get more votes
        base_votes = n_votes_range[1] * np.exp(-rank / (n_models / 3))
        vote_counts[idx] = max(n_votes_range[0], int(base_votes))

    # Generate wins matrix based on true ratings
    wins_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(i + 1, n_models):
            n_comparisons = int(
                (vote_counts[i] + vote_counts[j]) / (2 * n_models)
            )
            if n_comparisons > 0:
                p_i_wins = true_ratings[i] / (true_ratings[i] + true_ratings[j])
                wins_ij = rng.binomial(n_comparisons, p_i_wins)
                wins_matrix[i, j] = wins_ij
                wins_matrix[j, i] = n_comparisons - wins_ij

    # Compute Bradley-Terry ratings
    ratings = compute_bradley_terry_coefficients(wins_matrix)
    rankings = get_rankings(ratings, model_names)

    return LeaderboardState(
        model_names=model_names,
        wins_matrix=wins_matrix,
        ratings=ratings,
        rankings=rankings,
    )


class AdversarialSimulator:
    """
    Simulates adversarial voting attacks on the Chatbot Arena leaderboard.

    Implements the simulation described in Section 3.1 of the paper.
    """

    def __init__(
        self,
        leaderboard: LeaderboardState,
        config: Optional[SimulationConfig] = None,
    ):
        """
        Initialize the simulator.

        Args:
            leaderboard: Initial leaderboard state
            config: Simulation configuration
        """
        self.initial_leaderboard = leaderboard
        self.config = config or SimulationConfig()
        self.rng = np.random.RandomState(self.config.random_seed)

    def _sample_model_pair(self, wins_matrix: np.ndarray) -> tuple:
        """
        Sample a random pair of models for comparison.

        Models are sampled uniformly at random (as in Chatbot Arena).

        Args:
            wins_matrix: Current wins matrix

        Returns:
            Tuple of (model_a_idx, model_b_idx)
        """
        n_models = wins_matrix.shape[0]
        idx_a = self.rng.randint(0, n_models)
        idx_b = self.rng.randint(0, n_models - 1)
        if idx_b >= idx_a:
            idx_b += 1
        return idx_a, idx_b

    def _simulate_benign_vote(
        self,
        idx_a: int,
        idx_b: int,
        ratings: np.ndarray,
    ) -> Optional[tuple]:
        """
        Simulate a benign user vote based on Bradley-Terry probabilities.

        Args:
            idx_a: Index of model A
            idx_b: Index of model B
            ratings: Current Bradley-Terry ratings

        Returns:
            Tuple of (winner_idx, loser_idx) or None for tie
        """
        p_a_wins = ratings[idx_a] / (ratings[idx_a] + ratings[idx_b])
        if self.rng.random() < p_a_wins:
            return idx_a, idx_b
        else:
            return idx_b, idx_a

    def _simulate_adversarial_vote(
        self,
        idx_a: int,
        idx_b: int,
        target_idx: int,
        ratings: np.ndarray,
    ) -> Optional[tuple]:
        """
        Simulate an adversarial vote targeting a specific model.

        The adversary:
        1. Uses the detector to identify if the target model is in the pair
        2. If detected, votes for (or against) the target model
        3. If not detected, uses the non-detection strategy

        Args:
            idx_a: Index of model A
            idx_b: Index of model B
            target_idx: Index of the target model
            ratings: Current Bradley-Terry ratings

        Returns:
            Tuple of (winner_idx, loser_idx) or None for tie/abstain
        """
        # Check if target model is in the pair
        target_in_pair = (idx_a == target_idx) or (idx_b == target_idx)

        if target_in_pair:
            # Simulate detection with accuracy
            detected = self.rng.random() < self.config.detection_accuracy
        else:
            # False positive: incorrectly think target is present
            detected = self.rng.random() < self.config.false_positive_rate

        if detected and target_in_pair:
            # Correctly detected target model - cast adversarial vote
            if self.config.attack_direction == "up":
                # Vote FOR the target model
                if idx_a == target_idx:
                    return idx_a, idx_b
                else:
                    return idx_b, idx_a
            else:
                # Vote AGAINST the target model (demote)
                if idx_a == target_idx:
                    return idx_b, idx_a
                else:
                    return idx_a, idx_b

        elif detected and not target_in_pair:
            # False positive - incorrectly vote for a non-target model
            # This is a wasted vote; treat as random
            return self._simulate_benign_vote(idx_a, idx_b, ratings)

        else:
            # Target not detected - use non-detection strategy
            return self._handle_non_detection(idx_a, idx_b, ratings)

    def _handle_non_detection(
        self,
        idx_a: int,
        idx_b: int,
        ratings: np.ndarray,
    ) -> Optional[tuple]:
        """
        Handle the case where the target model is not detected.

        Args:
            idx_a: Index of model A
            idx_b: Index of model B
            ratings: Current Bradley-Terry ratings

        Returns:
            Vote result based on non-detection strategy
        """
        strategy = self.config.non_detection_strategy

        if strategy == "do_nothing":
            return None  # Abstain
        elif strategy == "random_upvote":
            # Randomly vote for one of the two models
            if self.rng.random() < 0.5:
                return idx_a, idx_b
            else:
                return idx_b, idx_a
        elif strategy == "vote_tie":
            return None  # Tie (no winner)
        elif strategy == "vote_tie_both_bad":
            return None  # Both bad (no winner)
        else:
            return None

    def simulate_attack(
        self,
        target_model: str,
        target_rank: int,
    ) -> dict:
        """
        Simulate an adversarial attack to move a model to a target rank.

        As described in Section 3.1, we iteratively simulate attacker interactions
        and adversarial votes, recalculating rankings after every 1,000 interactions.

        Args:
            target_model: Name of the model to manipulate
            target_rank: Target rank to achieve

        Returns:
            Dictionary with simulation results including votes and interactions needed
        """
        # Copy initial state
        wins_matrix = self.initial_leaderboard.wins_matrix.copy()
        model_names = self.initial_leaderboard.model_names
        ratings = self.initial_leaderboard.ratings.copy()

        target_idx = self.initial_leaderboard.get_model_index(target_model)
        initial_rank = self.initial_leaderboard.get_rank(target_model)

        logger.info(
            f"Starting attack: target={target_model}, "
            f"initial_rank={initial_rank}, target_rank={target_rank}"
        )

        total_interactions = 0
        total_votes = 0
        adversarial_votes = 0
        rank_history = [initial_rank]

        # Check if already at target rank
        current_rank = initial_rank
        if current_rank == target_rank:
            return {
                "target_model": target_model,
                "initial_rank": initial_rank,
                "target_rank": target_rank,
                "achieved": True,
                "total_interactions": 0,
                "total_votes": 0,
                "adversarial_votes": 0,
                "rank_history": rank_history,
            }

        # Determine attack direction
        if target_rank < initial_rank:
            self.config.attack_direction = "up"
        else:
            self.config.attack_direction = "down"

        while total_interactions < self.config.max_interactions:
            # Simulate a batch of interactions
            batch_votes = 0
            batch_adversarial = 0

            for _ in range(self.config.recalc_interval):
                # Sample a model pair
                idx_a, idx_b = self._sample_model_pair(wins_matrix)

                # Simulate adversarial vote
                vote_result = self._simulate_adversarial_vote(
                    idx_a, idx_b, target_idx, ratings
                )

                total_interactions += 1

                if vote_result is not None:
                    winner_idx, loser_idx = vote_result
                    wins_matrix[winner_idx, loser_idx] += 1
                    batch_votes += 1
                    total_votes += 1

                    # Check if this was an adversarial vote
                    if (
                        self.config.attack_direction == "up"
                        and winner_idx == target_idx
                    ) or (
                        self.config.attack_direction == "down"
                        and loser_idx == target_idx
                    ):
                        batch_adversarial += 1
                        adversarial_votes += 1

            # Recalculate rankings
            ratings = compute_bradley_terry_coefficients(wins_matrix)
            rankings = get_rankings(ratings, model_names)
            current_rank = next(
                rank for rank, name, _ in rankings if name == target_model
            )
            rank_history.append(current_rank)

            logger.debug(
                f"Interactions: {total_interactions}, "
                f"Votes: {total_votes}, "
                f"Adversarial votes: {adversarial_votes}, "
                f"Current rank: {current_rank}"
            )

            # Check if target rank achieved
            if (
                self.config.attack_direction == "up" and current_rank <= target_rank
            ) or (
                self.config.attack_direction == "down" and current_rank >= target_rank
            ):
                logger.info(
                    f"Target rank {target_rank} achieved after "
                    f"{total_interactions} interactions and "
                    f"{adversarial_votes} adversarial votes"
                )
                return {
                    "target_model": target_model,
                    "initial_rank": initial_rank,
                    "target_rank": target_rank,
                    "achieved": True,
                    "total_interactions": total_interactions,
                    "total_votes": total_votes,
                    "adversarial_votes": adversarial_votes,
                    "rank_history": rank_history,
                    "final_rank": current_rank,
                }

        logger.warning(
            f"Max interactions ({self.config.max_interactions}) reached "
            f"without achieving target rank {target_rank}"
        )
        return {
            "target_model": target_model,
            "initial_rank": initial_rank,
            "target_rank": target_rank,
            "achieved": False,
            "total_interactions": total_interactions,
            "total_votes": total_votes,
            "adversarial_votes": adversarial_votes,
            "rank_history": rank_history,
            "final_rank": current_rank,
        }

    def run_full_experiment(
        self,
        target_models: list,
        target_ranks: list,
    ) -> dict:
        """
        Run the full adversarial voting experiment for multiple models and target ranks.

        Replicates Tables 4 and 5 from the paper.

        Args:
            target_models: List of target model names
            target_ranks: List of target ranks to achieve

        Returns:
            Dictionary with results for all model-rank combinations
        """
        results = {}

        for target_model in target_models:
            results[target_model] = {}
            initial_rank = self.initial_leaderboard.get_rank(target_model)

            for target_rank in target_ranks:
                if target_rank == initial_rank:
                    results[target_model][target_rank] = {
                        "adversarial_votes": "N/A",
                        "total_interactions": "N/A",
                    }
                    continue

                logger.info(
                    f"Simulating: {target_model} -> rank {target_rank}"
                )

                # Reset random state for reproducibility
                self.rng = np.random.RandomState(self.config.random_seed)

                result = self.simulate_attack(target_model, target_rank)
                results[target_model][target_rank] = {
                    "adversarial_votes": result["adversarial_votes"]
                    if result["achieved"]
                    else ">max",
                    "total_interactions": result["total_interactions"]
                    if result["achieved"]
                    else ">max",
                    "achieved": result["achieved"],
                }

        return results
