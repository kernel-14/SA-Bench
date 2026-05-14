"""
simulation.py – Simulates the Elo‑based Chatbot Arena leaderboard and adversarial vote
manipulation as described in Section 3 of the paper.

Contains three classes:
  - Leaderboard   : maintains model ratings and updates them via Elo / Bradley‑Terry.
  - Attacker      : models an adversarial user with a de‑anonymisation detector.
  - SimulationRunner : orchestrates genuine vote generation and attack scenarios.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Forward reference to avoid circular import at type‑checking time.
# At runtime the Config instance is created in main.py and passed here.
from config import Config

logger = logging.getLogger(__name__)


# =========================================================================
# Leaderboard
# =========================================================================

class Leaderboard:
    """
    Elo‑based leaderboard that maintains a rating for each model and provides
    ranking updates after every match.

    Attributes:
        ratings  : Dict mapping model name → current Elo rating (float).
        k_factor : Elo K‑factor controlling rating adjustment magnitude.
    """

    def __init__(
        self,
        initial_ratings: Dict[str, float],
        k_factor: float,
    ) -> None:
        """
        Args:
            initial_ratings: Starting Elo ratings for all models.
            k_factor:        K‑factor used in Elo updates.
        """
        if not isinstance(initial_ratings, dict):
            raise TypeError("initial_ratings must be a dictionary")
        if k_factor <= 0:
            raise ValueError("k_factor must be positive")
        self.ratings: Dict[str, float] = initial_ratings.copy()
        self.k_factor: float = float(k_factor)

    # ------------------------------------------------------------------
    def expected_score(self, model_a: str, model_b: str) -> Tuple[float, float]:
        """
        Compute the probability that model_a wins against model_b using the
        logistic Bradley–Terry model with scale s = 1.

        P(A beats B) = 1 / (1 + exp(R_B - R_A))

        Returns:
            Tuple (E_A, E_B) where E_A + E_B = 1.
        """
        r_a = self.ratings[model_a]
        r_b = self.ratings[model_b]
        # Use math.exp for a scalar, stable calculation
        exp_diff = math.exp(r_b - r_a)
        e_a = 1.0 / (1.0 + exp_diff)
        e_b = 1.0 - e_a
        return e_a, e_b

    # ------------------------------------------------------------------
    def update(self, model_a: str, model_b: str, outcome: str) -> None:
        """
        Update the ratings for two models after a match.

        Valid outcomes:
          ``'win_a'``   – model_a wins,
          ``'win_b'``   – model_b wins,
          ``'tie'``     – tie (both good),
          ``'tie_both_bad'`` – tie (both bad), treated identically to ``'tie'``.

        Args:
            model_a: First model identifier.
            model_b: Second model identifier.
            outcome: Match outcome (must be one of the recognised strings).
        """
        e_a, e_b = self.expected_score(model_a, model_b)

        if outcome in ("win_a",):
            s_a, s_b = 1.0, 0.0
        elif outcome in ("win_b",):
            s_a, s_b = 0.0, 1.0
        elif outcome in ("tie", "tie_both_bad"):
            s_a, s_b = 0.5, 0.5
        else:
            raise ValueError(f"Unknown outcome: '{outcome}'")

        # Elo update
        self.ratings[model_a] += self.k_factor * (s_a - e_a)
        self.ratings[model_b] += self.k_factor * (s_b - e_b)

    # ------------------------------------------------------------------
    def get_rank(self) -> List[str]:
        """
        Return model names sorted by descending rating (rank 1 = highest rating).
        """
        # Sort items by rating descending
        sorted_items = sorted(
            self.ratings.items(), key=lambda item: item[1], reverse=True
        )
        return [model for model, _ in sorted_items]

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<Leaderboard: models={len(self.ratings)} | k={self.k_factor}>"


# =========================================================================
# Attacker
# =========================================================================

class Attacker:
    """
    Models an adversarial user who attempts to manipulate the leaderboard by
    selectively voting for or against a target model.

    The decision logic simulates a de‑anonymisation detector with the given
    accuracy and a fallback strategy when the target is not detected.

    Attributes:
        target_model        : Name of the model to manipulate.
        detector_accuracy   : Probability of correctly identifying the target.
        non_target_strategy : Action when no targeted vote is cast (``do_nothing``,
                              ``random_upvote``, ``vote_tie``, ``vote_tie_both_bad``).
        goal                : Either ``'up'`` (boost target) or ``'down'`` (harm target).
    """

    VALID_STRATEGIES = {"do_nothing", "random_upvote", "vote_tie", "vote_tie_both_bad"}
    VALID_GOALS = {"up", "down"}

    def __init__(
        self,
        target_model: str,
        detector_accuracy: float,
        goal: str = "up",
        non_target_strategy: str = "do_nothing",
    ) -> None:
        """
        Args:
            target_model:        The identifier of the model to boost/downgrade.
            detector_accuracy:   Accuracy of the attacker's model detector (0‑1).
            goal:                ``'up'`` or ``'down'``.
            non_target_strategy: Strategy for votes where the target is not acted on.
        """
        if not isinstance(target_model, str):
            raise TypeError("target_model must be a string")
        if not (0.0 <= detector_accuracy <= 1.0):
            raise ValueError("detector_accuracy must be between 0 and 1")
        if goal not in self.VALID_GOALS:
            raise ValueError(f"goal must be one of {self.VALID_GOALS}")
        if non_target_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"non_target_strategy must be one of {self.VALID_STRATEGIES}"
            )

        self.target_model: str = target_model
        self.detector_accuracy: float = float(detector_accuracy)
        self.goal: str = goal
        self.non_target_strategy: str = non_target_strategy

    # ------------------------------------------------------------------
    def decide_vote(self, model_a: str, model_b: str) -> Optional[str]:
        """
        Determine the outcome to submit for the current pair of anonymised models.

        The method first checks whether the target model is among the pair.  If it is,
        the simulated detector fires with probability *detector_accuracy*; if it does,
        a vote is returned to achieve the attacker's goal.  When the detector fails or
        the target is absent, the fallback ``non_target_strategy`` is applied.

        Args:
            model_a: First model in the pair.
            model_b: Second model in the pair.

        Returns:
            An outcome string accepted by :meth:`Leaderboard.update` (``'win_a'``,
            ``'win_b'``, ``'tie'``, ``'tie_both_bad'``) or ``None`` to abstain.
        """
        # 1. Target presence
        target_present = (model_a == self.target_model or model_b == self.target_model)

        # 2. Detection simulation
        detected = False
        if target_present:
            if random.random() < self.detector_accuracy:
                detected = True

        # 3. Vote when detected
        if detected:
            # Identify which one is the target
            if self.goal == "up":
                # Vote FOR the target
                return "win_a" if self.target_model == model_a else "win_b"
            else:  # goal == "down"
                # Vote AGAINST the target → make the opponent win
                return "win_b" if self.target_model == model_a else "win_a"

        # 4. Fallback strategy (target absent or detection failure)
        return self._apply_fallback()

    # ------------------------------------------------------------------
    def _apply_fallback(self) -> Optional[str]:
        """Implement the non‑target strategy."""
        if self.non_target_strategy == "do_nothing":
            return None
        elif self.non_target_strategy == "random_upvote":
            # 50‑50 choice between the two models
            if random.random() < 0.5:
                return "win_a"
            else:
                return "win_b"
        elif self.non_target_strategy == "vote_tie":
            return "tie"
        elif self.non_target_strategy == "vote_tie_both_bad":
            return "tie_both_bad"
        else:
            # Should never reach because constructor validates.
            raise ValueError(f"Unknown fallback strategy: {self.non_target_strategy}")

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<Attacker: target={self.target_model} goal={self.goal} "
            f"acc={self.detector_accuracy:.2f} fallback={self.non_target_strategy}>"
        )


# =========================================================================
# SimulationRunner
# =========================================================================

class SimulationRunner:
    """
    Orchestrates the full simulation: generates genuine baseline votes and then
    runs adversarial attacks to measure the number of interactions and votes
    required to shift a target model's rank by a specified amount.

    Attributes:
        config      : Global configuration (parsed from ``config.yaml``).
        leaderboard : The :class:`Leaderboard` instance.
        attacker    : The :class:`Attacker` instance used in the attack phase.
        model_names : List of all model identifiers (used for uniform sampling).
        tie_prob    : Probability of a genuine vote resulting in a tie.
    """

    def __init__(
        self,
        config: Config,
        leaderboard: Leaderboard,
        attacker: Attacker,
    ) -> None:
        """
        Args:
            config:      Application configuration (must contain simulation section).
            leaderboard: Pre‑built leaderboard with initial ratings.
            attacker:    Attacker with desired parameters (can be replaced later).
        """
        if not isinstance(config, Config):
            raise TypeError("config must be an instance of Config")
        if not isinstance(leaderboard, Leaderboard):
            raise TypeError("leaderboard must be an instance of Leaderboard")
        if not isinstance(attacker, Attacker):
            raise TypeError("attacker must be an instance of Attacker")

        self.config: Config = config
        self.leaderboard: Leaderboard = leaderboard
        self.attacker: Attacker = attacker
        self.model_names: List[str] = list(leaderboard.ratings.keys())
        self.tie_prob: float = config.simulation_tie_prob

        # For reproducibility, ensure the random module uses the global seed.
        # The main script sets it; we rely on that.
        logger.info(
            "SimulationRunner initialised with %d models. Tie probability = %.3f",
            len(self.model_names), self.tie_prob,
        )

    # ------------------------------------------------------------------
    def run_genuine(self, n_votes: int) -> None:
        """
        Simulate *n_votes* genuine user interactions on the leaderboard.

        Each interaction randomly selects two distinct models, computes the
        win probability via the Bradley‑Terry model, and produces an outcome
        (win_a, win_b, tie) according to the configured tie probability.
        The leaderboard is updated after each vote.

        Args:
            n_votes: Number of genuine interactions to simulate.
        """
        logger.info("Starting genuine vote simulation: %d votes", n_votes)
        for _ in range(n_votes):
            # 1. Uniform random pair of distinct models
            a, b = random.sample(self.model_names, 2)

            # 2. Expected scores
            e_a, e_b = self.leaderboard.expected_score(a, b)

            # 3. Determine outcome
            if random.random() < self.tie_prob:
                outcome = "tie"
            else:
                # Random outcome according to expected win probability
                r = random.random()
                outcome = "win_a" if r < e_a else "win_b"

            # 4. Update ratings
            self.leaderboard.update(a, b, outcome)

        logger.info("Genuine simulation complete.")

    # ------------------------------------------------------------------
    def run_attack(
        self,
        movement: int,
        direction: str = "up",
        max_interactions: int = 1_000_000,
    ) -> Dict[str, int]:
        """
        Run the adversarial voting loop until the target model has moved
        *movement* positions in the given *direction*, or until the
        interaction budget is exhausted.

        Args:
            movement:         Number of rank positions to shift (≥ 1).
            direction:        ``'up'`` (boost) or ``'down'`` (bury).
            max_interactions: Maximum number of interactions to simulate
                              (prevents infinite loops).

        Returns:
            Dictionary with keys ``'adversarial_votes'`` and
            ``'total_interactions'``.

        Raises:
            ValueError: If *direction* is invalid or *movement* < 1.
        """
        if direction not in ("up", "down"):
            raise ValueError("direction must be 'up' or 'down'")
        if movement < 1:
            raise ValueError("movement must be ≥ 1")

        # Update attacker's goal to match the desired direction
        self.attacker.goal = direction
        target = self.attacker.target_model

        # Compute target rank
        rank_list = self.leaderboard.get_rank()
        if target not in rank_list:
            raise ValueError(f"Target model '{target}' not in leaderboard")
        current_rank = rank_list.index(target) + 1  # 1‑indexed

        if direction == "up":
            target_rank = max(1, current_rank - movement)
        else:
            target_rank = min(len(rank_list), current_rank + movement)

        logger.info(
            "Attack start: target='%s', current_rank=%d, target_rank=%d, direction=%s",
            target, current_rank, target_rank, direction,
        )

        # Counters
        interactions = 0
        adversarial_votes = 0

        while interactions < max_interactions:
            # 1. Random pair
            a, b = random.sample(self.model_names, 2)

            # 2. Attacker decision
            outcome = self.attacker.decide_vote(a, b)
            interactions += 1

            if outcome is not None:
                adversarial_votes += 1
                # Normalise tie outcomes (both 'tie' and 'tie_both_bad' are ties)
                if outcome in ("tie_both_bad", "tie"):
                    outcome = "tie"
                self.leaderboard.update(a, b, outcome)

            # 3. Check termination every interaction (or every N for speed,
            #    but checking every time is fine for typical scales).
            if interactions % 1000 == 0:
                rank_list = self.leaderboard.get_rank()
                cur = rank_list.index(target) + 1
                logger.debug(
                    "Interaction %d: rank=%d, votes=%d",
                    interactions, cur, adversarial_votes,
                )
                # Early stop if reached goal?
                # Only break when interaction budget exhausted; we break below.

            # 4. Check if target rank achieved
            current_rank = self.leaderboard.get_rank().index(target) + 1
            if direction == "up" and current_rank <= target_rank:
                break
            elif direction == "down" and current_rank >= target_rank:
                break

        logger.info(
            "Attack finished. Interactions=%d, adversarial_votes=%d, final_rank=%d",
            interactions, adversarial_votes, current_rank,
        )
        return {
            "adversarial_votes": adversarial_votes,
            "total_interactions": interactions,
        }

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<SimulationRunner: models={len(self.model_names)}, "
            f"attacker={self.attacker}>"
        )

