```python
## simulation/attack_simulator.py
"""Adversarial voting simulation for the leaderboard manipulation paper.

This module implements the attack simulator from Section 3 of the paper
"Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards."
It simulates an adversary who casts targeted votes to shift a model's rank on
the Chatbot Arena leaderboard.

The simulation operates on real (or realistic) Bradley-Terry voting data and
models the attacker's behavior:
  1. Sample a model pair from the empirical pair distribution.
  2. Attempt to detect the target model using the de-anonymization classifier.
  3. Cast a directional vote if the target is detected.
  4. Every eval_interval interactions, re-fit the BT model and check rank.

Paper alignment:
  - Section 3.1: "We iteratively simulate attacker interactions and adversarial
    votes with the system. We calculate the Bradley-Terry coefficient and model
    ranking after every 1,000 interactions."
  - Section 3.1: "A detection accuracy of 95%, with symmetric false positive
    and false negative rates of 5%."
  - Tables 4, 5: High-ranked and low-ranked model experiments.
  - Appendix B.2, Tables 8, 9: Ablation studies.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import Config
from data_structures import SimulationResult
from simulation.bradley_terry import BradleyTerryModel
from simulation.voting_data_loader import VotingDataLoader
from utils.logger import get_logger

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Valid attack directions
# ---------------------------------------------------------------------------
_VALID_DIRECTIONS: frozenset = frozenset({"up", "down"})

# ---------------------------------------------------------------------------
# Valid non-target strategies (Appendix B.2, Table 9)
# ---------------------------------------------------------------------------
_VALID_NON_TARGET_STRATEGIES: frozenset = frozenset(
    {"do_nothing", "random_upvote", "vote_tie", "vote_both_bad"}
)

# ---------------------------------------------------------------------------
# Sentinel value for N/A entries in result DataFrames.
# Used when a target rank equals the current rank (no movement needed) or
# when the target rank is in the wrong direction for the experiment type.
# ---------------------------------------------------------------------------
_NA_SENTINEL: float = float("nan")


class AttackSimulator:
    """Simulates adversarial voting attacks on the Chatbot Arena leaderboard.

    Implements the attack simulation pipeline from Section 3 of the paper.
    For each simulation run, the attacker:
      1. Samples model pairs from the empirical pair distribution.
      2. Applies the de-anonymization detector to identify the target model.
      3. Casts directional votes when the target is detected.
      4. Periodically re-fits the Bradley-Terry model to track rank progress.

    The baseline BT model is never mutated — each simulation run works on a
    deep copy augmented with adversarial votes.

    Attributes:
        config: The global Config object from config.py.
        bt_model: The baseline fitted BradleyTerryModel (never mutated).
        voting_loader: VotingDataLoader for pair sampling and win matrix access.
        detection_accuracy: Default detector accuracy (0.95 per config.yaml).
        eval_interval: BT re-fitting frequency in interactions (1000 per config.yaml).
        max_interactions: Maximum interactions per simulation run (500000 per config.yaml).

    Example:
        >>> from config import Config
        >>> from simulation.bradley_terry import BradleyTerryModel
        >>> from simulation.voting_data_loader import VotingDataLoader
        >>> config = Config.from_yaml("config.yaml")
        >>> loader = VotingDataLoader(config.raw["simulation"]["voting_data_path"])
        >>> win_matrix, model_names = loader.get_win_matrix()
        >>> bt = BradleyTerryModel(scale_factor=config.bt_scale_factor)
        >>> bt.fit(win_matrix, model_names)
        >>> simulator = AttackSimulator(config, bt, loader)
        >>> result = simulator.simulate_attack("llama-13b", "up", 128)
        >>> isinstance(result, SimulationResult)
        True
    """

    def __init__(
        self,
        config: Config,
        bt_model: BradleyTerryModel,
        voting_loader: VotingDataLoader,
    ) -> None:
        """Initialize the AttackSimulator.

        Reads simulation parameters from the config object. Stores references
        to the baseline BT model and voting loader. No simulation is run at
        init time.

        Args:
            config: The global Config object. Provides simulation parameters
                via config.raw["simulation"] and config.random_state.
            bt_model: A fitted BradleyTerryModel instance representing the
                baseline leaderboard state. This object is never mutated —
                each simulate_attack call works on a deep copy.
            voting_loader: An initialized VotingDataLoader providing pair
                sampling and win matrix access for the simulation.
        """
        self.config: Config = config
        self.bt_model: BradleyTerryModel = bt_model
        self.voting_loader: VotingDataLoader = voting_loader

        # Read simulation parameters from the raw config dict.
        sim_cfg: Dict = config.raw.get("simulation", {})

        # Default detector accuracy: 0.95 per Section 3.1 and config.yaml.
        self.detection_accuracy: float = float(
            sim_cfg.get("detection_accuracy", 0.95)
        )

        # BT re-fitting frequency: every 1000 interactions per Section 3.1.
        self.eval_interval: int = int(sim_cfg.get("eval_interval", 1000))

        # Maximum interactions per run: 500000 per config.yaml.
        self.max_interactions: int = int(sim_cfg.get("max_interactions", 500000))

        # Pre-fetch the baseline win matrix and model names once.
        # These are used to construct augmented win matrices during simulation.
        logger.info(
            "AttackSimulator: fetching baseline win matrix from VotingDataLoader."
        )
        self._baseline_win_matrix: np.ndarray
        self._baseline_model_names: List[str]
        self._baseline_win_matrix, self._baseline_model_names = (
            voting_loader.get_win_matrix()
        )

        # Build a fast model-name-to-index lookup for O(1) access during
        # the simulation inner loop.
        self._model_to_idx: Dict[str, int] = {
            name: idx
            for idx, name in enumerate(self._baseline_model_names)
        }

        logger.info(
            "AttackSimulator initialized: detection_accuracy=%.2f, "
            "eval_interval=%d, max_interactions=%d, "
            "n_models=%d, n_pairs=%d.",
            self.detection_accuracy,
            self.eval_interval,
            self.max_interactions,
            len(self._baseline_model_names),
            len(voting_loader.get_pair_counts()),
        )

    # -----------------------------------------------------------------------
    # Private helper methods
    # -----------------------------------------------------------------------

    def _detect_target(
        self,
        sampled_model: str,
        target_model: str,
        rng: np.random.Generator,
        detection_accuracy: float,
    ) -> bool:
        """Simulate the de-anonymization detector for one model slot.

        Models the attacker's classifier with symmetric error rates:
          - If sampled_model IS the target: returns True with probability
            detection_accuracy (correct detection), False with probability
            1 - detection_accuracy (false negative).
          - If sampled_model is NOT the target: returns True with probability
            1 - detection_accuracy (false positive), False with probability
            detection_accuracy (correct rejection).

        Paper alignment: Section 3.1 — "A detection accuracy of 95%, with
        symmetric false positive and false negative rates of 5%."

        Args:
            sampled_model: The model name in this slot of the comparison.
            target_model: The model the attacker is trying to identify.
            rng: Seeded numpy random generator for reproducible sampling.
            detection_accuracy: Probability of correct detection/rejection.
                Symmetric: P(detect | is target) = P(reject | is not target).

        Returns:
            True if the attacker's classifier predicts this slot contains the
            target model. False otherwise.

        Example:
            >>> rng = np.random.default_rng(42)
            >>> simulator._detect_target("gpt-4o", "gpt-4o", rng, 0.95)
            True  # With 95% probability
            >>> simulator._detect_target("llama-3", "gpt-4o", rng, 0.95)
            False  # With 95% probability
        """
        if sampled_model == target_model:
            # True positive scenario: return True with probability detection_accuracy.
            return bool(rng.random() < detection_accuracy)
        else:
            # True negative scenario: return True (false positive) with probability
            # 1 - detection_accuracy.
            return bool(rng.random() < (1.0 - detection_accuracy))

    def _cast_vote(
        self,
        model_a: str,
        model_b: str,
        target_model: str,
        direction: str,
        non_target_strategy: str,
        rng: np.random.Generator,
        detection_accuracy: float,
    ) -> Optional[Tuple[str, str]]:
        """Determine the attacker's vote for a given model pair.

        Applies the de-anonymization detector to both slots, then casts a
        directional vote if the target is detected, or applies the non-target
        strategy if the target is not detected.

        Args:
            model_a: Name of the model in slot A.
            model_b: Name of the model in slot B.
            target_model: The model the attacker is targeting.
            direction: 'up' to upvote the target (boost rank) or 'down' to
                downvote the target (suppress rank).
            non_target_strategy: Action when target not detected. One of
                'do_nothing', 'random_upvote', 'vote_tie', 'vote_both_bad'.
            rng: Seeded numpy random generator.
            detection_accuracy: Detector accuracy for this simulation run.

        Returns:
            (winner, loser) tuple if a directional vote is cast.
            None if the attacker abstains (do_nothing, vote_tie, vote_both_bad).

        Example:
            >>> rng = np.random.default_rng(42)
            >>> vote = simulator._cast_vote(
            ...     "llama-13b", "gpt-4o", "llama-13b", "up",
            ...     "do_nothing", rng, 0.95
            ... )
            >>> vote  # ("llama-13b", "gpt-4o") with 95% probability
        """
        # Skip self-comparisons (should not occur in real data, but guard anyway).
        if model_a == model_b:
            logger.debug(
                "_cast_vote: model_a == model_b == '%s'. Skipping.", model_a
            )
            return None

        # --- Detect target model in each slot ---
        detected_a: bool = self._detect_target(
            model_a, target_model, rng, detection_accuracy
        )
        detected_b: bool = self._detect_target(
            model_b, target_model, rng, detection_accuracy
        )

        target_detected: bool = detected_a or detected_b

        if target_detected:
            # --- Determine which slot the attacker believes contains the target ---
            if detected_a and not detected_b:
                # Attacker believes target is in slot A.
                believed_target: str = model_a
                believed_other: str = model_b
            elif detected_b and not detected_a:
                # Attacker believes target is in slot B.
                believed_target = model_b
                believed_other = model_a
            else:
                # Both slots detected (one is a false positive).
                # Randomly pick one as the believed target.
                if rng.random() < 0.5:
                    believed_target = model_a
                    believed_other = model_b
                else:
                    believed_target = model_b
                    believed_other = model_a

            # --- Cast directional vote ---
            if direction == "up":
                # Upvote attack: vote FOR the believed target.
                return (believed_target, believed_other)
            else:
                # Downvote attack: vote AGAINST the believed target.
                return (believed_other, believed_target)

        else:
            # --- Target not detected: apply non-target strategy ---
            if non_target_strategy == "do_nothing":
                # Abstain — no vote cast.
                return None

            elif non_target_strategy == "random_upvote":
                # Randomly pick one of the two models as winner.
                if rng.random() < 0.5:
                    return (model_a, model_b)
                else:
                    return (model_b, model_a)

            elif non_target_strategy == "vote_tie":
                # Tie vote — no directional effect on BT win matrix.
                # Return None; the simulation loop can track ties separately
                # if needed, but they don't affect BT fitting directionally.
                return None

            elif non_target_strategy == "vote_both_bad":
                # Both-bad vote — equivalent to abstain for BT purposes.
                return None

            else:
                # Unknown strategy — treat as do_nothing.
                logger.warning(
                    "_cast_vote: unknown non_target_strategy '%s'. "
                    "Treating as 'do_nothing'.",
                    non_target_strategy,
                )
                return None

    def _build_augmented_win_matrix(
        self,
        adversarial_votes: List[Tuple[str, str]],
    ) -> np.ndarray:
        """Construct an augmented win matrix by adding adversarial votes to baseline.

        Creates a copy of the baseline win matrix and increments entries for
        each adversarial vote. This is used for periodic BT re-fitting during
        simulation without mutating the baseline.

        Args:
            adversarial_votes: List of (winner, loser) model name tuples
                accumulated during the simulation run so far.

        Returns:
            np.ndarray of shape (N, N) with dtype float64 representing the
            augmented win matrix. Models not in the baseline model list are
            silently skipped (logged at debug level).
        """
        # Start with a copy of the baseline win matrix.
        augmented: np.ndarray = self._baseline_win_matrix.copy()

        # Add each adversarial vote to the augmented matrix.
        for winner, loser in adversarial_votes:
            winner_idx: Optional[int] = self._model_to_idx.get(winner)
            loser_idx: Optional[int] = self._model_to_idx.get(loser)

            if winner_idx is None:
                logger.debug(
                    "_build_augmented_win_matrix: winner '%s' not in "
                    "baseline model list. Skipping vote.",
                    winner,
                )
                continue

            if loser_idx is None:
                logger.debug(
                    "_build_augmented_win_matrix: loser '%s' not in "
                    "baseline model list. Skipping vote.",
                    loser,
                )
                continue

            augmented[winner_idx, loser_idx] += 1.0

        return augmented

    def _get_initial_rank(
        self,
        target_model: str,
        working_bt: BradleyTerryModel,
    ) -> Optional[int]:
        """Get the initial rank of the target model in the working BT model.

        Args:
            target_model: Model name to look up.
            working_bt: The BT model to query.

        Returns:
            1-indexed rank of the target model, or None if the model is not
            found in the BT model's ratings.
        """
        try:
            return working_bt.get_rank(target_model)
        except (KeyError, RuntimeError) as exc:
            logger.warning(
                "_get_initial_rank: cannot get rank for model '%s': %s",
                target_model,
                exc,
            )
            return None

    def _check_stopping_condition(
        self,
        current_rank: int,
        target_rank: int,
        direction: str,
    ) -> bool:
        """Check whether the simulation's rank objective has been achieved.

        Args:
            current_rank: The model's current 1-indexed rank.
            target_rank: The desired 1-indexed rank.
            direction: 'up' (lower rank number = better) or 'down' (higher
                rank number = worse).

        Returns:
            True if the objective has been achieved, False otherwise.
        """
        if direction == "up":
            # Moving up means achieving a lower rank number (rank 1 is best).
            return current_rank <= target_rank
        else:
            # Moving down means achieving a higher rank number.
            return current_rank >= target_rank

    # -----------------------------------------------------------------------
    # Public simulation method
    # -----------------------------------------------------------------------

    def simulate_attack(
        self,
        target_model: str,
        direction: str,
        target_rank: int,
        non_target_strategy: str = "do_nothing",
        max_interactions: Optional[int] = None,
        eval_interval: Optional[int] = None,
        detection_accuracy: Optional[float] = None,
    ) -> SimulationResult:
        """Simulate an adversarial voting attack to shift a model's rank.

        Runs the main simulation loop: sample pairs, detect target, cast votes,
        periodically re-fit BT model, check stopping condition. Returns a
        SimulationResult capturing the outcome.

        Paper alignment: Section 3.1 — "We iteratively simulate attacker
        interactions and adversarial votes with the system. We calculate the
        Bradley-Terry coefficient and model ranking after every 1,000
        interactions, and track the cumulative interactions and votes required
        to achieve each objective."

        Args:
            target_model: Name of the model to attack. Must be present in the
                voting data's model list (from VotingDataLoader.get_model_list()).
            direction: 'up' to boost the model's rank (lower rank number) or
                'down' to suppress it (higher rank number).
            target_rank: The 1-indexed rank to achieve. For 'up', this should
                be less than the current rank. For 'down', greater.
            non_target_strategy: Action when target not detected. One of
                'do_nothing' (default), 'random_upvote', 'vote_tie',
                'vote_both_bad'. Default matches config.yaml
                default_non_target_strategy.
            max_interactions: Override for self.max_interactions. If None,
                uses the instance default (500000 per config.yaml).
            eval_interval: Override for self.eval_interval. If None, uses
                the instance default (1000 per config.yaml).
            detection_accuracy: Override for self.detection_accuracy. If None,
                uses the instance default (0.95 per config.yaml). Used by
                run_detector_accuracy_ablation to test different accuracies
                without mutating instance state.

        Returns:
            SimulationResult with:
              - target_model: the attacked model name
              - target_rank: the desired rank
              - achieved: True if target_rank was reached before max_interactions
              - n_votes: total adversarial votes cast
              - n_interactions: total interactions (votes + abstentions)
              - rank_history: list of ranks at each eval_interval checkpoint
              - vote_history: list of cumulative vote counts at each checkpoint
              - detection_accuracy: the accuracy used in this run
              - non_target_strategy: the strategy used in this run
              - direction: 'up' or 'down'

        Example:
            >>> result = simulator.simulate_attack(
            ...     target_model="llama-13b",
            ...     direction="up",
            ...     target_rank=128,
            ...     detection_accuracy=0.95,
            ... )
            >>> result.achieved
            True
            >>> result.n_votes < 200
            True
        """
        # --- Validate inputs ---
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"simulate_attack: direction='{direction}' is invalid. "
                f"Must be one of {sorted(_VALID_DIRECTIONS)}."
            )

        if non_target_strategy not in _VALID_NON_TARGET_STRATEGIES:
            raise ValueError(
                f"simulate_attack: non_target_strategy='{non_target_strategy}' "
                f"is invalid. Must be one of {sorted(_VALID_NON_TARGET_STRATEGIES)}."
            )

        # --- Resolve effective parameters (instance defaults or overrides) ---
        effective_max_interactions: int = (
            max_interactions if max_interactions is not None else self.max_interactions
        )
        effective_eval_interval: int = (
            eval_interval if eval_interval is not None else self.eval_interval
        )
        effective_detection_accuracy: float = (
            detection_accuracy
            if detection_accuracy is not None
            else self.detection_accuracy
        )

        logger.info(
            "simulate_attack: target_model='%s', direction='%s', "
            "target_rank=%d, non_target_strategy='%s', "
            "detection_accuracy=%.2f, max_interactions=%d, eval_interval=%d.",
            target_model,
            direction,
            target_rank,
            non_target_strategy,
            effective_detection_accuracy,
            effective_max_interactions,
            effective_eval_interval,
        )

        # --- Check that target model is in the voting data ---
        if target_model not in self._model_to_idx:
            logger.warning(
                "simulate_attack: target_model='%s' not found in voting data "
                "model list (%d models). Returning unachieved result.",
                target_model,
                len(self._baseline_model_names),
            )
            return SimulationResult(
                target_model=target_model,
                target_rank=target_rank,
                achieved=False,
                n_votes=0,
                n_interactions=0,
                rank_history=[],
                vote_history=[],
                detection_accuracy=effective_detection_accuracy,
                non_target_strategy=non_target_strategy,
                direction=direction,
            )

        # --- Initialize working BT model (deep copy of baseline) ---
        # We do NOT use copy.deepcopy on the full BT model because it contains
        # a potentially large _pairwise_data list. Instead, we create a fresh
        # BT model and fit it on the baseline win matrix.
        working_bt: BradleyTerryModel = BradleyTerryModel(
            scale_factor=self.bt_model.scale_factor
        )
        working_bt.fit(self._baseline_win_matrix, self._baseline_model_names)

        # --- Check initial rank ---
        initial_rank: Optional[int] = self._get_initial_rank(target_model, working_bt)

        if initial_rank is None:
            logger.warning(
                "simulate_attack: cannot determine initial rank for "
                "target_model='%s'. Returning unachieved result.",
                target_model,
            )
            return SimulationResult(
                target_model=target_model,
                target_rank=target_rank,
                achieved=False,
                n_votes=0,
                n_interactions=0,
                rank_history=[],
                vote_history=[],
                detection_accuracy=effective_detection_accuracy,
                non_target_strategy=non_target_strategy,
                direction=direction,
            )

        logger.info(
            "simulate_attack: initial rank of '%s' = %d, target rank = %d.",
            target_model,
            initial_rank,
            target_rank,
        )

        # --- Check if target rank is already achieved ---
        if self._check_stopping_condition(initial_rank, target_rank, direction):
            logger.info(
                "simulate_attack: target rank %d already achieved at start "
                "(current rank = %d). Returning immediately.",
                target_rank,
                initial_rank,
            )
            return SimulationResult(
                target_model=target_model,
                target_rank=target_rank,
                achieved=True,
                n_votes=0,
                n_interactions=0,
                rank_history=[initial_rank],
                vote_history=[0],
                detection_accuracy=effective_detection_accuracy,
                non_target_strategy=non_target_strategy,
                direction=direction,
            )

        # --- Initialize simulation state ---
        # Seeded RNG for reproducible simulation.
        rng: np.random.Generator = np.random.default_rng(
            self.config.random_state
        )

        n_interactions: int = 0
        n_votes: int = 0
        rank_history: List[int] = []
        vote_history: List[int] = []
        achieved: bool = False

        # Accumulate adversarial votes as (winner, loser) model name tuples.
        # These are added to the baseline win matrix at each eval checkpoint.
        adversarial_votes: List[Tuple[str, str]] = []

        # --- Main simulation loop ---
        while n_interactions < effective_max_interactions:
            # 1. Sample a model pair from the empirical pair distribution.
            model_a: str
            model_b: str
            model_a, model_b = self.voting_loader.sample_pair(rng)

            # 2. Attempt to cast a vote.
            vote: Optional[Tuple[str, str]] = self._cast_vote(
                model_a=model_a,
                model_b=model_b,
                target_model=target_model,
                direction=direction,
                non_target_strategy=non_target_strategy,
                rng=rng,
                detection_accuracy=effective_detection_accuracy,
            )

            # 3. Track the vote.
            if vote is not None:
                adversarial_votes.append(vote)
                n_votes += 1

            n_interactions += 1

            # 4. Every eval_interval interactions, re-fit BT and check rank.
            if n_interactions % effective_eval_interval == 0:
                # Build augmented win matrix = baseline + adversarial votes.
                augmented_matrix: np.ndarray = self._build_augmented_win_matrix(
                    adversarial_votes
                )

                # Re-fit the working BT model from scratch on the augmented data.
                # This matches the paper: "calculate the Bradley-Terry coefficient
                # and model ranking after every 1,000 interactions."
                try:
                    working_bt.fit(
                        augmented_matrix, self._baseline_model_names
                    )
                    current_rank: int = working_bt.get_rank(target_model)
                except (RuntimeError, KeyError, ValueError) as exc:
                    logger.warning(
                        "simulate_attack: BT re-fit or rank lookup failed at "
                        "interaction %d: %s. Skipping checkpoint.",
                        n_interactions,
                        exc,
                    )
                    continue

                rank_history.append(current_rank)
                vote_history.append(n_votes)

                logger.debug(
                    "simulate_attack: checkpoint at interaction=%d, "
                    "n_votes=%d, current_rank=%d (target=%d).",
                    n_interactions,
                    n_votes,
                    current_rank,
                    target_rank,
                )

                # 5. Check stopping condition.
                if self._check_stopping_condition(
                    current_rank, target_rank, direction
                ):
                    achieved = True
                    logger.info(
                        "simulate_attack: target rank %d achieved at "
                        "interaction=%d, n_votes=%d (current_rank=%d).",
                        target_rank,
                        n_interactions,
                        n_votes,
                        current_rank,
                    )
                    break

        if not achieved:
            logger.info(
                "simulate_attack: max_interactions=%d reached without "
                "achieving target rank %d for model '%s'. "
                "Final n_votes=%d.",
                effective_max_interactions,
                target_rank,
                target_model,
                n_votes,
            )

        return SimulationResult(
            target_model=target_model,
            target_rank=target_rank,
            achieved=achieved,
            n_votes=n_votes,
            n_interactions=n_interactions,
            rank_history=rank_history,
            vote_history=vote_history,
            detection_accuracy=effective_detection_accuracy,
            non_target_strategy=non_target_strategy,
            direction=direction,
        )

    # -----------------------------------------------------------------------
    # Experiment runner methods
    # -----------------------------------------------------------------------

    def run_high_ranked_experiments(
        self,
        target_models: Optional[List[str]] = None,
        target_ranks: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Run upvote experiments for high-ranked models (Table 4).

        Simulates the attacker boosting each high-ranked model to each target
        rank. Produces a DataFrame matching Table 4(a) (votes) and Table 4(b)
        (interactions) from the paper.

        Paper alignment: Table 4 — "The number of votes (a) and interactions
        (b) required to change the rankings of high-ranked models on the
        simulated leaderboard."

        Args:
            target_models: List of model names to attack. If None, reads from
                config.simulation.high_ranked_targets.
            target_ranks: List of target ranks to attempt. If None, reads from
                config.simulation.high_ranked_target_ranks.

        Returns:
            pd.DataFrame with MultiIndex columns (metric, target_rank) where
            metric is 'n_votes' or 'n_interactions', and rows are target models.
            N/A entries (diagonal and wrong-direction) are represented as
            np.nan. The caller can split into two sub-tables by selecting
            the appropriate metric column level.

        Example:
            >>> df = simulator.run_high_ranked_experiments()
            >>> df.shape[0]  # 5 high-ranked models
            5
        """
        sim_cfg: Dict = self.config.raw.get("simulation", {})

        # Resolve target models from config if not provided.
        if target_models is None:
            high_ranked_targets: List[Dict] = sim_cfg.get(
                "high_ranked_targets", []
            )
            target_models = [str(t["name"]) for t in high_ranked_targets]

        # Resolve target ranks from config if not provided.
        if target_ranks is None:
            target_ranks = [
                int(r) for r in sim_cfg.get("high_ranked_target_ranks", [1, 2, 3, 4, 5])
            ]

        # Build a lookup from model name to current rank.
        high_ranked_targets_list: List[Dict] = sim_cfg.get("high_ranked_targets", [])
        current_rank_lookup: Dict[str, int] = {
            str(t["name"]): int(t["current_rank"])
            for t in high_ranked_targets_list
        }

        logger.info(
            "run_high_ranked_experiments: %d models × %d target ranks.",
            len(target_models),
            len(target_ranks),
        )

        # Collect results: rows = models, cols = target ranks.
        votes_rows: Dict[str, Dict[int, object]] = {}
        interactions_rows: Dict[str, Dict[int, object]] = {}

        for model_name in target_models:
            current_rank: int = current_rank_lookup.get(model_name, -1)
            votes_row: Dict[int, object] = {}
            interactions_row: Dict[int, object] = {}

            for target_rank in target_ranks:
                # N/A if target rank equals current rank (no movement needed).