```python
## mitigations/malicious_user_detector.py
"""Malicious user detection mitigation for the adversarial leaderboard manipulation paper.

This module implements Section 4.2.3 (Scenario 1) of the paper
"Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards."

The core idea: given a user's vote sequence, determine whether they are a
benign user or an adversarial one using a likelihood-based hypothesis test.

Under the null hypothesis H_benign, votes are drawn i.i.d. from the known
benign distribution Pr(x_i | H_benign) estimated from historical voting data.
The test statistic T(x) = -2 * ln(L(x | H_benign)) is compared against a
simulated null distribution to compute an empirical p-value.

Paper alignment:
  - Section 4.2.3: "we assume that a defender can estimate the expected
    behaviour for benign users using historical data from previous votes."
  - Section 4.2.3: "T(x) = -2 * ln(L(x | H_benign))"
  - Section 4.2.3: "we use α = 0.01 in our evaluations"
  - Figure 4: Detection rate vs. number of malicious votes for naive vs.
    informed adversary under Scenario 1.
  - config.yaml: mitigations.malicious_user_detection.significance_level = 0.01
  - config.yaml: mitigations.malicious_user_detection.n_simulations = 10000
  - config.yaml: mitigations.malicious_user_detection.n_trials = 100
  - config.yaml: mitigations.malicious_user_detection.vote_counts = [10, 20, 50, 100, 200, 500, 1000]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from simulation.bradley_terry import BradleyTerryModel
from utils.logger import get_logger

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------
# Minimum probability floor to prevent log(0) in likelihood computation.
# Applied to models with zero empirical wins during fit_benign_distribution.
_PROB_FLOOR: float = 1e-10

# Default random seed for reproducibility (from config.reproducibility.random_state).
_DEFAULT_RANDOM_SEED: int = 42

# Default significance level (from config.yaml mitigations.malicious_user_detection).
_DEFAULT_SIGNIFICANCE_LEVEL: float = 0.01

# Default number of Monte Carlo simulations for p-value estimation.
_DEFAULT_N_SIMULATIONS: int = 10000

# Default number of trials per vote count in evaluate_scenario1.
_DEFAULT_N_TRIALS: int = 100

# Default vote counts to evaluate (from config.yaml).
_DEFAULT_VOTE_COUNTS: List[int] = [10, 20, 50, 100, 200, 500, 1000]

# Column names in the voting data DataFrame.
_COL_MODEL_A: str = "model_a"
_COL_MODEL_B: str = "model_b"
_COL_WINNER: str = "winner"
_WINNER_MODEL_A: str = "model_a"
_WINNER_MODEL_B: str = "model_b"
_WINNER_TIE: str = "tie"

# Approximate probability that the target model appears in a given vote pair
# under uniform pair sampling with N models: 2 / N_models.
# Used in adversary simulation to decide when the target model is "present."
_TARGET_PRESENCE_DENOMINATOR: float = 2.0


class MaliciousUserDetector:
    """Detects malicious users via likelihood-based hypothesis testing.

    Implements the malicious user detection approach from Section 4.2.3 of
    the paper. Given a user's vote sequence, tests whether it is consistent
    with the known benign voting distribution using a likelihood ratio test.

    Scenario 1 (this module): The defender knows the benign distribution
    from historical data. The test statistic T(x) = -2 * ln(L(x | H_benign))
    is compared against a simulated null distribution to compute a p-value.

    Attributes:
        bt_model: The fitted BradleyTerryModel providing ground-truth ratings
            for computing theoretically-grounded benign vote probabilities.
        significance_level: Alpha threshold for the hypothesis test. Default
            0.01 per config.yaml mitigations.malicious_user_detection.
        benign_probs: Dict mapping model name to empirical benign vote
            probability. Populated by fit_benign_distribution(). Empty until
            fit_benign_distribution() is called.
        bt_benign_probs: Dict mapping model name to BT-derived benign vote
            probability. Populated by fit_benign_distribution() via
            bt_model.compute_benign_vote_probs(). Used by PerturbedLeaderboard.
        model_names: Sorted list of all model names in the arena. Populated
            by fit_benign_distribution().

    Example:
        >>> from simulation.bradley_terry import BradleyTerryModel
        >>> bt = BradleyTerryModel()
        >>> bt.fit(win_matrix, model_names)
        >>> detector = MaliciousUserDetector(bt, significance_level=0.01)
        >>> detector.fit_benign_distribution(votes_df)
        >>> vote_seq = ["gpt-4o", "llama-3", "gpt-4o", "claude-3"]
        >>> detector.detect(vote_seq)
        False  # Likely benign
    """

    def __init__(
        self,
        bt_model: BradleyTerryModel,
        significance_level: float = _DEFAULT_SIGNIFICANCE_LEVEL,
    ) -> None:
        """Initialize the MaliciousUserDetector.

        Args:
            bt_model: A fitted BradleyTerryModel instance. Used to compute
                theoretically-grounded benign vote probabilities via
                bt_model.compute_benign_vote_probs(). Must have been fitted
                (bt_model.ratings must be non-empty) before calling
                fit_benign_distribution().
            significance_level: Alpha threshold for the hypothesis test.
                The null hypothesis H_benign is rejected (user flagged as
                malicious) when p-value < significance_level. Default 0.01
                per config.yaml mitigations.malicious_user_detection.
                significance_level and Section 4.2.3: "we use α = 0.01."

        Raises:
            ValueError: If significance_level is not in (0, 1).

        Example:
            >>> detector = MaliciousUserDetector(bt_model, significance_level=0.01)
            >>> detector.significance_level
            0.01
            >>> detector.benign_probs
            {}
        """
        if not (0.0 < significance_level < 1.0):
            raise ValueError(
                f"significance_level must be in (0, 1), got {significance_level}."
            )

        self.bt_model: BradleyTerryModel = bt_model
        self.significance_level: float = significance_level

        # Empirical benign vote probabilities from historical voting data.
        # Populated by fit_benign_distribution(). Maps model_name -> float.
        self.benign_probs: Dict[str, float] = {}

        # BT-derived benign vote probabilities from the theoretical model.
        # Populated by fit_benign_distribution() via bt_model.compute_benign_vote_probs().
        # Used by PerturbedLeaderboard for Scenario 2.
        self.bt_benign_probs: Dict[str, float] = {}

        # Sorted list of all model names in the arena.
        # Populated by fit_benign_distribution().
        self.model_names: List[str] = []

        logger.info(
            "MaliciousUserDetector initialized with significance_level=%.4f.",
            self.significance_level,
        )

    # -----------------------------------------------------------------------
    # Public interface methods
    # -----------------------------------------------------------------------

    def fit_benign_distribution(self, voting_data: pd.DataFrame) -> None:
        """Estimate the benign vote probability distribution from historical data.

        Computes the empirical probability that a benign user votes for each
        model, based on historical win counts from the voting data. This
        implements the defender's estimation of Pr(x_i | H_benign) from
        Section 4.2.3: "we assume that a defender can estimate the expected
        behaviour for benign users using historical data from previous votes."

        Also calls bt_model.compute_benign_vote_probs() to compute the
        theoretically-grounded BT-derived probabilities, stored separately
        as self.bt_benign_probs for use by PerturbedLeaderboard.

        Args:
            voting_data: DataFrame with columns ['model_a', 'model_b', 'winner']
                where winner is one of 'model_a', 'model_b', 'tie'. This is
                the output of VotingDataLoader.load_votes(). Ties are excluded
                from win counts since they do not constitute a vote for either
                model.

        Returns:
            None. Updates self.benign_probs, self.bt_benign_probs, and
            self.model_names in place.

        Raises:
            ValueError: If voting_data is empty or missing required columns.

        Example:
            >>> detector.fit_benign_distribution(votes_df)
            >>> abs(sum(detector.benign_probs.values()) - 1.0) < 1e-9
            True
            >>> all(v > 0 for v in detector.benign_probs.values())
            True
        """
        # --- Validate input ---
        if voting_data.empty:
            raise ValueError(
                "fit_benign_distribution: voting_data is empty. "
                "Cannot estimate benign distribution from empty data."
            )

        required_cols: frozenset = frozenset(
            {_COL_MODEL_A, _COL_MODEL_B, _COL_WINNER}
        )
        missing_cols: set = required_cols - set(voting_data.columns)
        if missing_cols:
            raise ValueError(
                f"fit_benign_distribution: voting_data is missing required "
                f"columns: {sorted(missing_cols)}. "
                f"Found columns: {sorted(voting_data.columns.tolist())}."
            )

        logger.info(
            "fit_benign_distribution: processing %d voting records.",
            len(voting_data),
        )

        # --- Extract all unique model names ---
        all_models_a: set = set(voting_data[_COL_MODEL_A].astype(str).unique())
        all_models_b: set = set(voting_data[_COL_MODEL_B].astype(str).unique())
        all_models: set = all_models_a | all_models_b
        self.model_names = sorted(list(all_models))

        logger.info(
            "fit_benign_distribution: found %d unique models in voting data.",
            len(self.model_names),
        )

        # --- Count wins per model (excluding ties) ---
        # Initialize win counts to 0 for all models.
        win_counts: Dict[str, float] = {model: 0.0 for model in self.model_names}

        # Count wins where model_a won.
        wins_a_mask: pd.Series = voting_data[_COL_WINNER] == _WINNER_MODEL_A
        wins_a: pd.Series = (
            voting_data.loc[wins_a_mask, _COL_MODEL_A]
            .astype(str)
            .value_counts()
        )
        for model_name, count in wins_a.items():
            if model_name in win_counts:
                win_counts[model_name] += float(count)

        # Count wins where model_b won.
        wins_b_mask: pd.Series = voting_data[_COL_WINNER] == _WINNER_MODEL_B
        wins_b: pd.Series = (
            voting_data.loc[wins_b_mask, _COL_MODEL_B]
            .astype(str)
            .value_counts()
        )
        for model_name, count in wins_b.items():
            if model_name in win_counts:
                win_counts[model_name] += float(count)

        # --- Compute total wins (excluding ties) ---
        total_wins: float = sum(win_counts.values())

        if total_wins <= 0.0:
            logger.warning(
                "fit_benign_distribution: no wins found in voting data "
                "(all votes may be ties). Using uniform distribution."
            )
            # Fall back to uniform distribution.
            n_models: int = len(self.model_names)
            uniform_prob: float = 1.0 / max(n_models, 1)
            self.benign_probs = {m: uniform_prob for m in self.model_names}
        else:
            # --- Normalize to get empirical probabilities ---
            # Apply probability floor to models with zero wins to prevent log(0).
            raw_probs: Dict[str, float] = {}
            for model_name in self.model_names:
                raw_count: float = win_counts.get(model_name, 0.0)
                # Apply floor: models with zero wins get a small smoothing value.
                raw_probs[model_name] = max(
                    raw_count / total_wins, _PROB_FLOOR
                )

            # Re-normalize after applying floor to ensure probabilities sum to 1.
            prob_sum: float = sum(raw_probs.values())
            self.benign_probs = {
                model_name: prob / prob_sum
                for model_name, prob in raw_probs.items()
            }

        # --- Compute BT-derived benign probabilities ---
        # These are the theoretically-grounded probabilities from the BT model.
        # Used by PerturbedLeaderboard for Scenario 2.
        try:
            self.bt_benign_probs = self.bt_model.compute_benign_vote_probs()
            logger.info(
                "fit_benign_distribution: computed BT-derived benign probs "
                "for %d models.",
                len(self.bt_benign_probs),
            )
        except RuntimeError as exc:
            logger.warning(
                "fit_benign_distribution: could not compute BT-derived benign "
                "probs (BT model may not be fitted): %s. "
                "bt_benign_probs will be empty.",
                exc,
            )
            self.bt_benign_probs = {}

        # --- Log summary statistics ---
        if self.benign_probs:
            top5_models: List[tuple] = sorted(
                self.benign_probs.items(), key=lambda x: x[1], reverse=True
            )[:5]
            logger.info(
                "fit_benign_distribution complete. "
                "Total wins processed: %.0f. "
                "Top-5 models by benign vote probability: %s.",
                total_wins,
                [(m, f"{p:.4f}") for m, p in top5_models],
            )

        # Verify normalization.
        prob_total: float = sum(self.benign_probs.values())
        if abs(prob_total - 1.0) > 1e-6:
            logger.warning(
                "fit_benign_distribution: benign_probs do not sum to 1.0 "
                "(sum=%.8f). Re-normalizing.",
                prob_total,
            )
            self.benign_probs = {
                m: p / prob_total for m, p in self.benign_probs.items()
            }

    def compute_log_likelihood(self, vote_sequence: List[str]) -> float:
        """Compute the log-likelihood of a vote sequence under H_benign.

        Implements ln(L(x | H_benign)) = sum_i ln(Pr(x_i | H_benign)) from
        Section 4.2.3 of the paper.

        Args:
            vote_sequence: List of model name strings, where each entry is
                the model that "won" in that vote (the model the user voted
                for). May be empty.

        Returns:
            Log-likelihood as a float (always <= 0.0 since all probabilities
            are in (0, 1]). Returns 0.0 for an empty sequence. Returns a
            very negative value for sequences containing models with very
            low benign probability.

        Raises:
            RuntimeError: If fit_benign_distribution() has not been called
                (self.benign_probs is empty).

        Example:
            >>> detector.fit_benign_distribution(votes_df)
            >>> ll = detector.compute_log_likelihood(["gpt-4o", "claude-3"])
            >>> ll <= 0.0
            True
            >>> detector.compute_log_likelihood([])
            0.0
        """
        if not self.benign_probs:
            raise RuntimeError(
                "MaliciousUserDetector.compute_log_likelihood(): "
                "benign_probs is empty. Call fit_benign_distribution() first."
            )

        # Empty sequence: log likelihood of empty observation is 0.
        if not vote_sequence:
            return 0.0

        log_likelihood: float = 0.0

        for model_name in vote_sequence:
            # Look up probability with floor to prevent log(0).
            p: float = self.benign_probs.get(model_name, _PROB_FLOOR)

            # Guard against zero or negative probabilities (should not happen
            # after fit_benign_distribution applies the floor, but defensive).
            if p <= 0.0:
                p = _PROB_FLOOR

            log_likelihood += float(np.log(p))

        return log_likelihood

    def compute_test_statistic(self, vote_sequence: List[str]) -> float:
        """Compute the test statistic T(x) = -2 * ln(L(x | H_benign)).

        Implements the test statistic from Section 4.2.3:
        "T(x) = -2 * ln(L(x | H_benign))"

        Larger values of T indicate the sequence is less likely under the
        benign hypothesis — i.e., more suspicious. The test statistic is
        always non-negative (since log likelihood is <= 0).

        Args:
            vote_sequence: List of model name strings representing the user's
                vote history. May be empty.

        Returns:
            Non-negative float. Returns 0.0 for an empty sequence.
            Returns a large positive value for sequences that are very
            unlikely under the benign distribution.

        Raises:
            RuntimeError: If fit_benign_distribution() has not been called.

        Example:
            >>> detector.fit_benign_distribution(votes_df)
            >>> T = detector.compute_test_statistic(["gpt-4o", "claude-3"])
            >>> T >= 0.0
            True
            >>> detector.compute_test_statistic([])
            0.0
        """
        log_likelihood: float = self.compute_log_likelihood(vote_sequence)
        # T(x) = -2 * ln(L(x | H_benign))
        # Since log_likelihood <= 0, T >= 0.
        return -2.0 * log_likelihood

    def compute_pvalue(
        self,
        vote_sequence: List[str],
        n_simulations: int = _DEFAULT_N_SIMULATIONS,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Compute the empirical p-value for a vote sequence under H_benign.

        Uses Monte Carlo simulation to estimate the p-value:
            p = (1/m) * sum_j I{T(s^j) >= T(x)}

        where s^j are simulated sequences drawn from the benign distribution
        and T is the test statistic. This matches the paper's formula from
        Section 4.2.3.

        Uses a vectorized batch approach for efficiency: all n_simulations
        sequences are generated at once using rng.choice with size=(n_simulations, n),
        avoiding a slow Python loop over 10,000 simulations.

        Args:
            vote_sequence: The observed vote sequence to test. May be empty
                (returns p-value of 1.0 for empty sequences).
            n_simulations: Number of Monte Carlo simulations for p-value
                estimation. Default 10000 per config.yaml
                mitigations.malicious_user_detection.n_simulations.
            rng: Optional numpy random generator for reproducibility. If None,
                creates a new generator seeded with _DEFAULT_RANDOM_SEED (42).
                Pass the same rng instance across calls to ensure reproducibility
                in the evaluate_scenario1 loop.

        Returns:
            Float in [0.0, 1.0] representing the empirical p-value. Small
            values (< significance_level) indicate the sequence is unlikely
            under H_benign (user is likely malicious).

        Raises:
            RuntimeError: If fit_benign_distribution() has not been called.

        Example:
            >>> detector.fit_benign_distribution(votes_df)
            >>> rng = np.random.default_rng(42)
            >>> p = detector.compute_pvalue(["gpt-4o"] * 100, rng=rng)
            >>> 0.0 <= p <= 1.0
            True
        """
        if not self.benign_probs:
            raise RuntimeError(
                "MaliciousUserDetector.compute_pvalue(): "
                "benign_probs is empty. Call fit_benign_distribution() first."
            )

        # Empty sequence: p-value is 1.0 (no evidence against H_benign).
        if not vote_sequence:
            return 1.0

        # Initialize RNG if not provided.
        if rng is None:
            rng = np.random.default_rng(_DEFAULT_RANDOM_SEED)

        # --- Compute observed test statistic ---
        T_obs: float = self.compute_test_statistic(vote_sequence)

        n: int = len(vote_sequence)

        # --- Prepare sampling arrays ---
        models: List[str] = list(self.benign_probs.keys())
        probs: np.ndarray = np.array(
            [self.benign_probs[m] for m in models], dtype=np.float64
        )

        # Normalize to ensure exact sum = 1.0 (guard against floating point drift).
        prob_sum: float = float(probs.sum())
        if prob_sum <= 0.0:
            logger.warning(
                "compute_pvalue: benign_probs sum to %.8f <= 0. "
                "Using uniform distribution.",
                prob_sum,
            )
            probs = np.ones(len(models), dtype=np.float64) / len(models)
        else:
            probs = probs / prob_sum

        n_models: int = len(models)

        # --- Vectorized batch simulation ---
        # Generate all n_simulations sequences at once as a 2D integer index array.
        # Shape: (n_simulations, n) where each row is one simulated sequence.
        # Using integer indices for efficiency, then map back to log-probs.
        try:
            # Sample integer indices into the models list.
            # Shape: (n_simulations, n)
            simulated_indices: np.ndarray = rng.choice(
                n_models,
                size=(n_simulations, n),
                p=probs,
            )
        except ValueError as exc:
            logger.warning(
                "compute_pvalue: vectorized rng.choice failed: %s. "
                "Falling back to sequential simulation.",
                exc,
            )
            return self._compute_pvalue_sequential(
                vote_sequence, T_obs, n_simulations, rng
            )

        # --- Compute log-probabilities for each model index ---
        # log_probs[i] = log(Pr(models[i] | H_benign))
        log_probs: np.ndarray = np.log(
            np.maximum(probs, _PROB_FLOOR)
        )  # Shape: (n_models,)

        # --- Compute test statistics for all simulated sequences ---
        # For each simulated sequence (row), sum the log-probs of the sampled
        # model indices, then multiply by -2.
        # simulated_log_likelihoods[j] = sum_i log_probs[simulated_indices[j, i]]
        # Shape: (n_simulations,)
        simulated_log_likelihoods: np.ndarray = log_probs[simulated_indices].sum(
            axis=1
        )
        # T_simulated[j] = -2 * simulated_log_likelihoods[j]
        T_simulated: np.ndarray = -2.0 * simulated_log_likelihoods

        # --- Compute empirical p-value ---
        # p = (1/m) * sum_j I{T(s^j) >= T(x)}
        p_value: float = float(np.mean(T_simulated >= T_obs))

        logger.debug(
            "compute_pvalue: n=%d, T_obs=%.4f, "
            "T_sim_mean=%.4f, T_sim_std=%.4f, p_value=%.6f.",
            n,
            T_obs,
            float(T_simulated.mean()),
            float(T_simulated.std()),
            p_value,
        )

        return p_value

    def detect(
        self,
        vote_sequence: List[str],
        n_simulations: int = _DEFAULT_N_SIMULATIONS,
    ) -> bool:
        """Determine whether a user is malicious based on their vote sequence.

        Rejects the null hypothesis H_benign (flags the user as malicious)
        when the empirical p-value is below the significance level α.

        Paper alignment: Section 4.2.3 — "We reject the null hypothesis (and
        conclude the user is likely not the known benign user) if the p-value
        is less than the desired significance level α. In particular we use
        α = 0.01 in our evaluations."

        Args:
            vote_sequence: The user's vote history as a list of model name
                strings (each entry is the model the user voted for).
            n_simulations: Number of Monte Carlo simulations for p-value
                estimation. Default 10000 per config.yaml.

        Returns:
            True if the user is flagged as malicious (p-value < α).
            False if the user appears benign (p-value >= α).

        Raises:
            RuntimeError: If fit_benign_distribution() has not been called.

        Example:
            >>> detector.fit_benign_distribution(votes_df)
            >>> # Sequence that always votes for the same model (suspicious)
            >>> suspicious_seq = ["gpt-4o"] * 100
            >>> detector.detect(suspicious_seq)
            True  # Likely flagged as malicious
            >>> # Short sequence (insufficient evidence)
            >>> short_seq = ["gpt-4o", "claude-3"]
            >>> detector.detect(short_seq)
            False  # Likely not flagged
        """
        p_value: float = self.compute_pvalue(
            vote_sequence=vote_sequence,
            n_simulations=n_simulations,
        )

        is_malicious: bool = p_value < self.significance_level

        logger.debug(
            "detect: sequence_length=%d, p_value=%.6f, "
            "significance_level=%.4f, is_malicious=%s.",
            len(vote_sequence),
            p_value,
            self.significance_level,
            is_malicious,
        )

        return is_malicious

    def simulate_naive_adversary(
        self,
        target_model: str,
        n_votes: int,
        rng: np.random.Generator,
    ) -> List[str]:
        """Generate a vote sequence for a naive adversary.

        The naive adversary always votes for the target model when it appears
        in a comparison, and votes uniformly at random among non-target models
        otherwise. This deviates from the benign distribution and makes the
        adversary detectable.

        Paper alignment: Section 4.3 — "the more naive adversary where the
        attacker randomly chooses between two non targeted models (and always
        prefers the targeted models)."

        Args:
            target_model: The model the adversary is trying to boost. Must be
                a model name present in self.model_names (populated by
                fit_benign_distribution()).
            n_votes: Number of votes to generate in the sequence.
            rng: Seeded numpy random generator for reproducible simulation.
                The same rng instance should be passed across calls in the
                evaluate_scenario1 loop.

        Returns:
            List of n_votes model name strings representing the adversary's
            vote sequence. Each entry is the model the adversary voted for.

        Raises:
            RuntimeError: If fit_benign_distribution() has not been called
                (self.model_names is empty).
            ValueError: If target_model is not in self.model_names.

        Example:
            >>> rng = np.random.default_rng(42)
            >>> seq = detector.simulate_naive_adversary("gpt-4o", 100, rng)
            >>> len(seq)
            100
            >>> "gpt-4o" in seq  # Target model appears in some votes
            True
        """
        if not self.model_names:
            raise RuntimeError(
                "MaliciousUserDetector.simulate_naive_adversary(): "
                "model_names is empty. Call fit_benign_distribution() first."
            )

        if target_model not in self.model_names:
            raise ValueError(
                f"simulate_naive_adversary: target_model='{target_model}' "
                f"not found in model_names. Available models: "
                f"{sorted(self.model_names)[:10]}..."
            )

        n_models: int = len(self.model_names)

        # Probability that the target model appears in a given vote pair.
        # Under uniform pair sampling with N models, each model appears in
        # approximately 2/N of all pairs.
        p_target_present: float = min(
            _TARGET_PRESENCE_DENOMINATOR / max(n_models, 1), 1.0
        )

        # Non-target models for uniform random voting.
        non_target_models: List[str] = [
            m for m in self.model_names if m != target_model
        ]

        if not non_target_models:
            # Edge case: only one model in the arena. All votes go to target.
            logger.warning(
                "simulate_naive_adversary: no non-target models available. "
                "All %d votes will be for target_model='%s'.",
                n_votes,
                target_model,
            )
            return [target_model] * n_votes

        vote_sequence: List[str] = []

        for _ in range(n_votes):
            # Determine whether the target model is present in this vote pair.
            target_present: bool = bool(rng.random() < p_target_present)

            if target_present:
                # Adversary votes for the target model.
                vote_sequence.append(target_model)
            else:
                # Adversary votes uniformly at random among non-target models.
                # This is the "naive" behavior — deviates from benign distribution.
                random_idx: int = int(rng.integers(0, len(non_target_models)))
                vote_sequence.append(non_target_models[random_idx])

        logger.debug(
            "simulate_naive_adversary: generated %d votes for target='%s'. "
            "Target vote fraction: %.3f.",
            n_votes,
            target_model,
            vote_sequence.count(target_model) / max(n_votes, 1),
        )

        return vote_sequence

    def simulate_informed_adversary