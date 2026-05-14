## Code: mitigations/perturbed_leaderboard.py

```python
## mitigations/perturbed_leaderboard.py
"""Perturbed leaderboard defense for the adversarial leaderboard manipulation paper.

This module implements Scenario 2 of Section 4.2.3 from the paper
"Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards."

The core defense: the defender releases a perturbed version of the leaderboard
(BT ratings with added Gaussian noise) to each user. An adversary who uses
these perturbed ratings to mimic benign behavior will have a detectable
statistical signature, because their vote distribution is calibrated to the
wrong (perturbed) ratings rather than the true ones.

The defender uses the Neyman-Pearson likelihood ratio test to distinguish
adversarial from benign voting patterns:
    Λ(x) = Pr_M(x) / Pr_B(x)

where Pr_M uses perturbed ratings (adversary's model) and Pr_B uses true
ratings (benign model).

Paper alignment:
  - Section 4.2.3 Scenario 2: "the defender modifies the rating of the model
    and releases the perturbed leaderboard."
  - Section 4.2.3: "we add scaled Gaussian noise to Bradley-Terry coefficient
    ratings before releasing the rating."
  - Section 4.2.3: "Λ(x) = Pr_M(x) / Pr_B(x)"
  - Section 4.3: "we measure utility as the average absolute change in the
    ranking of any item."
  - Figure 5: Detection rate vs. vote count for different noise scales.
  - Figure 6: Utility loss (average rank change) vs. noise scale.
  - config.yaml: mitigations.perturbed_leaderboard.noise_scales
  - config.yaml: mitigations.perturbed_leaderboard.n_trials
  - config.yaml: mitigations.perturbed_leaderboard.rng_seed
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simulation.bradley_terry import BradleyTerryModel
from utils.logger import get_logger

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------
# Epsilon floor applied to probabilities before taking log to prevent log(0).
# Applied in compute_likelihood_ratio to guard against floating-point underflow
# in the product formula for compute_benign_vote_probs.
_PROB_FLOOR: float = 1e-300

# Default noise scales to sweep over (from config.yaml
# mitigations.perturbed_leaderboard.noise_scales).
_DEFAULT_NOISE_SCALES: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# Default vote counts to evaluate (from config.yaml
# mitigations.malicious_user_detection.vote_counts).
_DEFAULT_VOTE_COUNTS: List[int] = [10, 20, 50, 100, 200, 500, 1000]

# Default number of trials per (noise_scale, vote_count) combination.
# From config.yaml mitigations.perturbed_leaderboard.n_trials.
_DEFAULT_N_TRIALS: int = 100

# Default random seed for reproducibility.
# From config.yaml mitigations.perturbed_leaderboard.rng_seed.
_DEFAULT_RNG_SEED: int = 42

# Default likelihood ratio threshold for detection.
# threshold=1.0 means log(threshold)=0, so detection fires when the adversary
# distribution is more likely than the benign distribution.
_DEFAULT_THRESHOLD: float = 1.0

# Maximum absolute value of the exponent argument to prevent overflow/underflow
# in math.exp during win probability computation.
_EXP_CLIP: float = 500.0


class PerturbedLeaderboard:
    """Implements the perturbed leaderboard defense from Section 4.2.3 Scenario 2.

    The defender adds Gaussian noise to the true Bradley-Terry ratings before
    publishing them. An adversary who uses these perturbed ratings to construct
    their voting strategy will have a detectable statistical signature via the
    Neyman-Pearson likelihood ratio test.

    This class is self-contained for Scenario 2 and does not depend on
    MaliciousUserDetector (Scenario 1). All random operations go through
    caller-provided numpy RNG instances for full reproducibility.

    Attributes:
        bt_model: The fitted BradleyTerryModel with true ratings. Never mutated.
        true_ratings: Snapshot of bt_model.ratings at construction time.
            Never mutated after __init__. All perturbations return new dicts.
        model_names: Sorted list of all model names from bt_model.model_names.
        scale_factor: The logistic scaling parameter s from bt_model.scale_factor.
            Used in the win probability formula:
            Pr(i preferred over j) = 1 / (1 + exp(-(Q_i - Q_j) / scale_factor))

    Example:
        >>> from simulation.bradley_terry import BradleyTerryModel
        >>> import numpy as np
        >>> bt = BradleyTerryModel(scale_factor=1.0)
        >>> bt.fit(win_matrix, model_names)
        >>> pl = PerturbedLeaderboard(bt)
        >>> rng = np.random.default_rng(42)
        >>> perturbed = pl.perturb_ratings(noise_scale=1.0, rng=rng)
        >>> utility_loss = pl.compute_utility_loss(perturbed)
        >>> utility_loss >= 0.0
        True
        >>> results = pl.evaluate_scenario2(
        ...     noise_scales=[0.5, 1.0],
        ...     vote_counts=[10, 50],
        ...     n_trials=10,
        ...     rng_seed=42,
        ... )
        >>> 'detection_rates' in results
        True
        >>> 'utility_losses' in results
        True
    """

    def __init__(self, bt_model: BradleyTerryModel) -> None:
        """Initialize the PerturbedLeaderboard with a fitted BradleyTerryModel.

        Takes a snapshot of the true BT ratings at construction time and stores
        them immutably. All subsequent perturbation operations return new dicts
        without modifying the stored true ratings.

        Args:
            bt_model: A fitted BradleyTerryModel instance. Must have non-empty
                ratings (bt_model.ratings must be populated by bt_model.fit()).
                The model's ratings, model_names, and scale_factor are read
                at construction time and stored locally.

        Raises:
            RuntimeError: If bt_model has not been fitted (bt_model.ratings
                is empty), since the perturbation defense requires true ratings
                to compute benign vote probabilities.
            ValueError: If bt_model has no model names (empty model_names list).

        Example:
            >>> bt = BradleyTerryModel(scale_factor=1.0)
            >>> bt.fit(win_matrix, ["model_a", "model_b", "model_c"])
            >>> pl = PerturbedLeaderboard(bt)
            >>> len(pl.true_ratings)
            3
            >>> pl.scale_factor
            1.0
        """
        if not bt_model.ratings:
            raise RuntimeError(
                "PerturbedLeaderboard.__init__(): bt_model has not been fitted "
                "(bt_model.ratings is empty). Call bt_model.fit() before "
                "constructing PerturbedLeaderboard."
            )

        if not bt_model.model_names:
            raise ValueError(
                "PerturbedLeaderboard.__init__(): bt_model.model_names is empty. "
                "The BT model must have at least one model."
            )

        self.bt_model: BradleyTerryModel = bt_model

        # Snapshot of true ratings — never mutated after __init__.
        # Using dict() to create a shallow copy (values are floats, so this
        # is a full independent copy).
        self.true_ratings: Dict[str, float] = dict(bt_model.ratings)

        # Sorted list of model names for deterministic iteration order.
        # Sorting ensures consistent ordering across all methods.
        self.model_names: List[str] = sorted(list(bt_model.model_names))

        # Logistic scaling parameter from the BT model.
        # Used in the win probability formula throughout this class.
        self.scale_factor: float = bt_model.scale_factor

        logger.info(
            "PerturbedLeaderboard initialized: %d models, scale_factor=%.4f. "
            "True rating range: [%.4f, %.4f].",
            len(self.model_names),
            self.scale_factor,
            min(self.true_ratings.values()),
            max(self.true_ratings.values()),
        )

    # -----------------------------------------------------------------------
    # Private helper methods
    # -----------------------------------------------------------------------

    def _win_prob(
        self,
        rating_i: float,
        rating_j: float,
    ) -> float:
        """Compute the Bradley-Terry win probability for model i over model j.

        Implements the logistic formula from Section 4.2.3 of the paper:
            Pr(i preferred over j) = 1 / (1 + exp(-(Q_i - Q_j) / scale_factor))

        Uses numerical clipping to prevent overflow/underflow in math.exp.

        Args:
            rating_i: Log-strength parameter Q_i for model i.
            rating_j: Log-strength parameter Q_j for model j.

        Returns:
            Float in (0.0, 1.0) representing the probability that model i
            beats model j. Returns 0.5 when ratings are equal.

        Example:
            >>> pl._win_prob(1.0, 0.0)  # i is stronger
            0.7310585786300049
            >>> pl._win_prob(0.0, 0.0)  # equal strength
            0.5
        """
        exponent: float = -(rating_i - rating_j) / self.scale_factor
        # Clip to prevent overflow/underflow in math.exp.
        exponent_clipped: float = max(-_EXP_CLIP, min(_EXP_CLIP, exponent))
        return 1.0 / (1.0 + math.exp(exponent_clipped))

    def _compute_vote_probs_from_ratings(
        self,
        ratings: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute normalized vote probabilities from a ratings dict.

        Implements the product formula from Section 4.2.3:
            Pr(i) = product over j≠i of Pr(i preferred over j | ratings)

        Uses log-space computation to prevent numerical underflow when
        multiplying many small probabilities (e.g., 21 terms for 22 models).
        Normalizes the result so all probabilities sum to 1.0.

        This is the core computation shared by both compute_benign_vote_probs
        (using true_ratings) and compute_adversary_vote_probs (using
        perturbed_ratings).

        Args:
            ratings: Dict mapping model name to BT log-strength parameter.
                May be true_ratings or perturbed_ratings.

        Returns:
            Dict mapping model name to normalized vote probability. All values
            are in (0.0, 1.0) and sum to 1.0 (within floating-point precision).
            Returns a uniform distribution if all log-probs are -inf (degenerate
            case that should not occur with finite ratings).

        Example:
            >>> probs = pl._compute_vote_probs_from_ratings(pl.true_ratings)
            >>> abs(sum(probs.values()) - 1.0) < 1e-9
            True
            >>> all(0.0 < p < 1.0 for p in probs.values())
            True
        """
        n_models: int = len(self.model_names)

        if n_models == 0:
            logger.warning(
                "_compute_vote_probs_from_ratings: no models. "
                "Returning empty dict."
            )
            return {}

        if n_models == 1:
            # Trivial case: only one model, probability = 1.0.
            return {self.model_names[0]: 1.0}

        # --- Log-space computation to prevent underflow ---
        # For each model i, compute:
        #   log_raw_prob[i] = sum over j≠i of log(Pr(i preferred over j | ratings))
        log_raw_probs: Dict[str, float] = {}

        for model_i in self.model_names:
            rating_i: float = ratings.get(model_i, 0.0)
            log_prob_i: float = 0.0

            for model_j in self.model_names:
                if model_i == model_j:
                    continue

                rating_j: float = ratings.get(model_j, 0.0)
                win_prob_ij: float = self._win_prob(rating_i, rating_j)

                # Guard against log(0) — clip to a small positive value.
                win_prob_ij = max(win_prob_ij, _PROB_FLOOR)
                log_prob_i += math.log(win_prob_ij)

            log_raw_probs[model_i] = log_prob_i

        # --- Log-sum-exp normalization ---
        # Subtract the maximum log-prob before exponentiating to prevent overflow.
        max_log_prob: float = max(log_raw_probs.values())

        # Compute unnormalized probabilities in a numerically stable way.
        raw_probs: Dict[str, float] = {}
        for model_name, log_prob in log_raw_probs.items():
            # exp(log_prob - max_log_prob) is in (0, 1] since log_prob <= max_log_prob.
            raw_probs[model_name] = math.exp(log_prob - max_log_prob)

        # Normalize to sum to 1.0.
        total: float = sum(raw_probs.values())

        if total <= 0.0:
            # Degenerate case: all probabilities underflowed to 0.
            # Fall back to uniform distribution.
            logger.warning(
                "_compute_vote_probs_from_ratings: all raw probabilities are 0 "
                "after normalization. Falling back to uniform distribution."
            )
            uniform_prob: float = 1.0 / n_models
            return {m: uniform_prob for m in self.model_names}

        normalized_probs: Dict[str, float] = {
            model_name: prob / total
            for model_name, prob in raw_probs.items()
        }

        return normalized_probs

    def _simulate_informed_adversary_sequence(
        self,
        vote_count: int,
        perturbed_ratings: Dict[str, float],
        rng: np.random.Generator,
    ) -> List[str]:
        """Generate a vote sequence for an informed adversary using perturbed ratings.

        The informed adversary knows the perturbed ratings they received and
        votes according to the adversary vote probability distribution derived
        from those perturbed ratings. This is the hardest adversary for the
        defender to detect in Scenario 2.

        Paper alignment: Section 4.3 — "when an adversary uses this perturbed
        leaderboard to choose between two untargeted models, their actions can
        still be detected."

        Args:
            vote_count: Number of votes to generate in the sequence.
            perturbed_ratings: The perturbed BT ratings the adversary received.
                Used to compute the adversary's vote probability distribution.
            rng: Seeded numpy random generator for reproducible sampling.

        Returns:
            List of vote_count model name strings. Each entry is the model
            the adversary voted for, sampled from the adversary vote
            probability distribution derived from perturbed_ratings.

        Example:
            >>> rng = np.random.default_rng(42)
            >>> perturbed = pl.perturb_ratings(1.0, rng)
            >>> seq = pl._simulate_informed_adversary_sequence(50, perturbed, rng)
            >>> len(seq)
            50
            >>> all(m in pl.model_names for m in seq)
            True
        """
        if vote_count <= 0:
            return []

        # Compute adversary vote probabilities from perturbed ratings.
        adversary_probs: Dict[str, float] = self.compute_adversary_vote_probs(
            perturbed_ratings
        )

        if not adversary_probs:
            logger.warning(
                "_simulate_informed_adversary_sequence: empty adversary_probs. "
                "Returning empty sequence."
            )
            return []

        # Extract model names and probabilities as parallel arrays for rng.choice.
        models: List[str] = list(adversary_probs.keys())
        probs: np.ndarray = np.array(
            [adversary_probs[m] for m in models], dtype=np.float64
        )

        # Normalize to ensure exact sum = 1.0 (guard against floating-point drift).
        prob_sum: float = float(probs.sum())
        if prob_sum <= 0.0:
            logger.warning(
                "_simulate_informed_adversary_sequence: adversary_probs sum "
                "to %.8f <= 0. Using uniform distribution.",
                prob_sum,
            )
            probs = np.ones(len(models), dtype=np.float64) / len(models)
        else:
            probs = probs / prob_sum

        # Sample vote_count model indices from the adversary distribution.
        try:
            sampled_indices: np.ndarray = rng.choice(
                len(models),
                size=vote_count,
                p=probs,
            )
            vote_sequence: List[str] = [models[int(idx)] for idx in sampled_indices]
        except ValueError as exc:
            logger.warning(
                "_simulate_informed_adversary_sequence: rng.choice failed: %s. "
                "Falling back to sequential sampling.",
                exc,
            )
            # Sequential fallback.
            vote_sequence = []
            for _ in range(vote_count):
                idx: int = int(rng.choice(len(models), p=probs))
                vote_sequence.append(models[idx])

        logger.debug(
            "_simulate_informed_adversary_sequence: generated %d votes. "
            "Most common model: '%s' (%.3f of votes).",
            vote_count,
            max(set(vote_sequence), key=vote_sequence.count) if vote_sequence else "N/A",
            vote_sequence.count(max(set(vote_sequence), key=vote_sequence.count))
            / max(vote_count, 1)
            if vote_sequence
            else 0.0,
        )

        return vote_sequence

    # -----------------------------------------------------------------------
    # Public interface methods
    # -----------------------------------------------------------------------

    def perturb_ratings(
        self,
        noise_scale: float,
        rng: np.random.Generator,
    ) -> Dict[str, float]:
        """Add scaled Gaussian noise to the true BT ratings.

        Implements the perturbation from Section 4.2.3 Scenario 2:
        "we add scaled Gaussian noise to Bradley-Terry coefficient ratings
        before releasing the rating."

        For each model i: perturbed_rating[i] = true_rating[i] + N(0, noise_scale²)

        The noise is independent per model (not correlated). The true_ratings
        are never mutated — a new dict is returned.

        Args:
            noise_scale: Standard deviation of the Gaussian noise added to
                each model's BT rating. Larger values produce more perturbation,
                increasing detection power but also utility loss.
                From config.yaml mitigations.perturbed_leaderboard.noise_scales.
                Must be non-negative. noise_scale=0.0 returns a copy of
                true_ratings with no perturbation.
            rng: Seeded numpy random generator for reproducible noise sampling.
                The caller (evaluate_scenario2) manages the RNG to ensure
                each trial gets a fresh, reproducible perturbation.

        Returns:
            Dict mapping model name to perturbed BT rating (float). Contains
            all models in self.model_names. The returned dict is independent
            of self.true_ratings — modifying it does not affect the stored
            true ratings.

        Raises:
            ValueError: If noise_scale is negative.

        Example:
            >>> rng = np.random.default_rng(42)
            >>> perturbed = pl.perturb_ratings(noise_scale=1.0, rng=rng)
            >>> len(perturbed) == len(pl.true_ratings)
            True
            >>> # Perturbed ratings differ from true ratings (with high probability)
            >>> any(abs(perturbed[m] - pl.true_ratings[m]) > 1e-10
            ...     for m in pl.model_names)
            True
            >>> # Zero noise returns copy of true ratings
            >>> zero_perturbed = pl.perturb_ratings(0.0, rng)
            >>> all(zero_perturbed[m] == pl.true_ratings[m] for m in pl.model_names)
            True
        """
        if noise_scale < 0.0:
            raise ValueError(
                f"perturb_ratings: noise_scale must be non-negative, "
                f"got {noise_scale}."
            )

        n_models: int = len(self.model_names)

        if n_models == 0:
            logger.warning(
                "perturb_ratings: no models in model_names. "
                "Returning empty dict."
            )
            return {}

        # Generate independent Gaussian noise for each model.
        # Shape: (n_models,) with mean=0, std=noise_scale.
        if noise_scale == 0.0:
            # No perturbation — return exact copy of true ratings.
            noise: np.ndarray = np.zeros(n_models, dtype=np.float64)
        else:
            noise = rng.normal(loc=0.0, scale=noise_scale, size=n_models)

        # Build perturbed ratings dict.
        perturbed_ratings: Dict[str, float] = {}
        for idx, model_name in enumerate(self.model_names):
            true_rating: float = self.true_ratings.get(model_name, 0.0)
            perturbed_ratings[model_name] = true_rating + float(noise[idx])

        logger.debug(
            "perturb_ratings: noise_scale=%.4f, "
            "noise_mean=%.4f, noise_std=%.4f. "
            "Perturbed rating range: [%.4f, %.4f].",
            noise_scale,
            float(noise.mean()),
            float(noise.std()),
            min(perturbed_ratings.values()),
            max(perturbed_ratings.values()),
        )

        return perturbed_ratings

    def compute_benign_vote_probs(
        self,
        ratings: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute normalized vote probabilities for a benign user from given ratings.

        Implements the formula from Section 4.2.3:
            Pr_B(i) = product over j≠i of Pr_B(i preferred over j | ratings)

        Uses log-space computation to prevent numerical underflow. Normalizes
        the result so all probabilities sum to 1.0.

        This method is parameterized by `ratings` (not hardcoded to true_ratings)
        so it can be reused for both benign (true ratings) and adversary
        (perturbed ratings) probability computations without code duplication.

        Args:
            ratings: Dict mapping model name to BT log-strength parameter.
                Pass self.true_ratings for benign vote probabilities.
                Pass perturbed_ratings for adversary vote probabilities.

        Returns:
            Dict mapping model name to normalized vote probability. All values
            are in (0.0, 1.0) and sum to 1.0 (within floating-point precision).

        Example:
            >>> probs = pl.compute_benign_vote_probs(pl.true_ratings)
            >>> abs(sum(probs.values()) - 1.0) < 1e-9
            True
            >>> all(0.0 < p < 1.0 for p in probs.values())
            True
        """
        return self._compute_vote_probs_from_ratings(ratings)

    def compute_adversary_vote_probs(
        self,
        perturbed_ratings: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute normalized vote probabilities for an adversary using perturbed ratings.

        Implements Pr_notB(i) from Section 4.2.3: the probability that an
        adversary (who received the perturbed leaderboard) votes for model i.
        The formula is identical to compute_benign_vote_probs but uses the
        perturbed ratings instead of the true ratings.

        This is a thin semantic wrapper around compute_benign_vote_probs that
        makes the API clear — callers can distinguish "benign probs from true
        ratings" from "adversary probs from perturbed ratings" without confusion.

        Args:
            perturbed_ratings: Dict mapping model name to perturbed BT rating.
                Produced by perturb_ratings(). The adversary uses these ratings
                to construct their voting strategy.

        Returns:
            Dict mapping model name to normalized adversary vote probability.
            All values are in (0.0, 1.0) and sum to 1.0.

        Example:
            >>> rng = np.random.default_rng(42)
            >>> perturbed = pl.perturb_ratings(1.0, rng)
            >>> adv_probs = pl.compute_adversary_vote_probs(perturbed)
            >>> abs(sum(adv_probs.values()) - 1.0) < 1e-9
            True
        """
        return self._compute_vote_probs_from_ratings(perturbed_ratings)

    def compute_likelihood_ratio(
        self,
        vote_sequence: List[str],
        perturbed_ratings: Dict[str, float],
    ) -> float:
        """Compute the log likelihood ratio Λ(x) for a vote sequence.

        Implements the Neyman-Pearson likelihood ratio from Section 4.2.3:
            Λ(x) = Pr_M(x) / Pr_B(x)

        Returns the LOG likelihood ratio (not the raw ratio) to prevent
        numerical overflow for long vote sequences. The detection threshold
        in detect() is adjusted accordingly (compare log_lr to log(threshold)).

        The log likelihood ratio is:
            log Λ(x) = sum_i [log(Pr_M(x_i)) - log(Pr_B(x_i))]

        where:
          - Pr_M(x_i) = adversary vote probability for model x_i (from perturbed ratings)
          - Pr_B(x_i) = benign vote probability for model x_i (from true ratings)

        Args:
            vote_sequence: List of model name strings. Each entry is the model
                that "won" in that vote (the model the user voted for). May be
                empty (returns 0.0 for empty sequences).
            perturbed_ratings: The perturbed BT ratings used to compute the
                adversary's vote probability distribution Pr_M.

        Returns:
            Log likelihood ratio as a float. Positive values indicate the
            sequence is more likely under the adversary distribution than the
            benign distribution. Returns 0.0 for empty sequences.

        Example:
            >>> rng = np.random.default_rng(42)
            >>> perturbed = pl.perturb_ratings(2.0, rng)
            >>> # Sequence generated by adversary should have positive log LR
            >>> adv_seq = pl._simulate_informed_adversary_sequence(50, perturbed, rng)
            >>> log_lr = pl.compute_likelihood_ratio(adv_seq, perturbed)
            >>> isinstance(log_lr, float)
            True
        """
        # Empty sequence: log likelihood ratio is 0 (no evidence either way).
        if not vote_sequence:
            return 0.0

        # Compute benign vote probabilities from true ratings.
        benign_probs: Dict[str, float] = self.compute_benign_vote_probs(
            self.true_ratings
        )

        # Compute adversary vote probabilities from perturbed ratings.
        adversary_probs: Dict[str, float] = self.compute_adversary_vote_probs(
            perturbed_ratings
        )

        # Compute log likelihood ratio: sum_i [log(Pr_M(x_i)) - log(Pr_B(x_i))]
        log_lr: float = 0.0

        for model_name in vote_sequence:
            # Get probabilities with floor to prevent log(0).
            p_benign: float = max(
                benign_probs.get(model_name, _PROB_FLOOR), _PROB_FLOOR
            )
            p_adversary: float = max(
                adversary_probs.get(model_name, _PROB_FLOOR), _PROB_FLOOR
            )

            # Accumulate log ratio contribution for this vote.
            log_lr += math.log(p_adversary) - math.log(p_benign)

        logger.debug(
            "compute_likelihood_ratio: sequence_length=%d, log_lr=%.4f.",
            len(vote_sequence),
            log_lr,
        )

        return log_lr

    def detect(
        self,
        vote_sequence: List[str],
        perturbed_ratings: Dict[str, float],
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> bool:
        """Detect whether a user is adversarial using the Neyman-Pearson test.

        Applies the likelihood ratio decision rule: reject H_benign (flag as
        adversarial) when the log likelihood ratio exceeds log(threshold).

        Paper alignment: Section 4.2.3 — "We can use the Neyman-Pearson Lemma
        to construct the hypothesis test. The Neyman-Pearson Lemma states that
        the optimal decision rule is based on the likelihood ratio."

        Args:
            vote_sequence: The user's vote history as a list of model name
                strings. Each entry is the model the user voted for.
            perturbed_ratings: The perturbed BT ratings used to compute the
                adversary's vote probability distribution. Should be the same
                perturbed ratings that were released to this user.
            threshold: Likelihood ratio threshold for detection. Default 1.0
                means log(threshold)=0, so detection fires when the adversary
                distribution is more likely than the benign distribution.
                Increasing threshold reduces false positives but also reduces
                detection power.

        Returns:
            True if the user is flagged as adversarial (log_lr > log(threshold)).
            False if the user appears benign.

        Raises:
            ValueError: If threshold <= 0 (log(threshold) would be undefined
                or -inf, making detection impossible).

        Example:
            >>> rng = np.random.default_rng(42)
            >>> perturbed = pl.perturb_ratings(2.0, rng)
            >>> # Benign user votes according to true distribution
            >>> benign_probs = pl.compute_benign_vote_probs(pl.true_ratings)
            >>> models = list(benign_probs.keys())
            >>> probs = [benign_probs[m] for m in models]
            >>> benign_seq = list(rng.choice(models, size=20, p=probs/np.sum(probs)))
            >>> pl.detect(benign_seq, perturbed)
            False  # Likely not flagged (benign behavior