"""
mitigation.py – Malicious user detection experiments (Section 4.3 of the paper).

Implements the :class:`MitigationSimulator` class, which uses statistical hypothesis
testing to distinguish adversarial voters from benign voters in a voting‑based
leaderboard.  Two scenarios are covered:

  1. **Known benign distribution** – naive attacker that consistently votes for a
     target model; detection via a likelihood test.
  2. **Known benign and malicious distributions** – the defender releases a perturbed
     version of the leaderboard; a likelihood‑ratio test detects smart attackers who
     mimic the perturbed distribution.

The module relies on the global configuration (from :mod:`config`) and on model
ratings obtained after the genuine‑vote simulation (from :mod:`simulation`).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Forward reference to Config type (avoid circular import at runtime).
from config import Config
from utils import set_all_seeds

logger = logging.getLogger(__name__)


class MitigationSimulator:
    """
    Simulates benign and adversarial voting behaviour and computes detection metrics
    for different noise scales and attacker strategies.

    Attributes:
        config:               The global configuration object.
        ratings:              True Bradley‑Terry ratings for all models.
        model_list:           Sorted list of model names (used for consistent indexing).
        p_benign:             1‑D array of marginal vote probabilities for a benign user,
                              computed from the true ratings via the product formula.
        current_noise_scale:  Standard deviation of Gaussian noise added to the ratings
                              when releasing a perturbed leaderboard.  Set by
                              :meth:`run_detection_experiment` before each
                              :meth:`simulate_attacker` call.
    """

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def __init__(self, config: Config, ratings: Dict[str, float]) -> None:
        """
        Initialise the simulation with configuration and true model ratings.

        Args:
            config:  The application configuration (must contain mitigation section).
            ratings: A dictionary mapping model identifiers to their true Bradley‑Terry
                     ratings (as obtained from the :class:`Leaderboard`).
        """
        if not isinstance(config, Config):
            raise TypeError("config must be an instance of Config")
        if not isinstance(ratings, dict):
            raise TypeError("ratings must be a dictionary")
        if len(ratings) == 0:
            raise ValueError("ratings dictionary cannot be empty")

        self.config: Config = config
        self.ratings: Dict[str, float] = ratings.copy()

        # ---------- Build sorted model list ----------
        self.model_list: List[str] = sorted(ratings.keys(), key=lambda x: x)

        # ---------- Compute benign marginal probabilities ----------
        self.p_benign: np.ndarray = self._compute_model_probs(self.ratings)

        # Placeholder for the noise scale used during the current iteration
        self.current_noise_scale: Optional[float] = None

        logger.info(
            "MitigationSimulator initialised: %d models, benign prob sum=%.4f",
            len(self.model_list), self.p_benign.sum(),
        )

    # ------------------------------------------------------------------
    # Static helper – product formula for marginal vote probability
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_model_probs(ratings_dict: Dict[str, float]) -> np.ndarray:
        """
        Compute approximate marginal vote probabilities using the product formula
        described in the paper (Section 4.2.3).

        For each model *i*, the unnormalised probability is

            P(i) ∝ ∏_{j ≠ i} logistic(r_i - r_j)

        where logistic(d) = 1 / (1 + exp(-d)) and the scale factor *s* is 1.

        The calculation is performed in log‑space to avoid underflow.

        Args:
            ratings_dict: Mapping from model name to rating.

        Returns:
            A 1‑D numpy array of normalised probabilities, indexed in the same order
            as ``sorted(ratings_dict.keys())``.
        """
        # Order models consistently
        model_names = sorted(ratings_dict.keys())
        n = len(model_names)
        ratings = np.array([ratings_dict[m] for m in model_names], dtype=float)

        # Pre‑compute logistic log‑probabilities : log(p_ij) where p_ij = P(i beats j)
        # Use the identity: log(1 / (1+exp(-d))) = -log(1 + exp(-d))
        # For numerical stability, if d >= 0: log_logistic(d) = -log(1 + exp(-d))
        #                    if d < 0 : log_logistic(d) = d - log(1 + exp(d))
        def log_logistic(d: float) -> float:
            if d >= 0:
                return -math.log1p(math.exp(-d))
            else:
                # d is negative, so d + log(1+exp(d)) = d - log(1+exp(- (-d) ))? Let's check:
                # log(1/(1+exp(-d))) = -log(1+exp(-d))
                # If d < 0, exp(-d) > 1, so -log(1+exp(-d)) is large negative.
                # Using identity: -log(1+exp(-d)) = d - log(1+exp(d)). Because d < 0, d and log(1+exp(d)) are both well-behaved.
                return d - math.log1p(math.exp(d))

        log_probs = np.zeros(n, dtype=float)

        for i in range(n):
            log_p_i = 0.0
            for j in range(n):
                if i == j:
                    continue
                diff = ratings[i] - ratings[j]
                log_p_i += log_logistic(diff)
            log_probs[i] = log_p_i

        # Convert to linear scale and normalise
        # Use np.exp with clipping to avoid overflow
        max_log = log_probs.max()
        if max_log == -np.inf:
            # All probabilities are zero (should not happen with realistic ratings)
            raise ValueError("Computed probabilities are all zero – check ratings.")

        exp_probs = np.exp(log_probs - max_log)  # subtract max for numerical safety
        probs = exp_probs / exp_probs.sum()
        return probs

    # ------------------------------------------------------------------
    # Benign user simulation
    # ------------------------------------------------------------------
    def simulate_benign_user(self, num_votes: int) -> List[str]:
        """
        Generate a sequence of votes that mimics a benign user.

        The votes are independent draws from the marginal distribution *p_benign*,
        which represents the overall probability that a benign user votes for each
        model in a single impression.

        Args:
            num_votes: Number of votes to generate.

        Returns:
            A list of model names (length *num_votes*) that the benign user voted for.
        """
        # Use a fresh random state for reproducibility (global seed already set)
        rng = np.random.RandomState(self.config.seed + 1)
        indices = rng.choice(
            len(self.model_list), size=num_votes, p=self.p_benign
        )
        return [self.model_list[i] for i in indices]

    # ------------------------------------------------------------------
    # Attacker simulation
    # ------------------------------------------------------------------
    def simulate_attacker(
        self,
        strategy: str,
        num_votes: int,
        target_model: Optional[str] = None,
    ) -> List[str]:
        """
        Generate a sequence of votes for an adversarial user.

        Two strategies are supported:

        * ``"naive"`` – the attacker uniformly samples two models; if the target is
          present, they always vote for it; otherwise they pick one randomly.
        * ``"smart"`` – the attacker uses the *perturbed* ratings implied by the
          current :attr:`current_noise_scale` to compute a vote distribution that
          mimics the released leaderboard, then draws votes i.i.d. from that distribution.

        Args:
            strategy:     ``"naive"`` or ``"smart"``.
            num_votes:    Number of votes to simulate.
            target_model: (Required for ``"naive"``) The model identifier that the
                          attacker wants to boost.

        Returns:
            A list of model names voted for.

        Raises:
            ValueError: if *strategy* is unknown or required parameters are missing.
        """
        if strategy not in ("naive", "smart"):
            raise ValueError(f"Unknown attacker strategy: '{strategy}'")

        rng = np.random.RandomState(
            self.config.seed + 2 + hash(strategy) % 1000
        )

        if strategy == "naive":
            if target_model is None:
                raise ValueError(
                    "'target_model' is required for 'naive' strategy"
                )
            # Ensure target is known
            if target_model not in self.model_list:
                raise ValueError(
                    f"Target model '{target_model}' not in model list"
                )

            votes: List[str] = []
            for _ in range(num_votes):
                # sample two distinct models uniformly
                pair = rng.choice(
                    self.model_list, size=2, replace=False
                ).tolist()
                if target_model in pair:
                    votes.append(target_model)
                else:
                    # random 50/50 between the two
                    chosen = pair[rng.randint(0, 2)]
                    votes.append(chosen)
            return votes

        else:  # "smart"
            # Generate a perturbed rating for each model using the current noise scale
            noise_scale = self.current_noise_scale
            if noise_scale is None:
                raise RuntimeError(
                    "'current_noise_scale' is not set. "
                    "Call run_detection_experiment() which sets it automatically."
                )

            # Use a seed that depends on noise scale to allow reproducibility of the
            # perturbed distribution across multiple calls.
            noise_rng = np.random.RandomState(
                self.config.seed + 100 + int(noise_scale * 100)
            )
            perturbed: Dict[str, float] = {}
            for model, rating in self.ratings.items():
                perturbed[model] = rating + noise_rng.normal(0, noise_scale)

            p_perturbed = self._compute_model_probs(perturbed)

            # sample i.i.d. votes from the perturbed distribution
            indices = rng.choice(
                len(self.model_list), size=num_votes, p=p_perturbed
            )
            return [self.model_list[i] for i in indices]

    # ------------------------------------------------------------------
    # Hypothesis test
    # ------------------------------------------------------------------
    def likelihood_test(
        self,
        user_votes: List[str],
        benign_probs: np.ndarray,
        alternative_probs: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute the empirical *p*‑value for the hypothesis that *user_votes* were
        drawn from the benign distribution.

        Two modes of operation:

        1. **Without** *alternative_probs* – uses the test statistic
           *T(x) = −2 Σ log(benign_prob(x_i))*.  Larger values indicate less
           compatibility with the benign distribution.

        2. **With** *alternative_probs* – uses the log‑likelihood ratio
           *log Λ(x) = Σ log(alt_prob(x_i)) − Σ log(benign_prob(x_i))*.
           Larger log‑ratios favour the alternative.

        In both cases the null distribution of the test statistic is obtained by
        simulating *num_null_simulations* sequences of the same length under the
        null hypothesis (benign).  The *p*‑value is defined as
        *(#null ≥ observed) / (num_simulations + 1)*.

        Args:
            user_votes:       Sequence of voted model names from a user.
            benign_probs:     Probability array for the benign distribution (length M).
            alternative_probs: Probability array for the alternative distribution.
                If ``None``, the simple negative‑log‑likelihood test is used.

        Returns:
            A *p*‑value between 0 and 1.  A small value (e.g., < 0.01) suggests
            the user is not benign.

        Raises:
            ValueError: If any vote is not in the model list.
        """
        M = len(self.model_list)
        if benign_probs.shape != (M,):
            raise ValueError(
                f"benign_probs must have length {M}, got {benign_probs.shape}"
            )

        # ---------- Map votes to indices ----------
        try:
            obs_idx = np.array(
                [self.model_list.index(v) for v in user_votes], dtype=int
            )
        except ValueError as exc:
            raise ValueError(
                "user_votes contains a model not present in model_list"
            ) from exc

        # ---------- Observed statistic ----------
        # Compute in log space for numerical stability
        eps = 1e-15  # avoid log(0)
        safe_benign = np.clip(benign_probs, eps, None)
        log_benign = np.log(safe_benign)
        observed_ll_benign = np.sum(log_benign[obs_idx])

        if alternative_probs is not None:
            if alternative_probs.shape != (M,):
                raise ValueError(
                    f"alternative_probs must have length {M}"
                )
            safe_alt = np.clip(alternative_probs, eps, None)
            log_alt = np.log(safe_alt)
            observed_ll_alt = np.sum(log_alt[obs_idx])
            # test statistic: log‑likelihood ratio
            obs_stat = observed_ll_alt - observed_ll_benign
        else:
            # test statistic: -2 * log‑likelihood under benign
            obs_stat = -2.0 * observed_ll_benign

        # ---------- Null distribution ----------
        num_null = self.config.mitigation.get("num_null_simulations", 1000)
        # Seed for null sampling, keep separate from simulation randomness
        null_rng = np.random.RandomState(self.config.seed + 300)
        N = len(user_votes)

        null_stats = np.empty(num_null, dtype=float)
        for sim in range(num_null):
            # generate a null sequence of the same length
            null_indices = null_rng.choice(M, size=N, p=benign_probs)
            null_ll_benign = np.sum(log_benign[null_indices])

            if alternative_probs is not None:
                null_ll_alt = np.sum(log_alt[null_indices])
                null_stats[sim] = null_ll_alt - null_ll_benign
            else:
                null_stats[sim] = -2.0 * null_ll_benign

        # ---------- p‑value ----------
        # Add 1 to numerator and denominator for a one‑sided test as in the paper
        greater_count = np.sum(null_stats >= obs_stat)
        p_value = (greater_count + 1) / (num_null + 1)

        logger.debug(
            "likelihood_test: obs_stat=%.2f, p=%.4f (Nnull=%d)",
            obs_stat, p_value, num_null,
        )
        return float(p_value)

    # ------------------------------------------------------------------
    # Full detection experiment
    # ------------------------------------------------------------------
    def run_detection_experiment(self) -> pd.DataFrame:
        """
        Execute the complete malicious‑user detection experiment across multiple
        noise scales and attacker strategies.

        For each combination:
        - The attacker vote sequence is simulated multiple times (controlled by
          the optional ``num_trials`` configuration key, default 100).
        - A hypothesis test is performed to decide whether the sequence is benign.
        - The detection rate is the fraction of trials where the attacker is detected.
        - The utility metric is the average absolute rank change between the true
          leaderboard and the perturbed leaderboard.

        Returns:
            A DataFrame with columns:
            ``noise_scale``, ``strategy``, ``detection_rate``, ``avg_rank_change``.
        """
        # Number of repeated attacker simulations to obtain stable detection rate.
        num_trials = self.config.mitigation.get("num_trials", 100)
        noise_scales = self.config.mitigation.get("noise_scales", [0, 10, 20, 50])
        strategies = self.config.mitigation.get("strategies", ["naive", "smart"])
        significance_level = self.config.mitigation.get(
            "significance_level", 0.01
        )
        num_attacker_votes = self.config.mitigation.get(
            "num_attacker_votes", 100
        )
        # Pick a default target model for naive attacker if not provided.
        default_target = self.config.mitigation.get(
            "target_model", self.model_list[0]
        )

        rows: List[Dict[str, Any]] = []

        for noise in noise_scales:
            self.current_noise_scale = float(noise)

            # --- Pre‑compute perturbed ratings once per noise level ---
            # Use a deterministic seed based on the noise scale.
            pert_rng = np.random.RandomState(
                self.config.seed + 200 + int(noise * 100)
            )
            perturbed_ratings: Dict[str, float] = {}
            for model, rating in self.ratings.items():
                perturbed_ratings[model] = rating + pert_rng.normal(0, noise)

            p_perturbed = self._compute_model_probs(perturbed_ratings)

            # --- Utility: average absolute rank change ---
            true_ranks = {
                m: i
                for i, m in enumerate(
                    sorted(self.ratings, key=lambda x: self.ratings[x], reverse=True)
                )
            }
            pert_ranks = {
                m: i
                for i, m in enumerate(
                    sorted(
                        perturbed_ratings,
                        key=lambda x: perturbed_ratings[x],
                        reverse=True,
                    )
                )
            }
            rank_differences = [
                abs(true_ranks[m] - pert_ranks[m]) for m in self.model_list
            ]
            avg_rank_change = float(np.mean(rank_differences))

            # --- Evaluate each strategy ---
            for strat in strategies:
                # seed for reproducibility of the detection experiment
                detection_count = 0
                for trial in range(num_trials):
                    # Per‑trial seed (combination of global seed + noise + strategy + trial)
                    trial_seed = (
                        self.config.seed + 400 + int(noise * 100) + hash(strat) % 500 + trial
                    )
                    np.random.seed(trial_seed)

                    if strat == "naive":
                        attacker_votes = self.simulate_attacker(
                            strategy="naive",
                            num_votes=num_attacker_votes,
                            target_model=default_target,
                        )
                        p_value = self.likelihood_test(
                            attacker_votes, self.p_benign
                        )
                    else:  # "smart"
                        # The smart attacker uses the perturbed distribution.
                        # Inside simulate_attacker, self.current_noise_scale is already set.
                        attacker_votes = self.simulate_attacker(
                            strategy="smart", num_votes=num_attacker_votes
                        )
                        p_value = self.likelihood_test(
                            attacker_votes,
                            self.p_benign,
                            alternative_probs=p_perturbed,
                        )

                    if p_value < significance_level:
                        detection_count += 1

                detection_rate = detection_count / num_trials
                rows.append(
                    {
                        "noise_scale": noise,
                        "strategy": strat,
                        "detection_rate": detection_rate,
                        "avg_rank_change": avg_rank_change,
                    }
                )
                logger.info(
                    "Noise=%.1f, strategy=%s, detection_rate=%.3f, avg_rank_change=%.2f",
                    noise, strat, detection_rate, avg_rank_change,
                )

        # restore global seed after experiment (optional)
        set_all_seeds(self.config.seed)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<MitigationSimulator: models={len(self.model_list)}, "
            f"benign_sum={self.p_benign.sum():.4f}>"
        )
