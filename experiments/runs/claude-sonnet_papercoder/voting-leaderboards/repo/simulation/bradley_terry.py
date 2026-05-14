```python
## simulation/bradley_terry.py
"""Bradley-Terry model for Chatbot Arena leaderboard ranking and simulation.

This module implements the Bradley-Terry model used by Chatbot Arena to rank
LLMs from pairwise comparison votes. It serves as the mathematical backbone
of the leaderboard manipulation simulation (Section 3 of the paper).

The module supports:
  1. Initial fitting from a win matrix produced by VotingDataLoader.
  2. Incremental vote accumulation and re-fitting for simulation.
  3. Win probability prediction for any model pair.
  4. Benign vote probability computation for the malicious user detection
     mitigations (Section 4.2.3).

Paper alignment:
  - Section 3.1: "Chatbot Arena ranks models using Bradley-Terry coefficients
    (Hunter, 2004) derived from user interactions."
  - Section 4.2.3: "Pr(i preferred over j) = 1 / (1 + exp(-(Q_i - Q_j) / s))"
  - Section 4.2.3: "Pr_B(i) = product over j of Pr_B(i preferred over j |
    true Bradley-Terry coefficient ratings)"
  - config.yaml: bt_scale_factor: 1.0

Dependencies:
  - choix: Bradley-Terry fitting via choix.ilsr_pairwise (MM algorithm).
  - numpy: Array operations and log-space arithmetic.
  - math: exp/log for win probability computation.
  - utils/logger.py: Centralized logging.
"""

from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

# ---------------------------------------------------------------------------
# Attempt to import choix. If unavailable, fall back to a simple MLE solver
# using scipy so the module remains functional without choix installed.
# ---------------------------------------------------------------------------
try:
    import choix  # type: ignore[import]
    _CHOIX_AVAILABLE: bool = True
except ImportError:
    _CHOIX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------
# Maximum absolute value of the exponent argument to prevent overflow/underflow
# in math.exp. Values outside [-_EXP_CLIP, _EXP_CLIP] are clipped.
_EXP_CLIP: float = 500.0

# Fallback rating assigned to models that choix cannot fit (e.g., isolated
# nodes with no wins or losses). Very negative so they rank last.
_FALLBACK_RATING: float = -1e6

# Minimum number of pairwise observations required before attempting to fit.
_MIN_PAIRWISE_OBS: int = 1

# Maximum number of iterations for the choix ILSR algorithm.
_CHOIX_MAX_ITER: int = 10000

# Convergence tolerance for the choix ILSR algorithm.
_CHOIX_ALPHA: float = 1e-8


class BradleyTerryModel:
    """Bradley-Terry model for pairwise comparison ranking.

    Fits log-strength parameters (ratings) for each model from pairwise
    comparison data using the MM algorithm (Hunter, 2004) via the choix
    library. Supports incremental vote accumulation for simulation.

    The win probability formula from Section 4.2.3 of the paper:
        Pr(A preferred over B) = 1 / (1 + exp(-(Q_A - Q_B) / scale_factor))

    where Q_A and Q_B are the fitted log-strength parameters (ratings) and
    scale_factor is the logistic scaling parameter s (default 1.0 per
    config.yaml bt_scale_factor).

    Attributes:
        scale_factor: Logistic scaling parameter s. Default 1.0 per config.yaml.
        ratings: Dict mapping model name to fitted log-strength parameter.
            Empty until fit() is called.
        model_names: Ordered list of model names establishing the index mapping
            between model names and integer indices used by choix.

    Example:
        >>> import numpy as np
        >>> bt = BradleyTerryModel(scale_factor=1.0)
        >>> win_matrix = np.array([[0, 3, 2], [1, 0, 4], [2, 0, 0]], dtype=float)
        >>> bt.fit(win_matrix, ["model_a", "model_b", "model_c"])
        >>> bt.get_rank("model_a")
        2
        >>> 0.0 < bt.predict_win_prob("model_a", "model_b") < 1.0
        True
    """

    def __init__(self, scale_factor: float = 1.0) -> None:
        """Initialize the BradleyTerryModel.

        Args:
            scale_factor: Logistic scaling parameter s in the win probability
                formula. Default 1.0 per config.yaml bt_scale_factor. A value
                of 1.0 means ratings are in log-odds space where a difference
                of 1.0 corresponds to ~73% win probability for the stronger model.

        Raises:
            ValueError: If scale_factor is not positive.

        Example:
            >>> bt = BradleyTerryModel(scale_factor=1.0)
            >>> bt.ratings
            {}
            >>> bt.model_names
            []
        """
        if scale_factor <= 0.0:
            raise ValueError(
                f"scale_factor must be positive, got {scale_factor}."
            )

        self.scale_factor: float = scale_factor

        # Fitted log-strength parameters. Empty until fit() is called.
        # Maps model_name -> float rating.
        self.ratings: Dict[str, float] = {}

        # Ordered list of model names. Establishes the index mapping between
        # model names and integer indices used by choix.ilsr_pairwise.
        self.model_names: List[str] = []

        # Accumulated pairwise comparison data as (winner_idx, loser_idx) tuples.
        # Populated during fit() from the win matrix and extended by
        # update_with_vote(). Re-fitting always uses the full accumulated history.
        self._pairwise_data: List[Tuple[int, int]] = []

        logger.info(
            "BradleyTerryModel initialized with scale_factor=%.4f.", scale_factor
        )

    # -----------------------------------------------------------------------
    # Private helper methods
    # -----------------------------------------------------------------------

    def _refit(self) -> None:
        """Re-fit the Bradley-Terry model from the accumulated pairwise data.

        Calls choix.ilsr_pairwise (or the scipy fallback) on the full
        accumulated _pairwise_data list and updates self.ratings. This is
        called by both fit() and update_with_vote().

        If choix raises a convergence error or any other exception, logs a
        warning and retains the previous ratings (if any). This ensures the
        simulation can continue even if a single re-fit fails.

        Returns:
            None. Updates self.ratings in place.
        """
        n_models: int = len(self.model_names)

        if n_models == 0:
            logger.warning("_refit: no models in model_names. Skipping.")
            return

        if len(self._pairwise_data) < _MIN_PAIRWISE_OBS:
            logger.warning(
                "_refit: insufficient pairwise observations (%d < %d). "
                "Skipping re-fit.",
                len(self._pairwise_data),
                _MIN_PAIRWISE_OBS,
            )
            return

        logger.debug(
            "_refit: fitting BT model with %d models and %d pairwise observations.",
            n_models,
            len(self._pairwise_data),
        )

        if _CHOIX_AVAILABLE:
            params: Optional[np.ndarray] = self._fit_with_choix(n_models)
        else:
            logger.warning(
                "_refit: choix not available. Using scipy MLE fallback."
            )
            params = self._fit_with_scipy_fallback(n_models)

        if params is None:
            logger.warning(
                "_refit: fitting failed. Retaining previous ratings."
            )
            return

        # Update self.ratings from the fitted parameters.
        # Handle NaN/Inf values by replacing with _FALLBACK_RATING.
        new_ratings: Dict[str, float] = {}
        for idx, model_name in enumerate(self.model_names):
            param_val: float = float(params[idx])
            if not math.isfinite(param_val):
                logger.warning(
                    "_refit: non-finite rating for model '%s' (value=%.4g). "
                    "Replacing with fallback rating %.4g.",
                    model_name,
                    param_val,
                    _FALLBACK_RATING,
                )
                param_val = _FALLBACK_RATING
            new_ratings[model_name] = param_val

        self.ratings = new_ratings

        # Log top-3 ranked models as a sanity check.
        top3: List[Tuple[str, float]] = self.get_rankings()[:3]
        logger.debug(
            "_refit complete. Top-3 models: %s.",
            [(name, f"{rating:.4f}") for name, rating in top3],
        )

    def _fit_with_choix(self, n_models: int) -> Optional[np.ndarray]:
        """Fit the BT model using choix.ilsr_pairwise.

        Args:
            n_models: Number of models (items) in the comparison.

        Returns:
            numpy array of shape (n_models,) with log-strength parameters,
            or None if fitting fails.
        """
        try:
            params: np.ndarray = choix.ilsr_pairwise(
                n_items=n_models,
                data=self._pairwise_data,
                alpha=_CHOIX_ALPHA,
                max_iter=_CHOIX_MAX_ITER,
            )
            return params
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "_fit_with_choix: choix.ilsr_pairwise failed: %s. "
                "Attempting scipy fallback.",
                exc,
            )
            # Try scipy fallback if choix fails.
            return self._fit_with_scipy_fallback(n_models)

    def _fit_with_scipy_fallback(self, n_models: int) -> Optional[np.ndarray]:
        """Fit the BT model using scipy MLE as a fallback when choix is unavailable.

        Implements the Bradley-Terry MLE via scipy.optimize.minimize using
        the negative log-likelihood of the pairwise comparison data.

        Args:
            n_models: Number of models (items) in the comparison.

        Returns:
            numpy array of shape (n_models,) with log-strength parameters,
            or None if optimization fails.
        """
        try:
            from scipy.optimize import minimize  # type: ignore[import]
        except ImportError:
            logger.error(
                "_fit_with_scipy_fallback: scipy not available. "
                "Cannot fit BT model. Returning None."
            )
            return None

        if not self._pairwise_data:
            return None

        def neg_log_likelihood(params: np.ndarray) -> float:
            """Negative log-likelihood of the BT model given pairwise data."""
            nll: float = 0.0
            for winner_idx, loser_idx in self._pairwise_data:
                # log Pr(winner beats loser) = log(exp(w) / (exp(w) + exp(l)))
                #                            = w - log(exp(w) + exp(l))
                #                            = -log(1 + exp(l - w))
                diff: float = float(params[loser_idx] - params[winner_idx])
                # Numerically stable log(1 + exp(diff)):
                if diff > 0:
                    nll += diff + math.log1p(math.exp(-diff))
                else:
                    nll += math.log1p(math.exp(diff))
            return nll

        # Initialize with zeros (uniform strength).
        x0: np.ndarray = np.zeros(n_models, dtype=np.float64)

        try:
            result = minimize(
                neg_log_likelihood,
                x0,
                method="L-BFGS-B",
                options={"maxiter": 1000, "ftol": 1e-10},
            )
            if result.success or result.fun < 1e10:
                # Center the parameters (subtract mean) for numerical stability.
                params: np.ndarray = result.x - result.x.mean()
                return params
            else:
                logger.warning(
                    "_fit_with_scipy_fallback: optimization did not converge. "
                    "Message: %s",
                    result.message,
                )
                return None
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "_fit_with_scipy_fallback: optimization failed: %s", exc
            )
            return None

    def _win_matrix_to_pairwise_data(
        self,
        win_matrix: np.ndarray,
        model_names: List[str],
    ) -> List[Tuple[int, int]]:
        """Convert a win count matrix to a list of (winner_idx, loser_idx) tuples.

        Each entry win_matrix[i][j] = k means model i beat model j k times.
        For fractional values (from tie splitting: 0.5 each), rounds to the
        nearest integer. This is a practical necessity since choix requires
        integer tuple counts.

        Args:
            win_matrix: np.ndarray of shape (N, N) where win_matrix[i][j] is
                the number of times model i beat model j. May contain fractional
                values from tie splitting (0.5 per tie).
            model_names: List of model name strings of length N. Establishes
                the mapping from integer index to model name.

        Returns:
            List of (winner_idx, loser_idx) integer tuples. The length of this
            list equals the sum of all rounded win counts in the matrix.

        Example:
            >>> bt = BradleyTerryModel()
            >>> matrix = np.array([[0, 2.5], [1.5, 0]], dtype=float)
            >>> data = bt._win_matrix_to_pairwise_data(matrix, ["a", "b"])
            >>> data.count((0, 1))  # round(2.5) = 2 or 3 depending on Python version
            2  # or 3
        """
        n_models: int = len(model_names)
        pairwise_data: List[Tuple[int, int]] = []

        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    continue
                count: float = float(win_matrix[i, j])
                if count <= 0.0:
                    continue
                # Round to nearest integer. Python's round() uses banker's
                # rounding (round half to even), which is fine for our purposes.
                int_count: int = int(round(count))
                if int_count > 0:
                    pairwise_data.extend([(i, j)] * int_count)

        return pairwise_data

    # -----------------------------------------------------------------------
    # Public interface methods
    # -----------------------------------------------------------------------

    def fit(self, win_matrix: np.ndarray, model_names: List[str]) -> None:
        """Fit the Bradley-Terry model from a win count matrix.

        Converts the win matrix to pairwise comparison data, stores the model
        name list, and calls the internal _refit() method to compute ratings.

        This is the primary initialization method, called once by Main after
        loading the Chatbot Arena voting data via VotingDataLoader.get_win_matrix().

        Args:
            win_matrix: np.ndarray of shape (N, N) where win_matrix[i][j] is
                the number of times model i beat model j. Ties are pre-split
                upstream in VotingDataLoader (0.5 added to both directions).
                Must be a square matrix with non-negative values.
            model_names: List of model name strings of length N. The order
                establishes the index mapping used internally by choix.
                Must match the model_names returned by VotingDataLoader.get_win_matrix().

        Raises:
            ValueError: If win_matrix is empty, not square, or all zeros.
            ValueError: If len(model_names) != win_matrix.shape[0].

        Example:
            >>> bt = BradleyTerryModel(scale_factor=1.0)
            >>> win_matrix = np.array([[0, 10, 5], [3, 0, 8], [7, 2, 0]], dtype=float)
            >>> bt.fit(win_matrix, ["gpt-4o", "claude-3", "llama-3"])
            >>> len(bt.ratings)
            3
            >>> bt.get_rank("gpt-4o") in [1, 2, 3]
            True
        """
        # --- Input validation ---
        if win_matrix.size == 0:
            raise ValueError(
                "fit: win_matrix is empty. Cannot fit BT model on empty data."
            )

        if win_matrix.ndim != 2 or win_matrix.shape[0] != win_matrix.shape[1]:
            raise ValueError(
                f"fit: win_matrix must be a square 2D array. "
                f"Got shape {win_matrix.shape}."
            )

        n_models: int = win_matrix.shape[0]

        if len(model_names) != n_models:
            raise ValueError(
                f"fit: len(model_names)={len(model_names)} does not match "
                f"win_matrix.shape[0]={n_models}."
            )

        if np.all(win_matrix == 0.0):
            raise ValueError(
                "fit: win_matrix is all zeros. Cannot fit BT model without "
                "any pairwise comparison data."
            )

        # --- Store model names and convert win matrix to pairwise data ---
        self.model_names = list(model_names)

        logger.info(
            "fit: converting win matrix (%d models, %.0f total comparisons) "
            "to pairwise data.",
            n_models,
            float(win_matrix.sum()),
        )

        self._pairwise_data = self._win_matrix_to_pairwise_data(
            win_matrix, model_names
        )

        logger.info(
            "fit: generated %d pairwise observations from win matrix.",
            len(self._pairwise_data),
        )

        # --- Fit the model ---
        self._refit()

        # Log summary statistics after fitting.
        if self.ratings:
            rankings: List[Tuple[str, float]] = self.get_rankings()
            top5: List[Tuple[str, float]] = rankings[:5]
            bottom3: List[Tuple[str, float]] = rankings[-3:]
            logger.info(
                "fit complete: %d models fitted. "
                "Top-5: %s. Bottom-3: %s.",
                len(self.ratings),
                [(name, f"{rating:.4f}") for name, rating in top5],
                [(name, f"{rating:.4f}") for name, rating in bottom3],
            )
        else:
            logger.warning("fit: no ratings computed after fitting.")

    def get_rankings(self) -> List[Tuple[str, float]]:
        """Return all models sorted by rating in descending order.

        The first element in the returned list is the highest-ranked model
        (rank 1). Ratings are the log-strength parameters from the fitted
        Bradley-Terry model.

        Returns:
            List of (model_name, rating) tuples sorted by rating descending.
            The first element is rank 1 (highest rated model).

        Raises:
            RuntimeError: If the model has not been fitted yet (ratings is empty).

        Example:
            >>> bt = BradleyTerryModel()
            >>> bt.fit(win_matrix, model_names)
            >>> rankings = bt.get_rankings()
            >>> rankings[0][0]  # Name of rank-1 model
            'some_model'
            >>> rankings[0][1] >= rankings[1][1]  # Descending order
            True
        """
        if not self.ratings:
            raise RuntimeError(
                "BradleyTerryModel.get_rankings(): model has not been fitted. "
                "Call fit() first."
            )

        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

    def get_rank(self, model_name: str) -> int:
        """Return the 1-indexed rank of a specific model.

        Rank 1 is the highest-rated model. Rank N is the lowest-rated model
        among N fitted models.

        Args:
            model_name: Exact model name string, e.g. "gpt-4o-2024-05-13".
                Must be present in the fitted ratings.

        Returns:
            1-indexed integer rank of the model. Returns 1 for the best model.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            KeyError: If model_name is not in the fitted ratings.

        Example:
            >>> bt = BradleyTerryModel()
            >>> bt.fit(win_matrix, ["model_a", "model_b", "model_c"])
            >>> rank = bt.get_rank("model_a")
            >>> 1 <= rank <= 3
            True
        """
        if not self.ratings:
            raise RuntimeError(
                "BradleyTerryModel.get_rank(): model has not been fitted. "
                "Call fit() first."
            )

        if model_name not in self.ratings:
            raise KeyError(
                f"BradleyTerryModel.get_rank(): model '{model_name}' not found "
                f"in fitted ratings. Available models: "
                f"{sorted(self.ratings.keys())[:10]}..."
            )

        # Sort all models by rating descending and find the position of model_name.
        rankings: List[Tuple[str, float]] = self.get_rankings()
        for rank_idx, (name, _) in enumerate(rankings):
            if name == model_name:
                return rank_idx + 1  # Convert 0-indexed to 1-indexed.

        # This should never be reached if model_name is in self.ratings.
        raise RuntimeError(
            f"BradleyTerryModel.get_rank(): model '{model_name}' found in "
            f"ratings dict but not in sorted rankings. This is a bug."
        )

    def predict_win_prob(self, model_a: str, model_b: str) -> float:
        """Compute the probability that model A beats model B.

        Uses the Bradley-Terry logistic formula from Section 4.2.3 of the paper:
            Pr(A preferred over B) = 1 / (1 + exp(-(Q_A - Q_B) / scale_factor))

        Args:
            model_a: Name of the first model (the one whose win probability
                is being computed).
            model_b: Name of the second model.

        Returns:
            Float in (0.0, 1.0) representing the probability that model_a
            beats model_b in a pairwise comparison. Returns 0.5 if both
            models have equal ratings.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            KeyError: If either model_a or model_b is not in the fitted ratings.

        Example:
            >>> bt = BradleyTerryModel()
            >>> bt.fit(win_matrix, model_names)
            >>> prob = bt.predict_win_prob("gpt-4o", "llama-3")
            >>> 0.0 < prob < 1.0
            True
            >>> bt.predict_win_prob("m", "m")  # Same model
            0.5
        """
        if not self.ratings:
            raise RuntimeError(
                "BradleyTerryModel.predict_win_prob(): model has not been "
                "fitted. Call fit() first."
            )

        if model_a not in self.ratings:
            raise KeyError(
                f"BradleyTerryModel.predict_win_prob(): model_a='{model_a}' "
                f"not found in fitted ratings."
            )

        if model_b not in self.ratings:
            raise KeyError(
                f"BradleyTerryModel.predict_win_prob(): model_b='{model_b}' "
                f"not found in fitted ratings."
            )

        Q_a: float = self.ratings[model_a]
        Q_b: float = self.ratings[model_b]

        # Compute the exponent argument with numerical clipping to prevent
        # overflow/underflow in math.exp.
        exponent: float = -(Q_a - Q_b) / self.scale_factor
        exponent_clipped: float = max(-_EXP_CLIP, min(_EXP_CLIP, exponent))

        win_prob: float = 1.0 / (1.0 + math.exp(exponent_clipped))
        return win_prob

    def update_with_vote(self, winner: str, loser: str) -> None:
        """Incorporate a new vote and re-fit the Bradley-Terry model.

        Appends the new (winner_idx, loser_idx) pair to the accumulated
        pairwise data and calls _refit() to update the ratings. This method
        is called by AttackSimulator at each eval_interval boundary (every
        1000 interactions) to incorporate batched adversarial votes.

        Per the Shared Knowledge spec: "The BT model is re-fit from scratch
        every eval_interval=1000 interactions during simulation (not
        incrementally updated) to ensure numerical stability." The AttackSimulator
        controls the call cadence — this method always re-fits when called.

        Args:
            winner: Name of the model that won the comparison. Must be in
                self.model_names (i.e., must have been present in the original
                win matrix passed to fit()).
            loser: Name of the model that lost the comparison. Must be in
                self.model_names.

        Raises:
            RuntimeError: If fit() has not been called yet (model_names is empty).
            ValueError: If winner or loser is not in self.model_names.

        Example:
            >>> bt = BradleyTerryModel()
            >>> bt.fit(win_matrix, model_names)
            >>> old_rank = bt.get_rank("llama-13b")
            >>> bt.update_with_vote("llama-13b", "some_other_model")
            >>> new_rank = bt.get_rank("llama-13b")
            >>> new_rank <= old_rank  # Rank improved or stayed same
            True
        """
        if not self.model_names:
            raise RuntimeError(
                "BradleyTerryModel.update_with_vote(): model has not been "
                "fitted. Call fit() first."
            )

        if winner not in self.ratings:
            raise ValueError(
                f"BradleyTerryModel.update_with_vote(): winner='{winner}' "
                f"not found in model_names. Available models: "
                f"{sorted(self.model_names)[:10]}..."
            )

        if loser not in self.ratings:
            raise ValueError(
                f"BradleyTerryModel.update_with_vote(): loser='{loser}' "
                f"not found in model_names. Available models: "
                f"{sorted(self.model_names)[:10]}..."
            )

        # Convert model names to integer indices for choix.
        winner_idx: int = self.model_names.index(winner)
        loser_idx: int = self.model_names.index(loser)

        # Append the new vote to the accumulated pairwise data.
        self._pairwise_data.append((winner_idx, loser_idx))

        logger.debug(
            "update_with_vote: appended (%s[%d], %s[%d]). "
            "Total pairwise observations: %d.",
            winner,
            winner_idx,
            loser,
            loser_idx,
            len(self._pairwise_data),
        )

        # Re-fit the model with the updated data.
        self._refit()

    def compute_benign_vote_probs(self) -> Dict[str, float]:
        """Compute the probability that a benign user votes for each model.

        Implements the formula from Section 4.2.3 of the paper:
            Pr_B(i) = product over j≠i of Pr(i preferred over j | true BT ratings)

        Then normalizes so all probabilities sum to 1.0. This represents the
        probability that a benign user's vote goes to model i (i.e., model i
        wins its matchup against the other model in the comparison).

        Uses log-space computation to prevent numerical underflow when
        multiplying many small probabilities together (with 129+ models,
        the product of 128 probabilities can easily underflow to 0).

        Returns:
            Dict mapping model name strings to normalized probability values.
            All values are in (0.0, 1.0) and sum to 1.0. Higher-ranked models
            (with higher BT ratings) have higher benign vote probabilities.

        Raises:
            RuntimeError: If the model has not been fitted yet.

        Example:
            >>> bt = BradleyTerryModel()
            >>> bt.fit(win_matrix, model_names)
            >>> probs = bt.compute_benign_vote_probs()
            >>> abs(sum(probs.values()) - 1.0) < 1e-9
            True
            >>> all(0.0 < p < 1.0 for p in probs.values())
            True
        """
        if not self.ratings:
            raise RuntimeError(
                "BradleyTerryModel.compute_benign_vote_probs(): model has not "
                "been fitted. Call fit() first."
            )

        n_models: int = len(self.model_names)

        if n_models == 0:
            logger.warning(
                "compute_benign_vote_probs: no models in model_names. "
                "Returning empty dict."
            )
            return {}

        if n_models == 1:
            # Trivial case: only one model, probability = 1.0.
            return {self.model_names[0]: 1.0}

        logger.debug(
            "compute_benign_vote_probs: computing log-space product formula "
            "for %d models.",
            n_models,
        )

        # --- Log-space computation to prevent underflow ---
        # For each model i, compute:
        #   log_prob_i = sum over j≠i of log(Pr(i preferred over j))
        # This is equivalent to log(product over j≠i of Pr(i preferred over j)).
        log_probs: Dict[str, float] = {}

        for model_i in self.model_names:
            log_prob_i: float = 0.0
            for model_j in self.model_names:
                if model_i == model_j:
                    continue
                # Pr(i preferred over j) from the BT logistic formula.
                win_prob_ij: float = self.predict_win_prob(model_i, model_j)

                # Guard against log(0) — should not happen with finite ratings,
                # but clip to a small positive value for numerical safety.
                win_prob_ij = max(win_prob_ij, 1e-300)
                log_prob_i += math.log(win_prob_ij)

            log_probs[model_i] = log_prob_i

        # --- Log-sum-exp normalization to prevent overflow/underflow ---
        # Subtract the maximum log-prob before exponentiating.
        max_log_prob: float = max(log_probs.values())

        raw_probs: Dict[str, float] = {}
        for model_name, log_prob