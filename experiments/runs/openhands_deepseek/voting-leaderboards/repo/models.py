"""Model definitions for voting-based leaderboard manipulation.

This paper does not use neural network components. Instead, it uses:
  - Logistic regression (scikit-learn) for the training-based detector
  - Bradley-Terry model for ranking (statistical model)
  - Likelihood ratio tests and Neyman-Pearson Lemma for malicious detection

This module provides wrapper classes for these statistical models,
matching the paper's algorithms described in Sections 2-4.
"""
import numpy as np
from typing import List, Optional
from sklearn.linear_model import LogisticRegression


class TargetModelClassifier:
    """Binary classifier for target model detection (Section 2.2).

    Wraps logistic regression with the paper's default hyperparameters:
      - scikit-learn LogisticRegression with default hyperparameters
      - random_state=42
    """

    def __init__(self, random_state: int = 42):
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before scoring.")
        return self.model.score(X, y)


class BradleyTerryModel:
    """Bradley-Terry model for pairwise comparison ranking (Section 3).

    Models the probability that model i is preferred over model j:
      Pr(i preferred over j) = 1 / (1 + exp(-(Q_i - Q_j) / s))

    where Q_i, Q_j are Bradley-Terry coefficients and s is a scaling factor.

    Uses the iterative MM algorithm (Hunter, 2004) to estimate ratings.
    """

    def __init__(self, n_models: int, scale: float = 1.0):
        self.n_models = n_models
        self.scale = scale
        self.ratings = np.zeros(n_models)
        self.win_matrix = np.zeros((n_models, n_models))
        self.total_matrix = np.zeros((n_models, n_models))

    def record_match(self, winner: int, loser: int):
        """Record a win for the winner over the loser."""
        self.win_matrix[winner, loser] += 1
        self.total_matrix[winner, loser] += 1
        self.total_matrix[loser, winner] += 1

    def record_tie(self, model_a: int, model_b: int):
        """Record a tie between two models."""
        self.total_matrix[model_a, model_b] += 1
        self.total_matrix[model_b, model_a] += 1

    def fit(self, max_iter: int = 100, tol: float = 1e-6):
        """Estimate Bradley-Terry ratings using MM algorithm."""
        n = self.n_models
        for _ in range(max_iter):
            old_ratings = self.ratings.copy()
            exp_r = np.exp(self.ratings / self.scale)

            for i in range(n):
                num = self.win_matrix[i, :].sum()
                denom = 0.0
                for j in range(n):
                    if i != j and self.total_matrix[i, j] > 0:
                        denom += self.total_matrix[i, j] / (exp_r[i] + exp_r[j])
                denom *= exp_r[i]
                if denom > 0:
                    self.ratings[i] = self.scale * np.log(num / denom)

            if np.max(np.abs(self.ratings - old_ratings)) < tol:
                break

        self.ratings -= self.ratings.mean()
        return self.ratings

    def get_probability(self, model_i: int, model_j: int) -> float:
        """Probability that model i is preferred over model j."""
        diff = self.ratings[model_i] - self.ratings[model_j]
        return 1.0 / (1.0 + np.exp(-diff / self.scale))

    def get_ranking(self) -> np.ndarray:
        """Return model indices sorted by rating (highest first)."""
        return np.argsort(-self.ratings)

    def get_rank(self, model_idx: int) -> int:
        """Return 1-indexed rank of a model."""
        ranking = self.get_ranking()
        return int(np.where(ranking == model_idx)[0][0]) + 1


class NeymanPearsonDetector:
    """Optimal hypothesis test based on Neyman-Pearson Lemma (Section 4.2.3).

    Used in Scenario 2 where the defender knows both benign and
    malicious voting distributions.

    The optimal decision rule is based on the likelihood ratio:
      Lambda(x) = Pr_M(x) / Pr_B(x)
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def compute_likelihood_ratio(
        self,
        observations: List[int],
        benign_probs: np.ndarray,
        malicious_probs: np.ndarray,
    ) -> float:
        """Compute log likelihood ratio for a sequence of observations."""
        log_lr = 0.0
        n = len(benign_probs)
        for obs in observations:
            if 0 <= obs < n:
                m_prob = max(malicious_probs[obs], 1e-10)
                b_prob = max(benign_probs[obs], 1e-10)
                log_lr += np.log(m_prob / b_prob)
        return log_lr

    def classify(
        self,
        observations: List[int],
        benign_probs: np.ndarray,
        malicious_probs: np.ndarray,
    ) -> bool:
        """Classify as malicious if likelihood ratio exceeds threshold."""
        lr = self.compute_likelihood_ratio(observations, benign_probs, malicious_probs)
        return lr > self.threshold
