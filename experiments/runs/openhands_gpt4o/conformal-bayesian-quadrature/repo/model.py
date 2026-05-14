import numpy as np
from scipy.stats import dirichlet

class BayesianConformalPredictor:
    def __init__(self, alpha: float, max_loss: float):
        self.alpha = alpha
        self.max_loss = max_loss

    def fit(self, losses: np.ndarray):
        self.losses = np.sort(losses)
        self.n = len(losses)

    def compute_upper_bound(self, beta: float = 0.95):
        weights = dirichlet.rvs([1] * (self.n + 1), size=10000)
        extended_losses = np.append(self.losses, self.max_loss)
        weighted_losses = np.dot(weights, extended_losses)
        return np.percentile(weighted_losses, beta * 100)

    def predict(self, new_loss: float):
        return new_loss <= self.compute_upper_bound()