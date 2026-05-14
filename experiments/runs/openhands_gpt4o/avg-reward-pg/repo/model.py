import numpy as np
from typing import Tuple

class PolicyGradient:
    def __init__(self, state_space: int, action_space: int, learning_rate: float):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.policy = np.ones((state_space, action_space)) / action_space

    def update_policy(self, gradient: np.ndarray):
        self.policy += self.learning_rate * gradient
        self.policy = self._project_policy(self.policy)

    def _project_policy(self, policy: np.ndarray) -> np.ndarray:
        projected_policy = np.clip(policy, 0, 1)
        row_sums = projected_policy.sum(axis=1, keepdims=True)
        return projected_policy / row_sums

    def get_action_probabilities(self, state: int) -> np.ndarray:
        return self.policy[state]

    def compute_gradient(self, state: int, action: int, advantage: float) -> np.ndarray:
        gradient = np.zeros_like(self.policy)
        gradient[state, action] = advantage
        return gradient

    def save_policy(self, file_path: str):
        np.save(file_path, self.policy)

    def load_policy(self, file_path: str):
        self.policy = np.load(file_path)