"""
Prior reward distributions for FRE training.

Three families of random unsupervised reward functions:
1. Singleton goal-reaching rewards
2. Random linear functions
3. Random MLP functions

The default FRE-all prior uses an equal mixture (1/3 each).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class GoalReachingReward:
    """
    Singleton goal-reaching reward.

    Reward is -1 for every timestep the goal is not reached, 0 when reached.
    Goal is sampled from the offline dataset.

    For AntMaze: goal is considered reached if within distance 2 of target XY.
    For ExORL: goal is reached if Euclidean distance < 0.1 (normalized by std).
    """

    def __init__(self, goal_state: np.ndarray, goal_threshold: float = 2.0,
                 state_std: Optional[np.ndarray] = None, xy_only: bool = False):
        """
        Args:
            goal_state: the goal state vector
            goal_threshold: distance threshold for goal achievement
            state_std: per-dimension std for normalization (ExORL)
            xy_only: if True, only use XY coordinates for distance (AntMaze)
        """
        self.goal_state = goal_state
        self.goal_threshold = goal_threshold
        self.state_std = state_std
        self.xy_only = xy_only

    def __call__(self, states: np.ndarray) -> np.ndarray:
        """
        Args:
            states: (N, state_dim) array of states
        Returns:
            rewards: (N,) array of rewards in {-1, 0}
        """
        if self.xy_only:
            diff = states[:, :2] - self.goal_state[:2]
        else:
            diff = states - self.goal_state
            if self.state_std is not None:
                diff = diff / (self.state_std + 1e-8)

        dist = np.linalg.norm(diff, axis=-1)
        rewards = np.where(dist < self.goal_threshold, 0.0, -1.0)
        return rewards


class RandomLinearReward:
    """
    Random linear reward function: r(s) = w^T s

    w is sampled uniformly from [-1, 1] with a sparse binary mask
    (each dimension zeroed with probability 0.9).

    For AntMaze, XY positions are excluded from the linear function
    to avoid instability from scale differences.
    """

    def __init__(self, state_dim: int, exclude_xy: bool = False, sparsity: float = 0.9,
                 rng: Optional[np.random.Generator] = None):
        if rng is None:
            rng = np.random.default_rng()
        self.w = rng.uniform(-1.0, 1.0, size=(state_dim,))
        mask = (rng.uniform(0, 1, size=(state_dim,)) > sparsity).astype(float)
        self.w = self.w * mask
        if exclude_xy:
            self.w[:2] = 0.0

    def __call__(self, states: np.ndarray) -> np.ndarray:
        """
        Args:
            states: (N, state_dim)
        Returns:
            rewards: (N,)
        """
        return states @ self.w


class RandomMLPReward:
    """
    Random MLP reward function.

    Architecture: (state_dim -> 32 -> 1) with tanh activation.
    Parameters sampled from N(0, scale) where scale = sqrt(2 / (in + out)).
    Output clipped to [-1, 1].
    """

    def __init__(self, state_dim: int, hidden_dim: int = 32,
                 rng: Optional[np.random.Generator] = None):
        if rng is None:
            rng = np.random.default_rng()

        # Layer 1: state_dim -> hidden_dim
        scale1 = np.sqrt(2.0 / (state_dim + hidden_dim))
        self.w1 = rng.normal(0, scale1, size=(state_dim, hidden_dim))
        self.b1 = rng.normal(0, scale1, size=(hidden_dim,))

        # Layer 2: hidden_dim -> 1
        scale2 = np.sqrt(2.0 / (hidden_dim + 1))
        self.w2 = rng.normal(0, scale2, size=(hidden_dim, 1))
        self.b2 = rng.normal(0, scale2, size=(1,))

    def __call__(self, states: np.ndarray) -> np.ndarray:
        """
        Args:
            states: (N, state_dim)
        Returns:
            rewards: (N,) clipped to [-1, 1]
        """
        h = np.tanh(states @ self.w1 + self.b1)
        out = h @ self.w2 + self.b2
        return np.clip(out.squeeze(-1), -1.0, 1.0)


class RewardPrior:
    """
    Mixture prior over reward functions.

    Supports the following named priors:
    - 'all': equal mix of goal-reaching, linear, MLP (FRE-all)
    - 'goals': only goal-reaching (FRE-goals)
    - 'lin': only linear (FRE-lin)
    - 'mlp': only MLP (FRE-mlp)
    - 'lin-mlp': equal mix of linear and MLP (FRE-lin-mlp)
    - 'goal-mlp': equal mix of goal-reaching and MLP (FRE-goal-mlp)
    - 'goal-lin': equal mix of goal-reaching and linear (FRE-goal-lin)
    - 'hint': domain-specific hint prior (FRE-hint)
    """

    PRIOR_WEIGHTS = {
        'all':      {'goal': 1/3, 'lin': 1/3, 'mlp': 1/3},
        'goals':    {'goal': 1.0, 'lin': 0.0, 'mlp': 0.0},
        'lin':      {'goal': 0.0, 'lin': 1.0, 'mlp': 0.0},
        'mlp':      {'goal': 0.0, 'lin': 0.0, 'mlp': 1.0},
        'lin-mlp':  {'goal': 0.0, 'lin': 0.5, 'mlp': 0.5},
        'goal-mlp': {'goal': 0.5, 'lin': 0.0, 'mlp': 0.5},
        'goal-lin': {'goal': 0.5, 'lin': 0.5, 'mlp': 0.0},
    }

    def __init__(
        self,
        prior_name: str,
        state_dim: int,
        dataset_states: np.ndarray,
        exclude_xy: bool = False,
        goal_threshold: float = 2.0,
        state_std: Optional[np.ndarray] = None,
        xy_only_goal: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        self.prior_name = prior_name
        self.state_dim = state_dim
        self.dataset_states = dataset_states
        self.exclude_xy = exclude_xy
        self.goal_threshold = goal_threshold
        self.state_std = state_std
        self.xy_only_goal = xy_only_goal
        self.rng = rng if rng is not None else np.random.default_rng()

        if prior_name not in self.PRIOR_WEIGHTS and prior_name != 'hint':
            raise ValueError(f"Unknown prior: {prior_name}. Choose from {list(self.PRIOR_WEIGHTS)}")

        self.weights = self.PRIOR_WEIGHTS.get(prior_name, self.PRIOR_WEIGHTS['all'])

    def sample(self):
        """Sample a random reward function from the prior."""
        choices = [k for k, v in self.weights.items() if v > 0]
        probs = [self.weights[k] for k in choices]
        choice = self.rng.choice(choices, p=probs)

        if choice == 'goal':
            idx = self.rng.integers(0, len(self.dataset_states))
            goal = self.dataset_states[idx]
            return GoalReachingReward(
                goal, self.goal_threshold, self.state_std, self.xy_only_goal
            )
        elif choice == 'lin':
            return RandomLinearReward(self.state_dim, self.exclude_xy, rng=self.rng)
        else:
            return RandomMLPReward(self.state_dim, rng=self.rng)

    def normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """
        Normalize rewards to [0, 1] for discretization.

        Rescales by shifting min to 0 and dividing by range.
        """
        r_min = rewards.min()
        r_max = rewards.max()
        if r_max - r_min < 1e-8:
            return np.zeros_like(rewards)
        return (rewards - r_min) / (r_max - r_min)
