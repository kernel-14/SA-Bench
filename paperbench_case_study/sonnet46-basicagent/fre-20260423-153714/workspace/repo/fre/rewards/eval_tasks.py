"""
Evaluation task reward functions for FRE benchmarks.

Covers:
- AntMaze: goal-reaching, directional, random-simplex, path tasks
- ExORL (cheetah/walker): velocity tasks, goal-reaching tasks
- Kitchen: sparse subtask rewards
"""

import numpy as np
from typing import Optional


# ============================================================
# AntMaze evaluation tasks
# ============================================================

# Goal locations on (X, Y) grid with origin at bottom-left
ANTMAZE_GOALS = {
    'goal-bottom':  np.array([28.0, 0.0]),
    'goal-left':    np.array([0.0, 15.0]),
    'goal-top':     np.array([35.0, 24.0]),
    'goal-center':  np.array([12.0, 24.0]),
    'goal-right':   np.array([33.0, 16.0]),
}

# Directional velocity targets (unit vectors in XY plane)
ANTMAZE_DIRECTIONS = {
    'vel_left':  np.array([-1.0, 0.0]),
    'vel_up':    np.array([0.0, 1.0]),
    'vel_down':  np.array([0.0, -1.0]),
    'vel_right': np.array([1.0, 0.0]),
}


class AntGoalReachingReward:
    """
    Reward -1 per timestep until goal is reached (within distance 2).
    Uses XY coordinates only.
    """

    def __init__(self, goal_xy: np.ndarray, threshold: float = 2.0):
        self.goal_xy = goal_xy
        self.threshold = threshold

    def __call__(self, states: np.ndarray) -> np.ndarray:
        """states: (N, state_dim), first two dims are XY."""
        dist = np.linalg.norm(states[:, :2] - self.goal_xy, axis=-1)
        return np.where(dist < self.threshold, 0.0, -1.0)


class AntDirectionalReward:
    """
    Reward based on dot product of agent velocity with target direction.
    Velocity is in the XY plane.
    """

    def __init__(self, target_velocity: np.ndarray):
        self.target_velocity = target_velocity

    def __call__(self, states: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        """
        Args:
            states: (N, state_dim) - not used directly
            velocities: (N, 2) XY velocities
        Returns:
            rewards: (N,) dot product with target direction
        """
        return velocities @ self.target_velocity


class AntRandomSimplexReward:
    """
    Random 2D noise reward using opensimplex.

    Assigns:
    - baseline -1 at each step
    - bonus for standing in higher "height" regions
    - additional bonus for moving in local preferred velocity direction
    """

    def __init__(self, seed: int = 1):
        try:
            from opensimplex import OpenSimplex
            self.noise = OpenSimplex(seed=seed)
        except ImportError:
            self.noise = None
            self.seed = seed
        self.seed = seed

    def height(self, x: float, y: float) -> float:
        if self.noise is not None:
            return self.noise.noise2(x * 0.05, y * 0.05)
        # Fallback: simple sine-based noise
        return np.sin(x * 0.05 + self.seed) * np.cos(y * 0.05 + self.seed)

    def preferred_velocity(self, x: float, y: float) -> np.ndarray:
        """Gradient of the height field gives preferred velocity direction."""
        eps = 0.1
        dx = (self.height(x + eps, y) - self.height(x - eps, y)) / (2 * eps)
        dy = (self.height(x, y + eps) - self.height(x, y - eps)) / (2 * eps)
        return np.array([dx, dy])

    def __call__(self, states: np.ndarray, velocities: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            states: (N, state_dim), first two dims are XY
            velocities: (N, 2) optional XY velocities
        Returns:
            rewards: (N,)
        """
        N = len(states)
        rewards = np.full(N, -1.0)
        for i in range(N):
            x, y = states[i, 0], states[i, 1]
            h = self.height(x, y)
            rewards[i] += h  # height bonus
            if velocities is not None:
                pv = self.preferred_velocity(x, y)
                pv_norm = np.linalg.norm(pv)
                if pv_norm > 1e-8:
                    pv = pv / pv_norm
                rewards[i] += np.dot(velocities[i], pv)
        return rewards


class AntPathReward:
    """
    Reward for moving along a hand-crafted corridor/path.

    Assigns reward proportional to how close the agent is to the path.
    """

    # Predefined path waypoints for each path type
    PATH_WAYPOINTS = {
        'center': [
            (12, 0), (12, 8), (12, 16), (12, 24),
        ],
        'loop': [
            (0, 0), (12, 0), (24, 0), (35, 0),
            (35, 8), (35, 16), (35, 24),
            (24, 24), (12, 24), (0, 24),
            (0, 16), (0, 8), (0, 0),
        ],
        'edges': [
            (0, 0), (35, 0), (35, 24), (0, 24), (0, 0),
        ],
    }

    def __init__(self, path_name: str, threshold: float = 3.0):
        self.waypoints = np.array(self.PATH_WAYPOINTS[path_name], dtype=float)
        self.threshold = threshold

    def __call__(self, states: np.ndarray) -> np.ndarray:
        """
        Args:
            states: (N, state_dim), first two dims are XY
        Returns:
            rewards: (N,) - 1 if near path, -1 otherwise
        """
        xy = states[:, :2]
        # Distance to nearest waypoint
        dists = np.linalg.norm(
            xy[:, None, :] - self.waypoints[None, :, :], axis=-1
        ).min(axis=-1)
        return np.where(dists < self.threshold, 0.0, -1.0)


# ============================================================
# ExORL evaluation tasks
# ============================================================

class CheetahVelocityReward:
    """
    Cheetah velocity reward.

    Reward is 1 if speed >= target_velocity, linearly decays to 0 below.
    If velocity is in opposite direction, reward is 0.
    """

    def __init__(self, target_velocity: float, forward: bool = True):
        self.target_velocity = target_velocity
        self.forward = forward

    def __call__(self, speeds: np.ndarray) -> np.ndarray:
        """
        Args:
            speeds: (N,) horizontal velocity (positive = forward)
        Returns:
            rewards: (N,)
        """
        if not self.forward:
            speeds = -speeds
        # Zero reward for wrong direction
        rewards = np.where(speeds < 0, 0.0, np.minimum(speeds / self.target_velocity, 1.0))
        return rewards


class WalkerVelocityReward:
    """
    Walker velocity reward.

    Reward is 1 if horizontal velocity >= threshold, linearly decays to 0 below.
    If velocity is in opposite direction, reward is 0.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, velocities: np.ndarray) -> np.ndarray:
        """
        Args:
            velocities: (N,) horizontal velocities
        Returns:
            rewards: (N,)
        """
        rewards = np.where(
            velocities < 0,
            0.0,
            np.minimum(velocities / max(self.threshold, 1e-8), 1.0)
        )
        return rewards


class ExOrlGoalReachingReward:
    """
    ExORL goal-reaching reward.

    Reward is -1 unless within Euclidean distance 0.1 of goal state
    (after normalizing by per-dimension std).
    """

    def __init__(self, goal_state: np.ndarray, state_std: np.ndarray, threshold: float = 0.1):
        self.goal_state = goal_state
        self.state_std = state_std
        self.threshold = threshold

    def __call__(self, states: np.ndarray) -> np.ndarray:
        """
        Args:
            states: (N, state_dim) - raw observation states (no augmented physics)
        Returns:
            rewards: (N,) in {-1, 0}
        """
        diff = (states - self.goal_state) / (self.state_std + 1e-8)
        dist = np.linalg.norm(diff, axis=-1)
        return np.where(dist < self.threshold, 0.0, -1.0)


# Named evaluation task sets
ANTMAZE_EVAL_TASKS = {
    'ant-goal-reaching': [
        ('goal-bottom', AntGoalReachingReward(ANTMAZE_GOALS['goal-bottom'])),
        ('goal-left',   AntGoalReachingReward(ANTMAZE_GOALS['goal-left'])),
        ('goal-top',    AntGoalReachingReward(ANTMAZE_GOALS['goal-top'])),
        ('goal-center', AntGoalReachingReward(ANTMAZE_GOALS['goal-center'])),
        ('goal-right',  AntGoalReachingReward(ANTMAZE_GOALS['goal-right'])),
    ],
    'ant-directional': [
        ('vel_left',  ANTMAZE_DIRECTIONS['vel_left']),
        ('vel_up',    ANTMAZE_DIRECTIONS['vel_up']),
        ('vel_down',  ANTMAZE_DIRECTIONS['vel_down']),
        ('vel_right', ANTMAZE_DIRECTIONS['vel_right']),
    ],
    'ant-random-simplex': [
        ('simplex-1', AntRandomSimplexReward(seed=1)),
        ('simplex-2', AntRandomSimplexReward(seed=2)),
        ('simplex-3', AntRandomSimplexReward(seed=3)),
        ('simplex-4', AntRandomSimplexReward(seed=4)),
        ('simplex-5', AntRandomSimplexReward(seed=5)),
    ],
    'ant-path-center': [('path-center', AntPathReward('center'))],
    'ant-path-loop':   [('path-loop',   AntPathReward('loop'))],
    'ant-path-edges':  [('path-edges',  AntPathReward('edges'))],
}

EXORL_CHEETAH_VELOCITY_TASKS = [
    ('cheetah-run',            CheetahVelocityReward(10.0, forward=True)),
    ('cheetah-run-backwards',  CheetahVelocityReward(10.0, forward=False)),
    ('cheetah-walk',           CheetahVelocityReward(1.0,  forward=True)),
    ('cheetah-walk-backwards', CheetahVelocityReward(1.0,  forward=False)),
]

EXORL_WALKER_VELOCITY_TASKS = [
    ('walker-vel-0.1', WalkerVelocityReward(0.1)),
    ('walker-vel-1',   WalkerVelocityReward(1.0)),
    ('walker-vel-4',   WalkerVelocityReward(4.0)),
    ('walker-vel-8',   WalkerVelocityReward(8.0)),
]
