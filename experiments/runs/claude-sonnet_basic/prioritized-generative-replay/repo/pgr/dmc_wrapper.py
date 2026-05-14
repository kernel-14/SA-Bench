"""
DeepMind Control Suite (DMC) environment wrapper.

Wraps DMC environments to provide a gymnasium-compatible interface.
Used for state-based DMC experiments in the paper.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any


class DMCWrapper:
    """
    Gymnasium-compatible wrapper for DeepMind Control Suite environments.

    Flattens observations from ordered dict to numpy array.
    Normalizes actions to [-1, 1].
    """

    def __init__(
        self,
        domain_name: str,
        task_name: str,
        seed: int = 0,
        frame_skip: int = 1,
        episode_length: int = 1000,
    ):
        from dm_control import suite
        import gymnasium as gym
        from gymnasium import spaces

        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={"random": seed},
        )
        self.frame_skip = frame_skip
        self.episode_length = episode_length
        self._step_count = 0

        # Build observation space
        obs_spec = self._env.observation_spec()
        obs_dim = sum(
            int(np.prod(spec.shape)) if len(spec.shape) > 0 else 1
            for spec in obs_spec.values()
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Build action space
        action_spec = self._env.action_spec()
        self.action_space = spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            dtype=np.float32,
        )

        self._obs_keys = list(obs_spec.keys())

    def _flatten_obs(self, time_step) -> np.ndarray:
        obs = time_step.observation
        parts = []
        for key in self._obs_keys:
            val = obs[key]
            if np.isscalar(val):
                parts.append(np.array([val], dtype=np.float32))
            else:
                parts.append(np.asarray(val, dtype=np.float32).flatten())
        return np.concatenate(parts)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            # Re-create env with new seed
            from dm_control import suite
            domain = self._env.physics.model.name.split('/')[0]
            # Note: DMC doesn't easily support re-seeding; we just reset
            pass
        time_step = self._env.reset()
        self._step_count = 0
        obs = self._flatten_obs(time_step)
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        reward = 0.0
        for _ in range(self.frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0.0
            if time_step.last():
                break

        self._step_count += 1
        obs = self._flatten_obs(time_step)
        terminated = time_step.last()
        truncated = self._step_count >= self.episode_length
        return obs, reward, terminated, truncated, {}

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        return self._env.physics.render(height=84, width=84, camera_id=0)
