## envs/gym_env.py
"""OpenAI Gym (gymnasium) environment wrapper for Prioritized Generative Replay (PGR).

Provides a unified interface for MuJoCo continuous control tasks used in the
OpenAI Gym benchmark (Table 2 of the paper): Walker2d-v2, HalfCheetah-v2,
and Hopper-v2. Consumed by PGRTrainer for online data collection and by
Evaluator for deterministic policy evaluation.

The interface mirrors DMCEnv so that PGRTrainer can treat both wrappers
interchangeably without conditional logic.
"""

from typing import Dict, Tuple

import gymnasium
import numpy as np


class GymEnv:
    """Wrapper around gymnasium MuJoCo environments for state-based RL.

    Handles the gymnasium v0.29 API (5-tuple step return, seed-on-reset),
    float32 casting, and action clipping. Stateless beyond the underlying
    gymnasium environment — episode tracking is the caller's responsibility.

    Supported environments (paper Table 2):
        - ``Walker2d-v2``   (obs_dim=17, action_dim=6)
        - ``HalfCheetah-v2`` (obs_dim=17, action_dim=6)
        - ``Hopper-v2``     (obs_dim=11, action_dim=3)

    Attributes:
        env_name: Gymnasium environment identifier string.
        seed: Random seed used to initialise the environment's internal RNG
            on the first ``reset()`` call.
    """

    def __init__(self, env_name: str = "Walker2d-v2", seed: int = 0) -> None:
        """Initialises the gymnasium environment wrapper.

        Instantiates the environment via ``gymnasium.make``, caches observation
        and action space dimensions, and stores action bounds for clipping.
        Does **not** call ``reset()`` — the caller must do so explicitly before
        the training loop begins.

        Args:
            env_name: Gymnasium environment identifier. Must be one of
                ``"Walker2d-v2"``, ``"HalfCheetah-v2"``, or ``"Hopper-v2"``
                for the experiments described in the paper (Table 2).
            seed: Random seed passed to the environment on the first
                ``reset()`` call via ``env.reset(seed=seed)``. Corresponds to
                ``env.seed`` in config.yaml.
        """
        self.env_name: str = env_name
        self.seed: int = seed

        # ── Instantiate the gymnasium environment ─────────────────────────────
        self._env: gymnasium.Env = gymnasium.make(env_name)

        # ── Cache observation space dimension ─────────────────────────────────
        # gymnasium MuJoCo envs expose Box observation spaces with shape (obs_dim,).
        self._obs_dim: int = int(self._env.observation_space.shape[0])

        # ── Cache action space dimension and bounds ───────────────────────────
        # Action bounds are used for clipping in step() and returned by
        # action_range() for policy rescaling.
        self._action_dim: int = int(self._env.action_space.shape[0])
        self._action_low: np.ndarray = self._env.action_space.low.copy().astype(
            np.float32
        )
        self._action_high: np.ndarray = self._env.action_space.high.copy().astype(
            np.float32
        )

        # ── First-reset flag for seed injection ──────────────────────────────
        # gymnasium v0.29 seeds the environment via reset(seed=seed).
        # We pass the seed only on the first call; subsequent resets carry
        # forward the internal RNG state.
        self._first_reset: bool = True

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Resets the environment and returns the initial observation.

        On the very first call, passes ``seed=self.seed`` to gymnasium to
        initialise the environment's internal RNG deterministically. All
        subsequent calls reset without re-seeding so that the RNG state
        evolves naturally across episodes.

        Returns:
            Float32 numpy array of shape ``(obs_dim,)`` representing the
            initial environment observation.
        """
        if self._first_reset:
            obs, _info = self._env.reset(seed=self.seed)
            self._first_reset = False
        else:
            obs, _info = self._env.reset()

        return obs.astype(np.float32)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Advances the environment by one step.

        Clips the incoming action to the valid range ``[action_low, action_high]``
        before passing it to the environment. This is a defensive guard — the
        policy already applies tanh squashing and rescaling, so in practice the
        action should already be in range.

        Merges gymnasium v0.29's separate ``terminated`` and ``truncated``
        flags into a single ``done`` boolean, matching the convention used by
        ``ReplayBuffer`` which stores a single done flag per transition.

        Args:
            action: Float numpy array of shape ``(action_dim,)`` representing
                the action to execute. Will be clipped to the valid action range.

        Returns:
            A tuple ``(next_obs, reward, done, info)`` where:
                - ``next_obs``: Float32 numpy array of shape ``(obs_dim,)``.
                - ``reward``: Scalar float reward.
                - ``done``: ``True`` if the episode ended (either terminated
                  due to a terminal MDP state, or truncated due to time limit).
                - ``info``: Info dictionary from gymnasium (passed through
                  unchanged; not used by PGRTrainer directly).
        """
        # Clip to valid action range — defensive guard against out-of-range
        # actions that could cause MuJoCo simulation errors.
        clipped_action: np.ndarray = np.clip(
            action, self._action_low, self._action_high
        ).astype(np.float32)

        # gymnasium v0.29 returns a 5-tuple from step().
        next_obs, reward, terminated, truncated, info = self._env.step(clipped_action)

        # Merge terminated and truncated into a single done flag.
        # terminated: episode ended due to a terminal MDP state (e.g. robot fell).
        # truncated: episode ended due to time limit (max_episode_steps reached).
        done: bool = bool(terminated) or bool(truncated)

        return next_obs.astype(np.float32), float(reward), done, info

    def observation_space_dim(self) -> int:
        """Returns the flat observation dimension.

        Cached at construction time from ``env.observation_space.shape[0]``.

        Returns:
            Integer observation dimension. Reference values:
                - Walker2d-v2: 17
                - HalfCheetah-v2: 17
                - Hopper-v2: 11
        """
        return self._obs_dim

    def action_space_dim(self) -> int:
        """Returns the number of action dimensions.

        Cached at construction time from ``env.action_space.shape[0]``.

        Returns:
            Integer action dimension. Reference values:
                - Walker2d-v2: 6
                - HalfCheetah-v2: 6
                - Hopper-v2: 3
        """
        return self._action_dim

    def action_range(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the valid action range as ``(min, max)`` numpy arrays.

        Used by ``REDQPolicy`` and ``SACPolicy`` to rescale tanh-squashed
        actor outputs (in ``[-1, 1]``) to the actual action range.

        Returns:
            Tuple of ``(action_low, action_high)`` float32 arrays of shape
            ``(action_dim,)``, copied from the gymnasium action space at
            construction time.
        """
        return self._action_low, self._action_high

    def __repr__(self) -> str:
        """Returns a concise string representation of the environment wrapper."""
        return (
            f"GymEnv(env_name='{self.env_name}', "
            f"obs_dim={self._obs_dim}, "
            f"action_dim={self._action_dim}, "
            f"seed={self.seed})"
        )
