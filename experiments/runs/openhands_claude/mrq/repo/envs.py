"""
Environment wrappers for MR.Q.

Supported benchmarks
--------------------
  gym          – OpenAI Gym / Gymnasium MuJoCo locomotion (vector obs, continuous)
  dmc_proprio  – DeepMind Control Suite proprioceptive (vector obs, continuous)
  dmc_visual   – DeepMind Control Suite visual (84×84 RGB, 3-frame stack, continuous)
  atari        – Atari-57 (84×84 grayscale, 4-frame stack, discrete)

All wrappers expose a unified interface:
    env.reset()  → np.ndarray  (observation)
    env.step(a)  → (obs, reward, done, info)
    env.obs_shape, env.action_dim, env.action_type
    env.action_low, env.action_high  (continuous only)
"""

from __future__ import annotations

import collections
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ── Gym / Gymnasium locomotion ────────────────────────────────────────────────

class GymEnv:
    """
    Thin wrapper around Gymnasium environments.
    Actions are expected in [-1, 1] and are rescaled to the environment range.
    """

    def __init__(self, env_name: str, seed: int = 0) -> None:
        import gymnasium as gym
        self._env = gym.make(env_name)
        self._env.action_space.seed(seed)
        self._env.observation_space.seed(seed)

        obs_space = self._env.observation_space
        act_space = self._env.action_space

        self.obs_shape: tuple = obs_space.shape
        self.action_dim: int = int(np.prod(act_space.shape))
        self.action_type: str = "continuous"
        self.action_low: np.ndarray = act_space.low.astype(np.float32)
        self.action_high: np.ndarray = act_space.high.astype(np.float32)

    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset()
        return obs.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # Rescale from [-1, 1] to environment action range
        scaled = self.action_low + (action + 1.0) * 0.5 * (
            self.action_high - self.action_low
        )
        obs, reward, terminated, truncated, info = self._env.step(scaled)
        done = bool(terminated or truncated)
        return obs.astype(np.float32), float(reward), done, info

    def close(self) -> None:
        self._env.close()


# ── DeepMind Control Suite ────────────────────────────────────────────────────

class DMCEnv:
    """
    Wrapper for DeepMind Control Suite environments.

    Supports both proprioceptive (vector) and visual (image) observations.
    Actions are expected in [-1, 1].
    """

    def __init__(
        self,
        domain: str,
        task: str,
        seed: int = 0,
        image_obs: bool = False,
        image_size: int = 84,
        frame_stack: int = 3,
        action_repeat: int = 2,
    ) -> None:
        from dm_control import suite
        from dm_control.suite.wrappers import pixels as pixels_wrapper

        self._action_repeat = action_repeat
        self._image_obs = image_obs
        self._image_size = image_size
        self._frame_stack = frame_stack

        self._env = suite.load(domain, task, task_kwargs={"random": seed})

        act_spec = self._env.action_spec()
        self.action_dim: int = int(np.prod(act_spec.shape))
        self.action_type: str = "continuous"
        self.action_low: np.ndarray = act_spec.minimum.astype(np.float32)
        self.action_high: np.ndarray = act_spec.maximum.astype(np.float32)

        if image_obs:
            # (frame_stack * 3, H, W) – RGB frames stacked along channel axis
            self.obs_shape: tuple = (frame_stack * 3, image_size, image_size)
            self._frames: collections.deque = collections.deque(
                maxlen=frame_stack
            )
        else:
            obs_dim = sum(
                int(np.prod(v.shape))
                for v in self._env.observation_spec().values()
            )
            self.obs_shape = (obs_dim,)

    def _get_obs(self, time_step) -> np.ndarray:
        if self._image_obs:
            frame = self._env.physics.render(
                height=self._image_size, width=self._image_size, camera_id=0
            )  # (H, W, 3) uint8
            frame = frame.transpose(2, 0, 1)  # (3, H, W)
            self._frames.append(frame)
            return np.concatenate(list(self._frames), axis=0).astype(np.uint8)
        else:
            obs_list = []
            for v in time_step.observation.values():
                obs_list.append(np.atleast_1d(v).astype(np.float32).ravel())
            return np.concatenate(obs_list)

    def reset(self) -> np.ndarray:
        time_step = self._env.reset()
        if self._image_obs:
            frame = self._env.physics.render(
                height=self._image_size, width=self._image_size, camera_id=0
            ).transpose(2, 0, 1)
            for _ in range(self._frame_stack):
                self._frames.append(frame)
        return self._get_obs(time_step)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # Rescale from [-1, 1] to environment action range
        scaled = self.action_low + (action + 1.0) * 0.5 * (
            self.action_high - self.action_low
        )
        total_reward = 0.0
        for _ in range(self._action_repeat):
            time_step = self._env.step(scaled)
            total_reward += float(time_step.reward or 0.0)
            if time_step.last():
                break
        done = time_step.last()
        obs = self._get_obs(time_step)
        return obs, total_reward, done, {}

    def close(self) -> None:
        self._env.close()


# ── Atari ─────────────────────────────────────────────────────────────────────

class AtariEnv:
    """
    Atari wrapper following standard preprocessing (Mnih et al. 2015;
    Machado et al. 2018; Castro et al. 2018).

    Preprocessing:
      - Sticky actions (p=0.25) via -v5 environments
      - Action repeat of 4
      - Grayscale + resize to 84×84
      - Max-pool over last 2 frames of each action repeat
      - Stack 4 observations → (4, 84, 84) uint8
    """

    def __init__(
        self,
        game: str,
        seed: int = 0,
        image_size: int = 84,
        frame_stack: int = 4,
        action_repeat: int = 4,
        noop_max: int = 30,
    ) -> None:
        import gymnasium as gym
        from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

        # Use -v5 for sticky actions (p=0.25 by default in ALE)
        env_id = f"ALE/{game}-v5"
        base_env = gym.make(env_id, frameskip=1, repeat_action_probability=0.25)
        base_env.action_space.seed(seed)

        # AtariPreprocessing handles: noop, frame skip, grayscale, resize, max-pool
        env = AtariPreprocessing(
            base_env,
            noop_max=noop_max,
            frame_skip=action_repeat,
            screen_size=image_size,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=False,
        )
        self._env = FrameStackObservation(env, stack_size=frame_stack)

        self.obs_shape: tuple = (frame_stack, image_size, image_size)
        self.action_dim: int = self._env.action_space.n
        self.action_type: str = "discrete"
        self.action_low: Optional[np.ndarray] = None
        self.action_high: Optional[np.ndarray] = None

    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset()
        return np.array(obs, dtype=np.uint8)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # action is a one-hot vector; take argmax to get integer action
        if action.ndim > 0:
            action_int = int(np.argmax(action))
        else:
            action_int = int(action)
        obs, reward, terminated, truncated, info = self._env.step(action_int)
        done = bool(terminated or truncated)
        return np.array(obs, dtype=np.uint8), float(reward), done, info

    def close(self) -> None:
        self._env.close()


# ── Factory ───────────────────────────────────────────────────────────────────

def make_env(cfg) -> Any:
    """
    Create an environment from a MRQConfig.

    Returns an environment object with the unified interface described above.
    """
    benchmark = cfg.benchmark.lower()

    if benchmark == "gym":
        return GymEnv(cfg.env_name, seed=cfg.seed)

    elif benchmark == "dmc_proprio":
        domain, task = _parse_dmc_name(cfg.env_name)
        return DMCEnv(
            domain=domain,
            task=task,
            seed=cfg.seed,
            image_obs=False,
            action_repeat=cfg.action_repeat,
        )

    elif benchmark == "dmc_visual":
        domain, task = _parse_dmc_name(cfg.env_name)
        return DMCEnv(
            domain=domain,
            task=task,
            seed=cfg.seed,
            image_obs=True,
            image_size=cfg.image_size,
            frame_stack=cfg.frame_stack,
            action_repeat=cfg.action_repeat,
        )

    elif benchmark == "atari":
        return AtariEnv(
            game=cfg.env_name,
            seed=cfg.seed,
            image_size=cfg.image_size,
            frame_stack=cfg.frame_stack,
            action_repeat=cfg.action_repeat,
        )

    else:
        raise ValueError(f"Unknown benchmark: {benchmark!r}")


def _parse_dmc_name(name: str) -> Tuple[str, str]:
    """
    Parse a DMC environment name like 'cheetah-run' or 'cheetah_run'
    into (domain, task).
    """
    name = name.replace("-", "_")
    parts = name.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"Cannot parse DMC environment name: {name!r}")
