# env_utils.py

"""
Environment wrapper and creation utilities for MR.Q benchmarks.

This module provides the `make_env` function that constructs a Gymnasium
environment with proper preprocessing, action repeat, frame stacking, and
sticky actions, exactly as required by the MR.Q paper.

Custom wrappers:
    - ActionRepeat: repeats the agent's action for a fixed number of frames.
    - ResizeObsAndFrameStack: resizes RGB observations and stacks the last
      N frames into a single channel-first tensor (for DM Control visual).

Supported benchmarks:
    - gym_locomotion
    - dmc_proprioceptive
    - dmc_visual
    - atari

Dependencies:
    - gymnasium
    - dm_control
    - shimmy
    - stable‑baselines3 (for Atari wrappers)
    - numpy
    - opencv‑python (cv2)
"""

from collections import deque
from typing import Tuple, Optional

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dm_control import suite
from shimmy import DmControlCompatibilityV0
from stable_baselines3.common.atari_wrappers import (
    StickyActionEnv,
    MaxAndSkipEnv,
    WarpFrame,
    FrameStack,
)


# ---------------------------------------------------------------------------
# Custom wrappers
# ---------------------------------------------------------------------------

class ActionRepeat(gym.Wrapper):
    """
    Repeat the same action for a fixed number of steps.

    Accumulates rewards and returns the observation from the last step.
    If the environment terminates or truncates, the loop breaks early.
    """

    def __init__(self, env: gym.Env, repeat: int = 1):
        super().__init__(env)
        if repeat < 1:
            raise ValueError("repeat must be >= 1")
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.repeat):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            terminated = terminated or term
            truncated = truncated or trunc
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class ResizeObsAndFrameStack(gym.Wrapper):
    """
    For DM Control visual tasks: resizes each RGB observation to
    `image_size` and stacks the last `num_stack` frames channel‑wise.

    Expected input: gym Env whose observation is a (H, W, 3) uint8 image.
    Output shape: (num_stack * 3, image_size[0], image_size[1]) uint8.
    """

    def __init__(
        self,
        env: gym.Env,
        image_size: Tuple[int, int] = (84, 84),
        num_stack: int = 3,
    ):
        super().__init__(env)
        self.image_size = image_size
        self.num_stack = num_stack
        # Deque to store the last num_stack frames, each with shape (3, H, W)
        self.frames = deque(maxlen=num_stack)
        # Observation space update
        new_shape = (num_stack * 3, *image_size)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8,
        )

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        """Resize incoming HWC frame, convert to CHW, append to deque, return stacked."""
        # obs shape (H, W, 3) uint8
        obs = cv2.resize(obs, self.image_size, interpolation=cv2.INTER_LINEAR)
        obs = np.transpose(obs, (2, 0, 1))  # (3, H, W)
        self.frames.append(obs)
        return np.concatenate(list(self.frames), axis=0)  # (num_stack*3, H, W)

    def reset(self, seed: Optional[int] = None, **kwargs):
        """Reset and fill frame buffer with copies of the initial frame."""
        obs, info = self.env.reset(seed=seed, **kwargs)
        init_obs = cv2.resize(obs, self.image_size, interpolation=cv2.INTER_LINEAR)
        init_obs = np.transpose(init_obs, (2, 0, 1))
        for _ in range(self.num_stack):
            self.frames.append(init_obs)
        stacked_obs = np.concatenate(list(self.frames), axis=0)
        return stacked_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._process_obs(obs)
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(benchmark: str, task: str, seed: int) -> gym.Env:
    """
    Create a Gymnasium environment for the requested benchmark and task.

    Parameters
    ----------
    benchmark : str
        One of {"gym_locomotion", "dmc_proprioceptive", "dmc_visual", "atari"}.
    task : str
        Environment name, e.g. "Ant-v4", "acrobot-swingup", "Alien".
    seed : int
        Random seed used to configure the environment (including DM Control's
        random seed and sticky action RNG).

    Returns
    -------
    gym.Env
        A fully wrapped environment that returns observations in the format
        expected by the MR.Q agent (uint8 for images, float32 for vectors).
    """
    if benchmark == "gym_locomotion":
        env = gym.make(task)
        env = ActionRepeat(env, repeat=1)
        env.reset(seed=seed)
        return env

    elif benchmark in ("dmc_proprioceptive", "dmc_visual"):
        # Parse DM Control domain and task name
        # Examples: "acrobot-swingup" -> domain="acrobot", task="swingup"
        #           "ball_in_cup-catch" -> domain="ball_in_cup", task="catch"
        domain, task_name = task.rsplit("-", 1)
        pixels = benchmark == "dmc_visual"

        # Create the underlying dm_control environment
        dm_env = suite.load(
            domain, task_name,
            task_kwargs={"random": seed},
            visualize_reward=False,
            pixels=pixels,
        )

        # Convert to a Gymnasium environment using shimmy
        if pixels:
            gym_env = DmControlCompatibilityV0(dm_env, render_mode="rgb_array")
        else:
            gym_env = DmControlCompatibilityV0(dm_env, render_mode=None)

        # Apply visual‑specific preprocessing: resize and frame‑stack
        if benchmark == "dmc_visual":
            gym_env = ResizeObsAndFrameStack(
                gym_env, image_size=(84, 84), num_stack=3
            )

        # Apply action repeat of 2 for all DMC tasks
        gym_env = ActionRepeat(gym_env, repeat=2)

        # Seed the full stack
        gym_env.reset(seed=seed)
        return gym_env

    elif benchmark == "atari":
        env = gym.make(f"ALE/{task}-v5")
        # Add sticky actions (probability 0.25)
        env = StickyActionEnv(env, action_repeat_probability=0.25)
        # Action repeat of 4 with max over two most recent frames
        env = MaxAndSkipEnv(env, skip=4)
        # Grayscale and resize to 84x84
        env = WarpFrame(env, width=84, height=84, grayscale=True)
        # Stack the last 4 frames (channel‑first) → shape (4, 84, 84)
        env = FrameStack(env, num_stack=4)
        # Seed
        env.reset(seed=seed)
        return env

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}. "
                         f"Valid options: gym_locomotion, dmc_proprioceptive, "
                         f"dmc_visual, atari.")
