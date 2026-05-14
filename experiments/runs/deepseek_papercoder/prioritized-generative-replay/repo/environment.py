## environment.py

"""
Unified environment interface for DeepMind Control Suite (state-based and pixel-based)
and OpenAI Gym tasks.

The class Environment wraps different backends:
- State‑based DMC tasks via `dm_control.suite`
- Pixel‑based DMC tasks via `dmc2gym` (must be installed separately)
- Standard Gym tasks (e.g., Walker2d‑v2, HalfCheetah‑v2, Hopper‑v2)

It provides a consistent API:
    obs = env.reset()
    next_obs, reward, done, info = env.step(action)
    env.seed(seed)

Pixel observations are returned in channel‑first format (C, H, W) to match PyTorch
conventions.

Note: The environment does **not** apply any state normalisation; that is handled
elsewhere in the pipeline.
"""

import gym
import numpy as np

try:
    import dmc2gym
    _DMC2GYM_AVAILABLE = True
except ImportError:
    _DMC2GYM_AVAILABLE = False

import dm_control.suite as suite


class Environment:
    """Unified wrapper for DMC (state/pixel) and Gym environments."""

    def __init__(self, env_name: str, state_based: bool, seed: int = 0):
        """
        Parameters
        ----------
        env_name : str
            Name of the environment. For DMC tasks use format "domain-task"
            (e.g. "cheetah-run", "walker-walk"). For Gym tasks use their standard
            IDs (e.g. "Walker2d-v2").
        state_based : bool
            If True, the environment returns proprioceptive state vectors.
            If False, it returns RGB pixel observations (84×84).
        seed : int, optional
            Random seed for reproducibility.
        """
        self.env_name = env_name
        self.state_based = state_based
        self.seed_val = seed

        # Determine backend type and initialise appropriately
        if env_name in ['Walker2d-v2', 'HalfCheetah-v2', 'Hopper-v2']:
            self.env_type = 'gym'
            self._init_gym()
        else:
            # All other names are assumed to be DMC tasks
            if self.state_based:
                self.env_type = 'dmc_state'
            else:
                if not _DMC2GYM_AVAILABLE:
                    raise ImportError(
                        "dmc2gym is required for pixel-based DMC tasks. "
                        "Please install it from: https://github.com/denisyarats/dmc2gym"
                    )
                self.env_type = 'dmc_pixel'
            self._init_dmc()

        # Only continuous action spaces are supported by PGR
        assert isinstance(self.action_space, gym.spaces.Box), \
            f"Action space must be gym.spaces.Box, got {type(self.action_space)}"

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _init_gym(self):
        """Create a standard Gym environment."""
        self.env = gym.make(self.env_name)
        self.state_based = True   # Gym tasks are state‑based
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.env.reset(seed=self.seed_val)
        self.env.action_space.seed(self.seed_val)

    def _init_dmc(self):
        """Create a DMC environment, either state‑based or pixel‑based."""
        domain, task = self.env_name.split('-', 1)

        if self.env_type == 'dmc_state':
            self.env = suite.load(domain, task,
                                  task_kwargs={'random': self.seed_val})
            # Observation space inferred from physics state
            state_dim = self.env.physics.get_state().shape[0]
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(state_dim,), dtype=np.float32
            )
            # Action space from the task specification
            act_spec = self.env.action_spec()
            self.action_space = gym.spaces.Box(
                low=act_spec.minimum, high=act_spec.maximum,
                shape=(act_spec.shape[0],), dtype=np.float32
            )
        else:  # dmc_pixel
            self.env = dmc2gym.make(
                domain_name=domain,
                task_name=task,
                seed=self.seed_val,
                from_pixels=True,
                height=84,
                width=84,
                frame_skip=2
            )
            # dmc2gym returns H×W×C images; we convert to C×H×W later
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=(3, 84, 84),
                dtype=np.uint8
            )
            self.action_space = self.env.action_space

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        if self.env_type == 'dmc_state':
            timestep = self.env.reset()
            obs = np.array(timestep.observation, dtype=np.float32)
            return obs
        else:
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            # Convert pixel images to channel‑first (C, H, W)
            if self.env_type == 'dmc_pixel' and obs.ndim == 3 and obs.shape[-1] == 3:
                obs = obs.transpose(2, 0, 1)
            return obs

    def step(self, action: np.ndarray):
        """Apply an action and return (next_obs, reward, done, info)."""
        if self.env_type == 'dmc_state':
            timestep = self.env.step(action)
            obs = np.array(timestep.observation, dtype=np.float32)
            reward = timestep.reward
            done = timestep.last()
            info = {}
            return obs, reward, done, info
        else:
            result = self.env.step(action)
            # Handle both old gym (4 returns) and new gym (5 returns)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            # Convert pixel images to channel‑first
            if self.env_type == 'dmc_pixel' and obs.ndim == 3 and obs.shape[-1] == 3:
                obs = obs.transpose(2, 0, 1)
            return obs, reward, done, info

    def seed(self, seed: int = None):
        """
        Set the random seed for the environment.

        For Gym environments the seed is applied to the existing instance.
        For DMC environments we recreate the environment with the new seed
        to guarantee reproducibility.
        """
        if seed is not None:
            self.seed_val = seed

        if self.env_type == 'gym':
            self.env.reset(seed=self.seed_val)
            self.env.action_space.seed(self.seed_val)
        elif self.env_type in ['dmc_state', 'dmc_pixel']:
            # Re‑initialise the DMC environment with the updated seed
            self._init_dmc()

    # ------------------------------------------------------------------
    # Optional rendering (for debugging)
    # ------------------------------------------------------------------
    def render(self, mode: str = 'human') -> None:
        """Render the current frame if supported."""
        if hasattr(self.env, 'render'):
            self.env.render()
        # dm_control suite does not have a generic render(). Pixel tasks
        # can be rendered via dmc2gym's render method.
        elif hasattr(self.env, '_render'):
            self.env._render()
