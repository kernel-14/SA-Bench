"""Environment wrappers for DMC, OpenAI Gym, and DMLab."""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym


def make_env(config) -> gym.Env:
    """Create environment based on config."""
    env_type = config.experiment

    if "dmc" in env_type:
        return make_dmc_env(config)
    elif "gym" in env_type:
        return make_gym_env(config)
    elif "dmlab" in env_type:
        return make_dmlab_env(config)
    else:
        raise ValueError(f"Unknown experiment type: {env_type}")


def make_dmc_env(config) -> gym.Env:
    """Create DeepMind Control Suite environment."""
    try:
        from dm_control import suite
        from dm_control.suite.wrappers import pixels
        import dm_env
    except ImportError:
        raise ImportError("dm_control not installed. Install with: pip install dm_control")

    domain = config.env.dmc_domain
    task = config.env.dmc_task

    is_pixel = "pixel" in config.experiment

    env = suite.load(domain, task)
    env = DMCWrapper(env, pixel_based=is_pixel, image_size=config.env.dmc_image_size)
    return env


def make_gym_env(config) -> gym.Env:
    """Create OpenAI Gymnasium environment."""
    env = gym.make(config.env.gym_env_name, max_episode_steps=config.env.gym_max_episode_steps)
    env = GymWrapper(env)
    return env


def make_dmlab_env(config) -> gym.Env:
    """Create DMLab randomized environment (Appendix A.2)."""
    try:
        import deepmind_lab
    except ImportError:
        raise ImportError("deepmind_lab not installed.")

    level_name = {
        "sparse": "contributed/dmlab30/explore_goal_locations_sparse",
        "very_sparse": "contributed/dmlab30/explore_goal_locations_very_sparse",
    }.get(config.env.dmlab_level, config.env.dmlab_level)

    env = DMLabWrapper(
        level_name=level_name,
        repeat=config.env.dmlab_repeat,
        episode_steps=config.env.dmlab_episode_steps,
    )
    return env


class DMCWrapper:
    """Wrapper for DeepMind Control Suite environments.

    Converts dm_control environments to gym-like interface.
    """

    def __init__(self, env, pixel_based: bool = False, image_size: int = 84):
        self._env = env
        self.pixel_based = pixel_based
        self.image_size = image_size

        action_spec = env.action_spec()
        self.action_space = gym.spaces.Box(
            low=action_spec.minimum,
            high=action_spec.maximum,
            shape=action_spec.shape,
            dtype=np.float32,
        )

        if pixel_based:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(3, image_size, image_size), dtype=np.uint8
            )
        else:
            obs_spec = env.observation_spec()
            obs_dim = sum(
                np.prod(s.shape) if len(s.shape) > 0 else 1
                for s in obs_spec.values()
            )
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )

    def reset(self) -> Tuple[np.ndarray, Dict]:
        time_step = self._env.reset()
        obs = self._get_obs(time_step)
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Map action from [-1, 1] to action spec range
        spec = self._env.action_spec()
        action = np.clip(action, -1.0, 1.0)
        action = spec.minimum + (action + 1.0) / 2.0 * (spec.maximum - spec.minimum)

        time_step = self._env.step(action)
        obs = self._get_obs(time_step)
        reward = float(time_step.reward) if time_step.reward is not None else 0.0
        terminated = time_step.last()
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _get_obs(self, time_step) -> np.ndarray:
        if self.pixel_based:
            obs = self._env.physics.render(
                height=self.image_size,
                width=self.image_size,
                camera_id=0,
            )
            obs = obs.transpose(2, 0, 1).astype(np.float32) / 255.0
            return obs
        else:
            obs_dict = time_step.observation
            obs_list = []
            for key in sorted(obs_dict.keys()):
                val = obs_dict[key]
                obs_list.append(val.ravel())
            return np.concatenate(obs_list).astype(np.float32)

    @property
    def unwrapped(self):
        return self

    def seed(self, seed: int):
        import numpy as np
        np.random.seed(seed)


class GymWrapper:
    """Wrapper for Gymnasium environments."""

    def __init__(self, env: gym.Env):
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self) -> Tuple[np.ndarray, Dict]:
        obs, info = self._env.reset()
        return obs.astype(np.float32), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    @property
    def unwrapped(self):
        return self._env

    def seed(self, seed: int):
        self._env.reset(seed=seed)


class DMLabWrapper:
    """Wrapper for DMLab environments with noisy TV (Appendix A.2).

    Replaces lower-right 42x42 pixels with random noise.
    """

    def __init__(self, level_name: str, repeat: int = 4, episode_steps: int = 1800):
        try:
            import deepmind_lab
        except ImportError:
            raise ImportError("deepmind_lab not installed.")

        import deepmind_lab
        self._lab = deepmind_lab
        self.level_name = level_name
        self.repeat = repeat
        self.episode_steps = episode_steps

        self.action_space = gym.spaces.Discrete(9)  # 9 discrete actions
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 84, 84), dtype=np.uint8
        )

        self._env = None
        self._episode_step = 0

    def reset(self) -> Tuple[np.ndarray, Dict]:
        if self._env is not None:
            self._env.close()
        self._env = self._lab.Lab(
            self.level_name,
            ["RGB_INTERLEAVED"],
            config={"width": "84", "height": "84", "fps": "60"},
        )
        self._env.reset()
        self._episode_step = 0
        obs = self._env.observations()["RGB_INTERLEAVED"]
        obs = self._add_noise(obs)
        return obs.astype(np.float32) / 255.0, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = int(np.argmax(action)) if isinstance(action, np.ndarray) else int(action)
        reward = 0.0
        for _ in range(self.repeat):
            r = self._env.step(action, num_steps=1)
            reward += r if r is not None else 0.0
        self._episode_step += 1

        obs = self._env.observations()["RGB_INTERLEAVED"]
        obs = self._add_noise(obs)

        terminated = self._episode_step >= self.episode_steps
        truncated = False
        return obs.astype(np.float32) / 255.0, float(reward), terminated, truncated, {}

    def _add_noise(self, obs: np.ndarray) -> np.ndarray:
        """Add stochastic noise to lower-right 42x42 pixels (noisy TV)."""
        obs = obs.copy()
        h, w = obs.shape[0], obs.shape[1]
        noise_h, noise_w = 42, 42
        noise = np.random.randint(0, 256, (noise_h, noise_w, 3), dtype=np.uint8)
        obs[h - noise_h:, w - noise_w:] = noise
        return obs

    @property
    def unwrapped(self):
        return self

    def close(self):
        if self._env is not None:
            self._env.close()
