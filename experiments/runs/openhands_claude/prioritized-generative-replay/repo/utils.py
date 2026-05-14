import os
import random
import numpy as np
import torch
from typing import Any, Dict, Optional, Tuple


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transition_dim(state_dim: int, action_dim: int) -> int:
    """Compute flat transition dimension: (s, a, s', r)."""
    return 2 * state_dim + action_dim + 1


def make_env(
    env_name: str,
    seed: int = 0,
    pixel_obs: bool = False,
    image_size: int = 84,
    frame_stack: int = 3,
    action_repeat: int = 2,
):
    """Create a DMC or Gym environment.

    For DMC environments, uses dm_control with the standard task naming convention
    (e.g., 'quadruped-walk' -> domain='quadruped', task='walk').
    For pixel-based tasks, wraps with frame stacking and pixel observations.
    """
    from config import ENV_ACTION_REPEAT, ENV_FRAME_STACK, GYM_ENVS

    if env_name in GYM_ENVS:
        return _make_gym_env(env_name, seed)
    elif env_name.startswith("dmlab"):
        return _make_dmlab_env(env_name, seed)
    else:
        return _make_dmc_env(env_name, seed, pixel_obs, image_size, frame_stack, action_repeat)


def _make_gym_env(env_name: str, seed: int):
    """Create an OpenAI Gym environment."""
    import gymnasium as gym
    env = gym.make(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def _make_dmc_env(
    env_name: str,
    seed: int,
    pixel_obs: bool = False,
    image_size: int = 84,
    frame_stack: int = 3,
    action_repeat: int = 2,
):
    """Create a DeepMind Control Suite environment."""
    from dm_control import suite
    import gymnasium as gym
    from gymnasium.wrappers import TimeLimit

    parts = env_name.split("-")
    domain = parts[0]
    task = "_".join(parts[1:])

    from config import ENV_ACTION_REPEAT, ENV_FRAME_STACK
    action_repeat = ENV_ACTION_REPEAT.get(env_name, action_repeat)
    frame_stack = ENV_FRAME_STACK.get(env_name, frame_stack) if pixel_obs else 1

    env = DMCWrapper(
        domain_name=domain,
        task_name=task,
        seed=seed,
        pixel_obs=pixel_obs,
        image_size=image_size,
        action_repeat=action_repeat,
        frame_stack=frame_stack,
    )
    return env


def _make_dmlab_env(env_name: str, seed: int):
    """Create a DMLab environment for noisy-TV experiments."""
    raise NotImplementedError(
        "DMLab environment requires deepmind_lab package. "
        "See Appendix A.2 for setup instructions."
    )


class DMCWrapper:
    """Gymnasium-compatible wrapper for DeepMind Control Suite.

    Supports both state-based and pixel-based observations.
    Implements action repeat and frame stacking for pixel tasks.
    """

    def __init__(
        self,
        domain_name: str,
        task_name: str,
        seed: int = 0,
        pixel_obs: bool = False,
        image_size: int = 84,
        action_repeat: int = 2,
        frame_stack: int = 3,
    ):
        from dm_control import suite
        import gymnasium as gym

        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={"random": seed},
        )
        self.pixel_obs = pixel_obs
        self.image_size = image_size
        self.action_repeat = action_repeat
        self.frame_stack = frame_stack
        self._frames = None

        action_spec = self._env.action_spec()
        self.action_space = gym.spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            dtype=np.float32,
        )

        if pixel_obs:
            obs_shape = (frame_stack * 3, image_size, image_size)
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8
            )
        else:
            obs_dim = sum(
                int(np.prod(v.shape))
                for v in self._env.observation_spec().values()
            )
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )

    def _get_obs(self) -> np.ndarray:
        if self.pixel_obs:
            frame = self._env.physics.render(
                height=self.image_size, width=self.image_size, camera_id=0
            )
            return frame.transpose(2, 0, 1).astype(np.uint8)
        else:
            obs_dict = self._env.observation()
            return np.concatenate([
                v.flatten() for v in obs_dict.values()
            ]).astype(np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        time_step = self._env.reset()
        obs = self._get_obs()
        if self.pixel_obs:
            self._frames = [obs] * self.frame_stack
            obs = np.concatenate(self._frames, axis=0)
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        total_reward = 0.0
        for _ in range(self.action_repeat):
            time_step = self._env.step(action)
            total_reward += time_step.reward or 0.0
            if time_step.last():
                break

        obs = self._get_obs()
        if self.pixel_obs:
            self._frames.pop(0)
            self._frames.append(obs)
            obs = np.concatenate(self._frames, axis=0)

        terminated = time_step.last()
        truncated = False
        return obs, total_reward, terminated, truncated, {}

    def close(self):
        self._env.close()


def evaluate_policy(
    agent,
    env,
    n_episodes: int = 10,
    deterministic: bool = True,
) -> float:
    """Evaluate policy and return mean episode return."""
    returns = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_return = 0.0
        done = False
        while not done:
            action = agent.select_action(state, deterministic=deterministic)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
        returns.append(episode_return)
    return float(np.mean(returns))


class Logger:
    """Simple logger supporting TensorBoard and Weights & Biases."""

    def __init__(
        self,
        log_dir: str,
        use_wandb: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        os.makedirs(log_dir, exist_ok=True)

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            self.writer = None

        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.get("wandb_project", "pgr") if config else "pgr",
                    name=config.get("exp_name", "run") if config else "run",
                    config=config,
                )
                self.wandb = wandb
            except ImportError:
                self.wandb = None
                self.use_wandb = False
        else:
            self.wandb = None

    def log(self, metrics: Dict[str, Any], step: int):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if self.writer is not None:
                    self.writer.add_scalar(key, value, step)
                if self.use_wandb and self.wandb is not None:
                    self.wandb.log({key: value}, step=step)

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.use_wandb and self.wandb is not None:
            self.wandb.finish()


def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float):
    """Polyak averaging: θ_target = τ·θ_source + (1-τ)·θ_target."""
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_obs(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (obs - mean) / (std + 1e-8)


def running_mean(values: list, window: int = 100) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.mean(values[-window:]))
