
import torch
import numpy as np
from collections import deque
import random
import gymnasium as gym
from dm_control import suite
from dm_control import _render
from dm_control.suite import wrapper as dm_wrapper
from typing import Tuple, Dict, Any, Optional

# Check if headless rendering is supported (for pixel-based DMControl tasks)
_render.ENABLE_HEADLESS_GL = True

class ReplayBuffer:
    """
    A standard replay buffer to store and sample transitions.
    """
    def __init__(self, capacity: int, observation_shape: Tuple[int, ...], action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device

        self.observations = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_observations = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool):
        self.observations[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_observations[self.idx] = next_obs
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self._to_tensor(idxs)

    def _to_tensor(self, idxs: np.ndarray) -> Dict[str, torch.Tensor]:
        return {
            "observations": torch.as_tensor(self.observations[idxs]).to(self.device),
            "actions": torch.as_tensor(self.actions[idxs]).to(self.device),
            "rewards": torch.as_tensor(self.rewards[idxs]).to(self.device),
            "next_observations": torch.as_tensor(self.next_observations[idxs]).to(self.device),
            "dones": torch.as_tensor(self.dones[idxs]).to(self.device)
        }
    
    def get_all_transitions(self) -> Dict[str, torch.Tensor]:
        """Returns all stored transitions as tensors."""
        if self.size == 0:
            return {
                "observations": torch.empty(0, *self.observations.shape[1:], device=self.device),
                "actions": torch.empty(0, *self.actions.shape[1:], device=self.device),
                "rewards": torch.empty(0, *self.rewards.shape[1:], device=self.device),
                "next_observations": torch.empty(0, *self.next_observations.shape[1:], device=self.device),
                "dones": torch.empty(0, *self.dones.shape[1:], device=self.device)
            }
        
        return {
            "observations": torch.as_tensor(self.observations[:self.size]).to(self.device),
            "actions": torch.as_tensor(self.actions[:self.size]).to(self.device),
            "rewards": torch.as_tensor(self.rewards[:self.size]).to(self.device),
            "next_observations": torch.as_tensor(self.next_observations[:self.size]).to(self.device),
            "dones": torch.as_tensor(self.dones[:self.size]).to(self.device)
        }

class DeepMindControlEnv:
    """
    Wrapper for DeepMind Control Suite environments.
    Handles pixel observations and action repeats.
    """
    def __init__(self, domain_name: str, task_name: str, obs_type: str = "state", action_repeat: int = 1, seed: int = 0):
        self.env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={'random_state': seed})
        self.obs_type = obs_type
        self.action_repeat = action_repeat
        self.seed = seed

        # Extract observation and action spaces
        self.observation_space = self._get_observation_space(self.env.observation_spec())
        self.action_space = self._get_action_space(self.env.action_spec())

        self.viewer = None # Placeholder for viewer

    def _get_observation_space(self, obs_spec: Dict[str, Any]) -> gym.spaces.Box:
        if self.obs_type == "state":
            # Combine all state observations into a single vector
            flat_dim = sum(spec.shape[0] if len(spec.shape) > 0 else 1 for spec in obs_spec.values())
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)
        elif self.obs_type == "pixel":
            # Assuming RGB images, 84x84 (common for DMC pixel tasks)
            # You might need to adjust height/width based on the specific task
            return gym.spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8) # C, H, W
        else:
            raise ValueError(f"Unknown observation type: {self.obs_type}")

    def _get_action_space(self, action_spec: Any) -> gym.spaces.Box:
        # Assuming continuous control
        return gym.spaces.Box(low=action_spec.minimum, high=action_spec.maximum, shape=action_spec.shape, dtype=action_spec.dtype)

    def _extract_observation(self, timestep: Any) -> np.ndarray:
        if self.obs_type == "state":
            # Flatten all observation components
            observations = []
            for k, v in timestep.observation.items():
                observations.append(v.flatten() if isinstance(v, np.ndarray) else np.array([v]))
            return np.concatenate(observations).astype(np.float32)
        elif self.obs_type == "pixel":
            # Render RGB image. Need to ensure rendering backend is available.
            camera_id = 0 # Default camera
            try:
                # Use offscreen rendering for pixel observations
                # Assuming 84x84 pixels for consistency with common RL benchmarks
                pixels = self.env.physics.render(height=84, width=84, camera_id=camera_id)
            except Exception as e:
                print(f"Warning: Could not render pixels. Ensure EGL/GL is configured. Error: {e}")
                # Fallback to dummy pixels if rendering fails, or raise error
                pixels = np.zeros((84, 84, 3), dtype=np.uint8)
            return pixels.transpose(2, 0, 1) # H, W, C -> C, H, W
        
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        timestep = self.env.reset()
        obs = self._extract_observation(timestep)
        return obs, {} # gym-like api returns obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = 0.0
        discount = 1.0
        for _ in range(self.action_repeat):
            timestep = self.env.step(action)
            reward += timestep.reward or 0.0
            discount *= timestep.discount
            if timestep.last():
                break
        
        obs = self._extract_observation(timestep)
        done = timestep.last()
        info = {"discount": discount}
        return obs, reward, done, False, info # obs, reward, terminated, truncated, info

    def close(self):
        # DMC environments typically don't need explicit closing beyond GC
        pass

def make_env(env_name: str, obs_type: str, action_repeat: int, seed: int, dmc_task: str) -> gym.Env:
    if env_name == "dmc":
        domain_name, task_name = dmc_task.split('-')
        return DeepMindControlEnv(domain_name, task_name, obs_type, action_repeat, seed)
    elif env_name == "gym":
        # For OpenAI Gym environments, we'd need a different wrapper or direct gym.make
        # This is a placeholder for potential future gym support as mentioned in paper
        env = gym.make(dmc_task) # dmc_task would be like "Walker2d-v2"
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    else:
        raise ValueError(f"Unsupported environment name: {env_name}")
