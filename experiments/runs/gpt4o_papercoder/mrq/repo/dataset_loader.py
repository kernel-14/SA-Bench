"""
dataset_loader.py
Handles loading of environments and data preprocessing to unify RL benchmarks (Gym, DMC, Atari) with specifications
outlined in the MR.Q reproducibility plan.
"""

import gym
import cv2
import numpy as np
from typing import Any, Dict
from utils import Utils


class DatasetLoader:
    """Class to handle dataset/environment loading and preprocessing for RL benchmarks."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DatasetLoader with configuration for environments and preprocessing.

        Args:
            config (Dict[str, Any]): Configuration dictionary parsed from config.yaml.
        """
        self.env_config = config.get("environments", {})
        self.preproc_config = config.get("preprocessing", {})  # e.g., image_size, grayscale, frame_stacking
        self.action_repeat = config["training"]["action_repeat"]  # Default action_repeat for DMC
        self.image_size = tuple(self.preproc_config.get("image_size", [84, 84]))  # (width, height) for resizing
        self.grayscale = self.preproc_config.get("grayscale", True)  # Boolean: Apply grayscale conversion or not
        self.frame_stacking = self.preproc_config.get("frame_stacking", {}).get("enabled", True)
        self.num_frames = self.preproc_config.get("frame_stacking", {}).get("num_frames", 4)

    def load_env(self, env_name: str) -> Any:
        """
        Load the environment for the given name, applying any relevant wrappers.

        Args:
            env_name (str): Name of the environment to load.

        Returns:
            Any: Loaded environment object with wrappers (if applicable).
        """
        # Identify the benchmark for the task
        if env_name in self.env_config.get("gym_tasks", []):
            env = gym.make(env_name)  # Load Gym's continuous control task
            return self.apply_wrappers(env, mode="vector")
        elif env_name in self.env_config.get("dm_control_proprioceptive_tasks", []):
            from dm_control import suite
            domain_task = env_name.split("-")
            env = suite.load(domain=domain_task[0], task=domain_task[1])
            return self.apply_wrappers(env, mode="vector", action_repeat=self.action_repeat)
        elif env_name in self.env_config.get("dm_control_visual_tasks", []):
            from dm_control import suite
            domain_task = env_name.split("-")
            env = suite.load(domain=domain_task[0], task=domain_task[1])
            return self.apply_wrappers(env, mode="image", action_repeat=self.action_repeat)
        elif env_name in self.env_config.get("atari_tasks", []):
            env = gym.make(env_name)
            return self.apply_wrappers(env, mode="image")
        else:
            raise ValueError(f"Environment '{env_name}' is not defined in the configuration.")

    def preprocess_data(self, data: Any, mode: str = "vector") -> Any:
        """
        Preprocess the observation data based on the mode (vector or image).

        Args:
            data (Any): Environment observation data.
            mode (str): Either 'vector' (raw vector data) or 'image' (pixel data).

        Returns:
            Any: Preprocessed observation.
        """
        if mode == "vector":
            # No changes for vector inputs; can add normalization if required
            return data
        elif mode == "image":
            # Handle image-based preprocessing
            if self.grayscale:
                data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(data, self.image_size, interpolation=cv2.INTER_AREA)
            normalized = resized / 255.0  # Scale pixel values to [0, 1]
            return np.expand_dims(normalized, axis=0)  # Add channel dimension for grayscale
        else:
            raise ValueError(f"Unsupported mode '{mode}'. Use 'vector' or 'image'.")

    def apply_wrappers(self, env: Any, mode: str, action_repeat: int = 1) -> Any:
        """
        Apply task-specific environment wrappers such as frame stacking and action repeats.

        Args:
            env (Any): The environment object.
            mode (str): Environment type mode ('vector' or 'image').
            action_repeat (int): Number of repeated actions in DMC (default: 1).

        Returns:
            Any: Wrapped environment.
        """
        # Apply action repeats for DMC environments
        if action_repeat > 1:
            from dm_control.rl.environment import TimeStep
            class ActionRepeatWrapper:
                def __init__(self, env, repeat_count):
                    self._env = env
                    self._repeat_count = repeat_count

                def step(self, action):
                    total_reward = 0.0
                    for _ in range(self._repeat_count):
                        timestep = self._env.step(action)
                        total_reward += timestep.reward
                        if timestep.last():
                            break
                    return TimeStep(timestep.step_type, total_reward, timestep.discount, timestep.observation)

                def reset(self):
                    return self._env.reset()

                def __getattr__(self, name):
                    return getattr(self._env, name)

            env = ActionRepeatWrapper(env, action_repeat)

        # Frame stacking for Atari or visual-based observations
        if mode == "image" and self.frame_stacking:
            from gym.wrappers.frame_stack import FrameStack
            env = FrameStack(env, num_stack=self.num_frames)

        return env

    def update_env_config(self, config: Dict[str, Any]) -> None:
        """
        Update environment configuration at runtime. Useful for changing preprocessing settings.

        Args:
            config (Dict[str, Any]): Updated configuration dictionary.
        """
        preprocessing_config = config.get("preprocessing", {})
        self.image_size = tuple(preprocessing_config.get("image_size", self.image_size))
        self.grayscale = preprocessing_config.get("grayscale", self.grayscale)
        self.frame_stacking = preprocessing_config.get("frame_stacking", {}).get("enabled", self.frame_stacking)
        self.num_frames = preprocessing_config.get("frame_stacking", {}).get("num_frames", self.num_frames)
        self.action_repeat = config.get("training", {}).get("action_repeat", self.action_repeat)
