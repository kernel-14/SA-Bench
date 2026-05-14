# data_loader.py

import numpy as np
import gym
from typing import Tuple, List, Dict, Any
from dm_control import suite
from dm_env import specs


class DataLoader:
    """
    Handles the setup of reinforcement learning environments (Gym/OpenAI/DMC).
    Includes environment initialization, data collection, and preprocessing.
    """

    def __init__(self, env_name: str, config: dict):
        """
        Initialize the DataLoader object.

        Args:
            env_name (str): Name of the environment type ('DMC' or 'Gym').
            config (dict): Configuration dictionary parsed from config.yaml.
        """
        self.env_name = env_name
        self.task = config["environment"]["task"]
        self.pixel_based = config["environment"]["pixel_based"]
        self.config = config

        # Internal environment state
        self.env = None
        self.is_dmc = self.env_name.lower() == "dmc"

    def setup_environment(self) -> Tuple[Any, dict]:
        """
        Sets up and initializes the environment as per the configuration.

        Returns:
            Tuple[Env, dict]: Initialized environment object and configuration dictionary.
        """
        if self.is_dmc:
            # Setup DeepMind Control Suite environment
            try:
                domain_name, task_name = self.task.split("-")
                self.env = suite.load(domain_name=domain_name, task_name=task_name)
                env_config = {
                    "observation_spec": self.env.observation_spec(),
                    "action_spec": self.env.action_spec(),
                }
            except Exception as e:
                raise ValueError(f"Error initializing DMC environment '{self.task}': {e}")
        else:
            # Setup Gym environment
            try:
                self.env = gym.make(self.task)
                env_config = {
                    "observation_space": self.env.observation_space,
                    "action_space": self.env.action_space,
                }
            except Exception as e:
                raise ValueError(f"Error initializing Gym environment '{self.task}': {e}")

        return self.env, env_config

    @staticmethod
    def preprocess_observation(observation: Any, pixel_based: bool) -> np.ndarray:
        """
        Preprocesses the observation depending on pixel or state-based tasks.

        Args:
            observation (Any): Raw observation from the environment.
            pixel_based (bool): Flag indicating if the task is pixel-based.

        Returns:
            np.ndarray: Preprocessed observation.
        """
        if pixel_based:
            # Normalize pixel observations to [0, 1]
            if isinstance(observation, dict):  # Handle DMC-style multi-channel pixel input
                observation = observation.get("pixels", observation)
            if isinstance(observation, np.ndarray):
                return observation.astype(np.float32) / 255.0  # Normalize image
            else:
                raise ValueError("Pixel-based observation is not a numpy array.")
        else:
            # For state-based tasks, convert to numpy array if not already
            if isinstance(observation, np.ndarray):
                return observation
            elif isinstance(observation, dict):
                # Handle DMC structured dict observations
                return np.concatenate([obs.flatten() for obs in observation.values()])
            else:
                return np.array(observation, dtype=np.float32)

    def get_real_transitions(self, num_transitions: int) -> List[Dict[str, Any]]:
        """
        Collect transitions from the environment.

        Args:
            num_transitions (int): Number of transitions to collect.

        Returns:
            List[Dict[str, Any]]: List of transitions in the form:
                {'s': current_state, 'a': action, 
                 's_prime': next_state, 'r': reward, 'done': done_flag}
        """
        if not self.env:
            raise RuntimeError("Environment is not initialized. Call setup_environment first.")

        transitions = []
        observation = self.reset_environment()  # Reset the environment to start
        
        for _ in range(num_transitions):
            action = self.sample_random_action()  # Replace with policy-derived action
            next_observation, reward, done, _ = self._step_environment(action)

            # Store transition
            transition = {
                "s": observation,
                "a": action,
                "s_prime": next_observation,
                "r": reward,
                "done": done,
            }
            transitions.append(transition)

            # Prepare for next iteration
            observation = next_observation
            if done:  # Reset the environment if the episode ends
                observation = self.reset_environment()

        return transitions

    def reset_environment(self) -> np.ndarray:
        """
        Resets the environment to its initial state.

        Returns:
            np.ndarray: Initial observation after resetting the environment.
        """
        if self.is_dmc:
            time_step = self.env.reset()
            return self.preprocess_observation(time_step.observation, self.pixel_based)
        else:
            observation = self.env.reset()
            return self.preprocess_observation(observation, self.pixel_based)

    def _step_environment(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Executes the given action in the environment.

        Args:
            action (np.ndarray): Action to execute.

        Returns:
            Tuple[np.ndarray, float, bool, dict]:
                - next_observation (np.ndarray)
                - reward (float)
                - done (bool): Whether the episode has terminated.
                - info (dict): Environment-specific metadata or diagnostics (ignored for now).
        """
        if self.is_dmc:
            time_step = self.env.step(action)
            next_observation = self.preprocess_observation(time_step.observation, self.pixel_based)
            reward = time_step.reward or 0.0  # Handle possible None reward
            done = time_step.last()  # Check terminal condition
            return next_observation, reward, done, {}
        else:
            next_observation, reward, done, info = self.env.step(action)
            next_observation = self.preprocess_observation(next_observation, self.pixel_based)
            return next_observation, reward, done, info

    def sample_random_action(self) -> np.ndarray:
        """
        Samples a random action from the environment's action space.

        Returns:
            np.ndarray: Random action from the action space.
        """
        if self.is_dmc:
            action_spec = self.env.action_spec()
            return np.random.uniform(
                action_spec.minimum, action_spec.maximum, action_spec.shape
            )
        else:
            return self.env.action_space.sample()
