import abc
import numpy as np
from typing import Any, Dict, Tuple, Optional

# Attempt to import Config from config.py
# If config.py is not yet available (e.g., during isolated testing or initial setup),
# a dummy Config class will be used to prevent circular imports and allow this module to be tested.
try:
    from config import Config
except ImportError:
    # Dummy Config class for self-testing or if config.py is not yet available
    class Config:
        def __init__(self, data: Dict = None):
            self._data = data if data is not None else {}
        def get(self, key: str, default: Any = None) -> Any:
            keys = key.split('.')
            current = self._data
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return default
            return current
        def set(self, key: str, value: Any) -> None:
            keys = key.split('.')
            d = self._data
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        def save(self, output_path: str) -> None: pass
    print("Warning: Could not import 'Config' from 'config.py'. Using a dummy Config class.")


class BaseEnvironment(abc.ABC):
    """
    Abstract base class for reinforcement learning environments.
    Establishes a common interface for all environments (e.g., Sokoban, Mini PacMan)
    to ensure consistent interaction with agents, trainers, and interpretability modules.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the base environment with configuration settings.

        Args:
            config (Config): The configuration object holding all relevant environment parameters.
        """
        self.config: Config = config
        self._current_state: Optional[np.ndarray] = None
        self._action_space_size: Optional[int] = None
        self._observation_space_shape: Optional[Tuple[int, ...]] = None

    @abc.abstractmethod
    def reset(self, level_seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to an initial state at the beginning of an episode.
        If a level_seed is provided, it should deterministically generate or load a specific level.

        Args:
            level_seed (Optional[int]): Seed for level generation/selection for reproducibility.
                                        If None, a random level (or default) is chosen.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - The initial observation (symbolic representation) of the environment.
                - An info dictionary containing any additional episode-specific details.
        """
        raise NotImplementedError("Subclasses must implement the 'reset' method.")

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Applies the chosen action to the environment, advancing its state by one step.

        Args:
            action (int): The action taken by the agent.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]:
                - next_observation (np.ndarray): The observation of the environment after taking the action.
                - reward (float): The reward received from taking the action.
                - done (bool): True if the episode has terminated, False otherwise.
                - info (Dict[str, Any]): An info dictionary containing diagnostic information.
        """
        raise NotImplementedError("Subclasses must implement the 'step' method.")

    @abc.abstractmethod
    def render(self) -> np.ndarray:
        """
        Generates a visual representation (e.g., pixel-based image) of the current environment state.

        Returns:
            np.ndarray: A NumPy array representing the RGB image of the environment.
        """
        raise NotImplementedError("Subclasses must implement the 'render' method.")

    @abc.abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Returns the current full internal state of the environment. This is typically the
        symbolic observation `x_t` that the agent perceives.

        Returns:
            np.ndarray: The symbolic observation `x_t` of the current environment state.
        """
        raise NotImplementedError("Subclasses must implement the 'get_state' method.")

    @abc.abstractmethod
    def set_state(self, state: np.ndarray) -> None:
        """
        Sets the environment's internal state to a provided state. This is crucial for
        interpretability tasks like concept labeling or interventions.

        Args:
            state (np.ndarray): A NumPy array representing a valid environment state
                                (e.g., a symbolic observation `x_t`).
        """
        raise NotImplementedError("Subclasses must implement the 'set_state' method.")

    @abc.abstractmethod
    def get_action_space_size(self) -> int:
        """
        Returns the total number of discrete actions available to the agent in this environment.

        Returns:
            int: The size of the action space.
        """
        raise NotImplementedError("Subclasses must implement the 'get_action_space_size' method.")

    @abc.abstractmethod
    def get_observation_space_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the observation tensor (`x_t`).

        Returns:
            Tuple[int, ...]: A tuple representing the dimensions of the observation space.
                             (e.g., (grid_height, grid_width, num_channels)).
        """
        raise NotImplementedError("Subclasses must implement the 'get_observation_space_shape' method.")

    @abc.abstractmethod
    def simulate_future_state(self, current_state: np.ndarray, action: int) -> np.ndarray:
        """
        Calculates the next state that would result from taking 'action' in 'current_state',
        without altering the actual environment's internal state. This method assumes
        deterministic transition dynamics.

        Args:
            current_state (np.ndarray): The symbolic observation of a hypothetical current state.
            action (int): The action to simulate.

        Returns:
            np.ndarray: The symbolic observation of the state that would result from the simulation.
        """
        raise NotImplementedError("Subclasses must implement the 'simulate_future_state' method.")


if __name__ == '__main__':
    print("--- Testing BaseEnvironment abstract class ---")

    # Create a dummy config for testing
    dummy_config_data = {
        'environment': {
            'name': 'DummyEnv',
            'dummy_param': 123
        }
    }
    dummy_config = Config(dummy_config_data)

    # Attempt to instantiate BaseEnvironment (should fail as it's abstract)
    try:
        env = BaseEnvironment(dummy_config)
        print("Error: Instantiated BaseEnvironment directly, which should not happen.")
    except TypeError as e:
        print(f"Successfully caught expected error when instantiating BaseEnvironment: {e}")

    # Define a minimal concrete implementation for testing purposes
    class ConcreteEnvironment(BaseEnvironment):
        def __init__(self, config: Config):
            super().__init__(config)
            self._current_state = np.zeros((1, 1, 1))
            self._action_space_size = 4
            self._observation_space_shape = (1, 1, 1)
            print(f"ConcreteEnvironment initialized with dummy_param: {self.config.get('environment.dummy_param')}")

        def reset(self, level_seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
            print(f"Resetting environment (seed: {level_seed})")
            self._current_state = np.ones((1, 1, 1)) * (level_seed if level_seed is not None else 0)
            return self._current_state, {"episode_info": "reset"}

        def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
            print(f"Stepping with action: {action}")
            self._current_state += action
            reward = float(action)
            done = bool(action >= 3)
            return self._current_state, reward, done, {"step_info": f"action_{action}"}

        def render(self) -> np.ndarray:
            print("Rendering environment")
            return np.zeros((10, 10, 3), dtype=np.uint8) # Dummy RGB image

        def get_state(self) -> np.ndarray:
            print("Getting state")
            return self._current_state

        def set_state(self, state: np.ndarray) -> None:
            print(f"Setting state to: {state.flatten()}")
            self._current_state = state

        def get_action_space_size(self) -> int:
            print("Getting action space size")
            return self._action_space_size

        def get_observation_space_shape(self) -> Tuple[int, ...]:
            print("Getting observation space shape")
            return self._observation_space_shape

        def simulate_future_state(self, current_state: np.ndarray, action: int) -> np.ndarray:
            print(f"Simulating future state from {current_state.flatten()} with action {action}")
            return current_state + action # Simple simulation

    # Instantiate the concrete environment
    print("\n--- Instantiating ConcreteEnvironment ---")
    concrete_env = ConcreteEnvironment(dummy_config)

    # Test all implemented methods
    print("\n--- Testing ConcreteEnvironment methods ---")
    initial_obs, info = concrete_env.reset(level_seed=10)
    print(f"Initial Observation: {initial_obs.flatten()}, Info: {info}")

    obs, reward, done, info = concrete_env.step(1)
    print(f"Obs: {obs.flatten()}, Reward: {reward}, Done: {done}, Info: {info}")

    obs, reward, done, info = concrete_env.step(2)
    print(f"Obs: {obs.flatten()}, Reward: {reward}, Done: {done}, Info: {info}")

    obs, reward, done, info = concrete_env.step(3)
    print(f"Obs: {obs.flatten()}, Reward: {reward}, Done: {done}, Info: {info}")

    concrete_env.render()

    current_state = concrete_env.get_state()
    print(f"Current State: {current_state.flatten()}")

    new_state_to_set = np.array([[[100.]]])
    concrete_env.set_state(new_state_to_set)
    print(f"State after set_state: {concrete_env.get_state().flatten()}")

    action_size = concrete_env.get_action_space_size()
    print(f"Action Space Size: {action_size}")

    obs_shape = concrete_env.get_observation_space_shape()
    print(f"Observation Space Shape: {obs_shape}")

    simulated_next_state = concrete_env.simulate_future_state(concrete_env.get_state(), 5)
    print(f"Simulated Next State: {simulated_next_state.flatten()}")

    print("\n--- BaseEnvironment testing complete ---")
