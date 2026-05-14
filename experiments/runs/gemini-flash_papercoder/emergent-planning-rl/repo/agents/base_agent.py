import abc
import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from typing import Any, Tuple, Optional

# Assuming Config class is available from config.py
# If config.py is not yet available (e.g., during isolated testing or initial setup),
# a dummy Config class will be used to prevent circular imports and allow this module to be tested.
try:
    from config import Config
except ImportError:
    # Dummy Config class for self-testing or if config.py is not yet available
    class Config:
        def __init__(self, data: dict = None):
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


class BaseAgentModel(nn.Module, abc.ABC):
    """
    Abstract base class for all reinforcement learning agent models.
    It defines the common interface for agents, including forward pass,
    action selection, and methods for accessing/modifying internal states
    for interpretability experiments.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the base agent model.

        Args:
            config (Config): A Config object containing hyperparameters and settings.
        """
        super().__init__()
        self.config: Config = config
        
        # Determine the device for computations
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get environment-specific dimensions from config
        env_name: str = self.config.get("environment.name", "Sokoban")
        
        if env_name == "Sokoban":
            self.action_space_size: int = self.config.get('environment.sokoban.action_space_size', 5) # 4 directions + no-op
            grid_size: list = self.config.get('environment.sokoban.grid_size', [8, 8])
            obs_channels: int = self.config.get('environment.sokoban.observation_channels', 7)
        elif env_name == "MiniPacMan":
            self.action_space_size: int = self.config.get('environment.mini_pacman.action_space_size', 5) # 4 directions + no-op
            grid_size: list = self.config.get('environment.mini_pacman.grid_size', [13, 13])
            obs_channels: int = self.config.get('environment.mini_pacman.observation_channels', 14)
        else:
            raise ValueError(f"Unsupported environment configured: {env_name}. Check config.yaml.")
        
        self.observation_space_shape: Tuple[int, ...] = tuple(grid_size) + (obs_channels,)
        self.grid_height: int = grid_size[0]
        self.grid_width: int = grid_size[1]
        self.obs_channels: int = obs_channels

        # Policy and Value heads will be defined in concrete subclasses.
        # They are initialized here as None and expected to be set by _build_model_architecture.
        self._policy_head: Optional[nn.Module] = None
        self._value_head: Optional[nn.Module] = None

        # Hidden state for recurrent agents. Will be updated by forward pass.
        # This will be instance-specific for an agent running in an environment.
        self.hidden_state: Any = None 

        # Build the specific model architecture (implemented by subclasses)
        self._build_model_architecture()

        # Move the entire model to the determined device
        self.to(self.device)

    @abc.abstractmethod
    def _build_model_architecture(self) -> None:
        """
        Abstract method to build the specific neural network architecture
        for the agent (e.g., encoder, recurrent core, policy/value heads).
        Concrete subclasses must implement this.
        This method should initialize self._policy_head and self._value_head.
        """
        pass

    @abc.abstractmethod
    def forward(self, obs: torch.Tensor, hidden_state: Any = None) -> Tuple[torch.Tensor, Any]:
        """
        Abstract method for the agent's forward pass.

        Args:
            obs (torch.Tensor): A batch of observations. Expected shape (batch_size, H, W, C).
                                The implementation should handle transposing if necessary (e.g., to (batch_size, C, H, W)).
            hidden_state (Any, optional): The agent's recurrent hidden state (e.g., (h, c) for LSTMs).
                                          Can be None for feedforward agents or initial state. Defaults to None.

        Returns:
            Tuple[torch.Tensor, Any]:
            - model_output_vector (torch.Tensor): The processed feature vector (o_t in the paper)
                                                  before being fed into the policy and value heads.
                                                  Expected shape (batch_size, feature_dim).
            - new_hidden_state (Any): The updated recurrent hidden state, or None if not applicable.
        """
        pass

    def get_action_logits(self, model_output_vector: torch.Tensor) -> torch.Tensor:
        """
        Computes action logits from the model's output vector.

        Args:
            model_output_vector (torch.Tensor): The feature vector (o_t) from the forward pass.
                                                Expected shape (batch_size, feature_dim).

        Returns:
            torch.Tensor: A tensor of action logits. Expected shape (batch_size, action_space_size).
        """
        if self._policy_head is None:
            raise NotImplementedError("Policy head must be initialized in _build_model_architecture before calling get_action_logits.")
        return self._policy_head(model_output_vector)

    def get_value_estimate(self, model_output_vector: torch.Tensor) -> torch.Tensor:
        """
        Computes the value estimate from the model's output vector.

        Args:
            model_output_vector (torch.Tensor): The feature vector (o_t) from the forward pass.
                                                Expected shape (batch_size, feature_dim).

        Returns:
            torch.Tensor: A tensor representing the estimated value. Expected shape (batch_size, 1).
        """
        if self._value_head is None:
            raise NotImplementedError("Value head must be initialized in _build_model_architecture before calling get_value_estimate.")
        return self._value_head(model_output_vector)

    @abc.abstractmethod
    def get_cell_state(self, layer_idx: int, tick_idx: int = -1) -> torch.Tensor:
        """
        Abstract method to retrieve a specific internal cell state or activation
        for probing. Concrete subclasses must implement this.

        Args:
            layer_idx (int): The 0-indexed number of the ConvLSTM layer or ResNet block.
            tick_idx (int, optional): For recurrent agents, the 0-indexed internal computational tick.
                                      -1 typically refers to the state after the final tick. Defaults to -1.

        Returns:
            torch.Tensor: A tensor representing the requested internal state.
                          Expected shape (H, W, G) for spatially-structured states.
                          Note: This should be a single (H, W, G) state, not batched.
        """
        pass

    @abc.abstractmethod
    def set_cell_state(self, layer_idx: int, tick_idx: int, new_state: torch.Tensor) -> None:
        """
        Abstract method to modify a specific internal cell state or activation
        for intervention experiments. Concrete subclasses must implement this.

        Args:
            layer_idx (int): The 0-indexed number of the ConvLSTM layer or ResNet block.
            tick_idx (int): For recurrent agents, the 0-indexed internal computational tick.
                            -1 typically refers to modifying the state after the final tick.
            new_state (torch.Tensor): The new tensor to set as the internal state.
                                      Expected shape (H, W, G) for spatially-structured states.
                                      Note: This new_state should be a single (H, W, G) state, not batched.
        """
        pass

    @torch.no_grad() # Ensure no gradients are computed for inference
    def act(self, obs: np.ndarray, hidden_state: Any = None, greedy: bool = True) -> Tuple[int, Any, torch.Tensor, torch.Tensor]:
        """
        Selects an action based on the current observation and agent policy.

        Args:
            obs (np.ndarray): The current observation as a numpy array. Expected shape (H, W, C).
            hidden_state (Any, optional): The agent's recurrent hidden state (e.g., (h, c) for LSTMs).
                                          Can be None for feedforward agents or initial state. Defaults to None.
            greedy (bool): If True, selects the action with the highest logit.
                           If False, samples an action from the policy distribution. Defaults to True.

        Returns:
            Tuple[int, Any, torch.Tensor, torch.Tensor]:
            - action (int): The chosen action.
            - new_hidden_state (Any): The updated recurrent hidden state, or None.
            - policy_logits (torch.Tensor): The raw logits from the policy head for the selected observation.
                                            Expected shape (action_space_size,).
            - value_estimate (torch.Tensor): The estimated value from the value head for the selected observation.
                                             Expected shape (1,).
        """
        # Convert numpy observation to torch tensor, add batch dimension, and move to device
        # Expected observation format is (H, W, C), which for Conv2d typically needs to be (C, H, W)
        torch_obs: torch.Tensor = torch.from_numpy(obs).float().to(self.device).permute(2, 0, 1).unsqueeze(0)

        # Perform the forward pass to get model output vector and new hidden state
        model_output_vector, new_hidden_state = self.forward(torch_obs, hidden_state)

        # Get policy logits and value estimate from the model output vector
        # Squeeze batch dimension (0) for return values as we're acting on a single observation
        policy_logits: torch.Tensor = self.get_action_logits(model_output_vector).squeeze(0)
        value_estimate: torch.Tensor = self.get_value_estimate(model_output_vector).squeeze(0)

        # Action selection
        if greedy:
            action: int = torch.argmax(policy_logits, dim=-1).item()
        else:
            action_distribution: dist.Categorical = dist.Categorical(logits=policy_logits)
            action: int = action_distribution.sample().item()
        
        return action, new_hidden_state, policy_logits, value_estimate

