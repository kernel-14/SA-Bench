```python
import numpy as np
import torch
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Callable, Optional, Union

# Attempt to import core dependencies
try:
    from config import Config
    from environments.base_environment import BaseEnvironment
    from agents.base_agent import BaseAgentModel
    from agents.drc_agent import DRCAgent
    from agents.resnet_agent import ResNetAgent
except ImportError:
    # Dummy classes for standalone testing or if dependencies are not yet available
    print("Warning: Could not import core dependencies. Using dummy classes for TrajectoryCollector.")

    class Config:
        """Dummy Config class for self-testing."""
        def __init__(self, data: Dict = None): self._data = data if data is not None else {}
        def get(self, key: str, default: Any = None) -> Any:
            keys = key.split('.')
            current = self._data
            for k in keys:
                if isinstance(current, dict) and k in current: current = current[k]
                else: return default
            return current
        def set(self, key: str, value: Any) -> None: pass
        def save(self, output_path: str) -> None: pass

    class BaseEnvironment:
        """Dummy BaseEnvironment class for self-testing."""
        def __init__(self, config: Config) -> None: self.config = config
        def reset(self, level_config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]: return np.zeros((8,8,7)), {}
        def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: # Added truncated for gymnasium API
             return np.zeros((8,8,7)), 0.0, False, False, {'is_success': False}
        def get_action_space_size(self) -> int: return 5
        def get_observation_space_shape(self) -> Tuple[int, ...]: return (8,8,7)
        @property
        def episode_length_max(self) -> int: return 120 # Dummy value

    class BaseAgentModel(torch.nn.Module):
        """Dummy BaseAgentModel class for self-testing."""
        def __init__(self, config: Config) -> None:
            super().__init__()
            self.config = config
            self.device = torch.device("cpu")
        def act(self, obs: np.ndarray, hidden_state: Any = None, greedy: bool = True) -> Tuple[int, Any, torch.Tensor, torch.Tensor]:
            return 0, None, torch.zeros(5), torch.zeros(1)
        def get_cell_state(self, layer_idx: int, tick_idx: int = -1) -> torch.Tensor:
            # Dummy state shape (H, W, G)
            H, W = self.config.get('environment.sokoban.grid_size', [8,8])
            G = self.config.get('agent.drc_agent.convlstm_channels', 32) # Common channel size
            return torch.randn(H, W, G)

    class DRCAgent(BaseAgentModel):
        """Dummy DRCAgent class for self-testing."""
        def __init__(self, config: Config) -> None:
            super().__init__(config)
            self.D = config.get('agent.drc_agent.D', 3)
            self.N = config.get('agent.drc_agent.N', 3)
            self.convlstm_channels = config.get('agent.drc_agent.convlstm_channels', 32)
            # Dummy lists to simulate _h_states_per_tick_layer and _c_states_per_tick_layer
            # Initialize with dummy tensors of correct shape (1, G_d, H, W)
            H, W = self.config.get('environment.sokoban.grid_size', [8,8])
            self._h_states_per_tick_layer = []
            self._c_states_per_tick_layer = []
            for _tick in range(self.N + 1):
                tick_h_states = []
                tick_c_states = []
                for _layer in range(self.D):
                    tick_h_states.append(torch.randn(1, self.convlstm_channels, H, W))
                    tick_c_states.append(torch.randn(1, self.convlstm_channels, H, W))
                self._h_states_per_tick_layer.append(tick_h_states)
                self._c_states_per_tick_layer.append(tick_c_states)
            
        def get_cell_state(self, layer_idx: int, tick_idx: int = -1) -> torch.Tensor:
            actual_tick_idx: int = tick_idx if tick_idx != -1 else self.N
            return self._c_states_per_tick_layer[actual_tick_idx][layer_idx].squeeze(0).permute(1,2,0) # (H,W,G)

        def _init_hidden_states(self, device: torch.device, channels: int, H: int, W: int) -> List[torch.Tensor]:
            return [torch.zeros(1, channels, H, W, device=device) for _ in range(self.D)]

    class ResNetAgent(BaseAgentModel):
        """Dummy ResNetAgent class for self-testing."""
        def __init__(self, config: Config) -> None:
            super().__init__(config)
            self.num_residual_blocks = config.get('agent.resnet_agent.num_residual_blocks', 24)
            self.block_channels = config.get('agent.resnet_agent.block_channels', 32)
            # Dummy _cached_activations
            H, W = self.config.get('environment.sokoban.grid_size', [8,8])
            self._cached_activations = {i: torch.randn(1, self.block_channels, H, W) for i in range(self.num_residual_blocks)}

        def get_cell_state(self, layer_idx: int, tick_idx: int = -1) -> torch.Tensor:
            # ResNet stores activations as (B, C, H, W), need to convert to (H, W, C)
            H, W = self.config.get('environment.sokoban.grid_size', [8,8])
            C = self.block_channels
            # Ensure _cached_activations is populated, otherwise provide default
            if layer_idx not in self._cached_activations:
                return torch.randn(H, W, C) # Fallback to random if not in cache
            return self._cached_activations[layer_idx].squeeze(0).permute(1,2,0) # (H,W,G)


class TrajectoryCollector:
    """
    The TrajectoryCollector class is responsible for running the agent in an environment
    and storing complete interaction sequences (trajectories). For interpretability,
    it meticulously saves not only observations, actions, and rewards, but also the
    agent's internal cell states at each internal tick (if applicable).
    """

    def __init__(self, agent_model: BaseAgentModel, env: BaseEnvironment, config: Config) -> None:
        """
        Initializes the TrajectoryCollector.

        Args:
            agent_model (BaseAgentModel): An instance of the agent model (e.g., DRCAgent, ResNetAgent).
            env (BaseEnvironment): An instance of the environment (e.g., SokobanEnv, MiniPacManEnv).
            config (Config): The configuration object containing experiment settings.
        """
        self.agent_model: BaseAgentModel = agent_model
        self.env: BaseEnvironment = env
        self.config: Config = config
        self.device: torch.device = agent_model.device

        self.inference_greedy: bool = self.config.get('rl_training.inference_greedy', True)

        # Determine agent-specific parameters for cell state collection
        agent_type: str = self.config.get('agent.type', 'DRCAgent')
        if agent_type == 'DRCAgent':
            self.num_layers: int = self.config.get('agent.drc_agent.D', 3)
            self.num_ticks: int = self.config.get('agent.drc_agent.N', 3) + 1 # +1 for tick 0 (s_t-1)
            self.cell_channels: int = self.config.get('agent.drc_agent.convlstm_channels', 32)
        elif agent_type == 'ResNetAgent':
            self.num_layers: int = self.config.get('agent.resnet_agent.num_residual_blocks', 24)
            self.num_ticks: int = 1 # ResNet has no internal ticks, so we collect state after each block
            self.cell_channels: int = self.config.get('agent.resnet_agent.block_channels', 32)
        else:
            raise ValueError(f"Unsupported agent type for TrajectoryCollector: {agent_type}")
        
        # Get grid dimensions for cell state shape checking
        env_name: str = self.config.get("environment.name", "Sokoban")
        if env_name == "Sokoban":
            self.grid_height: int = self.config.get('environment.sokoban.grid_size', [8, 8])[0]
            self.grid_width: int = self.config.get('environment.sokoban.grid_size', [8, 8])[1]
        elif env_name == "MiniPacMan":
            self.grid_height: int = self.config.get('environment.mini_pacman.grid_size', [13, 13])[0]
            self.grid_width: int = self.config.get('environment.mini_pacman.grid_size', [13, 13])[1]
        else:
            raise ValueError(f"Unsupported environment for TrajectoryCollector: {env_name}.")

    def collect_trajectories(
        self,
        num_episodes: int,
        behavior_policy: Callable[[np.ndarray, Any, bool], Tuple[int, Any, torch.Tensor, torch.Tensor]],
        store_cell_states: bool = True,
        level_configs: Optional[List[Dict[str, Any]]] = None # For specific evaluation levels
    ) -> List[Dict[str, Any]]:
        """
        Collects a specified number of trajectories (episodes) by running the agent in the environment.
        Optionally stores detailed internal cell states for interpretability.

        Args:
            num_episodes (int): The total number of episodes to collect.
            behavior_policy (Callable): A function that dictates the agent's action selection.
                                       It takes (obs, hidden_state, greedy) and returns
                                       (action, next_hidden_state, policy_logits, value_estimate).
            store_cell_states (bool, optional): If True, internal cell states at each layer and tick
                                                are collected. Defaults to True.
            level_configs (Optional[List[Dict[str, Any]]]): A list of dictionaries, where each dict
                                                           specifies the configuration for one level.
                                                           If None, random levels are used.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a complete
                                  episode's trajectory and contains lists of collected data.
        """
        all_trajectories: List[Dict[str, Any]] = []

        self.agent_model.eval() # Set agent to evaluation mode for data collection

        for episode_idx in tqdm(range(num_episodes), desc="Collecting Trajectories"):
            episode_data: Dict[str, Any] = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'next_observations': [],
                'dones': [],
                'infos': [],
                'hidden_states': [],        # Input hidden state to act()
                'policy_logits': [],
                'value_estimates': [],
                'cell_states': [] if store_cell_states else None
            }

            # If specific level configs are provided, use one for this episode
            current_level_config: Optional[Dict[str, Any]] = None
            if level_configs is not None and episode_idx < len(level_configs