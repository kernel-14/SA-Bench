import torch
import torch.nn.functional as F
import numpy as np

from config import AgentConfig
from model import DRC, ResNetAgent

class DRCAgent:
    """
    A wrapper for the DRC model to handle action selection, state management,
    and interaction with the environment.
    """
    def __init__(self, config: AgentConfig, drc_model: DRC, device: torch.device):
        self.config = config
        self.drc_model = drc_model.to(device)
        self.device = device
        self.hidden_states = None # Stores (h,c) for each ConvLSTM layer

    def reset(self):
        """Resets the internal hidden states of the ConvLSTM layers."""
        self.hidden_states = None

    def get_action(self, observation: np.ndarray, greedy: bool = True) -> int:
        """
        Selects an action based on the current observation.
        
        Args:
            observation (np.ndarray): The current environment observation.
                                      Shape: (H, W, Channels)
            greedy (bool): If True, selects the action with the highest logit.
                           If False, samples an action from the policy distribution.
        
        Returns:
            int: The selected action.
        """
        # Convert observation to PyTorch tensor and move to device
        # From (H, W, Channels) to (1, Channels, H, W) for batch_size=1
        obs_tensor = torch.from_numpy(observation).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value, new_states, all_cell_states_per_tick = self.drc_model(obs_tensor, self.hidden_states)
            self.hidden_states = new_states # Update internal states

            if greedy:
                action = torch.argmax(policy_logits, dim=1).item()
            else:
                action = torch.distributions.Categorical(logits=policy_logits).sample().item()
        
        return action

    def get_forward_pass_data(self, observation: np.ndarray, prev_states: list[tuple[torch.Tensor, torch.Tensor]] | None = None) -> \
            tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], list[list[tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Performs a forward pass and returns policy logits, value, new states, and all cell states
        per tick for probing. Used during training and evaluation.
        
        Args:
            observation (np.ndarray): Current environment observation.
            prev_states (list[tuple[torch.Tensor, torch.Tensor]] | None): Previous hidden states.
        
        Returns:
            tuple: (policy_logits, value, new_states, all_cell_states_per_tick)
        """
        obs_tensor = torch.from_numpy(observation).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        policy_logits, value, new_states, all_cell_states_per_tick = self.drc_model(obs_tensor, prev_states)
        return policy_logits, value, new_states, all_cell_states_per_tick
    
    def get_all_cell_states_for_tick(self, observation: np.ndarray, prev_states: list[tuple[torch.Tensor, torch.Tensor]] | None = None) -> \
            tuple[list[tuple[torch.Tensor, torch.Tensor]], list[list[tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Performs N internal ticks and returns the final (h,c) states and all (h,c) states per layer per tick.
        This is for data collection for probing.
        """
        obs_tensor = torch.from_numpy(observation).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, new_states, all_cell_states_per_tick = self.drc_model(obs_tensor, prev_states)
        return new_states, all_cell_states_per_tick
    
    def get_model(self) -> DRC:
        return self.drc_model

class ResNetActorCritic:
    """
    A wrapper for the ResNetAgent model for actor-critic training and action selection.
    """
    def __init__(self, config: AgentConfig, resnet_model: ResNetAgent, device: torch.device):
        self.config = config
        self.resnet_model = resnet_model.to(device)
        self.device = device

    def get_action(self, observation: np.ndarray, greedy: bool = True) -> int:
        """
        Selects an action based on the current observation.
        
        Args:
            observation (np.ndarray): The current environment observation.
                                      Shape: (H, W, Channels)
            greedy (bool): If True, selects the action with the highest logit.
                           If False, samples an action from the policy distribution.
        
        Returns:
            int: The selected action.
        """
        obs_tensor = torch.from_numpy(observation).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value, hidden_states_per_layer = self.resnet_model(obs_tensor)

            if greedy:
                action = torch.argmax(policy_logits, dim=1).item()
            else:
                action = torch.distributions.Categorical(logits=policy_logits).sample().item()
        
        return action

    def get_forward_pass_data(self, observation: np.ndarray) -> \
            tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Performs a forward pass and returns policy logits, value, and hidden states
        per layer for probing. Used during training and evaluation.
        
        Args:
            observation (np.ndarray): Current environment observation.
        
        Returns:
            tuple: (policy_logits, value, hidden_states_per_layer)
        """
        obs_tensor = torch.from_numpy(observation).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        policy_logits, value, hidden_states_per_layer = self.resnet_model(obs_tensor)
        return policy_logits, value, hidden_states_per_layer
    
    def get_model(self) -> ResNetAgent:
        return self.resnet_model

