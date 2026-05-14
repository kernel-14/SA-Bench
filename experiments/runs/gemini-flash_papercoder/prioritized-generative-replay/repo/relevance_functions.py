```python
import collections
import math
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.transform import resize

# Assuming these are available in the project structure
from config import Config
from replay_buffers import ReplayBuffer
from models.policy_nets import PolicyNetwork, QNetwork
from models.relevance_nets import (
    Comparator,
    EmbeddingNetwork,
    FeatureEncoder,
    ForwardDynamicsModel,
    PredictorNetwork,
    TargetNetwork,
)


# Helper function to preprocess pixel observations (for CTS and potentially others if needed)
def _preprocess_pixel_obs(
    obs: Union[np.ndarray, torch.Tensor],
    target_size: Tuple[int, int],
    context_bins: Optional[int] = None,
) -> Union[np.ndarray, Tuple[int, ...]]:
    """
    Resizes and optionally quantizes a pixel observation for CTS.
    Assumes input obs is (Batch, C, H, W) or (C, H, W) float32 [0,1].
    """
    # Convert to NumPy if it's a torch.Tensor
    if isinstance(obs, torch.Tensor):
        obs_np = obs.cpu().numpy()
    else:
        obs_np = obs

    # Handle batch dimension if present
    if obs_np.ndim == 4: # (B, C, H, W)
        obs_np = obs_np[0] # Take the first sample from the batch for individual processing
    elif obs_np.ndim != 3: # Expected (C, H, W)
        raise ValueError(f"Expected 3D (C,H,W) or 4D (B,C,H,W) observation, got {obs_np.shape}")

    # Permute to (H, W, C) for skimage.resize
    if obs_np.shape[0] < obs_np.shape[1] and obs_np.shape[0] < obs_np.shape[2]: # Assume C,H,W
        obs_np = np.transpose(obs_np, (1, 2, 0))

    # Resize
    # preserve_range=True to keep float [0,1] if input was float.
    # anti_aliasing=True for better image quality.
    resized_obs_np = resize(obs_np, target_size, anti_aliasing=True, preserve_range=True)

    if context_bins is not None:
        # Quantize to 0-context_bins-1 integer range
        # Scale float [0, 1] to integer [0, context_bins-1]
        quantized_obs_np = np.floor(resized_obs_np * (context_bins - 1)).astype(np.uint8)
        # Flatten and convert to tuple for a hashable key
        return tuple(quantized_obs_np.flatten())
    
    # Return resized float image if no quantization
    return resized_obs_np


# --- Abstract Base Class ---


class RelevanceFunction(ABC):
    """
    Abstract base class for all relevance functions.
    Defines common interface for computing scores and updating internal models.
    """

    def __init__(self, config: Config, state_dim: Union[int, Tuple[int, ...]], action_dim: int, device: torch.device):
        """
        Initializes the relevance function.

        Args:
            config (Config): Configuration object containing hyperparameters.
            state_dim (Union[int, Tuple[int, ...]]): Dimension of the state space.
                                                      int for vector states, Tuple for pixel states (C, H, W).
            action_dim (int): Dimension of the action space.
            device (torch.device): The device (e.g., 'cuda' or 'cpu') for models and tensors.
        """
        self.config: Config = config
        self.state_dim: Union[int, Tuple[int, ...]] = state_dim
        self.action_dim: int = action_dim
        self.device: torch.device = device
        self.pixel_based: bool = config.get_hyperparam('environment.pixel_based')

    @abstractmethod
    def compute_score(self, batch: Dict[str, torch.Tensor], policy_nets: Optional[Tuple[PolicyNetwork, QNetwork, Optional[QNetwork]]] = None) -> torch.Tensor:
        """
        Abstract method: Calculates and returns a batch of relevance scores (scalar 'c' values)
        for the given batch of transitions.

        Args:
            batch (Dict[str, torch.Tensor]): A dictionary of transition components.
            policy_nets (Optional[Tuple[PolicyNetwork, QNetwork, Optional[QNetwork]]]): An optional tuple containing
                                                                    the current PolicyNetwork (actor), QNetwork (critic),
                                                                    and optionally a target QNetwork for TD-error.
                                                                    The exact contents depend on the relevance function.

        Returns:
            torch.Tensor: A tensor of relevance scores (batch_size, 1).
        """
        pass

    @abstractmethod
    def update(self, real_buffer: ReplayBuffer, policy_nets: Optional[Tuple[PolicyNetwork, QNetwork, Optional[QNetwork]]] = None) -> None:
        """
        Abstract method: Updates the internal models/parameters of the relevance function
        using samples from the real replay buffer.

        Args:
            real_buffer (ReplayBuffer): The real replay buffer to sample from.
            policy_nets (Optional[Tuple[PolicyNetwork, QNetwork, Optional[QNetwork]]]): Optional policy and Q-networks.
        """
        pass

    @abstractmethod
    def get_params(self) -> List[torch.nn.Parameter]:
        """
        Abstract method: Returns a list of trainable parameters for the relevance function's
        internal models. Returns an empty list if not trainable.

        Returns:
            List[torch.nn.Parameter]: A list of trainable parameters.
        """
        pass


# --- Concrete Implementations ---


class ReturnRelevance(RelevanceFunction):
    """
    Relevance function based on the estimated Q-value of the current policy's action.
    F(s, a, s', r) = Q(s, pi(s))
    """

    def __init__(self, config: Config, state_dim: Union[int, Tuple[int, ...]], action_dim: int, device: torch.device):
        """
        Initializes the ReturnRelevance function.
        """
        super().__init__(config, state_dim, action_dim, device)
        # This relevance function has no internal trainable models.

    def compute_score(self, batch: Dict[str, torch.Tensor], policy_nets: Optional[Tuple[PolicyNetwork, QNetwork, Optional[QNetwork]]] = None) -> torch.Tensor:
        """
        Calculates relevance score as Q(s, pi(s)).

        Args:
            batch (Dict[str, torch.Tensor]): A dictionary of transition components, must contain 'state'.
            policy_nets (Optional[Tuple[PolicyNetwork, QNetwork, Optional[QNetwork]]]): Expected to provide
                                                                    (actor_net, critic_net).

        Returns:
            torch.Tensor: A tensor of relevance scores (batch_size, 1).
        """
        if policy_nets is None or len(policy_nets) < 2:
            raise ValueError("ReturnRelevance requires 'policy_nets' tuple (actor_net, critic_net).")

        actor_net, critic_net = policy_nets[0], policy_