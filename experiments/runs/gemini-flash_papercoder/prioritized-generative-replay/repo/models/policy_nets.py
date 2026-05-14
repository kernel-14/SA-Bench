import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

# Assuming common_utils and config are in accessible paths
from config import Config
from utils.common_utils import CNNEncoder, MLPBlock, NoisyLinear, get_activation_fn, init_weights


# --- Abstract Base Classes ---

class PolicyNetwork(nn.Module, ABC):
    """
    Abstract base class for all actor (policy) networks.
    Defines common interface for policy definition and action sampling.
    """

    def __init__(
        self,
        config: Config,
        state_dim: Union[int, Tuple[int, ...]],
        action_dim: int,
        pixel_based: bool,
        is_continuous: bool,
    ):
        """
        Initializes the PolicyNetwork.

        Args:
            config (Config): Configuration object containing hyperparameters.
            state_dim (Union[int, Tuple[int, ...]]): Dimension of the state space.
                                                      int for vector states, Tuple for pixel states (C, H, W).
            action_dim (int): Dimension of the action space.
            pixel_based (bool): True if observations are pixel-based.
            is_continuous (bool): True if action space is continuous, False for discrete.
        """
        super().__init__()
        self.config: Config = config
        self.state_dim: Union[int, Tuple[int, ...]] = state_dim
        self.action_dim: int = action_dim
        self.pixel_based: bool = pixel_based
        self.is_continuous: bool = is_continuous
        self.device: torch.device = config.get_hyperparam('experiment.device')

        self.policy_hidden_layers: int = config.get_hyperparam('rl_agent.policy_hidden_layers')
        self.policy_hidden_units: int = config.get_hyperparam('rl_agent.policy_hidden_units')

        self.noisy_enabled: bool = config.get_hyperparam('rl_agent.noisy_networks.enabled')
        self.noisy_std_init: float = config.get_hyperparam('rl_agent.noisy_networks.std_init')
        if self.noisy_std_init == "NOT_SPECIFIED": # Default if not specified in config
            self.noisy_std_init = 0.5

        # For continuous policies, we use log_std clipping for stability
        # These values are common in SAC implementations and not explicitly in paper.
        self.log_std_min: float = -20.0
        self.log_std_max: float = 2.0
        
        self._build_network()
        self.to(self.device)

    @abstractmethod
    def _build_network(self) -> None:
        """
        Constructs the specific neural network layers for the policy.
        This method must be implemented by concrete subclasses.
        """
        pass

    @abstractmethod
    def forward(self, state: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Performs the forward pass to compute policy parameters.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
                - If continuous: A tuple (mean, log_std) for a Gaussian distribution.
                - If discrete: A tensor of logits for a Categorical distribution.
        """
        pass

    @torch.no_grad()
    def get_action(self, state: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """
        Samples an action from the policy distribution given a state.

        Args:
            state (torch.Tensor): Input state tensor.
            deterministic (bool): If True, returns the deterministic action (mean for continuous,
                                  argmax for discrete). If False, samples from the distribution.

        Returns:
            torch.Tensor: The sampled action tensor.
        """
        if self.is_continuous:
            mean, log_std = self.forward(state)
            std = log_std.exp()
            normal = Normal(mean, std)

            if deterministic:
                z = mean # For deterministic, use mean directly
            else:
                z = normal.sample() # Sample from Gaussian

            action = torch.tanh(z)
            return action
        else: # Discrete action space
            logits = self.forward(state)
            
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()
            return action.unsqueeze(-1) # Ensure consistent shape (batch_size, 1)


class QNetwork(nn.Module, ABC):
    """
    Abstract base class for all critic (Q-value) networks.
    Defines common interface for Q-value prediction.
    """

    def __init__(
        self,
        config: Config,
        state_dim: Union[int, Tuple[int, ...]],
        action_dim: int,
        pixel_based: bool,
    ):
        """
        Initializes the QNetwork.

        Args:
            config (Config): Configuration object containing hyperparameters.
            state_dim (Union[int, Tuple[int, ...]]): Dimension of the state space.
                                                      int for vector states, Tuple for pixel states (C, H, W).
            action_dim (int): Dimension of the action space.
            pixel_based (bool): True if observations are pixel-based.
        """
        super().__init__()
        self.config: Config = config
        self.state_dim: Union[int, Tuple[int, ...]] = state_dim
        self.action_dim: int = action_dim
        self.pixel_based: bool = pixel_based
        self.device: torch.device = config.get_hyperparam('experiment.device')

        self.q_hidden_layers: int = config.get_hyperparam('rl_agent.q_hidden_layers')
        self.q_hidden_units: int = config.get_hyperparam('rl_agent.q_hidden_units')

        self.noisy_enabled: bool = config.get_hyperparam('rl_agent.noisy_networks.enabled')
        self.noisy_std_init: float = config.get_hyperparam('rl_agent.noisy_networks.std_init')
        if self.noisy_std_init == "NOT_SPECIFIED": # Default if not specified in config
            self.noisy_std_init = 0.5
        
        self._build_network()
        self.to(self.device)

    @abstractmethod
    def _build_network(self) -> None:
        """
        Constructs the specific neural network layers for the Q-value function.
        This method must be implemented by concrete subclasses.
        """
        pass

    @abstractmethod
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass to compute Q-values.

        Args:
            state (torch.Tensor): Input state tensor.
            action (torch.Tensor): Input action tensor.

        Returns:
            torch.Tensor: Predicted Q-value tensor.
        """
        pass


# --- MLP Implementations (for state-based tasks) ---

class MLPActor(PolicyNetwork):
    """
    MLP-based actor network for state-based environments with continuous action spaces.
    Outputs mean and log_std for a Gaussian policy.
    """

    def __init__(self, config: Config, state_dim: int, action_dim: int, is_continuous: bool = True):
        super().__init__(config, state_dim, action_dim, pixel_based=False, is_continuous=is_continuous)

    def _build_network(self) -> None:
        """
        Constructs the MLP for the actor network.
        """
        input_dim = self.state_dim
        
        # Shared MLP layers
        self.shared_mlp = MLPBlock(
            input_dim=input_dim,
            output_dim=self.policy_hidden_units, # Output of shared layers
            hidden_units=self.policy_hidden_units,
            num_hidden_layers=self.policy_hidden_layers - 1, # Last layer is handled below
            activation_fn_name="ReLU",
            output_activation_fn_name="ReLU"
        )
        
        if self.is_continuous:
            # Output layers for mean and log_std
            self.mean_layer = nn.Linear(self.policy_hidden_units, self.action_dim)
            self.log_std_layer = nn.Linear(self.policy_hidden_units, self.action_dim)
            init_weights(self.mean_layer)
            init_weights(self.log_std_layer)
        else: # Discrete
            self.logits_layer = nn.Linear(self.policy_hidden_units, self.action_dim)
            init_weights(self.logits_layer)

        # Apply noisy layers if enabled
        if self.noisy_enabled:
            # Replace nn.Linear layers in self.shared_mlp with NoisyLinear
            def replace_with_noisy(module):
                for name, child in module.named_children():
                    if isinstance(child, nn.Linear):
                        # Ensure to get the correct input/output features
                        noisy_layer = NoisyLinear(child.in_features, child.out_features, self.noisy_std_init)
                        setattr(module, name, noisy_layer)
                    else:
                        replace_with_noisy(child)
            replace_with_noisy(self.shared_mlp)

            if self.is_continuous:
                self.mean_layer = NoisyLinear(self.policy_hidden_units, self.action_dim, self.noisy_std_init)
                self.log_std_layer = NoisyLinear(self.policy_hidden_units, self.action_dim, self.noisy_std_init)
            else:
                self.logits_layer = NoisyLinear(self.policy_hidden_units, self.action_dim, self.noisy_std_init)

    def forward(self, state: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Performs the forward pass for the MLPActor.
        """
        x = self.shared_mlp(state)

        if self.is_continuous:
            mean = self.mean_layer(x)
            log_std = self.log_std_layer(x)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            return mean, log_std
        else:
            logits = self.logits_layer(x)
            return logits

    def get_action(self, state: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """
        Samples an action for the MLPActor, handling continuous and discrete cases.
        """
        state = state.to(self.device)
        return super().get_action(state, deterministic)


class MLPCritic(QNetwork):
    """
    MLP-based critic network for state-based environments.
    Outputs a single Q-value.
    """

    def __init__(self, config: Config, state_dim: int, action_dim: int):
        super().__init__(config, state_dim, action_dim, pixel_based=False)

    def _build_network(self) -> None:
        """
        Constructs the MLP for the critic network.
        """
        input_dim = self.state_dim + self.action_dim
        
        self.mlp = MLPBlock(
            input_dim=input_dim,
            output_dim=1, # Output a single Q-value
            hidden_units=self.q_hidden_units,
            num_hidden_layers=self.q_hidden_layers,
            activation_fn_name="ReLU",
            output_activation_fn_name=None # No activation for Q-value output
        )

        # Apply noisy layers if enabled
        if self.noisy_enabled:
            def replace_with_noisy(module):
                for name, child in module.named_children():
                    if isinstance(child, nn.Linear):
                        noisy_layer = NoisyLinear(child.in_features, child.out_features, self.noisy_std_init)
                        setattr(module, name, noisy_layer)
                    else:
                        replace_with_noisy(child)
            replace_with_noisy(self.mlp)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the MLPCritic.
        """
        state = state.to(self.device)
        action = action.to(self.device)
        x = torch.cat([state, action], dim=-1)
        return self.mlp(x)


# --- CNN Implementations (for pixel-based tasks) ---

class CNNActor(PolicyNetwork):
    """
    CNN-based actor network for pixel-based environments with continuous action spaces.
    Uses a CNN encoder to process pixel observations into a latent space, then an MLP for policy.
    """

    def __init__(self, config: Config, state_dim: Tuple[int, int, int], action_dim: int, is_continuous: bool = True):
        super().__init__(config, state_dim, action_dim, pixel_based=True, is_continuous=is_continuous)
        
    def _build_network(self) -> None:
        """
        Constructs the CNN encoder and MLP head for the actor network.
        """
        # CNN Encoder for pixel observations
        self.encoder = CNNEncoder(
            input_shape=self.state_dim, # (C, H, W)
            output_dim=self.config.get_hyperparam('environment.visual_encoder_output_dim')
        )
        self.latent_dim = self.encoder.output_dim

        # MLP head on top of the latent features
        self.mlp_head_shared = MLPBlock(
            input_dim=self.latent_dim,
            output_dim=self.policy_hidden_units,
            hidden_units=self.policy_hidden_units,
            num_hidden_layers=self.policy_hidden_layers - 1, # Last layer handled below
            activation_fn_name="ReLU",
            output_activation_fn_name="ReLU"
        )
        
        if self.is_continuous:
            self.mean_layer = nn.Linear(self.policy_hidden_units, self.action_dim)
            self.log_std_layer = nn.Linear(self.policy_hidden_units, self.action_dim)
            init_weights(self.mean_layer)
            init_weights(self.log_std_layer)
        else: # Discrete
            self.logits_layer = nn.Linear(self.policy_hidden_units, self.action_dim)
            init_weights(self.logits_layer)
        
        # Apply noisy layers if enabled
        if self.noisy_enabled:
            def replace_with_noisy(module):
                for name, child in module.named_children():
                    if isinstance(child, nn.Linear):
                        noisy_layer = NoisyLinear(child.in_features, child.out_features, self.noisy_std_init)
                        setattr(module, name, noisy_layer)
                    else:
                        replace_with_noisy(child)
            replace_with_noisy(self.mlp_head_shared)

            if self.is_continuous:
                self.mean_layer = NoisyLinear(self.policy_hidden_units, self.action_dim, self.noisy_std_init)
                self.log_std_layer = NoisyLinear(self.policy_hidden_units, self.action_dim, self.noisy_std_init)
            else:
                self.logits_layer = NoisyLinear(self.policy_hidden_units, self.action_dim, self.noisy_std_init)

    def forward(self, state: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Performs the forward pass for the CNNActor.
        """
        state = state.to(self.device)
        latent_features = self.encoder(state)
        x = self.mlp_head_shared(latent_features)

        if self.is_continuous:
            mean = self.mean_layer(x)
            log_std = self.log_std_layer(x)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            return mean, log_std
        else:
            logits = self.logits_layer(x)
            return logits

    def get_action(self, state: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """
        Samples an action for the CNNActor, handling continuous and discrete cases.
        """
        # State already moved to device in forward pass or by super().get_action implicitly
        return super().get_action(state, deterministic)


class CNNCritic(QNetwork):
    """
    CNN-based critic network for pixel-based environments.
    Uses a CNN encoder to process pixel observations into a latent space, then an MLP for Q-value prediction.
    """

    def __init__(self, config: Config, state_dim: Tuple[int, int, int], action_dim: int):
        super().__init__(config, state_dim, action_dim, pixel_based=True)
        
    def _build_network(self) -> None:
        """
        Constructs the CNN encoder and MLP head for the critic network.
        """
        # CNN Encoder for pixel observations
        self.encoder = CNNEncoder(
            input_shape=self.state_dim, # (C, H, W)
            output_dim=self.config.get_hyperparam('environment.visual_encoder_output_dim')
        )
        self.latent_dim = self.encoder.output_dim

        # MLP head on top of concatenated latent features and action
        input_mlp_dim = self.latent_dim + self.action_dim
        self.mlp_head = MLPBlock(
            input_dim=input_mlp_dim,
            output_dim=1, # Output a single Q-value
            hidden_units=self.q_hidden_units,
            num_hidden_layers=self.q_hidden_layers,
            activation_fn_name="ReLU",
            output_activation_fn_name=None # No activation for Q-value output
        )

        # Apply noisy layers if enabled
        if self.noisy_enabled:
            def replace_with_noisy(module):
                for name, child in module.named_children():
                    if isinstance(child, nn.Linear):
                        noisy_layer = NoisyLinear(child.in_features, child.out_features, self.noisy_std_init)
                        setattr(module, name, noisy_layer)
                    else:
                        replace_with_noisy(child)
            replace_with_noisy(self.mlp_head)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the CNNCritic.
        """
        state = state.to(self.device)
        action = action.to(self.device)
        latent_features = self.encoder(state)
        x = torch.cat([latent_features, action], dim=-1)
        return self.mlp_head(x)

