import torch
import torch.nn as nn
import torch.distributions as distributions
from typing import List, Tuple, Any

# Assuming Config class is available from config.py
# In a real project setup, you might need to adjust the import path:
# from your_project_name.config import Config
from config import Config


def _build_mlp_layers(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation_type: str,
    output_activation: bool = False,
) -> nn.Sequential:
    """
    Helper function to build a multi-layer perceptron (MLP).

    Args:
        input_dim: The dimension of the input layer.
        hidden_dims: A list of integers, where each integer represents the number of neurons
                     in a hidden layer.
        output_dim: The dimension of the output layer.
        activation_type: A string specifying the activation function for hidden layers (e.g., "ReLU", "ELU").
        output_activation: If True, applies the activation function to the output layer.

    Returns:
        A torch.nn.Sequential model representing the MLP.
    """
    layers: List[nn.Module] = []
    current_dim = input_dim

    # Get activation function module
    if activation_type == "ReLU":
        activation_fn = nn.ReLU()
    elif activation_type == "ELU":
        activation_fn = nn.ELU()
    elif activation_type == "Tanh":
        activation_fn = nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")

    # Hidden layers
    for h_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, h_dim))
        layers.append(activation_fn)
        current_dim = h_dim

    # Output layer
    layers.append(nn.Linear(current_dim, output_dim))
    if output_activation:
        layers.append(activation_fn)

    return nn.Sequential(*layers)


class PolicyModel(nn.Module):
    """
    Implements the actor network for policy optimization.
    Takes policy observations as input and outputs a Gaussian distribution
    (mean and log standard deviation) over actions.
    """

    def __init__(self, obs_policy_dim: int, action_dim: int, config: Config):
        """
        Initializes the PolicyModel.

        Args:
            obs_policy_dim: Dimension of the policy observation space.
            action_dim: Dimension of the action space.
            config: Configuration object containing model hyperparameters.
        """
        super().__init__()
        self.action_dim = action_dim
        self.device = config.global.device

        # Retrieve architecture details from config
        hidden_shape = config.mbpo_ppo.policy_value_architecture.hidden_shape
        activation_type = config.mbpo_ppo.policy_value_architecture.activation

        # Build the MLP for the action mean
        self.actor_net = _build_mlp_layers(
            input_dim=obs_policy_dim,
            hidden_dims=hidden_shape,
            output_dim=action_dim, # Output layer is action_dim for mean
            activation_type=activation_type,
            output_activation=False # No activation on the output of the mean head
        ).to(self.device)

        # Learnable parameter for log standard deviation
        # Initialize log_std to a small negative value to encourage exploration
        # and prevent initial actions from being too deterministic.
        # Max/Min bounds for stability (not in config, using reasonable defaults)
        self.log_std_min = -2.0 # Corresponds to std ~ 0.135
        self.log_std_max = 0.5  # Corresponds to std ~ 1.648
        # Initialize to min_log_std for initial exploration, as suggested by some PPO implementations
        self.log_std = nn.Parameter(torch.zeros(1, action_dim).float().to(self.device) + self.log_std_min) 


    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the actor network to get action distribution parameters.

        Args:
            obs: A batch of policy observations (batch_size, obs_policy_dim).

        Returns:
            A tuple containing:
            - action_mean: The mean of the Gaussian action distribution (batch_size, action_dim).
            - action_log_std: The log standard deviation of the Gaussian action distribution
                              (batch_size, action_dim), clamped for stability.
        """
        # Ensure obs is on the correct device
        obs = obs.to(self.device)

        action_mean = self.actor_net(obs)
        
        # Expand log_std to match batch size and clamp for numerical stability
        action_log_std = self.log_std.expand(obs.shape[0], -1)
        action_log_std = torch.clamp(action_log_std, self.log_std_min, self.log_std_max)

        return action_mean, action_log_std

    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Samples an action from the Gaussian distribution predicted by the policy.
        Uses the reparameterization trick for differentiability.

        Args:
            obs: A batch of policy observations (batch_size, obs_policy_dim).

        Returns:
            A sampled action tensor (batch_size, action_dim).
        """
        action_mean, action_log_std = self.forward(obs)
        std = torch.exp(action_log_std)
        
        # Create a normal distribution
        normal_dist = distributions.Normal(action_mean, std)
        
        # Sample action using reparameterization trick
        action = normal_dist.rsample()
        return action

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates the log probabilities and entropy of given actions under the current policy.

        Args:
            obs: A batch of policy observations (batch_size, obs_policy_dim).
            actions: A batch of actions to evaluate (batch_size, action_dim).

        Returns:
            A tuple containing:
            - log_probs: Log probabilities of the given actions (batch_size,).
            - entropy: Entropy of the action distribution (batch_size,).
            - action_mean: The mean of the Gaussian action distribution (batch_size, action_dim).
        """
        # Ensure inputs are on the correct device
        obs = obs.to(self.device)
        actions = actions.to(self.device)

        action_mean, action_log_std = self.forward(obs)
        std = torch.exp(action_log_std)

        normal_dist = distributions.Normal(action_mean, std)

        # Calculate log probabilities. Sum across action dimensions.
        log_probs = normal_dist.log_prob(actions).sum(axis=-1)
        
        # Calculate entropy. Sum across action dimensions.
        entropy = normal_dist.entropy().sum(axis=-1)

        return log_probs, entropy, action_mean


class ValueModel(nn.Module):
    """
    Implements the critic network for state-value prediction.
    Takes policy observations as input and outputs a single scalar value.
    """

    def __init__(self, obs_policy_dim: int, config: Config):
        """
        Initializes the ValueModel.

        Args:
            obs_policy_dim: Dimension of the policy observation space.
            config: Configuration object containing model hyperparameters.
        """
        super().__init__()
        self.device = config.global.device

        # Retrieve architecture details from config
        hidden_shape = config.mbpo_ppo.policy_value_architecture.hidden_shape
        activation_type = config.mbpo_ppo.policy_value_architecture.activation

        # Build the MLP for the value function
        self.critic_net = _build_mlp_layers(
            input_dim=obs_policy_dim,
            hidden_dims=hidden_shape,
            output_dim=1, # Output layer is 1 for scalar value
            activation_type=activation_type,
            output_activation=False # No activation on the output of the value head
        ).to(self.device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the critic network to predict the state-value.

        Args:
            obs: A batch of policy observations (batch_size, obs_policy_dim).

        Returns:
            A tensor of predicted state-values (batch_size, 1).
        """
        # Ensure obs is on the correct device
        obs = obs.to(self.device)
        value = self.critic_net(obs)
        return value

