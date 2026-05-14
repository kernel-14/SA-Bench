import abc
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional, Any

# Assuming these are available in the project structure
from config import Config
from environments import EnvironmentManager
from models.policy_nets import PolicyNetwork, QNetwork


class RLBaseAgent(abc.ABC):
    """
    Abstract base class for all reinforcement learning algorithms (SAC, REDQ, DRQ-V2, PPO).
    It defines a common interface for agent interaction, training, and state management.
    """

    def __init__(
        self,
        config: Config,
        env_manager: EnvironmentManager,
        policy_net: PolicyNetwork,
        q_net: QNetwork,
        device: torch.device,
    ):
        """
        Initializes the RLBaseAgent.

        Args:
            config (Config): Configuration object containing hyperparameters.
            env_manager (EnvironmentManager): An instance of EnvironmentManager to query environment properties.
            policy_net (PolicyNetwork): The policy network (actor) for the agent.
            q_net (QNetwork): The Q-value network (critic) for the agent.
            device (torch.device): The device (e.g., 'cuda' or 'cpu') for models and tensors.
        """
        self.config: Config = config
        self.env_manager: EnvironmentManager = env_manager
        self.policy_net: PolicyNetwork = policy_net.to(device)
        self.q_net: QNetwork = q_net.to(device)
        self.device: torch.device = device

        # Target Networks (deep copies, moved to device, and set to evaluation mode)
        # Parameters of target networks are not trained directly, but updated via soft_update
        self.target_policy_net: PolicyNetwork = copy.deepcopy(policy_net).to(device)
        self.target_policy_net.eval()
        for param in self.target_policy_net.parameters():
            param.requires_grad = False

        self.target_q_net: QNetwork = copy.deepcopy(q_net).to(device)
        self.target_q_net.eval()
        for param in self.target_q_net.parameters():
            param.requires_grad = False

        # Optimizers
        actor_lr: float = self.config.get_hyperparam('rl_agent.learning_rate.actor')
        critic_lr: float = self.config.get_hyperparam('rl_agent.learning_rate.critic')

        self.actor_optimizer: optim.Optimizer = optim.Adam(self.policy_net.parameters(), lr=actor_lr)
        self.critic_optimizer: optim.Optimizer = optim.Adam(self.q_net.parameters(), lr=critic_lr)

        # Alpha (Entropy Tuning for SAC/REDQ-like algorithms)
        self.log_alpha: Optional[nn.Parameter] = None
        self.alpha: Optional[torch.Tensor] = None
        self.alpha_optimizer: Optional[optim.Optimizer] = None
        self.target_entropy: Optional[float] = None

        try:
            alpha_lr: float = self.config.get_hyperparam('rl_agent.learning_rate.alpha')
            # Initialize log_alpha as a learnable parameter. `alpha` is `exp(log_alpha)`.
            self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
            self.alpha = self.log_alpha.exp().detach() # Initial alpha value, detached for initial use
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

            # Target entropy for automatic alpha tuning, depends on action space
            if self.policy_net.is_continuous:
                # For continuous action spaces, target entropy is typically negative of action dim
                action_space_shape = self.env_manager.get_action_space().shape
                if action_space_shape: # Ensure it's not an empty tuple for scalar actions
                    self.target_entropy = -float(torch.prod(torch.tensor(action_space_shape, device=device, dtype=torch.float32)))
                else: # Scalar continuous action, e.g., action_space = Box(low=-1.0, high=1.0, shape=())
                    self.target_entropy = -1.0 # Default to -1 for single scalar action
            else:
                # For discrete action spaces, target entropy is often -log(number of actions)
                num_actions = self.env_manager.get_action_space().n
                self.target_entropy = -math.log(num_actions)
        except KeyError:
            # If 'rl_agent.learning_rate.alpha' is not specified, entropy tuning is disabled
            pass

        # Hyperparameters
        self.gamma: float = self.config.get_hyperparam('rl_agent.discount')
        self.tau: float = self.config.get_hyperparam('rl_agent.target_update_tau')
        self.target_update_freq: int = self.config.get_hyperparam('rl_agent.target_update_freq')

    @abc.abstractmethod
    def get_action(self, state: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """
        Abstract method: Selects an action given a state observation.

        Args:
            state (torch.Tensor): The current state observation (batch_size, *state_dim).
            deterministic (bool): If True, return the deterministic action; otherwise, sample from the policy.

        Returns:
            torch.Tensor: The chosen action (batch_size, action_dim).
        """
        pass

    @abc.abstractmethod
    def train_step(self, real_batch: Dict[str, torch.Tensor],
                   synthetic_batch: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Abstract method: Performs one training step for the agent.

        Args:
            real_batch (Dict[str, torch.Tensor]): A dictionary of transition components
                                                  ('state', 'action', 'reward', 'next_state', 'done')
                                                  sampled from the real replay buffer.
            synthetic_batch (Optional[Dict[str, torch.Tensor]]): An optional dictionary with the same structure,
                                                                 sampled from the synthetic replay buffer.
                                                                 If None, training proceeds only with real_batch.

        Returns:
            Dict[str, float]: A dictionary of training metrics (e.g., 'actor_loss', 'critic_loss').
        """
        pass

    def get_policy_nets(self) -> Tuple[PolicyNetwork, QNetwork]:
        """
        Returns the agent's current policy and Q-networks.

        Returns:
            Tuple[PolicyNetwork, QNetwork]: A tuple containing the policy network and the Q-network.
        """
        return self.policy_net, self.q_net

    def sync_target_networks(self) -> None:
        """
        Performs a soft update of the target Q-network parameters towards the primary Q-network parameters.
        The update rule is: target_param = tau * param + (1 - tau) * target_param.
        """
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # While target_policy_net is copied, standard SAC/REDQ do not soft-update it.
        # It remains a fixed copy of the policy at an earlier point, or is not used for Q-target calculation directly.
        # If an algorithm requires it, it should be implemented in the specific agent subclass.
        # For now, it's a fixed copy from initialization.

    def save_checkpoint(self, path: str) -> None:
        """
        Saves the state of the agent's networks and optimizers to a file.

        Args:
            path (str): The file path where the checkpoint should be saved.
        """
        checkpoint_data: Dict[str, Any] = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'q_net_state_dict': self.q_net.state_dict(),
            'target_policy_net_state_dict': self.target_policy_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }
        if self.log_alpha is not None and self.alpha_optimizer is not None:
            checkpoint_data['log_alpha_state_dict'] = self.log_alpha.state_dict()
            checkpoint_data['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()

        torch.save(checkpoint_data, path)

    def load_checkpoint(self, path: str) -> None:
        """
        Loads the state of the agent's networks and optimizers from a file.

        Args:
            path (str): The file path from which to load the checkpoint.
        """
        checkpoint_data: Dict[str, Any] = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint_data['policy_net_state_dict'])
        self.q_net.load_state_dict(checkpoint_data['q_net_state_dict'])
        self.target_policy_net.load_state_dict(checkpoint_data['target_policy_net_state_dict'])
        self.target_q_net.load_state_dict(checkpoint_data['target_q_net_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint_data['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint_data['critic_optimizer_state_dict'])

        if 'log_alpha_state_dict' in checkpoint_data and self.log_alpha is not None:
            self.log_alpha.data.copy_(checkpoint_data['log_alpha_state_dict']['data']) # Load parameter data directly
            # Ensure self.alpha is updated if log_alpha was loaded
            self.alpha = self.log_alpha.exp().detach()
        if 'alpha_optimizer_state_dict' in checkpoint_data and self.alpha_optimizer is not None:
            self.alpha_optimizer.load_state_dict(checkpoint_data['alpha_optimizer_state_dict'])

        # Set networks back to their respective modes after loading
        self.policy_net.train()
        self.q_net.train()
        self.target_policy_net.eval()
        self.target_q_net.eval()

