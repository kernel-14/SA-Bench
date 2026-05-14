import copy
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import Config
from environments import EnvironmentManager
from models.policy_nets import CNNActor, CNNCritic, MLPActor, MLPCritic, PolicyNetwork, QNetwork
from rl_agents.base_agent import RLBaseAgent


class REDQAgent(RLBaseAgent):
    """
    Randomized Ensembled Double Q-learning (REDQ) agent implementation.
    Inherits from RLBaseAgent and implements REDQ-specific training logic,
    including an ensemble of Q-networks and support for various exploration bonuses.
    """

    def __init__(self, config: Config, env_manager: EnvironmentManager, device: torch.device):
        """
        Initializes the REDQ agent.

        Args:
            config (Config): Configuration object.
            env_manager (EnvironmentManager): The environment manager.
            device (torch.device): The device (CPU/GPU) to run the agent on.
        """
        self.config: Config = config
        self.env_manager: EnvironmentManager = env_manager
        self.device: torch.device = device

        # Determine state and action dimensions
        state_space: gym.Space = env_manager.get_observation_space()
        action_space: gym.Space = env_manager.get_action_space()

        pixel_based: bool = config.get_hyperparam('environment.pixel_based')
        is_continuous_action: bool = isinstance(action_space, gym.spaces.Box)

        # REDQ typically used for continuous action spaces, but can be adapted
        if not is_continuous_action:
            print("Warning: REDQ is commonly used with continuous action spaces. "
                  "Using it with discrete actions might require specific adaptations (e.g., Gumbel-Softmax). "
                  "This implementation assumes a continuous action space for the actor's output structure.")
            # For discrete, actor would output logits directly, not mean/log_std.
            # But the paper implies continuous for SAC/REDQ.
            # If it's discrete, Q-network would take action index as input, not a continuous action vector.
            # For now, we strictly follow the continuous assumption for SAC/REDQ as per common implementations.

        # State dimension can be an int (for vector states) or a tuple (for pixel states (C, H, W))
        state_dim: Union[int, Tuple[int, ...]]
        if pixel_based:
            state_dim = state_space.shape  # (C, H, W)
            action_dim = action_space.shape[0]  # Dimension of continuous action vector
        else:
            state_dim = state_space.shape[0]  # Flattened vector length
            action_dim = action_space.shape[0]  # Dimension of continuous action vector

        # --- REDQ-specific hyperparameters ---
        self.num_q_networks: int = self.config.get_hyperparam('rl_agent.num_q_networks')
        self.num_min_q_networks: int = self.config.get_hyperparam('rl_agent.num_min_q_networks')
        if self.num_min_q_networks > self.num_q_networks:
            raise ValueError(f"num_min_q_networks ({self.num_min_q_networks}) cannot be greater than num_q_networks ({self.num_q_networks}).")
        
        # Hyperparameters for optional exploration bonuses
        self.bootstrapped_q_enabled: bool = self.config.get_hyperparam('rl_agent.bootstrapped_q.enabled')
        # If bootstrapped_q is enabled, its num_heads should be consistent with num_q_networks
        if self.bootstrapped_q_enabled and self.config.get_hyperparam('rl_agent.bootstrapped_q.num_heads') != self.num_q_networks:
            print(f"Warning: Bootstrapped Q-values enabled, but num_heads ({self.config.get_hyperparam('rl_agent.bootstrapped_q.num_heads')}) "
                  f"does not match REDQ's num_q_networks ({self.num_q_networks}). Using REDQ's ensemble size.")

        self.exp_bonus_intrinsic_reward_weight: float = self.config.get_hyperparam('rl_agent.exploration_bonus_intrinsic_reward_weight')
        # This will be used if `intrinsic_reward` is present in the batch and this weight is > 0.

        # --- Network Initialization ---
        # Policy Network (Actor)
        if pixel_based:
            self.actor: PolicyNetwork = CNNActor(
                config=config, state_dim=state_dim, action_dim=action_dim, is_continuous=is_continuous_action
            ).to(device)
        else:
            self.actor: PolicyNetwork = MLPActor(
                config=config, state_dim=state_dim, action_dim=action_dim, is_continuous=is_continuous_action
            ).to(device)

        # Critic Ensemble
        self.q_networks: nn.ModuleList = nn.ModuleList()
        self.target_q_networks: nn.ModuleList = nn.ModuleList()
        for _ in range(self.num_q_networks):
            if pixel_based:
                q_net_instance = CNNCritic(config=config, state_dim=state_dim, action_dim=action_dim).to(device)
            else:
                q_net_instance = MLPCritic(config=config, state_dim=state_dim, action_dim=action_dim).to(device)
            
            self.q_networks.append(q_net_instance)
            
            target_q_net_instance = copy.deepcopy(q_net_instance).to(device)
            target_q_net_instance.eval()
            for param in target_q_net_instance.parameters():
                param.requires_grad = False
            self.target_q_networks.append(target_q_net_instance)

        # Initialize base class (RLBaseAgent) with the actor and the first Q-network.
        # The base class will manage optimizers for `self.actor` and `self.q_networks[0]`.
        # We will override the base class's critic optimizer management for the ensemble.
        super().__init__(config, env_manager, self.actor, self.q_networks[0], device) # q_networks[0] serves as base.q_net

        # --- Optimizers for the Q-network ensemble ---
        critic_lr: float = self.config.get_hyperparam('rl_agent.learning_rate.critic')
        self.q_optimizers: List[optim.Optimizer] = []
        for i, q_net in enumerate(self.q_networks):
            # If it's the first Q-network, its optimizer is already created in the base class.
            # Otherwise, create a new one.
            if i == 0:
                self.q_optimizers.append(self.critic_optimizer) # Use base class's critic_optimizer for the first one
            else:
                self.q_optimizers.append(optim.Adam(q_net.parameters(), lr=critic_lr))

        self._total_redq_updates: int = 0  # To track updates for target network syncing

        # REDQ typically updates the actor less frequently than the critic
        # The paper mentions UTD ratio for SAC, but not specifically for actor/critic updates within REDQ.
        # A common practice for REDQ is to update critic more frequently.
        # Let's assume a default of 2 critic updates per actor update if not explicitly configured.
        self.actor_update_freq: int = self.config.get_hyperparam('rl_agent.actor_update_freq') # Default to 1 (every update) in config, can be >1.
        if self.actor_update_freq == "NOT_SPECIFIED":
            self.actor_update_freq = 1

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Selects an action given a state observation.

        Args:
            state (torch.Tensor): The current state observation (batch_size, *state_dim).
            deterministic (bool): If True, return the deterministic action; otherwise, sample from the policy.

        Returns:
            torch.Tensor: The chosen action (batch_size, action_dim).
        """
        # If noisy networks are enabled, the policy_net's forward method handles the noise.
        # For deterministic, it will use mu. For non-deterministic, it will sample.
        # NoisyLinear layers internally reset their noise when in self.training mode.
        return self.actor.get_action(state, deterministic)

    def train_step(
        self,
        real_batch: Dict[str, torch.Tensor],
        synthetic_batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Performs one training step for the REDQ agent.

        Args:
            real_batch (Dict[str, torch.Tensor]): A dictionary of transition components
                                                  ('state', 'action', 'reward', 'next_state', 'done')
                                                  sampled from the real replay buffer.
            synthetic_batch (Optional[Dict[str, torch.Tensor]]): An optional dictionary with the same structure,
                                                                 sampled from the synthetic replay buffer.

        Returns:
            Dict[str, float]: A dictionary of training metrics.
        """
        # --- Data Preparation ---
        rl_batch_size: int = self.config.get_hyperparam('rl_agent.batch_size')
        synthetic_data_ratio: float = self.config.get_hyperparam('pgr_loop.synthetic_data_ratio')

        combined_batch: Dict[str, torch.Tensor] = {}

        if synthetic_batch is not None and synthetic_data_ratio > 0:
            num_synthetic_to_sample: int = int(rl_batch_size * synthetic_data_ratio)
            num_real_to_sample: int = rl_batch_size - num_synthetic_to_sample

            # Sample from real_batch
            real_len: int = len(real_batch['state'])
            if real_len > 0:
                real_indices = torch.randperm(real_len, device=self.device)[:num_real_to_sample]
                for k, v in real_batch.items():
                    combined_batch[k] = v[real_indices]
            else: # If no real samples, initialize with empty tensors
                for k, v in real_batch.items():
                    combined_batch[k] = torch.empty(0, *v.shape[1:], device=self.device)
                num_real_to_sample = 0 # Adjust count

            # Sample from synthetic_batch
            if num_synthetic_to_sample > 0:
                synthetic_len: int = len(synthetic_batch['state'])
                if synthetic_len > 0:
                    synthetic_indices = torch.randperm(synthetic_len, device=self.device)[:num_synthetic_to_sample]
                    synthetic_sampled = {k: v[synthetic_indices] for k, v in synthetic_batch.items()}
                else: # If no synthetic samples, initialize with empty tensors
                     synthetic_sampled = {k: torch.empty(0, *v.shape[1:], device=self.device) for k, v in synthetic_batch.items()}
                     num_synthetic_to_sample = 0 # Adjust count

                # Concatenate, ensuring state, action, etc. are on the correct device already from replay buffer
                for k in combined_batch.keys():
                    if combined_batch[k].shape[0] == 0: # If real batch was empty
                        combined_batch[k] = synthetic_sampled[k]
                    elif synthetic_sampled[k].shape[0] > 0: # If synthetic batch has data
                        combined_batch[k] = torch.cat([combined_batch[k], synthetic_sampled[k]], dim=0)

        else:
            # If no synthetic data or ratio is 0, use real_batch.
            real_len: int = len(real_batch['state'])
            if real_len >= rl_batch_size:
                real_indices = torch.randperm(real_len, device=self.device)[:rl_batch_size]
                combined_batch = {k: v[real_indices] for k, v in real_batch.items()}
            else: # Use all available real samples if less than batch_size
                combined_batch = real_batch

        # Now, ensure the combined batch is not empty for calculations
        if combined_batch['state'].shape[0] == 0:
            return {} # No data to train on

        state = combined_batch['state'].to(self.device)
        action = combined_batch['action'].to(self.device)
        reward = combined_batch['reward'].to(self.device) * self.reward_scale
        next_state = combined_batch['next_state'].to(self.device)
        done = combined_batch['done'].to(self.device).float()  # Convert bool to float for calculations

        metrics: Dict[str, float] = {}

        # --- Explicit Exploration Bonus (if present in batch) ---
        if self.exp_bonus_intrinsic_reward_weight > 0 and 'intrinsic_reward' in combined_batch:
            intrinsic_reward = combined_batch['intrinsic_reward'].to(self.device)
            reward += self.exp_bonus_intrinsic_reward_weight * intrinsic_reward

        # --- Critic (Q-function) Update ---
        with torch.no_grad():
            # Sample action and log-probability from current policy for next_state
            mean, log_std = self.actor(next_state)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            z = normal.sample()
            next_action = torch.tanh(z)

            log_prob_next_action = normal.log_prob(z) - torch.log(1 - next_action.pow(2) + 1e-6)
            log_prob_next_action = log_prob_next_action.sum(dim=-1, keepdim=True)

            # Randomly select `num_min_q_networks` target Q-networks for target Q-value calculation
            # This is the "randomized" part of REDQ
            q_network_indices = random.sample(range(self.num_q_networks), self.num_min_q_networks)
            
            # Compute Q-values from selected target critics
            q_target_values: List[torch.Tensor] = []
            for idx in q_network_indices:
                q_target_values.append(self.target_q_networks[idx](next_state, next_action))
            
            # Take the minimum of the selected target Q-values
            min_q_target = torch.min(torch.cat(q_target_values, dim=-1), dim=-1, keepdim=True)[0]

            # Compute Bellman target (Q_backup)
            # self.alpha is detached here as per SAC formulation
            q_backup = reward + self.gamma * (1 - done) * (min_q_target - self.alpha.detach() * log_prob_next_action)

        # Optimize all Q-networks
        critic_loss_sum: float = 0.0
        q_value_mean_sum: float = 0.0
        for i in range(self.num_q_networks):
            current_q_val = self.q_networks[i](state, action)
            q_loss = F.mse_loss(current_q_val, q_backup)

            self.q_optimizers[i].zero_grad()
            q_loss.backward()
            self.q_optimizers[i].step()

            critic_loss_sum += q_loss.item()
            q_value_mean_sum += current_q_val.mean().item()

        metrics['critic_loss'] = critic_loss_sum / self.num_q_networks
        metrics['q_value_mean'] = q_value_mean_sum / self.num_q_networks

        # --- Policy (Actor) Update ---
        # REDQ updates actor less frequently
        if self._total_redq_updates % self.actor_update_freq == 0:
            # No grad on critics during actor update
            for q_net in self.q_networks:
                for p in q_net.parameters():
                    p.requires_grad = False

            # Sample action and log-probability from current policy
            mean, log_std = self.actor(state)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            z = normal.sample()
            new_action = torch.tanh(z)

            log_prob_new_action = normal.log_prob(z) - torch.log(1 - new_action.pow(2) + 1e-6)
            log_prob_new_action = log_prob_new_action.sum(dim=-1, keepdim=True)

            # Get Q-values for new actions from ALL current critics (not just sampled ones)
            # The paper's Equation 3 for SAC has this min over all Qs
            # For REDQ actor update, often this uses all current Q-networks
            q_values_for_actor: List[torch.Tensor] = []
            for q_net in self.q_networks:
                q_values_for_actor.append(q_net(state, new_action))
            min_q_new_action = torch.min(torch.cat(q_values_for_actor, dim=-1), dim=-1, keepdim=True)[0]

            # Actor loss (maximize entropy and Q-value)
            # self.alpha is detached for actor loss as well
            actor_loss = (self.alpha.detach() * log_prob_new_action - min_q_new_action).mean()

            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_entropy'] = (-log_prob_new_action).mean().item()

            # Re-enable gradients for critics
            for q_net in self.q_networks:
                for p in q_net.parameters():
                    p.requires_grad = True

            # --- Temperature (Alpha) Update ---
            if self.log_alpha is not None and self.alpha_optimizer is not None and self.target_entropy is not None:
                alpha_loss = (self.alpha * (log_prob_new_action.detach() + self.target_entropy)).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                # Update alpha value from its learnable log_alpha parameter
                self.alpha = self.log_alpha.exp().detach()
                metrics['alpha_loss'] = alpha_loss.item()
                metrics['alpha'] = self.alpha.item()
            else:
                metrics['alpha'] = self.alpha.item() if self.alpha is not None else 0.0
        
        # --- Target Network Updates ---
        self._total_redq_updates += 1
        if self._total_redq_updates % self.target_update_freq == 0:
            self.sync_target_networks()

        return metrics

    def sync_target_networks(self) -> None:
        """
        Performs a soft update of all target Q-network parameters towards their primary Q-network counterparts.
        This overrides the RLBaseAgent's method to handle an ensemble of critics.
        """
        for i in range(self.num_q_networks):
            for param, target_param in zip(self.q_networks[i].parameters(), self.target_q_networks[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Note: The base class's self.target_q_net (which points to self.target_q_networks[0]) is updated
        # via this loop. No separate update needed.

    def save_checkpoint(self, path: str) -> None:
        """
        Saves the state of the agent's networks and optimizers to a file.
        Overrides RLBaseAgent to include the critic ensemble and their optimizers.

        Args:
            path (str): The file path where the checkpoint should be saved.
        """
        checkpoint_data: Dict[str, Any] = {
            'actor_state_dict': self.actor.state_dict(),
            'q_networks_state_dicts': [q_net.state_dict() for q_net in self.q_networks],
            'target_q_networks_state_dicts': [t_q_net.state_dict() for t_q_net in self.target_q_networks],
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'q_optimizers_state_dicts': [q_opt.state_dict() for q_opt in self.q_optimizers],
            '_total_redq_updates': self._total_redq_updates,
        }
        if self.log_alpha is not None and self.alpha_optimizer is not None:
            checkpoint_data['log_alpha_state_dict'] = self.log_alpha.state_dict()
            checkpoint_data['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()

        torch.save(checkpoint_data, path)
        # print(f"REDQAgent checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Loads the state of the agent's networks and optimizers from a file.
        Overrides RLBaseAgent to include the critic ensemble and their optimizers.

        Args:
            path (str): The file path from which to load the checkpoint.
        """
        checkpoint_data: Dict[str, Any] = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint_data['actor_state_dict'])
        
        q_networks_state_dicts = checkpoint_data['q_networks_state_dicts']
        for i, q_net_state_dict in enumerate(q_networks_state_dicts):
            self.q_networks[i].load_state_dict(q_net_state_dict)

        target_q_networks_state_dicts = checkpoint_data['target_q_networks_state_dicts']
        for i, t_q_net_state_dict in enumerate(target_q_networks_state_dicts):
            self.target_q_networks[i].load_state_dict(t_q_net_state_dict)

        self.actor_optimizer.load_state_dict(checkpoint_data['actor_optimizer_state_dict'])
        
        q_optimizers_state_dicts = checkpoint_data['q_optimizers_state_dicts']
        for i, q_opt_state_dict in enumerate(q_optimizers_state_dicts):
            self.q_optimizers[i].load_state_dict(q_opt_state_dict)
        
        self._total_redq_updates = checkpoint_data.get('_total_redq_updates', 0)

        if 'log_alpha_state_dict' in checkpoint_data and self.log_alpha is not None:
            # Need to copy data directly for Parameter type
            if 'data' in checkpoint_data['log_alpha_state_dict']: # PyTorch 2.0+ might store 'data' directly
                 self.log_alpha.data.copy_(checkpoint_data['log_alpha_state_dict']['data'])
            else: # Older PyTorch might just copy the entire state dict if it's simple
                self.log_alpha.load_state_dict(checkpoint_data['log_alpha_state_dict'])
            self.alpha = self.log_alpha.exp().detach()
        if 'alpha_optimizer_state_dict' in checkpoint_data and self.alpha_optimizer is not None:
            self.alpha_optimizer.load_state_dict(checkpoint_data['alpha_optimizer_state_dict'])

        # Set networks back to their respective modes after loading
        self.actor.train()
        for q_net in self.q_networks:
            q_net.train()
        for target_q_net in self.target_q_networks:
            target_q_net.eval()
        # print(f"REDQAgent checkpoint loaded from {path}")

    def get_policy_nets(self) -> Tuple[PolicyNetwork, QNetwork]:
        """
        Returns the agent's current policy and *one* of the Q-networks (the first one in the ensemble).
        This is primarily for compatibility with RelevanceFunction classes that expect a single QNetwork.
        """
        return self.actor, self.q_networks[0]

