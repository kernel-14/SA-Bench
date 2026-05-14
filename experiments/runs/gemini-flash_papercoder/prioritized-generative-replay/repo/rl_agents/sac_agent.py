import copy
import math
from typing import Any, Dict, Optional, Tuple, Union

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


class SACAgent(RLBaseAgent):
    """
    Soft Actor-Critic (SAC) agent implementation.
    Inherits from RLBaseAgent and implements SAC-specific training logic.
    """

    def __init__(self, config: Config, env_manager: EnvironmentManager, device: torch.device):
        """
        Initializes the SAC agent.

        Args:
            config (Config): Configuration object.
            env_manager (EnvironmentManager): The environment manager.
            device (torch.device): The device (CPU/GPU) to run the agent on.
        """
        self.config = config
        self.env_manager = env_manager
        self.device = device

        # Determine state and action dimensions
        state_space = env_manager.get_observation_space()
        action_space = env_manager.get_action_space()

        pixel_based: bool = config.get_hyperparam('environment.pixel_based')
        is_continuous_action: bool = isinstance(action_space, gym.spaces.Box)

        if not is_continuous_action:
            raise ValueError(
                f"SACAgent expects a continuous action space (gym.spaces.Box), but got {type(action_space)}."
            )

        # State dimension can be an int (for vector states) or a tuple (for pixel states (C, H, W))
        state_dim: Union[int, Tuple[int, ...]]
        if pixel_based:
            state_dim = state_space.shape  # (C, H, W)
            action_dim = action_space.shape[0]  # Dimension of continuous action vector
        else:
            state_dim = state_space.shape[0]  # Flattened vector length
            action_dim = action_space.shape[0]  # Dimension of continuous action vector

        # Policy Network (Actor)
        # MLPActor/CNNActor constructor handles hidden layers/units from config.
        # It also handles noisy_networks if enabled via config.
        if pixel_based:
            self.actor: PolicyNetwork = CNNActor(
                config=config, state_dim=state_dim, action_dim=action_dim, is_continuous=is_continuous_action
            ).to(device)
        else:
            self.actor: PolicyNetwork = MLPActor(
                config=config, state_dim=state_dim, action_dim=action_dim, is_continuous=is_continuous_action
            ).to(device)

        # Q-Networks (Critics) - SAC uses two critics to reduce overestimation bias
        # MLPCritic/CNNCritic constructor handles hidden layers/units from config.
        # It also handles noisy_networks if enabled via config.
        if pixel_based:
            self.critic1: QNetwork = CNNCritic(config=config, state_dim=state_dim, action_dim=action_dim).to(device)
            self.critic2: QNetwork = CNNCritic(config=config, state_dim=state_dim, action_dim=action_dim).to(device)
        else:
            self.critic1: QNetwork = MLPCritic(config=config, state_dim=state_dim, action_dim=action_dim).to(device)
            self.critic2: QNetwork = MLPCritic(config=config, state_dim=state_dim, action_dim=action_dim).to(device)

        # Pass primary actor (policy_net) and critic1 (q_net) to the base class constructor.
        # RLBaseAgent will handle `self.policy_net`, `self.q_net`, `self.target_policy_net` (which is unused
        # for SAC's Q-target calculation), `self.target_q_net` (for `critic1`'s target),
        # and their optimizers (`self.actor_optimizer`, `self.critic_optimizer`).
        super().__init__(config, env_manager, self.actor, self.critic1, device)

        # SAC-specific target critics (target_critic1 is already handled by RLBaseAgent as target_q_net)
        self.target_critic1: QNetwork = self.target_q_net  # This is the target for self.critic1 (self.q_net)

        self.target_critic2: QNetwork = copy.deepcopy(self.critic2).to(device)
        self.target_critic2.eval()
        for param in self.target_critic2.parameters():
            param.requires_grad = False

        # Optimizer for the second critic
        critic_lr: float = self.config.get_hyperparam('rl_agent.learning_rate.critic')
        self.critic2_optimizer: optim.Optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # SAC-specific parameters
        self.reward_scale: float = self.config.get_hyperparam('rl_agent.reward_scale')
        self._total_sac_updates: int = 0  # To track updates for target network syncing

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Selects an action given a state observation.

        Args:
            state (torch.Tensor): The current state observation (batch_size, *state_dim).
            deterministic (bool): If True, return the deterministic action; otherwise, sample from the policy.

        Returns:
            torch.Tensor: The chosen action (batch_size, action_dim).
        """
        return self.actor.get_action(state, deterministic)

    def train_step(
        self,
        real_batch: Dict[str, torch.Tensor],
        synthetic_batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Performs one training step for the SAC agent.

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
            num_synthetic: int = int(rl_batch_size * synthetic_data_ratio)
            num_real: int = rl_batch_size - num_synthetic

            # Sample from real_batch (ensure enough samples)
            real_len: int = len(real_batch['state'])
            if real_len >= num_real:
                real_indices = torch.randperm(real_len, device=self.device)[:num_real]
                for k, v in real_batch.items():
                    combined_batch[k] = v[real_indices]
            else: # If not enough real samples, take all real and adjust synthetic
                for k, v in real_batch.items():
                    combined_batch[k] = v
                num_real = real_len
                num_synthetic = rl_batch_size - num_real # Adjust synthetic to fill batch

            # Sample from synthetic_batch (ensure enough samples)
            if num_synthetic > 0:
                synthetic_len: int = len(synthetic_batch['state'])
                if synthetic_len >= num_synthetic:
                    synthetic_indices = torch.randperm(synthetic_len, device=self.device)[:num_synthetic]
                    synthetic_sampled = {k: v[synthetic_indices] for k, v in synthetic_batch.items()}
                else: # If not enough synthetic samples, take all synthetic and pad if necessary (though usually will be less than batch size)
                    synthetic_sampled = synthetic_batch

                # Concatenate
                for k in combined_batch.keys():
                    combined_batch[k] = torch.cat([combined_batch[k], synthetic_sampled[k].to(self.device)], dim=0)

        else:
            # If no synthetic data or ratio is 0, use real_batch.
            # Assuming real_batch is already of size rl_batch_size or larger, and we'll take a sub-sample.
            real_len: int = len(real_batch['state'])
            if real_len >= rl_batch_size:
                real_indices = torch.randperm(real_len, device=self.device)[:rl_batch_size]
                for k, v in real_batch.items():
                    combined_batch[k] = v[real_indices]
            else: # Use all available real samples if less than batch_size
                combined_batch = real_batch

        # Move all tensors in the combined_batch to self.device
        state = combined_batch['state'].to(self.device)
        action = combined_batch['action'].to(self.device)
        reward = combined_batch['reward'].to(self.device) * self.reward_scale
        next_state = combined_batch['next_state'].to(self.device)
        done = combined_batch['done'].to(self.device).float()  # Convert bool to float for calculations

        metrics: Dict[str, float] = {}

        # --- Critic (Q-function) Update ---
        with torch.no_grad():
            # Sample action and log-probability from current policy for next_state
            # The actor (PolicyNetwork) returns (mean, log_std) for continuous actions.
            mean, log_std = self.actor(next_state)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            z = normal.sample()
            next_action = torch.tanh(z)

            # Compute log_prob for the squashed action
            # 1e-6 added for numerical stability to avoid log(0)
            log_prob_next_action = normal.log_prob(z) - torch.log(1 - next_action.pow(2) + 1e-6)
            log_prob_next_action = log_prob_next_action.sum(dim=-1, keepdim=True)  # Sum across action dimensions

            # Get Q-values from target critics
            q_target1_val = self.target_critic1(next_state, next_action)
            q_target2_val = self.target_critic2(next_state, next_action)
            min_q_target = torch.min(q_target1_val, q_target2_val)

            # Compute target value (V_target)
            v_target = min_q_target - self.alpha.detach() * log_prob_next_action

            # Compute Bellman target (Q_backup)
            q_backup = reward + self.discount * (1 - done) * v_target

        # Get current Q-values from main critics
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        # Critic loss (MSE between current Q and target Q)
        critic1_loss = F.mse_loss(current_q1, q_backup)
        critic2_loss = F.mse_loss(current_q2, q_backup)
        critic_loss = critic1_loss + critic2_loss

        # Optimize critics
        self.critic_optimizer.zero_grad()  # For critic1 (self.q_net from base)
        self.critic2_optimizer.zero_grad()  # For critic2
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic2_optimizer.step()

        metrics['critic_loss'] = critic_loss.item()
        metrics['q_value_mean'] = current_q1.mean().item()

        # --- Policy (Actor) Update ---
        # No grad on critics during actor update
        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False

        # Sample action and log-probability from current policy
        mean, log_std = self.actor(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        new_action = torch.tanh(z)

        # Log probability correction for tanh
        log_prob_new_action = normal.log_prob(z) - torch.log(1 - new_action.pow(2) + 1e-6)
        log_prob_new_action = log_prob_new_action.sum(dim=-1, keepdim=True)

        # Get Q-values for new actions from current critics (these are used for policy gradient)
        q1_new_action = self.critic1(state, new_action)
        q2_new_action = self.critic2(state, new_action)
        min_q_new_action = torch.min(q1_new_action, q2_new_action)

        # Actor loss (maximize entropy and Q-value)
        actor_loss = (self.alpha.detach() * log_prob_new_action - min_q_new_action).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_entropy'] = (-log_prob_new_action).mean().item()

        # Re-enable gradients for critics
        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
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
            # If entropy tuning is disabled, alpha would be a fixed value or 0.0
            metrics['alpha'] = self.alpha.item() if self.alpha is not None else 0.0

        # --- Target Network Updates ---
        self._total_sac_updates += 1
        if self._total_sac_updates % self.target_update_freq == 0:
            self.sync_target_networks()

        return metrics

    def sync_target_networks(self) -> None:
        """
        Performs a soft update of the target Q-network parameters towards the primary Q-network parameters.
        This overrides the RLBaseAgent's method to handle two critics.
        """
        # Update critic1 target (self.target_q_net in RLBaseAgent)
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update critic2 target
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_checkpoint(self, path: str) -> None:
        """
        Saves the state of the agent's networks and optimizers to a file.
        Overrides RLBaseAgent to include critic2 and its optimizer.
        """
        checkpoint_data: Dict[str, Any] = {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            # self.critic_optimizer is for critic1 (self.q_net from base)
            'critic1_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            '_total_sac_updates': self._total_sac_updates,
        }
        if self.log_alpha is not None and self.alpha_optimizer is not None:
            checkpoint_data['log_alpha_state_dict'] = self.log_alpha.state_dict()
            checkpoint_data['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()

        torch.save(checkpoint_data, path)
        # print(f"SACAgent checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Loads the state of the agent's networks and optimizers from a file.
        Overrides RLBaseAgent to include critic2 and its optimizer.
        """
        checkpoint_data: Dict[str, Any] = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint_data['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint_data['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint_data['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint_data['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint_data['target_critic2_state_dict'])

        self.actor_optimizer.load_state_dict(checkpoint_data['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint_data['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint_data['critic2_optimizer_state_dict'])
        self._total_sac_updates = checkpoint_data.get('_total_sac_updates', 0)

        if 'log_alpha_state_dict' in checkpoint_data and self.log_alpha is not None:
            # Need to copy data directly for Parameter type, as state_dict returns a dict of data
            if 'data' in checkpoint_data['log_alpha_state_dict']: # PyTorch 2.0+ might store 'data' directly
                 self.log_alpha.data.copy_(checkpoint_data['log_alpha_state_dict']['data'])
            else: # Older PyTorch might just copy the entire state dict if it's simple
                self.log_alpha.load_state_dict(checkpoint_data['log_alpha_state_dict'])
            self.alpha = self.log_alpha.exp().detach()
        if 'alpha_optimizer_state_dict' in checkpoint_data and self.alpha_optimizer is not None:
            self.alpha_optimizer.load_state_dict(checkpoint_data['alpha_optimizer_state_dict'])

        # Set networks back to their respective modes after loading
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        self.target_critic1.eval()
        self.target_critic2.eval()
        # print(f"SACAgent checkpoint loaded from {path}")

    def get_policy_nets(self) -> Tuple[PolicyNetwork, QNetwork]:
        """
        Returns the agent's current policy and one of the Q-networks (critic1).
        This is used by relevance functions.
        """
        return self.actor, self.critic1

