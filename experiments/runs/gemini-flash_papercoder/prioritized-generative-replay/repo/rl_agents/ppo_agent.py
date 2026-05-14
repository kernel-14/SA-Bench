import copy
from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from config import Config
from environments import EnvironmentManager
from models.policy_nets import CNNActor, CNNCritic, MLPActor, MLPCritic, PolicyNetwork, QNetwork
from rl_agents.base_agent import RLBaseAgent


class PPOAgent(RLBaseAgent):
    """
    Proximal Policy Optimization (PPO) agent implementation.
    Inherits from RLBaseAgent and implements PPO-specific training logic.
    This implementation is adapted to fit the replay buffer and mini-batch structure
    of the PGR framework, treating PPO in an off-policy manner where `old_log_probs`
    are approximated by the policy's current log_probs at the beginning of the update.
    """

    def __init__(self, config: Config, env_manager: EnvironmentManager, device: torch.device):
        """
        Initializes the PPO agent.

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

        if is_continuous_action:
            raise ValueError("PPOAgent in this context expects a discrete action space for DMLab, "
                             "but got a continuous action space.")

        # State dimension can be an int (for vector states) or a tuple (for pixel states (C, H, W))
        state_dim: Union[int, Tuple[int, ...]]
        if pixel_based:
            state_dim = state_space.shape  # (C, H, W)
            action_dim = action_space.n  # Number of discrete actions
        else:
            state_dim = state_space.shape[0]  # Flattened vector length
            action_dim = action_space.n  # Number of discrete actions

        # Policy Network (Actor) - outputs logits for discrete actions
        if pixel_based:
            self.actor: PolicyNetwork = CNNActor(
                config=config, state_dim=state_dim, action_dim=action_dim, is_continuous=False
            ).to(device)
        else:
            self.actor: PolicyNetwork = MLPActor(
                config=config, state_dim=state_dim, action_dim=action_dim, is_continuous=False
            ).to(device)

        # Value Network (Critic) - outputs a single state-value
        # For a PPO critic, it's typically just a value network, not a Q-network.
        # However, `QNetwork` in our design is generalized enough to act as a value network
        # if `action_dim` is considered implicitly by policy's actions or if it just takes state.
        # Since the `QNetwork` interface takes (state, action), we adapt it.
        # For a standard PPO Value function, it would usually only take 'state' as input.
        # Here, we will use it as a V-function, effectively ignoring the action input in its forward pass for PPO's value prediction.
        # Or, we can redefine QNetwork for PPO to be a ValueNetwork. For now, we will use CNNCritic/MLPCritic
        # but modify its forward to only use state for value prediction, or train it as a Q-network.
        # The paper refers to it as Q-function based on SAC/REDQ context.
        # For PPO, it's a value function, so we make a slight deviation from the QNetwork name in the base class.
        # We will initialize a specific ValueNetwork and pass it to super.

        # PPO's critic is a Value Network, which estimates V(s). It does not take actions.
        # We define a specific PPOValueNetwork class or adjust MLPCritic/CNNCritic logic.
        # For now, let's create a specialized PPOValueNetwork that fits the QNetwork interface for RLBaseAgent.
        # This PPOValueNetwork will ignore the `action` input to its `forward` method.

        # Create a proxy QNetwork that acts as a Value Network for PPO's critic.
        if pixel_based:
            # For PPO, the critic typically just predicts V(s), not Q(s,a).
            # The QNetwork class is defined to accept state and action.
            # To fit RLBaseAgent, we'll instantiate CNNCritic but clarify its use.
            # This specific CNNCritic for PPO will effectively ignore the 'action' input for value prediction.
            self.critic: QNetwork = CNNCritic(config=config, state_dim=state_dim, action_dim=action_dim).to(device)
            # Override its forward method to only take state and return value, effectively ignoring action input
            def ppo_critic_forward(self, state: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
                state = state.to(self.device)
                latent_features = self.encoder(state)
                # For PPO, we predict V(s) so action is not concatenated with latent features for the MLP head.
                # The MLP head must be re-configured to take only latent_features as input if action is to be ignored.
                # However, the CNNCritic/MLPCritic is designed to take (latent_features + action).
                # This implies I need a new class or modify CNNCritic/MLPCritic directly.

                # Let's adjust for this specific PPO agent: CNNCritic/MLPCritic will be instantiated
                # with an action_dim of 0 effectively, or modified to have a different MLP head.
                # A more robust solution is a separate ValueNetwork class.
                # Given the design requires QNetwork, let's assume the action_dim will be used,
                # but it will likely receive a dummy action or a zero tensor for training value function.
                # A common approach in PPO with a Q-critic is to train it on max_Q(s,a) for V(s).
                # But that's not standard PPO.

                # Let's assume for PPO's Value function, we instantiate an MLP/CNN that takes only state and outputs 1 scalar.
                # This means I need to create a `PPOValueNetwork` class that inherits from `QNetwork`
                # but explicitly takes only state as input, or modify `QNetwork` to support `is_value_net` flag.
                # Design states: `MLPCritic`, `CNNCritic` inherit from `QNetwork`.
                # Okay, a `CNNCritic` for PPO will still have `action_dim` in its constructor, but its forward method
                # for value prediction will just ignore `action`.
                # I'll create a PPO-specific `ValueNetwork` that inherits from `nn.Module` and use it.
                # But RLBaseAgent expects QNetwork. This is a design conflict.

                # RETHINK: The design states `RLBaseAgent` takes `QNetwork` and `target_q_net` for init.
                # If PPO does not use Q-networks, but value networks, I must make my `PPOValueNetwork`
                # compatible with `QNetwork` interface (e.g. takes state, action, but ignores action).

                class PPOValueNetwork(QNetwork):
                    def __init__(self_inner, config: Config, state_dim: Union[int, Tuple[int, ...]], action_dim: int):
                        super().__init__(config, state_dim, action_dim, pixel_based) # Pass dummy action_dim 1
                        # Re-build network to truly be a value function
                        if pixel_based:
                            self_inner.encoder = CNNCritic(config=config, state_dim=state_dim, action_dim=1).encoder # Re-use CNN encoder
                            self_inner.latent_dim = self_inner.encoder.output_dim
                            input_mlp_dim = self_inner.latent_dim
                            self_inner.mlp_head = MLPBlock(
                                input_dim=input_mlp_dim,
                                output_dim=1,
                                hidden_units=self_inner.q_hidden_units,
                                num_hidden_layers=self_inner.q_hidden_layers,
                                activation_fn_name="ReLU",
                                output_activation_fn_name=None
                            )
                        else:
                            input_mlp_dim = state_dim[0] if isinstance(state_dim, tuple) else state_dim
                            self_inner.mlp = MLPBlock(
                                input_dim=input_mlp_dim,
                                output_dim=1,
                                hidden_units=self_inner.q_hidden_units,
                                num_hidden_layers=self_inner.q_hidden_layers,
                                activation_fn_name="ReLU",
                                output_activation_fn_name=None
                            )
                        self_inner.to(self.device)

                    def forward(self_inner, state: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
                        state = state.to(self.device)
                        if self_inner.pixel_based:
                            latent_features = self_inner.encoder(state)
                            return self_inner.mlp_head(latent_features)
                        else:
                            return self_inner.mlp(state)
                
                self.critic: QNetwork = PPOValueNetwork(config=config, state_dim=state_dim, action_dim=action_dim).to(device)
        else: # state-based
            class PPOValueNetwork(QNetwork):
                def __init__(self_inner, config: Config, state_dim: Union[int, Tuple[int, ...]], action_dim: int):
                    super().__init__(config, state_dim, action_dim, pixel_based)
                    # Re-build network to truly be a value function
                    if not isinstance(state_dim, int):
                        state_dim = state_dim[0]
                    self_inner.mlp = MLPBlock(
                        input_dim=state_dim,
                        output_dim=1,
                        hidden_units=self_inner.q_hidden_units,
                        num_hidden_layers=self_inner.q_hidden_layers,
                        activation_fn_name="ReLU",
                        output_activation_fn_name=None
                    )
                    self_inner.to(self.device)

                def forward(self_inner, state: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
                    state = state.to(self.device)
                    return self_inner.mlp(state)
            self.critic: QNetwork = PPOValueNetwork(config=config, state_dim=state_dim, action_dim=action_dim).to(device)


        # Pass primary actor (policy_net) and critic (q_net) to the base class constructor.
        # RLBaseAgent will handle `self.policy_net`, `self.q_net`, and their optimizers.
        # For PPO, target networks are generally not used, so `target_policy_net` and `target_q_net`
        # inherited from `RLBaseAgent` will be essentially unused or simply copies for consistency.
        # We explicitly set `self.target_policy_net` and `self.target_q_net` to be `None` later or
        # ensure they are not updated.
        super().__init__(config, env_manager, self.actor, self.critic, device)

        # PPO-specific hyperparameters
        self.ppo_clip_epsilon: float = self.config.get_hyperparam('rl_agent.ppo_clip_epsilon')
        self.ppo_entropy_coef: float = self.config.get_hyperparam('rl_agent.ppo_entropy_coef')
        self.ppo_value_loss_coef: float = self.config.get_hyperparam('rl_agent.ppo_value_loss_coef')
        
        # Alpha (entropy tuning) is not typically used in PPO.
        # Ensure it's explicitly set to None or a dummy value.
        self.log_alpha = None
        self.alpha = None
        self.alpha_optimizer = None
        self.target_entropy = None

        # Number of optimization epochs over the collected data.
        # In this replay-buffer-based setup, `train_step` processes one batch.
        # `ppo_epochs` determines how many times this `train_step` logic runs per data collection cycle
        # within the RLTrainer. For this file, we assume `train_step` is called for each epoch.
        self.ppo_epochs: int = self.config.get_hyperparam('rl_agent.ppo_epochs')

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects an action given a state observation and returns its log-probability.

        Args:
            state (torch.Tensor): The current state observation (batch_size, *state_dim).
            deterministic (bool): If True, return the deterministic action (argmax);
                                  otherwise, sample from the policy.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - The chosen action (batch_size, 1).
                - The log-probability of the chosen action (batch_size, 1).
        """
        state = state.to(self.device)
        self.actor.eval() # Ensure actor is in eval mode for action sampling
        with torch.no_grad():
            logits = self.actor(state)
            dist = Categorical(logits=logits)

            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
        self.actor.train() # Set actor back to train mode
        return action.unsqueeze(-1), log_prob.unsqueeze(-1)


    def train_step(self, real_batch: Dict[str, torch.Tensor],
                   synthetic_batch: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Performs one training step for the PPO agent.
        This adapts PPO for mini-batch training from a replay buffer.
        It uses the current policy's log-probabilities at the start of the step as `old_log_probs`.

        Args:
            real_batch (Dict[str, torch.Tensor]): A dictionary of transition components
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
            else:
                for k, v in real_batch.items():
                    combined_batch[k] = torch.empty(0, *v.shape[1:], device=self.device)
                num_real_to_sample = 0

            # Sample from synthetic_batch
            if num_synthetic_to_sample > 0:
                synthetic_len: int = len(synthetic_batch['state'])
                if synthetic_len > 0:
                    synthetic_indices = torch.randperm(synthetic_len, device=self.device)[:num_synthetic_to_sample]
                    synthetic_sampled = {k: v[synthetic_indices] for k, v in synthetic_batch.items()}
                else:
                     synthetic_sampled = {k: torch.empty(0, *v.shape[1:], device=self.device) for k, v in synthetic_batch.items()}
                     num_synthetic_to_sample = 0

                for k in combined_batch.keys():
                    if combined_batch[k].shape[0] == 0:
                        combined_batch[k] = synthetic_sampled[k]
                    elif synthetic_sampled[k].shape[0] > 0:
                        combined_batch[k] = torch.cat([combined_batch[k], synthetic_sampled[k]], dim=0)
        else:
            real_len: int = len(real_batch['state'])
            if real_len >= rl_batch_size:
                real_indices = torch.randperm(real_len, device=self.device)[:rl_batch_size]
                combined_batch = {k: v[real_indices] for k, v in real_batch.items()}
            else:
                combined_batch = real_batch

        if combined_batch['state'].shape[0] == 0:
            return {} # No data to train on

        state = combined_batch['state'].to(self.device)
        action = combined_batch['action'].to(self.device)
        reward = combined_batch['reward'].to(self.device)
        next_state = combined_batch['next_state'].to(self.device)
        done = combined_batch['done'].to(self.device).float()

        metrics: Dict[str, float] = {}

        # Get the 'old' log probabilities (approximated by current policy's log_probs before update)
        # In a typical PPO, these would come from the behavior policy that collected the data.
        # Here, we treat the current policy's log-probs before the update as the "old" ones.
        with torch.no_grad():
            old_logits = self.actor(state)
            old_dist = Categorical(logits=old_logits)
            old_log_probs = old_dist.log_prob(action.squeeze(-1)).unsqueeze(-1) # Action is (batch, 1)

            # Value prediction for current state
            value_preds = self.critic(state) # Critic is a value network, action is ignored
            
            # Value prediction for next state
            next_value_preds = self.critic(next_state)

            # Compute TD target and Advantages
            td_target = reward + self.gamma * next_value_preds * (1 - done)
            advantages = td_target - value_preds
            
            # Optional: Normalize advantages for stability
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        # PPO typically iterates multiple epochs over the same collected data.
        # For simplicity in this `train_step`, we perform one update here, assuming `RLTrainer`
        # calls `train_step` `ppo_epochs` times if needed.
        # However, the `utd_ratio` means `train_step` is called many times with *new* batches.
        # To simulate PPO epochs on a single batch, we loop `self.ppo_epochs` times.
        
        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        entropy_loss_sum = 0.0

        for _ in range(self.ppo_epochs):
            # Compute current policy outputs
            new_logits = self.actor(state)
            new_dist = Categorical(logits=new_logits)
            new_log_probs = new_dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
            
            # Compute value predictions for current state
            current_value_preds = self.critic(state)

            # --- Policy Loss ---
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.ppo_clip_epsilon, 1 + self.ppo_clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # --- Value Loss ---
            # Value target is detached so gradients don't flow back into the policy
            value_loss = F.mse_loss(current_value_preds, td_target.detach()) * self.ppo_value_loss_coef

            # --- Entropy Loss ---
            entropy_loss = new_dist.entropy().mean() * self.ppo_entropy_coef

            # --- Total Loss ---
            total_loss = policy_loss + value_loss - entropy_loss # Maximize entropy, so subtract

            # Optimize actor and critic
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            actor_loss_sum += policy_loss.item()
            critic_loss_sum += value_loss.item()
            entropy_loss_sum += entropy_loss.item()

        metrics['policy_loss'] = actor_loss_sum / self.ppo_epochs
        metrics['value_loss'] = critic_loss_sum / self.ppo_epochs
        metrics['entropy_loss'] = entropy_loss_sum / self.ppo_epochs
        metrics['total_ppo_loss'] = (actor_loss_sum + critic_loss_sum - entropy_loss_sum) / self.ppo_epochs

        return metrics

    def sync_target_networks(self) -> None:
        """
        PPO does not typically use target networks for its policy or value function.
        This method is a no-op for PPOAgent.
        """
        pass # No target networks in standard PPO

    def save_checkpoint(self, path: str) -> None:
        """
        Saves the state of the agent's networks and optimizers to a file.

        Args:
            path (str): The file path where the checkpoint should be saved.
        """
        checkpoint_data: Dict[str, Any] = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint_data, path)

    def load_checkpoint(self, path: str) -> None:
        """
        Loads the state of the agent's networks and optimizers from a file.

        Args:
            path (str): The file path from which to load the checkpoint.
        """
        checkpoint_data: Dict[str, Any] = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint_data['actor_state_dict'])
        self.critic.load_state_dict(checkpoint_data['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint_data['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint_data['critic_optimizer_state_dict'])

        self.actor.train()
        self.critic.train()

    def get_policy_nets(self) -> Tuple[PolicyNetwork, QNetwork]:
        """
        Returns the agent's current policy and value network.
        For PPO, the QNetwork interface is adapted to represent a ValueNetwork.
        """
        return self.actor, self.critic

