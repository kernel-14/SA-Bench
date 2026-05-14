## rl_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any
from replay_buffer import ReplayBuffer

class RLAgent:
    """
    Implements an RL agent for off-policy reinforcement learning, compatible with SAC and REDQ algorithms.
    Handles policies, replay buffers, and interactions with the environment.
    """

    def __init__(self, config: dict) -> None:
        """
        Initializes the RL agent with policy and critic networks, replay buffers, optimizers, 
        and hyperparameters based on the configuration.

        Args:
            config (dict): Configuration dictionary parsed from `config.yaml`.
        """
        # Load agent settings from config
        self.rl_algorithm = config["training"]["rl_algorithm"]
        self.gamma = config["training"]["discount_factor"]
        self.learning_rate = config["training"]["learning_rate"]
        self.batch_size = config["training"]["batch_size"]
        self.utd_ratio = config["training"]["update_to_data_ratio"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize actor (policy network)
        self.actor = self._build_actor(config)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        # Initialize critic networks
        if self.rl_algorithm == "SAC":
            self.critic = self._build_critic(config)
            self.target_critic = self._build_critic(config)  # Target Q-network
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
            self.update_target_network(self.target_critic, self.critic, tau=1.0)  # Initialize target
        elif self.rl_algorithm == "REDQ":
            self.critic_ensemble = [self._build_critic(config) for _ in range(10)]  # Ensemble of 10 Q-networks
            self.target_critic_ensemble = [self._build_critic(config) for _ in range(10)]  # Target ensemble
            self.critic_ensemble_optimizers = [
                optim.Adam(critic.parameters(), lr=self.learning_rate)
                for critic in self.critic_ensemble
            ]
            for target_critic, critic in zip(self.target_critic_ensemble, self.critic_ensemble):
                self.update_target_network(target_critic, critic, tau=1.0)  # Initialize target ensemble
        else:
            raise ValueError(f"Unsupported RL algorithm: {self.rl_algorithm}")

    def _build_actor(self, config: dict) -> nn.Module:
        """
        Builds the actor network (policy network).

        Args:
            config (dict): Configuration dictionary.

        Returns:
            nn.Module: Actor (policy) network.
        """
        return nn.Sequential(
            nn.Linear(config["curiosity_module"]["latent_dim"], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, config["replay_buffer"]["action_space"], bias=False),
            nn.Tanh()  # Squash action values to [-1, 1]
        ).to(self.device)

    def _build_critic(self, config: dict) -> nn.Module:
        """
        Builds the critic network (Q-value network).

        Args:
            config (dict): Configuration dictionary.

        Returns:
            nn.Module: Critic network.
        """
        return nn.Sequential(
            nn.Linear(config["curiosity_module"]["latent_dim"] + config["replay_buffer"]["action_space"], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Outputs a single Q-value score
        ).to(self.device)

    def collect_transitions(self, env: Any, replay_buffer: ReplayBuffer, num_steps: int) -> List[Dict[str, Any]]:
        """
        Collects transitions by interacting with the environment using the current policy.

        Args:
            env (Any): Initialized environment instance.
            replay_buffer (ReplayBuffer): Replay buffer to store collected transitions.
            num_steps (int): Number of steps to collect.

        Returns:
            List[Dict[str, Any]]: Collected transitions.
        """
        state = env.reset()
        collected_transitions = []

        for _ in range(num_steps):
            # Sample action from the policy (with exploration noise for SAC)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actor(state_tensor).cpu().detach().numpy()
            action += np.random.normal(0, 0.1, size=action.shape)  # Exploration noise

            # Step environment
            next_state, reward, done, _ = env.step(action)
            transition = {
                "state": state,
                "action": action,
                "next_state": next_state,
                "reward": reward,
                "done": done,
            }
            collected_transitions.append(transition)

            # Store transition in replay buffer
            replay_buffer.store(transition)

            state = next_state if not done else env.reset()

        return collected_transitions

    def update_policy(self, real_buffer: ReplayBuffer, synthetic_buffer: ReplayBuffer) -> None:
        """
        Updates the policy using transitions sampled from the replay buffers.

        Args:
            real_buffer (ReplayBuffer): Replay buffer containing real transitions.
            synthetic_buffer (ReplayBuffer): Replay buffer containing synthetic transitions.
        """
        # Sample mixed batch of transitions
        batch = ReplayBuffer.mix_batches(real_buffer, synthetic_buffer, self.utd_ratio / (self.utd_ratio + 1))

        # Process batch into tensors
        state_batch = torch.tensor(
            np.array([transition["state"] for transition in batch]),
            dtype=torch.float32, device=self.device
        )
        action_batch = torch.tensor(
            np.array([transition["action"] for transition in batch]),
            dtype=torch.float32, device=self.device
        )
        reward_batch = torch.tensor(
            np.array([transition["reward"] for transition in batch]),
            dtype=torch.float32, device=self.device
        )
        next_state_batch = torch.tensor(
            np.array([transition["next_state"] for transition in batch]),
            dtype=torch.float32, device=self.device
        )
        done_batch = torch.tensor(
            np.array([transition["done"] for transition in batch]),
            dtype=torch.float32, device=self.device
        )

        # Critic and Actor Updates (Depends on RL Algorithm)
        if self.rl_algorithm == "SAC":
            self._update_sac(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        elif self.rl_algorithm == "REDQ":
            self._update_redq(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def _update_sac(self, states, actions, rewards, next_states, dones):
        """
        Updates critic and actor networks using SAC-specific loss functions.

        Args:
            states (torch.Tensor): Batch of states.
            actions (torch.Tensor): Batch of actions.
            rewards (torch.Tensor): Batch of rewards.
            next_states (torch.Tensor): Batch of next states.
            dones (torch.Tensor): Batch of done flags.
        """
        # Critic update
        target_q_values = rewards + self.gamma * (1 - dones) * self.target_critic(torch.cat([next_states, actions], dim=1))
        critic_loss = nn.MSELoss()(self.critic(torch.cat([states, actions], dim=1)), target_q_values.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_actions = self.actor(states)
        actor_loss = -self.critic(torch.cat([states, actor_actions], dim=1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target critic
        self.update_target_network(self.target_critic, self.critic, tau=0.005)

    def _update_redq(self, states, actions, rewards, next_states, dones):
        """
        Updates critic and actor networks using REDQ-specific loss functions.

        Args:
            states (torch.Tensor): Batch of states.
            actions (torch.Tensor): Batch of actions.
            rewards (torch.Tensor): Batch of rewards.
            next_states (torch.Tensor): Batch of next states.
            dones (torch.Tensor): Batch of done flags.
        """
        # Critic updates
        target_q_values = rewards + self.gamma * (1 - dones) * torch.min(
            torch.stack([target(torch.cat([next_states, actions], dim=1)) for target in self.target_critic_ensemble], dim=0),
            dim=0
        ).values

        for critic, optimizer in zip(self.critic_ensemble, self.critic_ensemble_optimizers):
            critic_loss = nn.MSELoss()(critic(torch.cat([states, actions], dim=1)), target_q_values.detach())
            optimizer.zero_grad()
            critic_loss.backward()
            optimizer.step()

        # Actor update
        actor_actions = self.actor(states)
        actor_loss = -torch.mean(torch.min(
            torch.stack([critic(torch.cat([states, actor_actions], dim=1)) for critic in self.critic_ensemble], dim=0),
            dim=0
        ).values)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target critics
        for target_critic, critic in zip(self.target_critic_ensemble, self.critic_ensemble):
            self.update_target_network(target_critic, critic, tau=0.005)

    def update_target_network(self, target_network: nn.Module, source_network: nn.Module, tau: float) -> None:
        """
        Updates the target network parameters using a soft update mechanism.

        Args:
            target_network (nn.Module): Target network.
            source_network (nn.Module): Source network.
            tau (float): Soft update coefficient.
        """
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)
