# ppo.py

import torch
from torch import nn
from torch.optim import Adam
from typing import List, Dict, Tuple
from termination import Termination
from model import Model


class PPO:
    def __init__(self, policy_model: Model, critic_model: Model, termination: Termination, params: Dict):
        """
        Initialize the PPO class for Macro Action RLHF.
        
        Args:
            policy_model (Model): The policy model responsible for generating token probabilities.
            critic_model (Model): The critic model responsible for evaluating state values.
            termination (Termination): Termination strategy used to define macro-actions.
            params (Dict): Dictionary containing training hyperparameters.
        """
        self.policy_model = policy_model
        self.critic_model = critic_model
        self.termination = termination
        self.params = params

        # Hyperparameters
        self.gamma = self.params.get("gamma", 0.99)  # Discount factor
        self.clip_range = self.params.get("clip_range", 0.2)  # PPO clipping range
        self.learning_rate = self.params.get("learning_rate", 1e-5)

        # Optimizers for both policy and critic models
        self.optimizer_policy = Adam(self.policy_model.trainable_parameters(), lr=self.learning_rate)
        self.optimizer_critic = Adam(self.critic_model.trainable_parameters(), lr=self.learning_rate)

    def compute_loss(self, data: List[Dict], macro_actions: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute policy loss and value loss for MA-PPO.
        
        Args:
            data (List[Dict]): Batch data containing input sequences and associated rewards.
            macro_actions (List[List[str]]): List of macro-actions segmented for each sequence.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Policy loss and value loss tensors.
        """
        # Initialize loss accumulators
        policy_loss, value_loss = 0.0, 0.0
        total_macro_actions = 0

        for entry, macro_action_list in zip(data, macro_actions):
            input_ids = entry["input_ids"]
            rewards = torch.tensor(entry["rewards"], dtype=torch.float32)

            # Retrieve token-level model outputs
            logits = self.policy_model.forward(input_ids)
            values = self.critic_model.forward(input_ids)

            # Decode logits into log probabilities
            logprobs = torch.log_softmax(logits, dim=-1)

            # Compute macro-action log probabilities and values
            macro_logprobs = self.compute_macro_action_logprobs(macro_action_list, logprobs)
            macro_values = self.compute_macro_action_values(macro_action_list, values)

            # Compute rewards at macro-action level
            macro_rewards = self.compute_macro_rewards(macro_action_list, rewards)

            # Calculate advantages using Generalized Advantage Estimation (GAE)
            advantages = self.compute_advantages(macro_rewards, macro_values)

            # Compute policy ratio for each macro-action
            macro_ratios = self.compute_macro_action_ratios(macro_logprobs, entry["old_logprobs"])

            # Calculate policy loss
            policy_loss_entry = self.policy_loss_macro_action(macro_ratios, advantages)
            policy_loss += policy_loss_entry

            # Calculate value loss
            value_loss_entry = self.value_loss_macro_action(macro_values, advantages)
            value_loss += value_loss_entry

            total_macro_actions += len(macro_action_list)

        # Normalize losses by total number of macro-actions
        if total_macro_actions > 0:
            policy_loss /= total_macro_actions
            value_loss /= total_macro_actions

        return policy_loss, value_loss

    def optimize(self, policy_loss: torch.Tensor, value_loss: torch.Tensor) -> None:
        """
        Perform optimization for both policy and critic models.

        Args:
            policy_loss (torch.Tensor): Computed PPO policy loss.
            value_loss (torch.Tensor): Computed value loss.
        """
        # Zero gradients for policy optimizer and update weights
        self.optimizer_policy.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.optimizer_policy.step()

        # Zero gradients for critic optimizer and update weights
        self.optimizer_critic.zero_grad()
        value_loss.backward()
        self.optimizer_critic.step()

    def update_macro_actions(self, data: List[Dict]) -> List[List[str]]:
        """
        Compute macro-actions for each input sequence in the dataset using Termination strategies.
        
        Args:
            data (List[Dict]): Dataset containing tokenized input sequences.
        
        Returns:
            List[List[str]]: Macro-actions segmented per sequence.
        """
        macro_actions = []
        for entry in data:
            sequence = entry["input_ids"]
            macro_actions.append(self.termination.get_macro_actions(sequence))
        return macro_actions

    def compute_macro_action_values(self, macro_actions: List[List[str]], values: torch.Tensor) -> torch.Tensor:
        """
        Compute aggregated values for macro-actions based on token-level values.

        Args:
            macro_actions (List[List[str]]): List of macro-actions.
            values (torch.Tensor): Token-level values.

        Returns:
            torch.Tensor: Macro-action values aggregated by position or equal contributions.
        """
        macro_values = []
        for macro_action in macro_actions:
            token_values = [values[token_idx] for token_idx in macro_action]
            macro_values.append(sum(token_values) / len(token_values))  # Equal contribution
        return torch.tensor(macro_values, dtype=torch.float32)

    def compute_advantages(self, macro_rewards: torch.Tensor, macro_values: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages using macro rewards and values.

        Args:
            macro_rewards (torch.Tensor): Macro-level rewards.
            macro_values (torch.Tensor): Macro-level values.

        Returns:
            torch.Tensor: Computed advantages for macro-actions.
        """
        return macro_rewards - macro_values

    def compute_macro_action_logprobs(self, macro_actions: List[List[str]], logprobs: torch.Tensor) -> torch.Tensor:
        """
        Compute aggregated log probabilities for macro-actions.

        Args:
            macro_actions (List[List[str]]): List of macro-actions.
            logprobs (torch.Tensor): Token-level log probabilities.

        Returns:
            torch.Tensor: Macro-action log probabilities aggregated across constituent tokens.
        """
        macro_logprobs = []
        for macro_action in macro_actions:
            macro_logprobs.append(sum(logprobs[token_idx] for token_idx in macro_action))
        return torch.tensor(macro_logprobs, dtype=torch.float32)

    def compute_macro_action_ratios(self, macro_logprobs: torch.Tensor, old_logprobs: torch.Tensor) -> torch.Tensor:
        """
        Compute policy ratios for macro-actions.

        Args:
            macro_logprobs (torch.Tensor): Current macro-action log probabilities.
            old_logprobs (torch.Tensor): Previous macro-action log probabilities.

        Returns:
            torch.Tensor: Ratios computed for macro-actions.
        """
        return torch.exp(macro_logprobs - old_logprobs)

    def compute_macro_rewards(self, macro_actions: List[List[str]], rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute aggregated macro-level rewards from token-level rewards.

        Args:
            macro_actions (List[List[str]]): List of macro-actions.
            rewards (torch.Tensor): Token-level rewards.

        Returns:
            torch.Tensor: Macro-level rewards aggregated by summation across tokens.
        """
        macro_rewards = []
        for macro_action in macro_actions:
            macro_rewards.append(sum(rewards[token_idx] for token_idx in macro_action))
        return torch.tensor(macro_rewards, dtype=torch.float32)

    def policy_loss_macro_action(self, ratios: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        """
        Compute policy loss for macro-actions using clipped ratios.

        Args:
            ratios (torch.Tensor): Policy ratios for macro-actions.
            advantages (torch.Tensor): Computed advantages for macro-actions.

        Returns:
            torch.Tensor: Policy loss for macro-actions.
        """
        clipped_ratios = torch.clamp(ratios, 1.0 - self.clip_range, 1.0 + self.clip_range)
        loss = torch.min(ratios * advantages, clipped_ratios * advantages)
        return -torch.sum(loss)

    def value_loss_macro_action(self, macro_values: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        """
        Compute value loss for macro-actions.

        Args:
            macro_values (torch.Tensor): Macro-action state values.
            advantages (torch.Tensor): Computed advantages for macro-actions.

        Returns:
            torch.Tensor: Value loss for macro-actions.
        """
        return nn.MSELoss()(macro_values, advantages)
