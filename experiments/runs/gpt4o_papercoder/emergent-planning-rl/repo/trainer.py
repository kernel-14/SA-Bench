# trainer.py

from typing import Optional
import os
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from drc_model import DRCModel
from sokoban_environment import SokobanEnvironment
import logging

class Trainer:
    """
    Trainer class for managing the training of the DRCModel on the Sokoban environment 
    using the IMPALA reinforcement learning algorithm.
    """

    def __init__(self, model: DRCModel, env: SokobanEnvironment, config: dict):
        """
        Initializes the Trainer class.

        Args:
            model (DRCModel): The model to train.
            env (SokobanEnvironment): The Sokoban environment instance.
            config (dict): Training and environment configurations.
        """
        self.model = model
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training parameters
        self.learning_rate = config["training"]["learning_rate"]
        self.weight_decay = config["training"]["weight_decay"]
        self.entropy_penalty = config["training"]["entropy_penalty"]
        self.discount_rate = config["training"]["discount_rate"]
        self.lambda_vtrace = config["training"]["lambda_vtrace"]
        self.max_grad_norm = 40.0  # Gradient clipping threshold

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.model.to(self.device)

        # Logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s :: %(levelname)s :: %(message)s",
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger("Trainer")

    def train(self, num_epochs: int) -> None:
        """
        Executes the primary training loop for the model.

        Args:
            num_epochs (int): Number of training epochs.
        """
        self.logger.info("Starting training for %d epochs.", num_epochs)

        for epoch in range(num_epochs):
            episode_reward_total = 0
            total_loss, policy_loss_sum, value_loss_sum, entropy_loss_sum = 0, 0, 0, 0
            episode_count = 0

            # Reset environment and initialize buffers
            state = self.env.reset()

            while episode_count < len(self.config["dataset"]["training_levels"]):
                # Collect a trajectory
                trajectory = self._collect_trajectory()
                # Compute loss and update model
                total_loss, policy_loss, value_loss, entropy_loss = self._update_model(trajectory)

                # Logging progress for this episode
                episode_reward_total += np.sum(trajectory["rewards"])
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_loss_sum += entropy_loss.item()
                episode_count += 1

            # Log metrics after each epoch
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Total Reward: {episode_reward_total:.2f}, "
                f"Policy Loss: {policy_loss_sum:.4f}, "
                f"Value Loss: {value_loss_sum:.4f}, "
                f"Entropy Loss: {entropy_loss_sum:.4f}"
            )

            # Save model checkpoint periodically
            if (epoch + 1) % 10 == 0:
                self.save_model(f"checkpoint_epoch_{epoch + 1}.pth")

        self.logger.info("Training complete.")

    def _collect_trajectory(self) -> dict:
        """
        Collects an episode trajectory by interacting with the environment.

        Returns:
            dict: Trajectory containing observations, actions, rewards, and log probabilities.
        """
        observations, actions, rewards, log_probs = [], [], [], []
        done = False
        state = self.env.reset()

        while not done:
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Forward pass to get policy and value
            policy_logits, _ = self.model(state_tensor)
            
            # Softmax to calculate probabilities and sample an action
            policy_probs = torch.softmax(policy_logits, dim=-1)
            action_dist = torch.distributions.Categorical(probs=policy_probs)
            action = action_dist.sample()

            # Perform step in the environment
            next_state, reward, done, _ = self.env.step(action.item())

            # Store trajectory elements
            observations.append(state)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(action_dist.log_prob(action))

            # Move to the next state
            state = next_state

        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "log_probs": log_probs,
        }

    def _update_model(self, trajectory: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Updates the model parameters using a collected trajectory.

        Args:
            trajectory (dict): An episode trajectory containing states, actions, rewards, and log probabilities.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Total loss, policy loss, value loss, and entropy loss.
        """
        # Extract components from trajectory
        observations = torch.tensor(np.array(trajectory["observations"]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(trajectory["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.tensor(trajectory["rewards"], dtype=torch.float32, device=self.device)
        log_probs = torch.stack(trajectory["log_probs"])

        # Compute discounted rewards (using V-trace if applicable)
        discounted_rewards = self._compute_discounted_rewards(rewards)

        # Forward pass through the model to get policy logits and value estimates
        policy_logits, values = self.model(observations)

        # Calculate value loss (critic loss)
        value_loss = F.mse_loss(values.squeeze(), discounted_rewards)

        # Calculate policy loss (actor loss)
        action_log_probs = torch.gather(torch.log_softmax(policy_logits, dim=-1), 1, actions.unsqueeze(1)).squeeze(1)
        advantages = discounted_rewards - values.squeeze()
        policy_loss = -(action_log_probs * advantages.detach()).mean()

        # Calculate entropy loss (for exploration)
        entropy = -torch.sum(torch.softmax(policy_logits, dim=-1) * torch.log_softmax(policy_logits, dim=-1), dim=-1)
        entropy_loss = -self.entropy_penalty * entropy.mean()

        # Total loss
        total_loss = policy_loss + value_loss + entropy_loss

        # Backpropagation and optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)  # Clip gradients
        self.optimizer.step()

        return total_loss, policy_loss, value_loss, entropy_loss

    def _compute_discounted_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Computes discounted rewards for a trajectory.

        Args:
            rewards (torch.Tensor): Tensor of rewards from the trajectory.

        Returns:
            torch.Tensor: Discounted rewards.
        """
        discounted = torch.zeros_like(rewards, device=self.device)
        running_add = 0.0

        for t in reversed(range(len(rewards))):
            running_add = rewards[t] + self.discount_rate * running_add
            discounted[t] = running_add

        return discounted

    def save_model(self, filepath: str) -> None:
        """
        Saves the current model and optimizer state to a file.

        Args:
            filepath (str): Path to save the model checkpoint.
        """
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, filepath)
        self.logger.info(f"Model checkpoint saved to {filepath}.")

    def load_model(self, filepath: str) -> None:
        """
        Loads the model and optimizer state from a checkpoint.

        Args:
            filepath (str): Path to the checkpoint file.
        """
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.logger.info(f"Model checkpoint loaded from {filepath}.")
        else:
            raise FileNotFoundError(f"Checkpoint file {filepath} not found.")
