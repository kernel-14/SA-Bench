"""
trainer.py
Manages the training lifecycle of the MR.Q algorithm, including sampling from replay buffer,
loss computation, backpropagation, target network updates, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Tuple
from replay_buffer import ReplayBuffer
from model import Model
from utils import Utils


class Trainer:
    """
    Trainer class for managing the training of the MR.Q algorithm.
    """

    def __init__(self, model: Model, optimizer: optim.Optimizer, replay_buffer: ReplayBuffer, config: Dict[str, Any]):
        """
        Initialize the Trainer.

        Args:
            model (Model): The MR.Q deep RL model.
            optimizer (optim.Optimizer): Optimizer for training.
            replay_buffer (ReplayBuffer): An instance of prioritized experience replay buffer.
            config (Dict[str, Any]): Configuration dictionary parsed from `config.yaml`.
        """
        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Target network for stabilization
        self.target_network = Model(config["model"]).to(self.device)
        self.target_network.load_state_dict(self.model.state_dict())
        self.target_network.eval()

        # Hyperparameters from config
        training_config = self.config["training"]
        self.batch_size = training_config["batch_size"]
        self.gamma = training_config["gamma"]
        self.target_update_frequency = training_config["target_network_update_frequency"]
        self.prioritized_replay_alpha = training_config["prioritized_replay_alpha"]
        self.action_repeat = training_config["action_repeat"]

        loss_weights = self.config["loss_weights"]
        self.lambda_reward = loss_weights["reward_loss_lambda"]
        self.lambda_dynamics = loss_weights["dynamics_loss_lambda"]
        self.lambda_terminal = loss_weights["terminal_loss_lambda"]
        self.pre_act_reg_lambda = loss_weights["pre_activation_regularization_lambda"]

        # TensorBoard logger
        self.logger = Utils.setup_logger(log_dir=config["logging"]["log_dir"])

        # Miscellaneous
        self.step_counter = 0
        self.checkpoint_dir = config["logging"]["checkpoint_dir"]

    def train_one_epoch(self):
        """
        Train the model for one epoch using batches sampled from the ReplayBuffer.
        """
        if self.replay_buffer.size < self.batch_size:
            raise ValueError("Not enough data in ReplayBuffer to sample a full batch.")

        # Sample a batch of transitions
        transitions, indices, is_weights = self.replay_buffer.sample_batch(batch_size=self.batch_size)
        transitions = self._prepare_batch(transitions)

        # Forward pass through the model
        z_s = self.model.forward_state(transitions["state"])
        z_sa = self.model.forward_state_action(transitions["state"], transitions["action"])
        reward_preds, next_state_preds, terminal_preds = torch.split(z_sa, [1, self.model.zs_dim, 1], dim=-1)

        # Forward pass through target model (next state embeddings)
        with torch.no_grad():
            target_embeds = self.target_network.forward_state(transitions["next_state"])

        # Compute losses
        losses, td_errors = self.compute_loss(
            transitions, reward_preds, next_state_preds, terminal_preds, target_embeds
        )

        total_loss = (
            self.lambda_reward * losses["reward_loss"]
            + self.lambda_dynamics * losses["dynamics_loss"]
            + self.lambda_terminal * losses["terminal_loss"]
        )

        # Backpropagation and gradient update
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config["optimizer"]["clip_gradient_norm"])
        self.optimizer.step()

        # Update priorities and log metrics
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        self.log_metrics(losses, td_errors)

        # Update step counter and target network periodically
        self.step_counter += 1
        if self.step_counter % self.target_update_frequency == 0:
            self.update_target_network()

    def update_target_network(self):
        """
        Synchronize the weights of the target network with the current model.
        """
        self.target_network.load_state_dict(self.model.state_dict())
        print("Target network updated.")

    def compute_loss(
        self,
        transitions: Dict[str, torch.Tensor],
        reward_preds: torch.Tensor,
        next_state_preds: torch.Tensor,
        terminal_preds: torch.Tensor,
        target_embeds: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute the reward, dynamics, and terminal prediction losses.

        Args:
            transitions (Dict[str, torch.Tensor]): Batch of transitions from the replay buffer.
            reward_preds (torch.Tensor): Model's predicted rewards.
            next_state_preds (torch.Tensor): Model's predicted next-state embeddings.
            terminal_preds (torch.Tensor): Model's predicted terminal signals.
            target_embeds (torch.Tensor): Target network embeddings for the next state.

        Returns:
            Tuple containing:
            - A dictionary of individual losses.
            - Tensor of batch-average TD errors for prioritized replay.
        """
        # Reward Loss
        reward_targets = transitions["reward"]
        reward_targets_encoded = torch.tensor(
            [
                Utils.preprocess_reward(reward.item(), bins=self.config["reward_binning"]["bins"])
                for reward in reward_targets
            ],
            dtype=torch.float32,
        ).to(self.device)
        reward_loss = F.cross_entropy(reward_preds.squeeze(-1), reward_targets_encoded)

        # Dynamics Loss
        dynamics_loss = F.mse_loss(next_state_preds, target_embeds)

        # Terminal Loss
        terminal_targets = transitions["terminal"]
        terminal_loss = F.mse_loss(terminal_preds.squeeze(-1), terminal_targets)

        # Compute TD Errors (for prioritized experience replay)
        with torch.no_grad():
            target_q_values = transitions["reward"] + self.gamma * target_embeds.max(dim=1)[0] * (1 - terminal_targets)
            td_errors = torch.abs(transitions["q_values"] - target_q_values)

        return {
            "reward_loss": reward_loss,
            "dynamics_loss": dynamics_loss,
            "terminal_loss": terminal_loss,
        }, td_errors

    def log_metrics(self, losses: Dict[str, float], td_errors: torch.Tensor):
        """
        Log key metrics (losses, TD errors) to TensorBoard.

        Args:
            losses (Dict[str, float]): Dictionary of individual losses.
            td_errors (torch.Tensor): Tensor of TD errors.
        """
        current_step = self.step_counter
        self.logger.add_scalar("Loss/Reward", losses["reward_loss"].item(), current_step)
        self.logger.add_scalar("Loss/Dynamics", losses["dynamics_loss"].item(), current_step)
        self.logger.add_scalar("Loss/Terminal", losses["terminal_loss"].item(), current_step)
        self.logger.add_scalar("TD_Error/Mean", td_errors.mean().item(), current_step)
        self.logger.add_scalar("TD_Error/Max", td_errors.max().item(), current_step)

    def _prepare_batch(self, transitions: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Convert a batch of transitions to tensors for use in training.

        Args:
            transitions (List[Dict[str, Any]]): List of sampled transitions.

        Returns:
            Dict[str, torch.Tensor]: Tensorized transitions.
        """
        batch = {key: torch.tensor([sample[key] for sample in transitions], device=self.device) for key in transitions[0]}
        return batch
