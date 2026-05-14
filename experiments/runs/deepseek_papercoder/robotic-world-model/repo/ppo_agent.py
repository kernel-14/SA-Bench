"""
ppo_agent.py

Implements PPOAgent: a Proximal Policy Optimization (PPO) actor‑critic agent
for continuous action spaces, used in the MBPO‑PPO pipeline of the RWM paper.

Architecture follows Table S9 and hyper‑parameters from Table S11.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class PPOAgent:
    """PPO agent with Gaussian policy and a state‑value critic.

    The policy outputs the mean of a diagonal Gaussian; a separate trainable
    log‑standard‑deviation parameter is shared across observations.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = (128, 128, 128),
        activation: str = "elu",
        lr: float = 0.001,
        clip_range: float = 0.2,
        entropy_coef: float = 0.005,
        device: str = "cuda",
    ):
        """
        Args:
            obs_dim:       Dimensionality of the observation space.
            act_dim:       Dimensionality of the action space.
            hidden_dims:   List of hidden sizes for the actor/critic MLPs.
            activation:    'relu' or 'elu'.
            lr:            Learning rate (Adam).
            clip_range:    PPO clipping parameter ε.
            entropy_coef:  Coefficient for entropy bonus.
            device:        Torch device to place the networks on.
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)

        # ---------- Activation function ----------
        if activation.lower() not in ("relu", "elu"):
            raise ValueError(f"Unsupported activation: {activation}")
        act_fn = nn.ReLU if activation.lower() == "relu" else nn.ELU

        # ---------- Actor (policy) ----------
        actor_layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            actor_layers.append(nn.Linear(in_dim, h))
            actor_layers.append(act_fn())
            in_dim = h
        # Final linear layer to action mean (no activation)
        actor_layers.append(nn.Linear(in_dim, act_dim))
        self.actor = nn.Sequential(*actor_layers)

        # Trainable log‑std per action dimension (initialised to 0 ⇒ std = 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim, device=self.device, dtype=torch.float32))

        # ---------- Critic (value function) ----------
        critic_layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            critic_layers.append(nn.Linear(in_dim, h))
            critic_layers.append(act_fn())
            in_dim = h
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # ---------- Optimizer (actor, log_std, critic) ----------
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + [self.log_std] + list(self.critic.parameters()),
            lr=lr,
            weight_decay=0.0,   # as per Table S11
        )

        # Move networks to device
        self.actor.to(self.device)
        self.critic.to(self.device)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action from the current policy.

        Args:
            obs: Observation tensor of shape (batch_size, obs_dim) or (obs_dim,).

        Returns:
            action:    Sampled action, shape (batch_size, act_dim).
            log_prob:  Log probability of the sampled action under the policy,
                       shape (batch_size, 1).
        """
        # Ensure batch dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device, dtype=torch.float32)

        # Compute action mean
        action_mean = self.actor(obs)              # (B, act_dim)

        # Standard deviation from learned parameter
        std = torch.exp(self.log_std)              # (act_dim,)

        # Create distribution and sample
        dist = Normal(action_mean, std)
        action = dist.sample()                     # (B, act_dim)

        # Log probability (sum across action dimensions)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)  # (B, 1)

        return action, log_prob

    def evaluate(
        self, obs: torch.Tensor, act: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the policy and value function for given observation-action pairs.

        Args:
            obs: Observation tensor of shape (batch_size, obs_dim).
            act: Action tensor of shape (batch_size, act_dim).

        Returns:
            action_mean: The mean of the policy distribution, shape (batch_size, act_dim).
            log_prob:    Log probability of the given actions under the current policy,
                         shape (batch_size, 1).
            value:       State‑value estimates, shape (batch_size, 1).
        """
        obs = obs.to(self.device, dtype=torch.float32)
        act = act.to(self.device, dtype=torch.float32)

        action_mean = self.actor(obs)              # (B, act_dim)

        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)

        log_prob = dist.log_prob(act).sum(dim=-1, keepdim=True)  # (B, 1)

        value = self.critic(obs)                   # (B, 1)

        return action_mean, log_prob, value

    def update(
        self,
        rollout_buffer: Dict[str, torch.Tensor],
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = 0.01,
        ppo_epochs: int = 5,
        ppo_minibatches: int = 4,
    ) -> Dict[str, float]:
        """Perform PPO update on the given rollout data.

        The rollout buffer is expected to contain:
            'obs'           : shape (N, obs_dim)
            'act'           : shape (N, act_dim)
            'old_log_prob'  : shape (N, 1)
            'adv'           : shape (N, 1)   (normalised advantages)
            'ret'           : shape (N, 1)   (returns)

        Args:
            rollout_buffer: Dict with the above keys holding torch tensors.
            value_loss_coef: Weight of the value function loss.
            max_grad_norm:  Maximum gradient norm for clipping.
            target_kl:      If not None, early‑stops an epoch if KL exceeds this value.
            ppo_epochs:     Number of epochs over the full buffer (config: 5).
            ppo_minibatches: Number of minibatches per epoch (config: 4).

        Returns:
            Dictionary of average losses and metrics for logging.
        """
        obs = rollout_buffer["obs"].to(self.device, dtype=torch.float32)
        act = rollout_buffer["act"].to(self.device, dtype=torch.float32)
        old_log_prob = rollout_buffer["old_log_prob"].to(self.device, dtype=torch.float32)
        advantages = rollout_buffer["adv"].to(self.device, dtype=torch.float32)
        returns = rollout_buffer["ret"].to(self.device, dtype=torch.float32)

        N = obs.shape[0]
        minibatch_size = max(1, N // ppo_minibatches)

        losses = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
        }

        self.actor.train()
        self.critic.train()

        for epoch in range(ppo_epochs):
            # Shuffle indices
            perm = torch.randperm(N, device=self.device)
            obs_shuf = obs[perm]
            act_shuf = act[perm]
            old_log_prob_shuf = old_log_prob[perm]
            advantages_shuf = advantages[perm]
            returns_shuf = returns[perm]

            epoch_policy_loss = epoch_value_loss = epoch_entropy_loss = 0.0
            epoch_count = 0

            for start in range(0, N, minibatch_size):
                end = min(start + minibatch_size, N)
                mb_obs = obs_shuf[start:end]
                mb_act = act_shuf[start:end]
                mb_old_log_prob = old_log_prob_shuf[start:end]
                mb_adv = advantages_shuf[start:end]
                mb_ret = returns_shuf[start:end]

                # Evaluate current policy on the minibatch
                _, new_log_prob, values = self.evaluate(mb_obs, mb_act)

                # Ratio
                ratio = torch.exp(new_log_prob - mb_old_log_prob)

                # Clipped surrogate objective
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = ((mb_ret - values) ** 2).mean()

                # Entropy bonus (using the same std)
                # We need the action distribution again from the actor mean
                action_mean = self.actor(mb_obs)        # (mb, act_dim)
                std = torch.exp(self.log_std)
                dist = Normal(action_mean, std)
                entropy = dist.entropy().sum(dim=-1).mean()  # scalar
                entropy_loss = -self.entropy_coef * entropy

                # Total loss
                loss = policy_loss + value_loss_coef * value_loss + entropy_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                self.optimizer.step()

                # Track statistics
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy_loss += entropy_loss.item()
                epoch_count += 1

            # Average losses for the epoch
            mean_policy_loss = epoch_policy_loss / epoch_count
            mean_value_loss = epoch_value_loss / epoch_count
            mean_entropy_loss = epoch_entropy_loss / epoch_count

            # Approximate KL divergence for early stopping
            with torch.no_grad():
                # Use the whole dataset for KL estimation
                _, new_log_prob_all, _ = self.evaluate(obs, act)
                ratio_all = torch.exp(new_log_prob_all - old_log_prob)
                approx_kl = ((ratio_all - 1) - torch.log(ratio_all)).mean().item()

            # Aggregate over epochs
            losses["policy_loss"] += mean_policy_loss
            losses["value_loss"] += mean_value_loss
            losses["entropy_loss"] += mean_entropy_loss
            losses["total_loss"] += mean_policy_loss + mean_value_loss + mean_entropy_loss
            losses["approx_kl"] += approx_kl

            # Early stop if KL exceeds target (if provided)
            if target_kl is not None and approx_kl > target_kl:
                break

        # Average over completed epochs
        for key in losses:
            losses[key] /= (epoch + 1)

        return losses

    def parameters(self):
        """Return all trainable parameters (for gradient clipping)."""
        return list(self.actor.parameters()) + [self.log_std] + list(self.critic.parameters())

    def save(self, path: str) -> None:
        """Save the agent to the given path."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "log_std": self.log_std,
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load the agent from the given path."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.log_std = checkpoint["log_std"]
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Ensure obs/act dims match (they could be stored in the checkpoint)
        assert self.obs_dim == checkpoint.get("obs_dim", self.obs_dim), "Inconsistent obs_dim"
        assert self.act_dim == checkpoint.get("act_dim", self.act_dim), "Inconsistent act_dim"
