"""Baseline implementations: PER with various priority criteria, exploration bonuses.

Note: MBPO and DREAMER-V3 are external baselines referenced from prior work.
We provide PER and exploration bonus baselines as described in Section 5.1.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

from models.policy import REDQPolicy, SACPolicy
from models.curiosity import ICM, PixelICM, RND
from replay import ReplayBuffer, PrioritizedReplayBuffer


class PER_Baseline:
    """Prioritized Experience Replay baseline (Schaul et al. 2015).

    Uses REDQ policy with prioritized replay buffer.
    Priority can be TD-error (Eq. 4) or curiosity (Eq. 5).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        priority_type: str = "td_error",  # or "curiosity"
        hidden_dim: int = 256,
        n_layers: int = 2,
        n_critics: int = 10,
        n_target_critics: int = 2,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 1_000_000,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.priority_type = priority_type
        self.gamma = gamma

        # Policy
        self.policy = REDQPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_critics=n_critics,
            n_target_critics=n_target_critics,
            gamma=gamma,
            tau=tau,
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(
            sum([list(c.parameters()) for c in self.policy.critics], []), lr=3e-4
        )
        self.alpha_optimizer = torch.optim.Adam([self.policy.log_alpha], lr=3e-4)

        # Prioritized replay buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            state_dim=state_dim,
            action_dim=action_dim,
            alpha=per_alpha,
            beta=per_beta,
        )

        # Optional curiosity module for priority computation
        if priority_type == "curiosity":
            self.curiosity = ICM(
                state_dim=state_dim,
                action_dim=action_dim,
            ).to(self.device)
        else:
            self.curiosity = None

        self.total_env_steps = 0

    def compute_priority(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> float:
        """Compute priority for a transition."""
        if self.priority_type == "td_error":
            # Eq. (4): TD-error
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            a = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            r = torch.FloatTensor([reward]).unsqueeze(0).to(self.device)
            ns = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            d = torch.FloatTensor([done]).unsqueeze(0).to(self.device)

            with torch.no_grad():
                next_actions, next_log_probs, _ = self.policy.actor.sample(ns)
                q1 = self.policy.critics[0](ns, next_actions)
                q2 = self.policy.critics[1](ns, next_actions)
                q_next = torch.min(q1, q2)
                q_current = self.policy.critics[0](s, a)
                td = r + self.gamma * (1 - d) * q_next - q_current
                return td.abs().item()

        elif self.priority_type == "curiosity":
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            a = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            ns = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.curiosity.compute_relevance(s, a, ns).item()

        return 1.0

    def update(self, batch_size: int, utd: int = 1):
        """Single policy update step."""
        if len(self.buffer) < batch_size:
            return

        s, a, r, ns, d, weights, indices = self.buffer.sample(batch_size, self.device)

        for _ in range(utd):
            # Critic update
            critic_loss = self.policy.critic_loss(s, a, r, ns, d)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor update
            actor_loss, alpha_loss = self.policy.actor_loss(s)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.policy.update_targets()

        # Update priorities
        with torch.no_grad():
            s, a, r, ns, d, _, indices = self.buffer.sample(batch_size, self.device)
            if self.priority_type == "td_error":
                next_actions, next_log_probs, _ = self.policy.actor.sample(ns)
                q1 = self.policy.critics[0](ns, next_actions)
                q2 = self.policy.critics[1](ns, next_actions)
                q_next = torch.min(q1, q2)
                q_current = self.policy.critics[0](s, a)
                td = r + self.gamma * (1 - d) * q_next - q_current
                new_priorities = td.abs().cpu().numpy().flatten() + 1e-6
            else:
                new_priorities = self.curiosity.compute_relevance(s, a, ns).cpu().numpy().flatten()

            self.buffer.update_priorities(indices, new_priorities)


class ExplorationBonusBaseline:
    """Baseline: REDQ + intrinsic curiosity reward bonus.

    Adds exploration bonus to extrinsic reward (Section 5.1, Fig. 3b).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        bonus_type: str = "curiosity",  # or "rnd"
        intrinsic_weight: float = 0.1,
        hidden_dim: int = 256,
        n_layers: int = 2,
        n_critics: int = 10,
        n_target_critics: int = 2,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 1_000_000,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.intrinsic_weight = intrinsic_weight
        self.gamma = gamma

        # Policy
        self.policy = REDQPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_critics=n_critics,
            n_target_critics=n_target_critics,
            gamma=gamma,
            tau=tau,
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(
            sum([list(c.parameters()) for c in self.policy.critics], []), lr=3e-4
        )
        self.alpha_optimizer = torch.optim.Adam([self.policy.log_alpha], lr=3e-4)

        # Replay buffer
        self.buffer = ReplayBuffer(
            capacity=buffer_capacity,
            state_dim=state_dim,
            action_dim=action_dim,
        )

        # Intrinsic bonus module
        if bonus_type == "curiosity":
            self.bonus = ICM(
                state_dim=state_dim,
                action_dim=action_dim,
                intrinsic_reward_weight=intrinsic_weight,
            ).to(self.device)
        elif bonus_type == "rnd":
            self.bonus = RND(
                input_dim=state_dim,
            ).to(self.device)

        self.total_env_steps = 0

    def compute_reward(self, state, action, next_state, extrinsic_reward: float) -> float:
        """Add intrinsic bonus to extrinsic reward."""
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        a = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        ns = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            intrinsic = self.bonus.compute_intrinsic_reward(s, a, ns)
        return extrinsic_reward + intrinsic.item()
