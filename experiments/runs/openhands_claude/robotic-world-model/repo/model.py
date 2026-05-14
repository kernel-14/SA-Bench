"""
Top-level model classes that wrap the modules with loss computation and
autoregressive training logic.

Implements:
  - WorldModelLoss: multi-step prediction loss (Eq. 2)
  - RWM: full Robotic World Model with training/inference API
  - MLPBaseline / RSSMBaseline / TransformerBaseline: wrapped baselines
  - ActorCritic: combined policy + value for MBPO-PPO
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    ExperimentConfig,
    MLPBaselineConfig,
    MBPOPPOConfig,
    PolicyArchConfig,
    RSSMConfig,
    RWMArchConfig,
    RWMTrainingConfig,
    TransformerConfig,
    ValueArchConfig,
)
from modules import (
    MLPWorldModel,
    PolicyNetwork,
    RSSMWorldModel,
    RWMCore,
    TransformerWorldModel,
    ValueNetwork,
)


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class WorldModelLoss(nn.Module):
    """
    Multi-step prediction loss (Eq. 2):

        L = (1/N) * sum_{k=1}^{N} alpha^k * [L_o(o'_{t+k}, o_{t+k})
                                               + L_c(c'_{t+k}, c_{t+k})]

    L_o and L_c are negative log-likelihoods under Gaussian distributions.
    alpha=1.0 means no decay (all steps weighted equally).
    """

    def __init__(self, forecast_decay: float = 1.0):
        super().__init__()
        self.forecast_decay = forecast_decay

    def gaussian_nll(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        dist = torch.distributions.Normal(mean, std)
        return -dist.log_prob(target).sum(dim=-1).mean()

    def forward(
        self,
        obs_means: List[torch.Tensor],
        obs_stds: List[torch.Tensor],
        obs_targets: torch.Tensor,
        priv_means: List[torch.Tensor],
        priv_stds: List[torch.Tensor],
        priv_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            obs_means:    list of N tensors (B, obs_dim)
            obs_stds:     list of N tensors (B, obs_dim)
            obs_targets:  (B, N, obs_dim)
            priv_means:   list of N tensors (B, priv_dim)
            priv_stds:    list of N tensors (B, priv_dim)
            priv_targets: (B, N, priv_dim)

        Returns:
            total_loss, metrics dict
        """
        N = len(obs_means)
        total_loss = torch.tensor(0.0, device=obs_means[0].device)
        obs_loss_sum = 0.0
        priv_loss_sum = 0.0

        for k in range(N):
            weight = (self.forecast_decay ** (k + 1)) / N
            l_o = self.gaussian_nll(obs_means[k], obs_stds[k], obs_targets[:, k])
            l_c = self.gaussian_nll(priv_means[k], priv_stds[k], priv_targets[:, k])
            total_loss = total_loss + weight * (l_o + l_c)
            obs_loss_sum += l_o.item()
            priv_loss_sum += l_c.item()

        metrics = {
            "loss": total_loss.item(),
            "obs_loss": obs_loss_sum / N,
            "priv_loss": priv_loss_sum / N,
        }
        return total_loss, metrics


class RSSMLoss(nn.Module):
    """
    RSSM training loss: reconstruction NLL + KL divergence between
    posterior and prior categorical distributions.
    """

    def __init__(self, kl_weight: float = 1.0, free_nats: float = 1.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.free_nats = free_nats

    def categorical_kl(
        self, post_logits: torch.Tensor, prior_logits: torch.Tensor, num_categories: int
    ) -> torch.Tensor:
        B, T, D = post_logits.shape
        cat_size = D // num_categories
        post = post_logits.view(B * T, num_categories, cat_size)
        prior = prior_logits.view(B * T, num_categories, cat_size)
        post_probs = F.softmax(post, dim=-1)
        prior_probs = F.softmax(prior, dim=-1)
        kl = (post_probs * (torch.log(post_probs + 1e-8) - torch.log(prior_probs + 1e-8)))
        kl = kl.sum(dim=-1).sum(dim=-1)
        return kl.mean()

    def forward(
        self,
        obs_means: torch.Tensor,
        obs_stds: torch.Tensor,
        obs_targets: torch.Tensor,
        prior_logits: torch.Tensor,
        post_logits: torch.Tensor,
        num_categories: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        recon_loss = -torch.distributions.Normal(obs_means, obs_stds).log_prob(
            obs_targets
        ).sum(dim=-1).mean()
        kl = self.categorical_kl(post_logits, prior_logits, num_categories)
        kl_clamped = torch.clamp(kl, min=self.free_nats)
        total = recon_loss + self.kl_weight * kl_clamped
        return total, {
            "loss": total.item(),
            "recon_loss": recon_loss.item(),
            "kl": kl.item(),
        }


# ---------------------------------------------------------------------------
# RWM: Full Robotic World Model
# ---------------------------------------------------------------------------

class RWM(nn.Module):
    """
    Robotic World Model (RWM) — the main model from the paper.

    Wraps RWMCore with the multi-step autoregressive training loss.
    Supports both autoregressive (AR) and teacher-forcing (TF) training modes.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        privileged_dim: int,
        arch_cfg: RWMArchConfig,
        train_cfg: RWMTrainingConfig,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.privileged_dim = privileged_dim
        self.history_horizon = train_cfg.history_horizon
        self.forecast_horizon = train_cfg.forecast_horizon

        self.core = RWMCore(
            obs_dim=obs_dim,
            action_dim=action_dim,
            privileged_dim=privileged_dim,
            gru_hidden_size=arch_cfg.gru_hidden_size,
            gru_num_layers=arch_cfg.gru_num_layers,
            head_hidden_size=arch_cfg.head_hidden_size,
            head_activation=arch_cfg.head_activation,
        )
        self.loss_fn = WorldModelLoss(forecast_decay=train_cfg.forecast_decay)

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        privileged: torch.Tensor,
        autoregressive: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss on a batch of trajectories.

        Args:
            obs:        (B, M+N, obs_dim)
            actions:    (B, M+N, action_dim)
            privileged: (B, M+N, priv_dim)
            autoregressive: if False, use teacher-forcing (N=1 effectively)

        Returns:
            loss, metrics
        """
        M = self.history_horizon
        N = self.forecast_horizon

        obs_history = obs[:, :M]
        action_history = actions[:, :M]
        action_forecast = actions[:, M : M + N]
        obs_targets = obs[:, M : M + N]
        priv_targets = privileged[:, M : M + N]

        if autoregressive:
            obs_means, obs_stds, priv_means, priv_stds = self.core.autoregressive_rollout(
                obs_history, action_history, action_forecast
            )
        else:
            # Teacher-forcing: predict each step from ground-truth history
            obs_means, obs_stds, priv_means, priv_stds = [], [], [], []
            hidden = self.core.forward_history(obs_history, action_history)
            for k in range(N):
                a_k = action_forecast[:, k]
                obs_mean, obs_std, priv_mean, priv_std, hidden = self.core.step(
                    obs_targets[:, k - 1] if k > 0 else obs_history[:, -1],
                    a_k,
                    hidden,
                )
                obs_means.append(obs_mean)
                obs_stds.append(obs_std)
                priv_means.append(priv_mean)
                priv_stds.append(priv_std)

        return self.loss_fn(
            obs_means, obs_stds, obs_targets,
            priv_means, priv_stds, priv_targets,
        )

    def predict(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        action_forecast: torch.Tensor,
        use_mean: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive rollout for inference.

        Returns:
            obs_preds:  (B, N, obs_dim)
            priv_preds: (B, N, priv_dim)
        """
        with torch.no_grad():
            obs_means, _, priv_means, _ = self.core.autoregressive_rollout(
                obs_history, action_history, action_forecast, use_mean=use_mean
            )
        return torch.stack(obs_means, dim=1), torch.stack(priv_means, dim=1)

    def imagine(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        policy: nn.Module,
        horizon: int,
        use_mean: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Imagination rollout driven by a policy (Eq. 3).

        Args:
            obs_history:    (B, M, obs_dim)
            action_history: (B, M, action_dim)
            policy:         PolicyNetwork
            horizon:        T imagination steps
            use_mean:       deterministic world model

        Returns:
            obs_traj:  (B, T, obs_dim)
            act_traj:  (B, T, action_dim)
        """
        hidden = self.core.forward_history(obs_history, action_history)
        current_obs = obs_history[:, -1]

        obs_traj, act_traj = [], []
        for _ in range(horizon):
            action, _ = policy.act(current_obs)
            obs_mean, obs_std, _, _, hidden = self.core.step(current_obs, action, hidden)
            if use_mean:
                next_obs = obs_mean
            else:
                eps = torch.randn_like(obs_mean)
                next_obs = obs_mean + obs_std * eps
            obs_traj.append(next_obs)
            act_traj.append(action)
            current_obs = next_obs

        return torch.stack(obs_traj, dim=1), torch.stack(act_traj, dim=1)


# ---------------------------------------------------------------------------
# Baseline Wrappers
# ---------------------------------------------------------------------------

class MLPBaseline(nn.Module):
    """MLP world model baseline with autoregressive training."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        privileged_dim: int,
        history_horizon: int,
        forecast_horizon: int,
        cfg: MLPBaselineConfig,
        forecast_decay: float = 1.0,
    ):
        super().__init__()
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        self.core = MLPWorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            privileged_dim=privileged_dim,
            history_horizon=history_horizon,
            hidden_sizes=cfg.hidden_sizes,
            activation=cfg.activation,
        )
        self.loss_fn = WorldModelLoss(forecast_decay=forecast_decay)

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        privileged: torch.Tensor,
        autoregressive: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        M = self.history_horizon
        N = self.forecast_horizon
        obs_history = obs[:, :M]
        action_history = actions[:, :M]
        action_forecast = actions[:, M : M + N]
        obs_targets = obs[:, M : M + N]
        priv_targets = privileged[:, M : M + N]

        if autoregressive:
            obs_means, obs_stds, priv_means, priv_stds = self.core.autoregressive_rollout(
                obs_history, action_history, action_forecast
            )
        else:
            obs_mean, obs_std, priv_mean, priv_std = self.core.forward(
                obs_history, action_history
            )
            obs_means = [obs_mean]
            obs_stds = [obs_std]
            priv_means = [priv_mean]
            priv_stds = [priv_std]
            obs_targets = obs_targets[:, :1]
            priv_targets = priv_targets[:, :1]

        return self.loss_fn(
            obs_means, obs_stds, obs_targets,
            priv_means, priv_stds, priv_targets,
        )

    def predict(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        action_forecast: torch.Tensor,
        use_mean: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            obs_means, _, priv_means, _ = self.core.autoregressive_rollout(
                obs_history, action_history, action_forecast, use_mean=use_mean
            )
        return torch.stack(obs_means, dim=1), torch.stack(priv_means, dim=1)


class RSSMBaseline(nn.Module):
    """RSSM world model baseline (teacher-forcing, as traditionally implemented)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        privileged_dim: int,
        history_horizon: int,
        forecast_horizon: int,
        cfg: RSSMConfig,
    ):
        super().__init__()
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        self.core = RSSMWorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            privileged_dim=privileged_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            latent_dim=cfg.latent_dim,
            num_categories=cfg.num_categories,
            category_size=cfg.category_size,
        )
        self.loss_fn = RSSMLoss()

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        privileged: torch.Tensor,
        autoregressive: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        M = self.history_horizon
        N = self.forecast_horizon
        seq = obs[:, :M + N]
        act = actions[:, :M + N]

        result = self.core.forward_sequence(seq, act, use_posterior=True)
        # Reconstruction targets are the same time-step observations (not shifted)
        T = result["obs_means"].size(1)
        obs_targets = seq[:, :T]

        return self.loss_fn(
            result["obs_means"],
            result["obs_stds"],
            obs_targets,
            result["prior_logits"],
            result["post_logits"],
            self.core.num_categories,
        )

    def predict(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        action_forecast: torch.Tensor,
        use_mean: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            obs_means, _, priv_means, _ = self.core.autoregressive_rollout(
                obs_history, action_history, action_forecast, use_mean=use_mean
            )
        return torch.stack(obs_means, dim=1), torch.stack(priv_means, dim=1)


class TransformerBaseline(nn.Module):
    """Transformer world model baseline (teacher-forcing)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        privileged_dim: int,
        history_horizon: int,
        forecast_horizon: int,
        cfg: TransformerConfig,
        forecast_decay: float = 1.0,
    ):
        super().__init__()
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        self.core = TransformerWorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            privileged_dim=privileged_dim,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            context_length=cfg.context_length,
            dropout=cfg.dropout,
        )
        self.loss_fn = WorldModelLoss(forecast_decay=forecast_decay)

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        privileged: torch.Tensor,
        autoregressive: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        M = self.history_horizon
        N = self.forecast_horizon
        obs_history = obs[:, :M]
        action_history = actions[:, :M]
        action_forecast = actions[:, M : M + N]
        obs_targets = obs[:, M : M + N]
        priv_targets = privileged[:, M : M + N]

        # Teacher-forcing: predict each step from ground-truth context
        obs_means, obs_stds, priv_means, priv_stds = [], [], [], []
        for k in range(N):
            if k == 0:
                o_ctx = obs_history
                a_ctx = action_history
            else:
                o_ctx = torch.cat([obs_history[:, k:], obs_targets[:, :k]], dim=1)
                a_ctx = torch.cat([action_history[:, k:], action_forecast[:, :k]], dim=1)
            obs_mean, obs_std, priv_mean, priv_std = self.core.forward(o_ctx, a_ctx)
            obs_means.append(obs_mean)
            obs_stds.append(obs_std)
            priv_means.append(priv_mean)
            priv_stds.append(priv_std)

        return self.loss_fn(
            obs_means, obs_stds, obs_targets,
            priv_means, priv_stds, priv_targets,
        )

    def predict(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        action_forecast: torch.Tensor,
        use_mean: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            obs_means, _, priv_means, _ = self.core.autoregressive_rollout(
                obs_history, action_history, action_forecast, use_mean=use_mean
            )
        return torch.stack(obs_means, dim=1), torch.stack(priv_means, dim=1)


# ---------------------------------------------------------------------------
# Actor-Critic for MBPO-PPO
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Combined policy and value network for MBPO-PPO."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        policy_cfg: PolicyArchConfig,
        value_cfg: ValueArchConfig,
    ):
        super().__init__()
        self.policy = PolicyNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=policy_cfg.hidden_sizes,
            activation=policy_cfg.activation,
            log_std_init=policy_cfg.log_std_init,
            log_std_min=policy_cfg.log_std_min,
            log_std_max=policy_cfg.log_std_max,
        )
        self.value = ValueNetwork(
            obs_dim=obs_dim,
            hidden_sizes=value_cfg.hidden_sizes,
            activation=value_cfg.activation,
        )

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action, log_prob = self.policy.act(obs, deterministic)
        value = self.value(obs)
        return action, log_prob, value

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_prob, entropy = self.policy.evaluate_actions(obs, actions)
        value = self.value(obs)
        return log_prob, entropy, value


def build_rwm(cfg: ExperimentConfig) -> RWM:
    robot = cfg.get_robot_config()
    return RWM(
        obs_dim=robot.obs_dim,
        action_dim=robot.action_dim,
        privileged_dim=robot.privileged_dim,
        arch_cfg=cfg.rwm_arch,
        train_cfg=cfg.rwm_training,
    )


def build_mlp_baseline(cfg: ExperimentConfig) -> MLPBaseline:
    robot = cfg.get_robot_config()
    return MLPBaseline(
        obs_dim=robot.obs_dim,
        action_dim=robot.action_dim,
        privileged_dim=robot.privileged_dim,
        history_horizon=cfg.rwm_training.history_horizon,
        forecast_horizon=cfg.rwm_training.forecast_horizon,
        cfg=cfg.mlp_baseline,
        forecast_decay=cfg.rwm_training.forecast_decay,
    )


def build_rssm_baseline(cfg: ExperimentConfig) -> RSSMBaseline:
    robot = cfg.get_robot_config()
    return RSSMBaseline(
        obs_dim=robot.obs_dim,
        action_dim=robot.action_dim,
        privileged_dim=robot.privileged_dim,
        history_horizon=cfg.rwm_training.history_horizon,
        forecast_horizon=cfg.rwm_training.forecast_horizon,
        cfg=cfg.rssm,
    )


def build_transformer_baseline(cfg: ExperimentConfig) -> TransformerBaseline:
    robot = cfg.get_robot_config()
    return TransformerBaseline(
        obs_dim=robot.obs_dim,
        action_dim=robot.action_dim,
        privileged_dim=robot.privileged_dim,
        history_horizon=cfg.rwm_training.history_horizon,
        forecast_horizon=cfg.rwm_training.forecast_horizon,
        cfg=cfg.transformer,
        forecast_decay=cfg.rwm_training.forecast_decay,
    )


def build_actor_critic(cfg: ExperimentConfig) -> ActorCritic:
    robot = cfg.get_robot_config()
    return ActorCritic(
        obs_dim=robot.policy_obs_dim,
        action_dim=robot.action_dim,
        policy_cfg=cfg.policy_arch,
        value_cfg=cfg.value_arch,
    )
