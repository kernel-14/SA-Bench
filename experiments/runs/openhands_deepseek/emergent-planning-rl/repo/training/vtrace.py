"""
V-trace off-policy actor-critic algorithm.

Based on Espeholt et al. (2018), "IMPALA: Scalable Distributed Deep-RL
with Importance Weighted Actor-Learner Architectures".

Used for training the DRC agent in Sokoban, following the setup in
Guez et al. (2019).
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def compute_vtrace_returns(
    behaviour_logits: torch.Tensor,
    target_logits: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    discounts: torch.Tensor,
    lambda_: float = 0.97,
    rho_clip: float = 1.0,
    c_clip: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute V-trace returns and advantages.

    Args:
        behaviour_logits: (T, B, num_actions) logits of behavior policy
        target_logits: (T, B, num_actions) logits of target policy
        actions: (T, B) actions taken
        rewards: (T, B) rewards received
        values: (T, B) value function estimates
        bootstrap_value: (B,) value of final state (or next state)
        discounts: (T, B) discount factors (gamma or 0 for terminal)
        lambda_: V-trace lambda parameter
        rho_clip: clipping for importance weights (rho)
        c_clip: clipping for trace cutting (c)

    Returns:
        vtrace_returns: (T+1, B) V-trace returns (includes bootstrap)
        advantages: (T, B) advantages for policy gradient
    """
    T, B = rewards.shape
    device = rewards.device

    # Compute importance sampling weights
    behaviour_probs = F.softmax(behaviour_logits, dim=-1)
    target_probs = F.softmax(target_logits, dim=-1)

    # \rho_t = \pi(a_t) / \mu(a_t)
    behaviour_action_probs = behaviour_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    target_action_probs = target_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    rho = target_action_probs / (behaviour_action_probs + 1e-8)

    # Apply clipping
    clipped_rho = torch.clamp(rho, max=rho_clip)
    clipped_c = torch.clamp(rho, max=c_clip)

    # Compute V-trace recursively backwards
    # v_s = V(x_s) + \delta_s V + \gamma c_s (v_{s+1} - V(x_{s+1}))
    # \delta_s V = \rho_s (r_s + \gamma V(x_{s+1}) - V(x_s))
    vs = bootstrap_value  # (B,)
    vtrace_returns_list = []

    for t in reversed(range(T)):
        vs = values[t] + discounts[t] * clipped_c[t] * (vs - values[t]) + clipped_rho[t] * (rewards[t] + discounts[t] * vs - values[t])
        vtrace_returns_list.append(vs)

    vtrace_returns_list.reverse()
    vtrace_returns = torch.stack(vtrace_returns_list, dim=0)  # (T, B)

    # Add bootstrap
    vtrace_returns_with_bootstrap = torch.cat([
        vtrace_returns,
        bootstrap_value.unsqueeze(0),
    ], dim=0)  # (T+1, B)

    # Advantages: \rho_s (r_s + \gamma v_{s+1} - V(x_s))
    advantages_list = []
    vs_next = bootstrap_value
    for t in reversed(range(T)):
        vs_cur = values[t]
        vs = vtrace_returns[t]
        advantage = clipped_rho[t] * (rewards[t] + discounts[t] * vs_next - vs_cur)
        advantages_list.append(advantage)
        vs_next = vs_cur

    advantages_list.reverse()
    advantages = torch.stack(advantages_list, dim=0)  # (T, B)

    return vtrace_returns_with_bootstrap, advantages


def compute_vtrace_loss(
    target_logits: torch.Tensor,
    behaviour_logits: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    discounts: torch.Tensor,
    lambda_: float = 0.97,
    rho_clip: float = 1.0,
    c_clip: float = 1.0,
    baseline_cost: float = 0.5,
    entropy_cost: float = 0.01,
    action_l2_penalty: float = 1e-3,
    head_l2_penalty: float = 1e-5,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the total IMPALA loss with V-trace.

    Returns:
        total_loss: scalar loss
        metrics: dict of component losses for logging
    """
    T, B = rewards.shape

    vtrace_returns, advantages = compute_vtrace_returns(
        behaviour_logits=behaviour_logits,
        target_logits=target_logits,
        actions=actions,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        discounts=discounts,
        lambda_=lambda_,
        rho_clip=rho_clip,
        c_clip=c_clip,
    )

    # V-trace returns for T steps (excluding bootstrap)
    vs = vtrace_returns[:T]  # (T, B)

    # Policy loss: negative log-prob * advantage
    target_probs = F.softmax(target_logits, dim=-1)
    log_probs = torch.log(target_probs + 1e-8)
    action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    policy_loss = -(action_log_probs * advantages.detach()).mean()

    # Value loss: MSE between values and V-trace returns
    value_loss = F.mse_loss(values, vs.detach()) * baseline_cost

    # Entropy bonus
    probs = F.softmax(target_logits, dim=-1)
    log_probs_all = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs_all).sum(dim=-1).mean()
    entropy_loss = -entropy_cost * entropy

    # L2 penalty on action logits
    l2_logits = target_logits.pow(2).mean() * action_l2_penalty

    total_loss = policy_loss + value_loss + entropy_loss + l2_logits

    metrics = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "entropy_loss": entropy_loss.item(),
        "l2_logits": l2_logits.item(),
        "total_loss": total_loss.item(),
        "mean_advantage": advantages.mean().item(),
    }

    return total_loss, metrics
