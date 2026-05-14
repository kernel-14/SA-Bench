import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_vtrace_targets(
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    log_rhos: torch.Tensor,
    discount: float = 0.97,
    vtrace_lambda: float = 0.97,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute V-trace targets and policy gradient advantages.
    
    Implements V-trace as in Espeholt et al. (2018) IMPALA.
    
    Args:
        rewards: (T, B) rewards
        values: (T, B) value estimates from learner
        bootstrap_value: (B,) value estimate for state after last step
        log_rhos: (T, B) log importance sampling ratios log(pi/mu)
        discount: discount factor gamma
        vtrace_lambda: lambda for V-trace (lambda in paper)
        clip_rho_threshold: rho_bar for V-trace targets
        clip_pg_rho_threshold: c_bar for policy gradient
    Returns:
        vtrace_returns: (T, B) V-trace return targets
        pg_advantages: (T, B) policy gradient advantages
    """
    T, B = rewards.shape
    device = rewards.device

    rhos = torch.exp(log_rhos)
    clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
    cs = torch.clamp(rhos, max=vtrace_lambda)

    # Append bootstrap value
    values_t_plus_1 = torch.cat([values[1:], bootstrap_value.unsqueeze(0)], dim=0)

    # Temporal differences
    deltas = clipped_rhos * (rewards + discount * values_t_plus_1 - values)

    # Compute V-trace targets backwards
    vtrace_returns = torch.zeros_like(rewards)
    acc = torch.zeros(B, device=device)

    for t in reversed(range(T)):
        acc = deltas[t] + discount * cs[t] * acc
        vtrace_returns[t] = acc + values[t]

    # Policy gradient advantages
    clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
    pg_advantages = clipped_pg_rhos * (
        rewards + discount * values_t_plus_1 - values
    )

    return vtrace_returns, pg_advantages


def impala_loss(
    policy_logits: torch.Tensor,
    values: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    bootstrap_value: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    discount: float = 0.97,
    vtrace_lambda: float = 0.97,
    entropy_coef: float = 1e-2,
    logit_l2_penalty: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    """
    IMPALA loss with V-trace targets.
    
    Args:
        policy_logits: (T, B, A) learner policy logits
        values: (T, B) learner value estimates
        actions: (T, B) actions taken by actor
        rewards: (T, B) rewards received
        bootstrap_value: (B,) value of state after last step
        behavior_log_probs: (T, B) log probs of actions under behavior policy
        discount: gamma
        vtrace_lambda: lambda for V-trace
        entropy_coef: entropy regularization coefficient
        logit_l2_penalty: L2 penalty on action logits
    Returns:
        dict with total_loss and component losses
    """
    T, B, A = policy_logits.shape

    log_probs = F.log_softmax(policy_logits, dim=-1)
    learner_log_probs = log_probs.gather(
        -1, actions.unsqueeze(-1)
    ).squeeze(-1)

    log_rhos = learner_log_probs - behavior_log_probs

    vtrace_returns, pg_advantages = compute_vtrace_targets(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        log_rhos=log_rhos,
        discount=discount,
        vtrace_lambda=vtrace_lambda,
    )

    # Policy gradient loss
    pg_loss = -(pg_advantages.detach() * learner_log_probs).mean()

    # Value function loss
    value_loss = 0.5 * F.mse_loss(values, vtrace_returns.detach())

    # Entropy bonus
    probs = F.softmax(policy_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    entropy_loss = -entropy_coef * entropy

    # L2 penalty on logits
    logit_l2_loss = logit_l2_penalty * (policy_logits ** 2).mean()

    total_loss = pg_loss + value_loss + entropy_loss + logit_l2_loss

    return {
        "total_loss": total_loss,
        "pg_loss": pg_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "logit_l2_loss": logit_l2_loss,
        "entropy": entropy.detach(),
        "vtrace_returns": vtrace_returns.detach(),
    }
