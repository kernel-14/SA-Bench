"""
Macro-Action PPO (MA-PPO) for RLHF.

Implements the core MA-PPO algorithm as described in Section 3.2.2 and
Appendix E of the paper. The key idea is to use macro actions (sequences
of tokens) instead of individual tokens for policy optimization, while
maintaining the same action space (token vocabulary).

Key components:
- Macro action joint probability: π_θ(ω_τ | s_τ) = ∏ π_θ(a_t | a_<t)
- Macro action value function: V^π(s_τ, ω_τ)
- MA-PPO clipped objective with macro-action-level importance sampling
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Literal
import numpy as np


def policy_loss_macro_action(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    sequence: List[int],
    cliprange: float = 0.2,
) -> torch.Tensor:
    """
    Compute MA-PPO policy loss at the macro-action level.
    
    The objective is:
    L^{MA-PPO}(θ) = E_τ[min(π_θ(ω_τ|s_τ)/π_θold(ω_τ|s_τ) * Â_τ,
                              clip(π_θ(ω_τ|s_τ)/π_θold(ω_τ|s_τ), 1-ε, 1+ε) * Â_τ)]
    
    The joint probability ratio for a macro action is computed as the product
    of token-level probability ratios within that macro action.
    
    Args:
        logprobs: Current policy log probabilities, shape (batch, seq_len).
        old_logprobs: Old policy log probabilities, shape (batch, seq_len).
        advantages: Macro-action-level advantages, shape (batch, num_macro_actions).
        mask: Attention mask for response tokens, shape (batch, seq_len).
        sequence: List of macro action boundary positions.
        cliprange: PPO clipping parameter ε (default 0.2).
    
    Returns:
        Scalar policy loss.
    
    Reference: Equation (3) and Appendix E.
    """
    # Log ratio of probabilities: log(π_θ / π_θold) for each token
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    
    # Split ratios by macro action boundaries
    split_list = torch.diff(torch.tensor(sequence)).tolist()
    split_ratio = torch.split(ratio, split_list, dim=-1)
    split_mask = torch.split(mask, split_list, dim=-1)
    
    pg_loss = 0.0
    total_mask_sum = 0.0
    
    for i in range(len(split_list)):
        ratio_i = split_ratio[i]        # (batch, macro_len)
        mask_i = split_mask[i]           # (batch, macro_len)
        advantages_i = advantages[:, i]  # (batch,)
        
        # The macro action ratio is the product of token ratios
        # log(macro_ratio) = sum(log(token_ratio))
        # macro_ratio = exp(sum(log(token_ratio)))
        macro_log_ratio = (log_ratio[:, :split_list[i]]) if i == 0 else None
        # We use the sum of log ratios for numerical stability
        # But in practice, we apply the advantage to each token within the macro action
        
        # PPO clipped objective applied to each token's ratio with the macro advantage
        pg_loss1 = -advantages_i.unsqueeze(-1) * ratio_i
        pg_loss2 = -advantages_i.unsqueeze(-1) * torch.clamp(
            ratio_i, 1.0 - cliprange, 1.0 + cliprange
        )
        
        pg_loss += torch.sum(torch.max(pg_loss1, pg_loss2) * mask_i)
        total_mask_sum += mask_i.sum()
    
    pg_loss = pg_loss / (total_mask_sum + 1e-8)
    
    return pg_loss


def policy_loss_macro_action_joint(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    sequence: List[int],
    cliprange: float = 0.2,
) -> torch.Tensor:
    """
    Alternative MA-PPO policy loss using true macro-action joint probability ratio.
    
    This computes the ratio at the macro action level:
    π_θ(ω_τ|s_τ) / π_θold(ω_τ|s_τ) = ∏ π_θ(a_t|a_<t) / π_θold(a_t|a_<t)
    
    Args:
        logprobs: Current policy log probabilities, shape (batch, seq_len).
        old_logprobs: Old policy log probabilities, shape (batch, seq_len).
        advantages: Macro-action-level advantages, shape (batch, num_macro_actions).
        mask: Attention mask for response tokens, shape (batch, seq_len).
        sequence: List of macro action boundary positions.
        cliprange: PPO clipping parameter ε (default 0.2).
    
    Returns:
        Scalar policy loss.
    
    Reference: Equation (3).
    """
    log_ratio = (logprobs - old_logprobs) * mask
    
    split_list = torch.diff(torch.tensor(sequence)).tolist()
    split_log_ratio = torch.split(log_ratio, split_list, dim=-1)
    split_mask = torch.split(mask, split_list, dim=-1)
    
    pg_loss = 0.0
    total_count = 0
    
    for i in range(len(split_list)):
        log_ratio_i = split_log_ratio[i]
        mask_i = split_mask[i]
        advantages_i = advantages[:, i]  # (batch,)
        
        # Sum log ratios over tokens in macro action to get macro log ratio
        macro_log_ratio = (log_ratio_i * mask_i).sum(dim=-1)  # (batch,)
        # Where mask is all zero, set log ratio to 0 (ratio = 1)
        macro_log_ratio = torch.where(
            mask_i.sum(dim=-1) > 0, macro_log_ratio, torch.zeros_like(macro_log_ratio)
        )
        macro_ratio = torch.exp(torch.clamp(macro_log_ratio, -20, 20))
        
        # PPO clipped objective
        pg_loss1 = -advantages_i * macro_ratio
        pg_loss2 = -advantages_i * torch.clamp(
            macro_ratio, 1.0 - cliprange, 1.0 + cliprange
        )
        
        pg_loss += torch.sum(torch.max(pg_loss1, pg_loss2))
        total_count += advantages_i.size(0)
    
    pg_loss = pg_loss / (total_count + 1e-8)
    
    return pg_loss


def critic_loss_macro_action(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    sequence: List[int],
    clip_value_loss: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute MA-PPO critic (value) loss at the macro-action level.
    
    Args:
        values: Current value predictions, shape (batch, seq_len).
        old_values: Old value predictions, shape (batch, seq_len).
        returns: Macro-action-level returns, shape (batch, num_macro_actions).
        mask: Attention mask, shape (batch, seq_len).
        sequence: List of macro action boundary positions.
        clip_value_loss: If set, clip the value loss (value function clipping).
    
    Returns:
        Scalar critic loss (MSE).
    
    Reference: Appendix E.
    """
    split_list = torch.diff(torch.tensor(sequence)).tolist()
    split_values = torch.split(values, split_list, dim=-1)
    split_old_values = torch.split(old_values, split_list, dim=-1)
    split_mask = torch.split(mask, split_list, dim=-1)
    
    value_loss = 0.0
    total_mask_sum = 0.0
    
    for i in range(len(split_list)):
        value_i = split_values[i]          # (batch, macro_len)
        old_value_i = split_old_values[i]  # (batch, macro_len)
        mask_i = split_mask[i]             # (batch, macro_len)
        returns_i = returns[:, i]          # (batch,)
        
        if clip_value_loss is not None:
            # Value clipping as in standard PPO
            values_clipped = old_value_i + torch.clamp(
                value_i - old_value_i, -clip_value_loss, clip_value_loss
            )
            v_loss1 = (value_i - returns_i.unsqueeze(-1)) ** 2
            v_loss2 = (values_clipped - returns_i.unsqueeze(-1)) ** 2
            v_loss = torch.max(v_loss1, v_loss2)
        else:
            v_loss = (value_i - returns_i.unsqueeze(-1)) ** 2
        
        value_loss += torch.sum(v_loss * mask_i)
        total_mask_sum += mask_i.sum()
    
    value_loss = value_loss / (total_mask_sum + 1e-8)
    
    return value_loss


def compute_macro_action_returns_and_advantages(
    macro_values: torch.Tensor,
    macro_rewards: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute returns and advantages using Generalized Advantage Estimation (GAE)
    at the macro action level.
    
    Args:
        macro_values: Macro action values of shape (batch, num_macro_actions).
        macro_rewards: Macro action rewards of shape (batch, num_macro_actions).
        gamma: Discount factor (γ in GAE, default 1.0 as per paper).
        lam: GAE lambda parameter (λ in GAE, default 0.95 as per paper).
    
    Returns:
        advantages: Macro-level advantages of shape (batch, num_macro_actions).
        returns: Macro-level returns of shape (batch, num_macro_actions).
    
    Reference: Section 3.2.2 and Appendix E.
    """
    batch_size, num_steps = macro_values.shape
    advantages = torch.zeros_like(macro_values)
    returns = torch.zeros_like(macro_values)
    
    gae = 0.0
    for step in reversed(range(num_steps)):
        if step == num_steps - 1:
            next_value = 0.0
        else:
            next_value = macro_values[:, step + 1]
        
        delta = macro_rewards[:, step] + gamma * next_value - macro_values[:, step]
        gae = delta + gamma * lam * gae
        
        advantages[:, step] = gae
        returns[:, step] = gae + macro_values[:, step]
    
    return advantages, returns


def compute_macro_rewards(
    token_rewards: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    sequence: List[int],
    rho: float = 1.0,
) -> torch.Tensor:
    """
    Compute macro action rewards from token-level rewards.
    
    R_τ = E[Σ_{i=0}^{|ω_τ|-1} ρ^i * r_{t_τ+i} | s_τ]
    
    With ρ = 1 (as used in the paper), this is simply the sum of token rewards
    within each macro action.
    
    Args:
        token_rewards: Token-level rewards, shape (batch, seq_len).
        mask: Attention mask, shape (batch, seq_len).
        start: Starting index (prompt length - 1).
        sequence: List of macro action boundary positions.
        rho: Discount factor within macro action (default 1.0 as per paper).
    
    Returns:
        Macro action rewards of shape (batch, num_macro_actions).
    """
    split_list = torch.diff(torch.tensor(sequence)).tolist()
    splited_rewards = torch.split(token_rewards[:, start:], split_list, dim=-1)
    splited_mask = torch.split(mask[:, start:], split_list, dim=-1)
    
    batch_size = token_rewards.size(0)
    macro_rewards = torch.zeros(batch_size, len(split_list), 
                                 dtype=token_rewards.dtype, device=token_rewards.device)
    
    for idx, (reward_i, mask_i) in enumerate(zip(splited_rewards, splited_mask)):
        # With ρ=1, macro reward is sum of token rewards
        macro_rewards[:, idx] = (reward_i * mask_i).sum(dim=-1)
    
    return macro_rewards


def broadcast_advantages_to_tokens(
    macro_advantages: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    sequence: List[int],
) -> torch.Tensor:
    """
    Broadcast macro-action advantages back to individual tokens.
    
    Each token within a macro action gets the same advantage (the macro advantage).
    This is used when applying the advantage to the token-level policy gradient.
    
    Args:
        macro_advantages: Macro-level advantages, shape (batch, num_macro_actions).
        mask: Attention mask for token positions, shape (batch, seq_len).
        start: Starting index.
        sequence: List of macro action boundary positions.
    
    Returns:
        Token-level advantages, shape (batch, seq_len).
    """
    batch_size = mask.size(0)
    seq_len = mask.size(1)
    token_advantages = torch.zeros(batch_size, seq_len, 
                                    dtype=macro_advantages.dtype, 
                                    device=macro_advantages.device)
    
    split_list = torch.diff(torch.tensor(sequence)).tolist()
    split_mask = torch.split(mask[:, start:], split_list, dim=-1)
    
    for idx, (mask_i, split_len) in enumerate(zip(split_mask, split_list)):
        if idx < macro_advantages.size(1):
            adv = macro_advantages[:, idx:idx+1]  # (batch, 1)
            # Broadcast to all tokens in this macro action
            token_advantages[:, start:start+split_len] += adv * mask_i
    
    return token_advantages
