"""
RLHF training utilities for MA-RLHF.

Implements the standard RLHF components as described in Section 2.2:
- KL divergence penalty computation
- Reward shaping with KL penalty
- Reward model training loss
- Data formatting utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict


def compute_kl_penalty(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token KL divergence penalty between policy and reference model.
    
    D_KL(π_θ(·|x) || π_sft(·|x)) = Σ_a π_θ(a|x) * log(π_θ(a|x) / π_sft(a|x))
    
    Using the approximation:
    KL_t = log(π_θ(a_t|s_t)) - log(π_sft(a_t|s_t))
    
    Args:
        log_probs: Policy model log probabilities, shape (batch, seq_len).
        ref_log_probs: Reference (SFT) model log probabilities, shape (batch, seq_len).
        mask: Attention mask, shape (batch, seq_len).
    
    Returns:
        Per-token KL penalties, shape (batch, seq_len).
    """
    kl = log_probs - ref_log_probs
    return kl * mask


def compute_shaped_reward(
    rm_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.05,
) -> torch.Tensor:
    """
    Compute the shaped reward with KL penalty.
    
    R(x, y) = r_φ(x, y) - β * D_KL(π_θ(·|x) || π_sft(·|x))
    
    The RM score is applied only at the final token, while the KL penalty
    is applied at each token position.
    
    Args:
        rm_scores: Reward model scores, shape (batch,).
        log_probs: Policy log probabilities, shape (batch, response_len).
        ref_log_probs: Reference log probabilities, shape (batch, response_len).
        mask: Response attention mask, shape (batch, response_len).
        beta: KL penalty coefficient (default 0.05 as per Table 5).
    
    Returns:
        Shaped per-token rewards, shape (batch, response_len).
    """
    batch_size, seq_len = log_probs.shape
    
    # KL penalty per token
    kl_penalty = compute_kl_penalty(log_probs, ref_log_probs, mask)
    
    # Initialize rewards with -β * KL at each position
    shaped_rewards = -beta * kl_penalty
    
    # Add RM score at the final non-padded position for each sample
    for b in range(batch_size):
        valid_positions = mask[b].nonzero(as_tuple=True)[0]
        if len(valid_positions) > 0:
            final_pos = valid_positions[-1]
            shaped_rewards[b, final_pos] += rm_scores[b]
    
    return shaped_rewards


def compute_reward_model_loss(
    rm_scores_chosen: torch.Tensor,
    rm_scores_rejected: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the ranking loss for reward model training.
    
    L_RM = -log σ(log(r_φ(x, y_+)) - log(r_φ(x, y_-)))
    
    Args:
        rm_scores_chosen: RM scores for chosen responses, shape (batch,).
        rm_scores_rejected: RM scores for rejected responses, shape (batch,).
    
    Returns:
        Scalar RM loss.
    """
    # The paper uses log(r_φ(x, y)) scores directly
    loss = -torch.log(torch.sigmoid(rm_scores_chosen - rm_scores_rejected))
    return loss.mean()


def compute_reward_model_loss_with_margin(
    rm_scores_chosen: torch.Tensor,
    rm_scores_rejected: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Alternative RM loss with margin.
    
    L_RM = -log σ(log(r_φ(x, y_+)) - log(r_φ(x, y_-)) - margin)
    """
    loss = -torch.log(torch.sigmoid(rm_scores_chosen - rm_scores_rejected - margin))
    return loss.mean()


def compute_program_synthesis_reward(
    compile_success: bool,
    runtime_error: bool,
    n_pass: int,
    n_fail: int,
) -> float:
    """
    Compute the adaptive compiler signal reward for program synthesis.
    
    R(x, y) = 
        -0.3 + 1.3 * N_pass / (N_pass + N_fail)  if successfully compiled
        -0.6                                      if runtime error
        -1.0                                      if compile error
    
    Args:
        compile_success: Whether the code compiled successfully.
        runtime_error: Whether the code had a runtime error.
        n_pass: Number of passed unit tests.
        n_fail: Number of failed unit tests.
    
    Returns:
        Scalar reward value.
    
    Reference: Equation (4) in Appendix B.5.
    """
    if compile_success and not runtime_error:
        if n_pass + n_fail > 0:
            return -0.3 + 1.3 * (n_pass / (n_pass + n_fail))
        else:
            return -0.3
    elif runtime_error:
        return -0.6
    else:
        return -1.0
