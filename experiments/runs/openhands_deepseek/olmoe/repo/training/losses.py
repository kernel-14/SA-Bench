import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def compute_total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    router_logits_list: List[torch.Tensor],
    router_probs_list: List[torch.Tensor],
    num_experts: int,
    load_balancing_weight: float = 0.01,
    router_z_weight: float = 0.001,
    ignore_index: int = -100,
) -> dict:
    """
    Compute total training loss (Equation 2):
    
    L = L_CE + alpha * L_LB + beta * L_RZ
    
    Args:
        logits: (B, T, vocab_size) output logits
        labels: (B, T) target token indices
        router_logits_list: list of router logits per MoE layer
        router_probs_list: list of router probs per MoE layer
        num_experts: number of experts
        load_balancing_weight: alpha (default 0.01)
        router_z_weight: beta (default 0.001)
        ignore_index: label index to ignore
    Returns:
        dict with keys: loss, ce_loss, lb_loss, rz_loss
    """
    from ..model.moe import compute_load_balancing_loss, compute_router_z_loss

    # Cross-entropy loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )

    # Load balancing loss (Equation 3)
    lb_loss = torch.tensor(0.0, device=logits.device)
    if load_balancing_weight > 0 and len(router_logits_list) > 0:
        for rl in router_logits_list:
            lb_loss = lb_loss + compute_load_balancing_loss(
                rl, num_experts,
                num_activated_experts=8,  # k=8 for OLMoE-1B-7B
            )
        lb_loss = lb_loss / len(router_logits_list)

    # Router z-loss (Equation 4)
    rz_loss = torch.tensor(0.0, device=logits.device)
    if router_z_weight > 0 and len(router_logits_list) > 0:
        for rl in router_logits_list:
            rz_loss = rz_loss + compute_router_z_loss(rl)
        rz_loss = rz_loss / len(router_logits_list)

    # Total loss
    total_loss = ce_loss + load_balancing_weight * lb_loss + router_z_weight * rz_loss

    return {
        "loss": total_loss,
        "ce_loss": ce_loss.detach(),
        "lb_loss": lb_loss.detach(),
        "rz_loss": rz_loss.detach(),
    }


def dpo_loss(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_mask: torch.Tensor,
    beta: float = 0.1,
    ignore_index: int = -100,
) -> dict:
    """
    Direct Preference Optimization loss (Rafailov et al., 2023).
    
    Implementation based on the DPO paper, used with beta=0.1.
    """
    batch_size = policy_logits.size(0)

    # Compute per-token log probabilities
    shift_logps = F.log_softmax(policy_logits[..., :-1, :], dim=-1)
    shift_logps_r = F.log_softmax(reference_logits[..., :-1, :], dim=-1)
    shift_labels = labels[..., 1:]

    per_token_logps = torch.gather(
        shift_logps, dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    per_token_logps_r = torch.gather(
        shift_logps_r, dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Sum log probs over sequence (excluding padding)
    seq_logps = (per_token_logps * (shift_labels != ignore_index)).sum(dim=-1)
    seq_logps_r = (per_token_logps_r * (shift_labels != ignore_index)).sum(dim=-1)

    # DPO objective
    log_ratio = seq_logps - seq_logps_r
    loss = -F.logsigmoid(beta * log_ratio).mean()

    # Accuracy metric
    chosen_rewards = beta * seq_logps
    rejected_rewards = beta * seq_logps_r
    accuracy = (chosen_rewards > rejected_rewards).float().mean()

    return {"loss": loss, "accuracy": accuracy.detach()}


def kto_loss(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.1,
    ignore_index: int = -100,
) -> dict:
    """
    Kahneman-Tversky Optimization loss (Ethayarajh et al., 2024).
    
    Simplified implementation based on the KTO paper.
    """
    shift_logps = F.log_softmax(policy_logits[..., :-1, :], dim=-1)
    shift_logps_r = F.log_softmax(reference_logits[..., :-1, :], dim=-1)
    shift_labels = labels[..., 1:]

    per_token_logps = torch.gather(
        shift_logps, dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    per_token_logps_r = torch.gather(
        shift_logps_r, dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    seq_logps = (per_token_logps * (shift_labels != ignore_index)).sum(dim=-1)
    seq_logps_r = (per_token_logps_r * (shift_labels != ignore_index)).sum(dim=-1)

    log_ratio = seq_logps - seq_logps_r
    loss = -F.logsigmoid(beta * log_ratio).mean()

    return {"loss": loss}
