
import torch
import torch.nn.functional as F

def load_balancing_loss(router_logits, routing_weights_all_experts, selected_experts, num_experts):
    """
    Calculates the Load Balancing Loss (L_LB) as described in the paper (Eq. 3).

    Args:
        router_logits (torch.Tensor): Logits from the router before softmax (num_tokens, num_experts).
        routing_weights_all_experts (torch.Tensor): Softmax output of router_logits (num_tokens, num_experts).
        selected_experts (torch.Tensor): Indices of the top-k selected experts (num_tokens, num_experts_per_token).
        num_experts (int): Total number of experts.

    Returns:
        torch.Tensor: The calculated load balancing loss.
    """
    # Convert selected_experts to one-hot encoding for easier calculation of expert load
    # (num_tokens, num_experts_per_token) -> (num_tokens, num_experts)
    expert_indicators = F.one_hot(selected_experts, num_classes=num_experts).sum(dim=1).float()

    # Fraction of tokens routed to each expert (f_i)
    # Sum of expert_indicators for each expert, then divide by total tokens
    # (num_experts)
    f_i = expert_indicators.sum(dim=0) / expert_indicators.shape[0]

    # Total routing probability allocated to each expert (P_i)
    # Sum of routing_weights_all_experts for each expert over all tokens
    # (num_experts)
    P_i = routing_weights_all_experts.sum(dim=0) / routing_weights_all_experts.shape[0]
    
    # L_LB = N_E * sum(f_i * P_i)
    loss = num_experts * torch.sum(f_i * P_i)
    return loss

def router_z_loss(router_logits):
    """
    Calculates the Router Z-loss (L_RZ) as described in the paper (Section 4.1.7).

    Args:
        router_logits (torch.Tensor): Logits from the router before softmax (num_tokens, num_experts).

    Returns:
        torch.Tensor: The calculated router z-loss.
    """
    # Z-loss penalizes large router logits to prevent instability
    # L_RZ = sum(router_logits^2)
    return torch.sum(router_logits * router_logits)

def olmoe_loss(lm_logits, labels, router_logits_list, routing_weights_all_experts_list, selected_experts_list, num_experts, alpha=0.01, beta=0.001):
    """
    Calculates the total OLMoE loss, combining Cross-Entropy, Load Balancing, and Router Z-loss.

    Args:
        lm_logits (torch.Tensor): Output logits from the language model (batch_size, seq_len, vocab_size).
        labels (torch.Tensor): Ground truth labels (batch_size, seq_len).
        router_logits_list (list): List of router_logits from each MoE layer.
        routing_weights_all_experts_list (list): List of routing_weights_all_experts from each MoE layer.
        selected_experts_list (list): List of selected_experts from each MoE layer.
        num_experts (int): Total number of experts in MoE layer.
        alpha (float): Weight for Load Balancing Loss (default: 0.01).
        beta (float): Weight for Router Z-loss (default: 0.001).

    Returns:
        torch.Tensor: The total calculated loss.
    """
    # 1. Cross-Entropy Loss
    # Reshape for cross_entropy: (N, C) and (N)
    ce_loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

    # 2. Load Balancing Loss
    lb_loss_total = 0.0
    for router_l, routing_w_all, selected_e in zip(router_logits_list, routing_weights_all_experts_list, selected_experts_list):
        lb_loss_total += load_balancing_loss(router_l, routing_w_all, selected_e, num_experts)
    
    # 3. Router Z-loss
    rz_loss_total = 0.0
    for router_l in router_logits_list:
        rz_loss_total += router_z_loss(router_l)

    # Combine losses
    total_loss = ce_loss + alpha * lb_loss_total + beta * rz_loss_total

    return total_loss


