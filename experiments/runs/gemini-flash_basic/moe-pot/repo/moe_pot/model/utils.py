import torch

def calculate_load_balancing_loss(gating_weights_all, num_routed_experts, weight_balance=0.1):
    # gating_weights_all: (B, num_routed_experts, H_p, W_p) - gating weights BEFORE Top-K selection and normalization
    # This corresponds to w^l(z_0^l(x)) in equation 131

    # Reshape to (B * H_p * W_p, num_routed_experts) to treat each spatial location independently
    # for calculating importance per expert across the batch.
    batch_size, _, h_p, w_p = gating_weights_all.shape
    gating_weights_reshaped = gating_weights_all.permute(0, 2, 3, 1).reshape(-1, num_routed_experts) # (B*H_p*W_p, N_r)

    # Importance_i^l = sum_{b=1 to B} w_{i,b}^l(x) (Equation 151)
    # Here, we sum over the batch and spatial locations to get total importance for each expert.
    # Sum across all spatial locations and batch items for each expert.
    importance_per_expert = torch.sum(gating_weights_reshaped, dim=0) # (num_routed_experts,)

    # Calculate coefficient of variation (CV)
    # CV = std_dev / mean
    mean_importance = torch.mean(importance_per_expert)
    std_importance = torch.std(importance_per_expert)

    # Avoid division by zero if all importances are zero
    if mean_importance == 0:
        cv = torch.tensor(0.0, device=gating_weights_all.device)
    else:
        cv = std_importance / mean_importance

    # L_balance^l = w_bal * CV(Importance_i^l)^2 (Equation 157)
    loss_balance = weight_balance * (cv ** 2)

    return loss_balance
