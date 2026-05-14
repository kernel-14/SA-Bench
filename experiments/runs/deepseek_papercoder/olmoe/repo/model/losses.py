# model/losses.py
"""
Auxiliary loss functions for the OLMoE Mixture-of-Experts language model.

These functions implement the paper's Equations 3 and 4:
    - Load balancing loss (Shazeer et al., 2017) – encourages uniform expert usage.
    - Router z‑loss (Zoph et al., 2022) – penalises large router logits for stability.

Both operate on the raw router logits (before softmax) and are designed to be
called by the trainer **per MoE layer**.  The final weighted loss is assembled
in the trainer after applying the coefficients from ``config.yaml``:
    α = model.moe.load_balancing_weight  (default 0.01)
    β = model.moe.router_z_loss_weight   (default 0.001)
"""

from __future__ import annotations

import torch
from typing import Optional


def load_balancing_loss(
    router_logits: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Compute the un‑weighted load balancing loss for a single MoE layer.

    Implements Equation 3 of the OLMoE paper:
        L_LB = N_E · Σ_i (f_i · P_i)
    where
        f_i = fraction of tokens routed to expert i
        P_i = mean gating probability for expert i (over all tokens)

    Args:
        router_logits:  Raw router outputs, shape ``(T, N_E)``, where T is the
            total number of tokens in the current (micro‑)batch.
        topk_indices:   Index tensor of shape ``(T, k)`` containing the indices
            of the selected experts for each token. Values must be in
            ``[0, N_E-1]``.
        num_experts:    Total number of experts, N_E.

    Returns:
        Scalar tensor containing the load balancing loss.
    """
    assert router_logits.dim() == 2, (
        "router_logits must be 2‑dimensional (T, N_E)"
    )
    T, N_E_logits = router_logits.shape
    if N_E_logits != num_experts:
        raise ValueError(
            f"router_logits has {N_E_logits} experts but "
            f"num_experts={num_experts} was provided."
        )
    T2, k = topk_indices.shape
    if T2 != T:
        raise ValueError(
            f"router_logits and topk_indices must have the same first "
            f"dimension (got {T} vs {T2})"
        )

    device = router_logits.device
    dtype = router_logits.dtype

    # Compute gating probabilities in full precision for numerical stability.
    probs = torch.softmax(router_logits.float(), dim=-1)          # (T, N_E)

    # P_i = average probability per expert across tokens.
    P_i = probs.mean(dim=0)                                      # (N_E,)

    # f_i = fraction of tokens that include expert i in their top‑k.
    # Create a per‑token one‑hot mask and average over tokens.
    token_has_expert = torch.zeros(T, num_experts, device=device, dtype=torch.float32)
    token_indices = (
        torch.arange(T, device=device, dtype=torch.long)
        .unsqueeze(1)                                  # (T, 1)
        .expand(-1, k)                                 # (T, k)
        .reshape(-1)                                   # (T*k,)
    )
    expert_indices = topk_indices.reshape(-1)           # (T*k,)
    token_has_expert[token_indices, expert_indices] = 1.0
    f_i = token_has_expert.mean(dim=0)                  # (N_E,)

    # Load balancing loss (un‑weighted)
    loss = num_experts * torch.dot(f_i, P_i)

    return loss.to(dtype)   # return in original precision for consistency


def router_z_loss(
    router_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the un‑weighted router z‑loss for a single MoE layer.

    Implements Equation 4 of the OLMoE paper:
        L_RZ = (1/T) · Σ_t [ log( Σ_j exp(logits_{t,j}) ) ]²

    Args:
        router_logits:  Raw router outputs, shape ``(T, N_E)``.

    Returns:
        Scalar tensor containing the router z‑loss.
    """
    assert router_logits.dim() == 2, (
        "router_logits must be 2‑dimensional (T, N_E)"
    )

    dtype = router_logits.dtype

    # Compute log‑sum‑exp per token in full precision to avoid overflow.
    lse = torch.logsumexp(router_logits.float(), dim=-1)   # (T,)
    loss = (lse ** 2).mean()

    return loss.to(dtype)   # return in original precision for consistency


# ---------------------------------------------------------------------------
# Quick sanity tests (executed when this file is run directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy inputs
    T = 16
    N = 64
    k = 8

    logits = torch.randn(T, N, requires_grad=True)
    probs = torch.softmax(logits, dim=-1)
    topk_indices = torch.topk(probs, k, dim=-1).indices

    lbl = load_balancing_loss(logits, topk_indices, N)
    rzl = router_z_loss(logits)

    print("Load balancing loss:", lbl.item())
    print("Router z‑loss:", rzl.item())

    # Check gradients
    (lbl + rzl).backward()
    print("Gradient exists:", logits.grad is not None)
