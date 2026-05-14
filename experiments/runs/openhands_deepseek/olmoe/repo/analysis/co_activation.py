import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_expert_co_activation(
    model,
    dataloader,
    k: int = 8,
) -> Dict[int, np.ndarray]:
    """
    Compute expert co-activation matrix (Equation 6, Section 5.2).

    Expert co-activation(E_i, E_j) = N_{E_i,E_j} / N_{E_i}

    Where:
        N_{E_i,E_j}: number of times experts E_i and E_j are activated together
        N_{E_i}: total number of times expert E_i is activated

    A co-activation of 100% means if E_i is activated, E_j is always activated.
    0% means they never co-occur.

    Args:
        model: OLMoE model
        dataloader: data loader
        k: number of top-k experts

    Returns:
        dict mapping layer_idx -> (num_experts, num_experts) co-activation matrix
    """
    model.eval()
    device = next(model.parameters()).device
    num_experts = model.num_experts

    # Accumulators per layer
    co_activation_counts: Dict[int, np.ndarray] = {}
    expert_counts: Dict[int, np.ndarray] = {}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            B, T = input_ids.shape

            _, _, router_probs_list = model(input_ids)

            for layer_idx, router_probs in enumerate(router_probs_list):
                # router_probs: (B, T, num_experts)
                _, topk_indices = torch.topk(router_probs, k, dim=-1)  # (B, T, k)
                topk_indices_flat = topk_indices.view(-1, k).cpu().numpy()  # (B*T, k)

                if layer_idx not in co_activation_counts:
                    co_activation_counts[layer_idx] = np.zeros((num_experts, num_experts), dtype=np.float64)
                    expert_counts[layer_idx] = np.zeros(num_experts, dtype=np.float64)

                for token_experts in topk_indices_flat:
                    for i, e_i in enumerate(token_experts):
                        expert_counts[layer_idx][e_i] += 1
                        for j, e_j in enumerate(token_experts):
                            if i != j:
                                co_activation_counts[layer_idx][e_i, e_j] += 1

    # Normalize to get co-activation percentages
    co_activation_matrices = {}
    for layer_idx in co_activation_counts:
        counts = co_activation_counts[layer_idx]
        expert_total = expert_counts[layer_idx]
        # Normalize: co_activation = N_{E_i,E_j} / N_{E_i}
        co_activation = np.zeros_like(counts)
        for i in range(num_experts):
            if expert_total[i] > 0:
                co_activation[i] = counts[i] / expert_total[i]
        co_activation_matrices[layer_idx] = co_activation

    return co_activation_matrices


def compute_cross_layer_co_activation(
    model,
    dataloader,
    k: int = 1,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Compute co-activation between experts across different layers.

    This analyzes whether experts in different layers tend to process
    the same tokens (Appendix G, Figure 35).

    Args:
        model: OLMoE model
        dataloader: data loader
        k: top-k experts to consider (default 1 for top-1 routing)

    Returns:
        dict mapping (layer_i, layer_j) -> (num_experts_i, num_experts_j) matrix
    """
    model.eval()
    device = next(model.parameters()).device
    num_experts = model.num_experts
    n_layers = model.n_layers

    # Get all layer expert selections
    all_topk: Dict[int, List] = {}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            B, T = input_ids.shape

            _, _, router_probs_list = model(input_ids)

            for layer_idx, router_probs in enumerate(router_probs_list):
                _, topk_indices = torch.topk(router_probs, k, dim=-1)
                topk_flat = topk_indices.view(-1, k).cpu().numpy()
                if layer_idx not in all_topk:
                    all_topk[layer_idx] = []
                all_topk[layer_idx].append(topk_flat)

    # Concatenate all tokens
    for layer_idx in all_topk:
        all_topk[layer_idx] = np.concatenate(all_topk[layer_idx], axis=0)  # (N, k)

    # Compute cross-layer co-activation
    cross_coactivation = {}
    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            co_act = np.zeros((num_experts, num_experts), dtype=np.float64)
            expert_total_i = np.zeros(num_experts, dtype=np.float64)

            tokens_i = all_topk[i]  # (N, k)
            tokens_j = all_topk[j]  # (N, k)
            n_tokens = min(tokens_i.shape[0], tokens_j.shape[0])

            for t in range(n_tokens):
                for ki in range(k):
                    e_i = tokens_i[t, ki]
                    expert_total_i[e_i] += 1
                    for kj in range(k):
                        e_j = tokens_j[t, kj]
                        co_act[e_i, e_j] += 1

            # Normalize
            for ei in range(num_experts):
                if expert_total_i[ei] > 0:
                    co_act[ei] /= expert_total_i[ei]

            cross_coactivation[(i, j)] = co_act

    return cross_coactivation
