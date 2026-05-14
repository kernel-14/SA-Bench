"""MoE routing analysis tools.

Implements the four analysis methods from Section 5 of the paper:
1. Router saturation (§5.1)
2. Expert co-activation (§5.2)
3. Domain specialization (§5.3)
4. Vocabulary specialization (§5.4)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def compute_expert_assignments(
    router_logits: torch.Tensor,
    k: int = 8,
) -> torch.Tensor:
    """Get top-k expert assignments from router logits.

    Args:
        router_logits: (num_tokens, num_experts) tensor of router logits
        k: Number of experts to activate per token

    Returns:
        expert_indices: (num_tokens, k) tensor of expert indices
    """
    probs = F.softmax(router_logits, dim=-1)
    _, indices = torch.topk(probs, k, dim=-1)
    return indices


def router_saturation(
    current_assignments: torch.Tensor,
    final_assignments: torch.Tensor,
    k: int = 8,
) -> float:
    """Compute router saturation (§5.1, Equation 5).

    Router saturation is the proportion of expert activations at checkpoint t
    that match the expert IDs activated at the final checkpoint T.

    RouterSaturation(t) = 1/N * Σ_i |E_i^(t) ∩ E_i^(T)| / k

    Args:
        current_assignments: (N, k) expert indices at checkpoint t
        final_assignments: (N, k) expert indices at final checkpoint T
        k: Number of active experts

    Returns:
        Saturation value between 0 and 1 (random chance = k/num_experts)
    """
    N = current_assignments.shape[0]
    total_overlap = 0

    for i in range(N):
        current_set = set(current_assignments[i].tolist())
        final_set = set(final_assignments[i].tolist())
        overlap = len(current_set.intersection(final_set))
        total_overlap += overlap

    saturation = total_overlap / (N * k)
    return saturation


def router_saturation_by_layer(
    model,
    dataloader,
    checkpoint_assignments: Dict[int, torch.Tensor],
    final_assignments: Dict[int, torch.Tensor],
    k: int = 8,
    device: str = "cuda",
) -> Dict[int, Dict[int, float]]:
    """Compute router saturation for each layer across checkpoints.

    As shown in Figure 20 of the paper.

    Args:
        model: OLMoE model
        dataloader: DataLoader yielding batches of input_ids
        checkpoint_assignments: Dict mapping checkpoint step -> {layer_idx: assignments}
        final_assignments: Dict mapping layer_idx -> assignments at final checkpoint
        k: Number of active experts
        device: Device to run on

    Returns:
        Nested dict: checkpoint_step -> layer_idx -> saturation
    """
    results = {}
    for step, layer_assignments in checkpoint_assignments.items():
        results[step] = {}
        for layer_idx in layer_assignments:
            results[step][layer_idx] = router_saturation(
                layer_assignments[layer_idx],
                final_assignments[layer_idx],
                k=k,
            )
    return results


def expert_coactivation(
    assignments: torch.Tensor,
    num_experts: int,
) -> np.ndarray:
    """Compute expert co-activation matrix (§5.2, Equation 6).

    Expert co-activation(E_i, E_j) = N_{E_i,E_j} / N_{E_i}

    Where N_{E_i,E_j} is the number of times experts i and j are activated
    together, and N_{E_i} is the total number of times expert i is activated.

    A co-activation of 100% means j is always activated when i is.

    Args:
        assignments: (N, k) tensor of expert indices per token
        num_experts: Total number of experts

    Returns:
        coact: (num_experts, num_experts) numpy array of co-activation scores
    """
    N, k = assignments.shape
    assignments_np = assignments.cpu().numpy()

    # Count individual and co-activations
    N_i = np.zeros(num_experts)
    N_ij = np.zeros((num_experts, num_experts))

    for token_idx in range(N):
        experts = assignments_np[token_idx]
        for i in experts:
            N_i[i] += 1
            for j in experts:
                if i != j:
                    N_ij[i, j] += 1

    # Compute co-activation = N_ij / N_i
    coact = np.zeros((num_experts, num_experts))
    for i in range(num_experts):
        if N_i[i] > 0:
            coact[i] = N_ij[i] / N_i[i]

    return coact


def domain_specialization(
    router_logits_by_domain: Dict[str, torch.Tensor],
    num_experts: int,
    k: int = 8,
) -> Dict[str, np.ndarray]:
    """Compute domain specialization (§5.3, Equation 7).

    Domain specialization(E_i, D) = N_{E_i,D}^{(k)} / N_D

    Where N_{E_i,D}^{(k)} is the number of tokens from domain D for which
    E_i is among the top-k selected experts, and N_D is the total number
    of tokens from domain D.

    A value of 100% means all data from that domain is routed to E_i.
    0% means the expert is never used for that domain.

    Args:
        router_logits_by_domain: Dict mapping domain name -> (N, num_experts) logits
        num_experts: Total number of experts
        k: Top-k experts considered

    Returns:
        specialization_by_domain: Dict mapping domain -> (num_experts,) array of
                                  specialization scores
    """
    specialization = {}

    for domain, logits in router_logits_by_domain.items():
        N_D = logits.shape[0]
        if N_D == 0:
            specialization[domain] = np.zeros(num_experts)
            continue

        # Get top-k assignments
        assignments = compute_expert_assignments(logits, k=k)  # (N, k)

        # Count how many tokens route to each expert
        N_E = np.zeros(num_experts)
        for token_idx in range(N_D):
            experts = assignments[token_idx].cpu().numpy()
            for e in experts:
                N_E[e] += 1

        # Domain specialization = N_E / N_D
        specialization[domain] = N_E / N_D

    return specialization


def vocabulary_specialization(
    token_ids: torch.LongTensor,
    assignments: torch.Tensor,
    num_experts: int,
    k: int = 1,
) -> np.ndarray:
    """Compute vocabulary specialization (§5.4, Equation 8).

    VocabSpecialization(E_i, x) = N_{x,E_i}^{(k)} / N_x

    Where N_{x,E_i}^{(k)} is the number of times input data is routed to
    E_i for token x, and N_x is the total number of times input data is
    routed across all experts for x.

    Args:
        token_ids: (N,) tensor of input token IDs
        assignments: (N, k) tensor of expert indices per token
        num_experts: Total number of experts
        k: Top-k experts to consider

    Returns:
        vocab_spec: (num_experts, vocab_size) array of specialization scores
    """
    token_ids_np = token_ids.cpu().numpy()
    assignments_np = assignments.cpu().numpy()
    N = len(token_ids_np)

    # We don't know vocab_size here, so compute it from token_ids
    vocab_size = int(token_ids_np.max()) + 1

    # Count token occurrences and token-expert co-occurrences
    N_x = np.zeros(vocab_size)
    N_xe = np.zeros((vocab_size, num_experts))

    for i in range(N):
        token = token_ids_np[i]
        N_x[token] += 1
        for e in assignments_np[i]:
            N_xe[token, e] += 1

    # Compute specialization
    vocab_spec = np.zeros((num_experts, vocab_size))
    for token in range(vocab_size):
        if N_x[token] > 0:
            for e in range(num_experts):
                vocab_spec[e, token] = N_xe[token, e] / N_x[token]

    return vocab_spec


def vocabulary_specialization_by_layer(
    model,
    dataloader,
    num_experts: int = 64,
    k: int = 1,
    device: str = "cuda",
) -> Dict[int, np.ndarray]:
    """Compute vocabulary specialization for all layers.

    As shown in Figure 23 of the paper.

    Args:
        model: OLMoE model
        dataloader: DataLoader yielding (input_ids, ...) batches
        num_experts: Number of experts per layer
        k: Number of experts to consider
        device: Device to run on

    Returns:
        layer_spec: Dict mapping layer_idx -> (num_experts, vocab_size) array
    """
    num_layers = len(model.layers)
    # Initialize accumulators
    N_x = [{} for _ in range(num_layers)]  # token counts per layer
    N_xe = [{} for _ in range(num_layers)]  # token-expert co-occurrences

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0] if isinstance(batch, (list, tuple)) else batch
            input_ids = input_ids.to(device)

            # Get embeddings
            x = model.token_embeddings(input_ids)
            bsz, seq_len, hidden = x.shape

            # Process layer by layer
            for layer_idx, layer in enumerate(model.layers):
                if layer.moe is None:
                    continue

                # Get normalized input to MoE
                x_norm = layer.ffn_norm(x.reshape(-1, hidden))
                router_logits = layer.moe.router(x_norm)
                assignments = compute_expert_assignments(router_logits, k=k)

                # Accumulate token-expert statistics
                flat_ids = input_ids.reshape(-1).cpu().numpy()
                assignments_np = assignments.cpu().numpy()

                for i in range(len(flat_ids)):
                    token = int(flat_ids[i])
                    N_x[layer_idx][token] = N_x[layer_idx].get(token, 0) + 1
                    for e in assignments_np[i]:
                        key = (token, e)
                        N_xe[layer_idx][key] = N_xe[layer_idx].get(key, 0) + 1

                # Forward through attention for next layer
                if layer_idx < num_layers - 1:
                    # Create causal mask
                    causal_mask = torch.triu(
                        torch.full((seq_len, seq_len), float("-inf"), device=device),
                        diagonal=1,
                    ).unsqueeze(0).unsqueeze(0)
                    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                    x, _ = layer(x, causal_mask, position_ids)

    # Compute specialization from accumulators
    layer_spec = {}
    max_vocab = max(
        max(N_x[l].keys()) if N_x[l] else 0
        for l in range(num_layers)
    ) + 1

    for layer_idx in range(num_layers):
        spec = np.zeros((num_experts, max_vocab))
        for token, count in N_x[layer_idx].items():
            if count > 0:
                for e in range(num_experts):
                    spec[e, token] = N_xe[layer_idx].get((token, e), 0) / count
        layer_spec[layer_idx] = spec

    return layer_spec
