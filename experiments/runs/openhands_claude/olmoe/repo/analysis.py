"""MoE routing analysis for OLMoE (§5).

Implements four analysis metrics defined in the paper:

1. Router Saturation (§5.1, Eq. 5):
   Proportion of expert activations at checkpoint t that match the final
   checkpoint T. Measures how early routing decisions stabilize.

2. Expert Co-activation (§5.2, Eq. 6):
   Proportion of times two experts E_i and E_j are simultaneously activated.
   Low co-activation suggests little redundancy across experts.

3. Domain Specialization (§5.3, Eq. 7):
   Proportion of tokens from domain D routed to expert E_i.
   High values indicate experts specialize in specific data domains.

4. Vocabulary Specialization (§5.4, Eq. 8):
   Proportion of tokens with token ID x routed to expert E_i.
   Distinguishes input token ID vs predicted output token ID specialization.

Key findings:
- After 1% of pretraining (~20B tokens), ~60% of top-8 routing has saturated
- Routing in later layers saturates earlier
- Layer 0 saturates significantly more slowly (linked to load balancing)
- Experts show strong domain specialization (arXiv, GitHub, books)
- Vocabulary specialization is higher in later layers
- Later layers specialize more on predicted output tokens vs input tokens
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import OLMoE


# ---------------------------------------------------------------------------
# Router Saturation (§5.1, Equation 5)
# ---------------------------------------------------------------------------

def compute_router_saturation(
    model_t: OLMoE,
    model_T: OLMoE,
    dataloader: DataLoader,
    device: torch.device,
    k_values: List[int] = [1, 8],
    max_tokens: int = 100_000,
) -> Dict[str, Dict[int, List[float]]]:
    """Compute router saturation comparing checkpoint t to final checkpoint T.

    Router Saturation(t) = (1/N) * sum_{i=1}^{N} |E_i^(t) ∩ E_i^(T)| / k

    where E_i^(t) is the set of k experts activated for token i at checkpoint t.

    Args:
        model_t: intermediate checkpoint model
        model_T: final checkpoint model
        dataloader: data to evaluate on (paper uses 0.5% of C4 validation)
        device: compute device
        k_values: list of k values to compute saturation for (paper: k=1 and k=8)
        max_tokens: maximum tokens to process
    Returns:
        dict mapping k -> layer_idx -> saturation score
    """
    model_t.eval()
    model_T.eval()

    # Collect routing decisions for both models
    # routing_t[layer][token_idx] = set of expert indices
    routing_t: Dict[int, List[torch.Tensor]] = defaultdict(list)
    routing_T: Dict[int, List[torch.Tensor]] = defaultdict(list)

    tokens_processed = 0

    with torch.no_grad():
        for batch in dataloader:
            if tokens_processed >= max_tokens:
                break

            input_ids = batch["input_ids"].to(device)
            bsz, seq_len = input_ids.shape

            out_t = model_t(input_ids=input_ids, return_router_info=True)
            out_T = model_T(input_ids=input_ids, return_router_info=True)

            for layer_idx in range(model_t.config.n_layers):
                # topk_indices: (bsz * seq_len, k)
                indices_t = out_t["router_info"][layer_idx]["topk_indices"]
                indices_T = out_T["router_info"][layer_idx]["topk_indices"]
                routing_t[layer_idx].append(indices_t.cpu())
                routing_T[layer_idx].append(indices_T.cpu())

            tokens_processed += bsz * seq_len

    # Compute saturation for each k value
    results: Dict[str, Dict[int, float]] = {}

    for k in k_values:
        layer_saturations: Dict[int, float] = {}

        for layer_idx in range(model_t.config.n_layers):
            all_t = torch.cat(routing_t[layer_idx], dim=0)  # (N, k_full)
            all_T = torch.cat(routing_T[layer_idx], dim=0)

            # Use only top-k experts (already sorted by probability in topk)
            top_t = all_t[:, :k]  # (N, k)
            top_T = all_T[:, :k]

            # Compute intersection size per token
            n_tokens = top_t.shape[0]
            intersection_sizes = []

            for i in range(n_tokens):
                set_t = set(top_t[i].tolist())
                set_T = set(top_T[i].tolist())
                intersection_sizes.append(len(set_t & set_T))

            saturation = np.mean(intersection_sizes) / k
            layer_saturations[layer_idx] = saturation

        results[f"k={k}"] = layer_saturations

    return results


# ---------------------------------------------------------------------------
# Expert Co-activation (§5.2, Equation 6)
# ---------------------------------------------------------------------------

def compute_expert_coactivation(
    model: OLMoE,
    dataloader: DataLoader,
    device: torch.device,
    layer_indices: Optional[List[int]] = None,
    max_tokens: int = 100_000,
) -> Dict[int, np.ndarray]:
    """Compute expert co-activation matrix for specified layers.

    Co-activation(E_i, E_j) = N_{E_i, E_j} / N_{E_i}

    where N_{E_i, E_j} = times E_i and E_j are activated together,
          N_{E_i} = total activations of E_i.

    Args:
        model: OLMoE model
        dataloader: evaluation data
        device: compute device
        layer_indices: which layers to analyze (default: all)
        max_tokens: maximum tokens to process
    Returns:
        dict mapping layer_idx -> (n_experts, n_experts) co-activation matrix
    """
    model.eval()
    n_experts = model.config.n_experts
    n_layers = model.config.n_layers

    if layer_indices is None:
        layer_indices = list(range(n_layers))

    # Count co-activations: coact[layer][i][j] = count(E_i and E_j both active)
    coact_counts = {l: np.zeros((n_experts, n_experts)) for l in layer_indices}
    expert_counts = {l: np.zeros(n_experts) for l in layer_indices}

    tokens_processed = 0

    with torch.no_grad():
        for batch in dataloader:
            if tokens_processed >= max_tokens:
                break

            input_ids = batch["input_ids"].to(device)
            bsz, seq_len = input_ids.shape

            out = model(input_ids=input_ids, return_router_info=True)

            for layer_idx in layer_indices:
                topk_indices = out["router_info"][layer_idx]["topk_indices"].cpu().numpy()
                # topk_indices: (bsz * seq_len, k)

                for token_experts in topk_indices:
                    for ei in token_experts:
                        expert_counts[layer_idx][ei] += 1
                        for ej in token_experts:
                            coact_counts[layer_idx][ei][ej] += 1

            tokens_processed += bsz * seq_len

    # Normalize: co-activation(E_i, E_j) = count(i,j) / count(i)
    coactivation_matrices = {}
    for layer_idx in layer_indices:
        counts = coact_counts[layer_idx]
        totals = expert_counts[layer_idx][:, np.newaxis]
        # Avoid division by zero
        matrix = np.where(totals > 0, counts / totals, 0.0)
        # Zero out diagonal (self-coactivation is trivially 100%)
        np.fill_diagonal(matrix, 0.0)
        coactivation_matrices[layer_idx] = matrix

    return coactivation_matrices


# ---------------------------------------------------------------------------
# Domain Specialization (§5.3, Equation 7)
# ---------------------------------------------------------------------------

def compute_domain_specialization(
    model: OLMoE,
    domain_dataloaders: Dict[str, DataLoader],
    device: torch.device,
    k: int = 8,
    layer_indices: Optional[List[int]] = None,
    max_tokens_per_domain: int = 50_000,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Compute domain specialization for each expert and domain.

    Domain Specialization(E_i, D) = N_{E_i, D}^(k) / N_D

    where N_{E_i, D}^(k) = tokens from D for which E_i is among top-k experts,
          N_D = total tokens from D.

    Args:
        model: OLMoE model
        domain_dataloaders: dict mapping domain name -> DataLoader
        device: compute device
        k: number of top experts to consider
        layer_indices: which layers to analyze
        max_tokens_per_domain: max tokens per domain
    Returns:
        dict: layer_idx -> domain -> (n_experts,) specialization array
    """
    model.eval()
    n_experts = model.config.n_experts
    n_layers = model.config.n_layers

    if layer_indices is None:
        layer_indices = list(range(n_layers))

    # expert_domain_counts[layer][domain][expert] = count
    expert_domain_counts: Dict[int, Dict[str, np.ndarray]] = {
        l: {d: np.zeros(n_experts) for d in domain_dataloaders}
        for l in layer_indices
    }
    domain_total_counts: Dict[str, int] = {d: 0 for d in domain_dataloaders}

    with torch.no_grad():
        for domain, dataloader in domain_dataloaders.items():
            tokens_processed = 0

            for batch in dataloader:
                if tokens_processed >= max_tokens_per_domain:
                    break

                input_ids = batch["input_ids"].to(device)
                bsz, seq_len = input_ids.shape

                out = model(input_ids=input_ids, return_router_info=True)

                for layer_idx in layer_indices:
                    topk_indices = out["router_info"][layer_idx]["topk_indices"].cpu().numpy()
                    # topk_indices: (bsz * seq_len, k_full), use top-k
                    for token_experts in topk_indices[:, :k]:
                        for ei in token_experts:
                            expert_domain_counts[layer_idx][domain][ei] += 1

                n_new = bsz * seq_len
                domain_total_counts[domain] += n_new
                tokens_processed += n_new

    # Normalize
    specialization: Dict[int, Dict[str, np.ndarray]] = {}
    for layer_idx in layer_indices:
        specialization[layer_idx] = {}
        for domain in domain_dataloaders:
            total = domain_total_counts[domain]
            if total > 0:
                specialization[layer_idx][domain] = (
                    expert_domain_counts[layer_idx][domain] / total
                )
            else:
                specialization[layer_idx][domain] = np.zeros(n_experts)

    return specialization


# ---------------------------------------------------------------------------
# Vocabulary Specialization (§5.4, Equation 8)
# ---------------------------------------------------------------------------

def compute_vocabulary_specialization(
    model: OLMoE,
    dataloader: DataLoader,
    device: torch.device,
    k: int = 1,
    layer_indices: Optional[List[int]] = None,
    max_tokens: int = 100_000,
    use_output_tokens: bool = False,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Compute vocabulary specialization per expert and token ID.

    Vocabulary Specialization(E_i, x) = N_{x, E_i}^(k) / N_x

    where N_{x, E_i}^(k) = times token x is routed to E_i (among top-k),
          N_x = total times token x appears.

    Args:
        model: OLMoE model
        dataloader: evaluation data
        device: compute device
        k: number of top experts to consider (paper: k=1 for main figure)
        layer_indices: which layers to analyze
        max_tokens: maximum tokens to process
        use_output_tokens: if True, use next token ID (output) instead of input
    Returns:
        dict: layer_idx -> token_id -> (n_experts,) specialization array
    """
    model.eval()
    n_experts = model.config.n_experts
    n_layers = model.config.n_layers

    if layer_indices is None:
        layer_indices = list(range(n_layers))

    # token_expert_counts[layer][token_id][expert] = count
    token_expert_counts: Dict[int, Dict[int, np.ndarray]] = {
        l: defaultdict(lambda: np.zeros(n_experts))
        for l in layer_indices
    }
    token_total_counts: Dict[int, int] = defaultdict(int)

    tokens_processed = 0

    with torch.no_grad():
        for batch in dataloader:
            if tokens_processed >= max_tokens:
                break

            input_ids = batch["input_ids"].to(device)
            bsz, seq_len = input_ids.shape

            out = model(input_ids=input_ids, return_router_info=True)

            # Determine which token IDs to use
            if use_output_tokens:
                # Next token IDs (shift by 1)
                token_ids_flat = input_ids[:, 1:].contiguous().view(-1).cpu().numpy()
                # Trim router info to match (seq_len - 1 tokens)
                valid_len = seq_len - 1
            else:
                token_ids_flat = input_ids.view(-1).cpu().numpy()
                valid_len = seq_len

            for layer_idx in layer_indices:
                topk_indices = out["router_info"][layer_idx]["topk_indices"].cpu().numpy()
                # topk_indices: (bsz * seq_len, k_full)

                if use_output_tokens:
                    # Reshape and trim last token position
                    topk_reshaped = topk_indices.reshape(bsz, seq_len, -1)
                    topk_trimmed = topk_reshaped[:, :valid_len, :].reshape(-1, topk_indices.shape[-1])
                else:
                    topk_trimmed = topk_indices

                for i, token_id in enumerate(token_ids_flat):
                    if i >= len(topk_trimmed):
                        break
                    token_id = int(token_id)
                    token_total_counts[token_id] += 1
                    for ei in topk_trimmed[i, :k]:
                        token_expert_counts[layer_idx][token_id][ei] += 1

            tokens_processed += bsz * seq_len

    # Normalize
    specialization: Dict[int, Dict[int, np.ndarray]] = {}
    for layer_idx in layer_indices:
        specialization[layer_idx] = {}
        for token_id, counts in token_expert_counts[layer_idx].items():
            total = token_total_counts[token_id]
            if total > 0:
                specialization[layer_idx][token_id] = counts / total

    return specialization


# ---------------------------------------------------------------------------
# Summary utilities
# ---------------------------------------------------------------------------

def get_top_specialized_tokens(
    vocab_spec: Dict[int, np.ndarray],
    expert_id: int,
    tokenizer,
    top_n: int = 10,
    min_count: int = 8,
) -> List[Tuple[str, float]]:
    """Get the top-n tokens most specialized to a given expert.

    Reproduces Table 8 in the paper.
    """
    results = []
    for token_id, expert_probs in vocab_spec.items():
        prob = expert_probs[expert_id]
        if prob > 0:
            try:
                token_str = tokenizer.decode([token_id])
            except Exception:
                token_str = f"<id:{token_id}>"
            results.append((token_str, prob))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def print_saturation_summary(saturation_results: Dict[str, Dict[int, float]]):
    """Print router saturation summary (Figure 20 in paper)."""
    print("\n=== Router Saturation ===")
    for k_str, layer_sats in saturation_results.items():
        avg_sat = np.mean(list(layer_sats.values()))
        print(f"{k_str}: avg={avg_sat:.3f}")
        for layer_idx, sat in sorted(layer_sats.items()):
            print(f"  Layer {layer_idx:2d}: {sat:.3f}")


def print_coactivation_summary(
    coact_matrices: Dict[int, np.ndarray],
    top_n: int = 5,
):
    """Print top co-activated expert pairs per layer (Figure 21 in paper)."""
    print("\n=== Expert Co-activation (top pairs) ===")
    for layer_idx, matrix in sorted(coact_matrices.items()):
        n = matrix.shape[0]
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                coact = (matrix[i, j] + matrix[j, i]) / 2
                if coact > 0:
                    pairs.append((i, j, coact))
        pairs.sort(key=lambda x: x[2], reverse=True)
        print(f"Layer {layer_idx}:")
        for ei, ej, coact in pairs[:top_n]:
            print(f"  E{ei} <-> E{ej}: {coact:.3f}")
