import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Set


def compute_domain_specialization(
    model,
    dataloader,
    domain_labels: List[str],
    k: int = 8,
) -> Dict[int, Dict[int, Dict[str, float]]]:
    """
    Compute domain specialization (Equation 7, Section 5.3).

    Domain specialization(E_i, D) = N_{E_i,D}^{(k)} / N_D

    Where:
        E_i: the i-th expert
        D: domain from which data originates
        k: number of top-k experts considered
        N_{E_i,D}^{(k)}: number of tokens from domain D where E_i is in top-k
        N_D: total number of tokens from domain D

    Returns:
        dict mapping layer_idx -> expert_idx -> domain_name -> specialization_score
    """
    model.eval()
    device = next(model.parameters()).device
    num_experts = model.num_experts

    # Accumulators
    domain_total: Dict[str, int] = defaultdict(int)
    expert_domain_counts: Dict[int, Dict[int, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            B, T = input_ids.shape
            domain = domain_labels[batch_idx % len(domain_labels)]

            _, _, router_probs_list = model(input_ids)

            for layer_idx, router_probs in enumerate(router_probs_list):
                # router_probs: (B, T, num_experts)
                _, topk_indices = torch.topk(router_probs, k, dim=-1)  # (B, T, k)
                topk_flat = topk_indices.view(-1, k)  # (B*T, k)

                for token_idx in range(topk_flat.size(0)):
                    domain_total[domain] += 1
                    for expert_idx in topk_flat[token_idx].tolist():
                        expert_domain_counts[layer_idx][expert_idx][domain] += 1

    # Normalize
    results = {}
    for layer_idx in expert_domain_counts:
        results[layer_idx] = {}
        for expert_idx in range(num_experts):
            results[layer_idx][expert_idx] = {}
            for domain, total in domain_total.items():
                if total > 0:
                    count = expert_domain_counts[layer_idx][expert_idx].get(domain, 0)
                    results[layer_idx][expert_idx][domain] = count / total
                else:
                    results[layer_idx][expert_idx][domain] = 0.0

    return results


def compute_vocabulary_specialization(
    model,
    dataloader,
    k: int = 1,
    min_token_freq: int = 8,
) -> Dict[int, Dict[int, Dict[int, float]]]:
    """
    Compute vocabulary specialization (Equation 8, Section 5.4).

    Vocabulary specialization(E_i, x) = N_{x,E_i}^{(k)} / N_x

    Where:
        E_i: the i-th expert
        x: token ID (vocabulary element)
        k: number of top-k experts
        N_{x,E_i}: number of times data is routed to E_i for token x
        N_x: total number of times data is routed for token x

    Args:
        model: OLMoE model
        dataloader: data loader
        k: number of top-k experts (1 for input specialization, 8 for all active)
        min_token_freq: minimum token frequency to include

    Returns:
        dict mapping layer_idx -> expert_idx -> token_id -> specialization_score
    """
    model.eval()
    device = next(model.parameters()).device
    num_experts = model.num_experts

    # Accumulators
    token_total: Dict[int, int] = defaultdict(int)
    expert_token_counts: Dict[int, Dict[int, Dict[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)  # (B, T)
            B, T = input_ids.shape

            logits, _, router_probs_list = model(input_ids)

            for layer_idx, router_probs in enumerate(router_probs_list):
                _, topk_indices = torch.topk(router_probs, k, dim=-1)  # (B, T, k)

                for b in range(B):
                    for t in range(T):
                        token_id = input_ids[b, t].item()
                        token_total[token_id] += 1
                        for expert_idx in topk_indices[b, t].tolist():
                            expert_token_counts[layer_idx][expert_idx][token_id] += 1

    # Normalize
    results = {}
    for layer_idx in expert_token_counts:
        results[layer_idx] = {}
        for expert_idx in range(num_experts):
            results[layer_idx][expert_idx] = {}
            for token_id, total in token_total.items():
                if total >= min_token_freq:
                    count = expert_token_counts[layer_idx][expert_idx].get(token_id, 0)
                    results[layer_idx][expert_idx][token_id] = count / total

    return results


def compute_output_vocabulary_specialization(
    model,
    dataloader,
    k: int = 1,
    min_token_freq: int = 8,
) -> Dict[int, Dict[int, Dict[int, float]]]:
    """
    Compute vocabulary specialization based on PREDICTED output token IDs.

    This distinguishes from input specialization: routing is based on the
    token the model is about to predict (output), not the original input token.
    In later layers, output specialization dominates (Section 5.4).

    Args:
        model: OLMoE model
        dataloader: data loader
        k: number of top-k experts
        min_token_freq: minimum token frequency

    Returns:
        dict mapping layer_idx -> expert_idx -> predicted_token_id -> specialization_score
    """
    model.eval()
    device = next(model.parameters()).device
    num_experts = model.num_experts

    predicted_token_total: Dict[int, int] = defaultdict(int)
    expert_pred_token_counts: Dict[int, Dict[int, Dict[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            B, T = input_ids.shape

            logits, _, router_probs_list = model(input_ids)

            # Get predicted token IDs at each position
            pred_tokens = logits.argmax(dim=-1)  # (B, T)

            for layer_idx, router_probs in enumerate(router_probs_list):
                _, topk_indices = torch.topk(router_probs, k, dim=-1)  # (B, T, k)

                for b in range(B):
                    for t in range(T):
                        pred_id = pred_tokens[b, t].item()
                        predicted_token_total[pred_id] += 1
                        for expert_idx in topk_indices[b, t].tolist():
                            expert_pred_token_counts[layer_idx][expert_idx][pred_id] += 1

    results = {}
    for layer_idx in expert_pred_token_counts:
        results[layer_idx] = {}
        for expert_idx in range(num_experts):
            results[layer_idx][expert_idx] = {}
            for pred_id, total in predicted_token_total.items():
                if total >= min_token_freq:
                    count = expert_pred_token_counts[layer_idx][expert_idx].get(pred_id, 0)
                    results[layer_idx][expert_idx][pred_id] = count / total

    return results


def get_top_vocabulary_tokens(
    specialization: Dict[int, Dict[int, Dict[int, float]]],
    tokenizer,
    top_n: int = 10,
    threshold: float = 0.8,
) -> Dict[int, Dict[int, List[tuple]]]:
    """
    Get the top vocabulary tokens for each expert, showing what they specialize on.

    This reproduces analysis like Table 8 in the paper.

    Args:
        specialization: output from compute_vocabulary_specialization
        tokenizer: tokenizer for decoding token IDs
        top_n: number of top tokens to return
        threshold: minimum specialization score to include

    Returns:
        dict mapping layer_idx -> expert_idx -> [(token_str, specialization_score), ...]
    """
    results = {}
    for layer_idx, experts in specialization.items():
        results[layer_idx] = {}
        for expert_idx, tokens_dict in experts.items():
            if not tokens_dict:
                continue
            # Sort by specialization score
            sorted_tokens = sorted(tokens_dict.items(), key=lambda x: x[1], reverse=True)
            top_tokens = sorted_tokens[:top_n]
            decoded = [(tokenizer.decode([token_id]), score) for token_id, score in top_tokens]
            results[layer_idx][expert_idx] = [(t, s) for t, s in decoded if s >= threshold]

    return results
