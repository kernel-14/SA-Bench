"""
MoE Analysis for OLMoE

Implements the four analysis metrics from Section 5 of the paper:
1. Router Saturation (Section 5.1)
2. Expert Co-activation (Section 5.2)
3. Domain Specialization (Section 5.3)
4. Vocabulary Specialization (Section 5.4)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def compute_router_saturation(
    model_intermediate,
    model_final,
    dataset,
    k: int = 8,
    k1: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute router saturation between an intermediate checkpoint and the final checkpoint.

    Router Saturation(t) = (1/N) * sum_{i=1}^{N} |E_i^(t) ∩ E_i^(T)| / k

    where:
    - N: total number of tokens
    - k: number of top-k experts activated per token
    - E_i^(t): set of k experts activated for token i at checkpoint t
    - E_i^(T): set of k experts activated for token i at final checkpoint T

    From Section 5.1 of the paper:
    - After 1% of pretraining (~20B tokens), ~60% of top-8 routing has saturated
    - After 40% of pretraining, saturation reaches ~80%
    - Later layers saturate earlier
    - Layer 0 is an outlier, saturating more slowly

    Args:
        model_intermediate: Model at intermediate checkpoint
        model_final: Model at final checkpoint
        dataset: Dataset to evaluate on (paper uses 0.5% of C4 validation)
        k: Number of top-k experts to consider
        k1: Also compute saturation for k=1 (top-1 expert)

    Returns:
        Dictionary with saturation scores per layer
    """
    model_intermediate.eval()
    model_final.eval()

    num_layers = model_intermediate.config.num_hidden_layers
    saturation_k = defaultdict(list)  # layer -> list of per-token saturation
    saturation_k1 = defaultdict(list)  # layer -> list of per-token saturation (k=1)

    with torch.no_grad():
        for batch in dataset:
            input_ids = batch["input_ids"]
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            # Get routing decisions from both models
            routing_intermediate = get_routing_decisions(model_intermediate, input_ids, k)
            routing_final = get_routing_decisions(model_final, input_ids, k)

            # Compute saturation per layer
            for layer_idx in range(num_layers):
                experts_inter = routing_intermediate[layer_idx]  # [B*T, k]
                experts_final = routing_final[layer_idx]  # [B*T, k]

                # For each token, compute intersection size / k
                for token_idx in range(experts_inter.shape[0]):
                    inter_set = set(experts_inter[token_idx].tolist())
                    final_set = set(experts_final[token_idx].tolist())
                    intersection = len(inter_set & final_set)
                    saturation_k[layer_idx].append(intersection / k)

                    if k1:
                        # Top-1 saturation: does the top-1 expert match?
                        top1_inter = experts_inter[token_idx, 0].item()
                        top1_final = experts_final[token_idx, 0].item()
                        saturation_k1[layer_idx].append(float(top1_inter == top1_final))

    # Average saturation per layer
    results = {}
    for layer_idx in range(num_layers):
        results[f"layer_{layer_idx}_k{k}"] = np.mean(saturation_k[layer_idx])
        if k1:
            results[f"layer_{layer_idx}_k1"] = np.mean(saturation_k1[layer_idx])

    return results


def get_routing_decisions(
    model,
    input_ids: torch.Tensor,
    k: int,
) -> Dict[int, torch.Tensor]:
    """
    Extract routing decisions (which experts are selected) for each layer.

    Returns:
        Dict mapping layer_idx -> selected_experts tensor [B*T, k]
    """
    routing_decisions = {}

    # Hook to capture router outputs
    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output is (hidden_states, lb_loss, rz_loss)
            # We need to re-run the router to get decisions
            hidden_states = input[0]
            batch_size, seq_len, hidden_size = hidden_states.shape
            hidden_flat = hidden_states.view(-1, hidden_size)
            router_logits = module.router(hidden_flat)
            _, selected = torch.topk(router_logits, k, dim=-1)
            routing_decisions[layer_idx] = selected.detach().cpu()
        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        hook = layer.moe.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    with torch.no_grad():
        model(input_ids)

    for hook in hooks:
        hook.remove()

    return routing_decisions


def compute_expert_coactivation(
    model,
    dataset,
    layer_idx: int,
    k: int = 8,
) -> np.ndarray:
    """
    Compute expert co-activation matrix for a specific layer.

    Expert co-activation(E_i, E_j) = N_{E_i, E_j} / N_{E_i}

    where:
    - N_{E_i, E_j}: number of times experts E_i and E_j are activated together
    - N_{E_i}: total number of times expert E_i is activated

    From Section 5.2:
    - No strong co-activation among experts in one layer (few exceptions)
    - Layers 7 and 15 show similar co-activation patterns
    - Low co-activation suggests little redundancy across experts

    Args:
        model: OLMoE model
        dataset: Dataset to evaluate on (paper uses 0.5% of C4 validation)
        layer_idx: Which layer to analyze
        k: Number of top-k experts

    Returns:
        Co-activation matrix [num_experts, num_experts]
    """
    model.eval()
    num_experts = model.config.num_experts

    # Count co-activations
    coactivation_counts = np.zeros((num_experts, num_experts))
    activation_counts = np.zeros(num_experts)

    with torch.no_grad():
        for batch in dataset:
            input_ids = batch["input_ids"]
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            routing = get_routing_decisions(model, input_ids, k)
            selected = routing[layer_idx].numpy()  # [B*T, k]

            for token_experts in selected:
                # Update activation counts
                for e in token_experts:
                    activation_counts[e] += 1

                # Update co-activation counts
                for i, e_i in enumerate(token_experts):
                    for j, e_j in enumerate(token_experts):
                        if i != j:
                            coactivation_counts[e_i, e_j] += 1

    # Normalize: co-activation(E_i, E_j) = N_{E_i, E_j} / N_{E_i}
    coactivation_matrix = np.zeros((num_experts, num_experts))
    for i in range(num_experts):
        if activation_counts[i] > 0:
            coactivation_matrix[i] = coactivation_counts[i] / activation_counts[i]

    return coactivation_matrix


def compute_domain_specialization(
    model,
    domain_datasets: Dict[str, any],
    k: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Compute domain specialization for each expert in each layer.

    Domain specialization(E_i, D) = N_{E_i, D}^(k) / N_D

    where:
    - N_{E_i, D}^(k): number of tokens from domain D for which E_i is among top-k
    - N_D: total number of tokens from domain D

    From Section 5.3:
    - Many experts are activated significantly above/below random chance for specific domains
    - arXiv: first expert in layer 0 is nearly 100% specialized
    - GitHub and arXiv often activate the same experts in layer 7
    - Generic domains (C4) show more balanced expert activations
    - Mixtral shows little domain specialization (likely due to upcycling)

    Args:
        model: OLMoE model
        domain_datasets: Dict mapping domain name -> dataset
        k: Number of top-k experts to consider

    Returns:
        Dict mapping domain -> specialization matrix [num_layers, num_experts]
    """
    model.eval()
    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_experts

    results = {}

    for domain_name, dataset in domain_datasets.items():
        # Count activations per expert per layer
        expert_counts = np.zeros((num_layers, num_experts))
        total_tokens = 0

        with torch.no_grad():
            for batch in dataset:
                input_ids = batch["input_ids"]
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()

                routing = get_routing_decisions(model, input_ids, k)
                batch_tokens = input_ids.numel()
                total_tokens += batch_tokens

                for layer_idx in range(num_layers):
                    selected = routing[layer_idx].numpy()  # [B*T, k]
                    for token_experts in selected:
                        for e in token_experts:
                            expert_counts[layer_idx, e] += 1

        # Normalize by total tokens
        if total_tokens > 0:
            specialization = expert_counts / total_tokens
        else:
            specialization = expert_counts

        results[domain_name] = specialization

    return results


def compute_vocabulary_specialization(
    model,
    dataset,
    k: int = 1,
    use_output_tokens: bool = False,
) -> Dict[int, np.ndarray]:
    """
    Compute vocabulary specialization for each expert in each layer.

    Vocabulary specialization(E_i, x) = N_{x, E_i}^(k) / N_x

    where:
    - N_{x, E_i}^(k): number of times input data is routed to E_i for token x
    - N_x: total number of times token x appears

    From Section 5.4:
    - Vocabulary specialization is higher in later layers
    - Later layers specialize more on predicted output token IDs
    - Expert 27 specializes on non-alphabetic tokens (Cyrillic, Devanagari)
    - Expert 43 specializes on geographic terms
    - Expert 7 specializes on religious terms
    - Expert 37 specializes on time/sports terms

    Args:
        model: OLMoE model
        dataset: Dataset to evaluate on
        k: Number of top-k experts (k=1 for top-1 routing)
        use_output_tokens: If True, use next token IDs instead of input token IDs

    Returns:
        Dict mapping layer_idx -> specialization matrix [vocab_size, num_experts]
    """
    model.eval()
    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_experts
    vocab_size = model.config.vocab_size

    # Count activations per token ID per expert per layer
    # Using sparse representation for efficiency
    token_expert_counts = {
        layer_idx: defaultdict(lambda: np.zeros(num_experts))
        for layer_idx in range(num_layers)
    }
    token_counts = defaultdict(int)

    with torch.no_grad():
        for batch in dataset:
            input_ids = batch["input_ids"]
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            routing = get_routing_decisions(model, input_ids, k)
            input_ids_flat = input_ids.view(-1).cpu().numpy()

            if use_output_tokens:
                # Use next token as the "output" token
                # Shift: output token for position i is input_ids[i+1]
                output_ids = input_ids.view(-1).cpu().numpy()
                token_ids = output_ids[1:]  # Shift by 1
                num_tokens = len(token_ids)
            else:
                token_ids = input_ids_flat
                num_tokens = len(token_ids)

            for layer_idx in range(num_layers):
                selected = routing[layer_idx].numpy()  # [B*T, k]
                # Align with token_ids
                selected_aligned = selected[:num_tokens]

                for token_pos, (token_id, experts) in enumerate(
                    zip(token_ids, selected_aligned)
                ):
                    token_counts[token_id] += 1
                    for e in experts:
                        token_expert_counts[layer_idx][token_id][e] += 1

    # Normalize
    results = {}
    for layer_idx in range(num_layers):
        # Build specialization matrix for tokens that appear enough times
        specialization = {}
        for token_id, expert_counts in token_expert_counts[layer_idx].items():
            total = token_counts[token_id]
            if total > 0:
                specialization[token_id] = expert_counts / total
        results[layer_idx] = specialization

    return results


def get_top_specialized_tokens(
    vocab_specialization: Dict[int, np.ndarray],
    layer_idx: int,
    expert_idx: int,
    tokenizer,
    top_n: int = 10,
    min_count: int = 8,
) -> List[Tuple[str, float]]:
    """
    Get the top tokens that are most specialized for a given expert.

    Args:
        vocab_specialization: Output of compute_vocabulary_specialization
        layer_idx: Layer to analyze
        expert_idx: Expert to analyze
        tokenizer: Tokenizer to decode token IDs
        top_n: Number of top tokens to return
        min_count: Minimum number of occurrences to include a token

    Returns:
        List of (token_string, specialization_score) tuples
    """
    layer_spec = vocab_specialization[layer_idx]

    token_scores = []
    for token_id, expert_scores in layer_spec.items():
        score = expert_scores[expert_idx]
        if score > 0:
            token_str = tokenizer.decode([token_id])
            token_scores.append((token_str, score))

    # Sort by specialization score
    token_scores.sort(key=lambda x: x[1], reverse=True)
    return token_scores[:top_n]


def analyze_routing_saturation_over_training(
    checkpoints: List[str],
    final_checkpoint: str,
    dataset,
    k: int = 8,
) -> Dict[str, Dict[str, float]]:
    """
    Analyze how routing saturation changes over training.

    From Figure 20 in the paper:
    - Measures saturation at 1%, 10%, 20%, 40% of pretraining
    - Compares to final checkpoint

    Args:
        checkpoints: List of checkpoint paths (intermediate)
        final_checkpoint: Path to final checkpoint
        dataset: Evaluation dataset
        k: Number of top-k experts

    Returns:
        Dict mapping checkpoint_name -> saturation_per_layer
    """
    from src.model import OLMoEForCausalLM, OLMoEConfig

    # Load final model
    final_state = torch.load(final_checkpoint, map_location="cpu")
    config = OLMoEConfig()
    final_model = OLMoEForCausalLM(config)
    final_model.load_state_dict(final_state["model_state_dict"])
    if torch.cuda.is_available():
        final_model = final_model.cuda()

    results = {}

    for ckpt_path in checkpoints:
        state = torch.load(ckpt_path, map_location="cpu")
        inter_model = OLMoEForCausalLM(config)
        inter_model.load_state_dict(state["model_state_dict"])
        if torch.cuda.is_available():
            inter_model = inter_model.cuda()

        saturation = compute_router_saturation(
            inter_model, final_model, dataset, k=k
        )
        ckpt_name = ckpt_path.split("/")[-1]
        results[ckpt_name] = saturation

    return results
