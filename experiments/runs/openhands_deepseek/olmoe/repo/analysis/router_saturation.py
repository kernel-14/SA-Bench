import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np


def compute_router_saturation(
    model,
    dataloader,
    final_checkpoint_expert_ids: Dict[int, List[torch.Tensor]],
    k: int = 8,
) -> Dict[int, float]:
    """
    Compute router saturation (Equation 5, Section 5.1).

    Router Saturation(t) = (1/N) * sum_i=1^N |E_i^(t) ∩ E_i^(T)| / k

    Where:
        N: total number of tokens
        k: number of top-k experts (8 for training, also analyzed for k=1)
        E_i^(t): set of top-k experts for token i at checkpoint t
        E_i^(T): set of top-k experts for token i at final checkpoint T

    Saturation measures what proportion of expert activations at an intermediary
    checkpoint match the final checkpoint routing.

    Args:
        model: OLMoE model
        dataloader: data loader providing batches of input_ids
        final_checkpoint_expert_ids: pre-computed expert IDs from final checkpoint
        k: number of top-k experts to consider

    Returns:
        dict mapping layer_idx -> saturation score (0 to 100%)
    """
    model.eval()
    device = next(model.parameters()).device

    # Storage for per-layer expert selections
    layer_topk: Dict[int, List[torch.Tensor]] = {}
    num_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            B, T = input_ids.shape

            logits, _, router_probs_list = model(input_ids)

            for layer_idx, router_probs in enumerate(router_probs_list):
                # router_probs: (B, T, num_experts)
                _, topk_indices = torch.topk(router_probs, k, dim=-1)  # (B, T, k)
                if layer_idx not in layer_topk:
                    layer_topk[layer_idx] = []
                layer_topk[layer_idx].append(topk_indices.cpu())

            num_tokens += B * T

    # Compute saturation for each layer
    saturation_scores = {}
    for layer_idx in layer_topk:
        current_ids = torch.cat(layer_topk[layer_idx], dim=0)  # (total_tokens, k)
        final_ids = final_checkpoint_expert_ids[layer_idx]  # (total_tokens, k)

        # Count matching experts per token
        matches = 0
        for i in range(min(current_ids.size(0), final_ids.size(0))):
            current_set = set(current_ids[i].tolist())
            final_set = set(final_ids[i].tolist())
            matches += len(current_set.intersection(final_set))

        saturation = (matches / (min(current_ids.size(0), final_ids.size(0)) * k)) * 100.0
        saturation_scores[layer_idx] = saturation

    return saturation_scores


def precompute_final_expert_ids(
    model,
    dataloader,
    k: int = 8,
) -> Dict[int, torch.Tensor]:
    """
    Pre-compute top-k expert IDs from the final checkpoint for router saturation analysis.

    Returns dict mapping layer_idx -> tensor of shape (total_tokens, k).
    """
    model.eval()
    device = next(model.parameters()).device

    layer_topk: Dict[int, List[torch.Tensor]] = {}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)

            _, _, router_probs_list = model(input_ids)

            for layer_idx, router_probs in enumerate(router_probs_list):
                _, topk_indices = torch.topk(router_probs, k, dim=-1)  # (B, T, k)
                if layer_idx not in layer_topk:
                    layer_topk[layer_idx] = []
                layer_topk[layer_idx].append(topk_indices.cpu())

    return {k: torch.cat(v, dim=0) for k, v in layer_topk.items()}


def analyze_saturation_across_checkpoints(
    checkpoints: List[str],
    dataloader,
    k: int = 8,
) -> Dict[str, Dict[int, float]]:
    """
    Analyze router saturation across multiple checkpoints.

    This implements the analysis from Figure 20, computing saturation
    at 1%, 10%, 20%, 40% of pretraining vs. final checkpoint.

    Args:
        checkpoints: list of checkpoint paths [ckpt_T, ckpt_t1, ckpt_t2, ...]
        dataloader: data loader
        k: top-k experts

    Returns:
        dict mapping checkpoint_name -> layer_saturation_scores
    """
    from model.olmoe_model import OLMoEModel

    final_ckpt = checkpoints[0]
    final_model = OLMoEModel.from_pretrained(final_ckpt)
    final_ids = precompute_final_expert_ids(final_model, dataloader, k=k)
    del final_model

    results = {}
    for ckpt in checkpoints[1:]:
        model = OLMoEModel.from_pretrained(ckpt)
        saturation = compute_router_saturation(model, dataloader, final_ids, k=k)
        results[ckpt] = saturation
        del model

    return results
