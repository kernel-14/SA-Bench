"""
Interpretability analysis for MoE-POT router-gating network.

Implements the dataset classification algorithm from Section 5.4 and Appendix B.4:

1. For each dataset, compute the average expert selection distribution Y_i
2. For a new input X, compute its routing distribution I_0
3. Classify X as belonging to dataset i_0 = argmin_i f(I_0, Y_i)
   where f is the cross-entropy loss

This analysis demonstrates that the router-gating network implicitly learns
to distinguish between different PDE datasets with ~98% accuracy.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .model import MoEPOT


def compute_average_routing_distributions(
    model: MoEPOT,
    dataset_loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    block_idx: int = 1,  # Block 2 in 1-indexed (0-indexed = 1)
) -> Dict[str, torch.Tensor]:
    """
    Compute average expert selection distributions for each dataset.

    For each dataset k, computes:
        Y_k = (1/N_k) * sum_j Y_{kj}

    where Y_{kj} is the routing weight vector for the j-th sample in dataset k.

    Args:
        model: Trained MoEPOT model.
        dataset_loaders: Dict mapping dataset name -> DataLoader.
        device: Computation device.
        block_idx: Which block to analyze (0-indexed, default=1 for Block 2).

    Returns:
        Dict mapping dataset name -> average routing distribution (N_r,).
    """
    model.eval()
    avg_distributions = {}

    with torch.no_grad():
        for dataset_name, loader in dataset_loaders.items():
            all_weights = []

            for batch in loader:
                u_input = batch["input"].to(device)  # (B, T, C, H, W)

                # Get routing weights from all blocks
                routing_weights = model.get_routing_weights_all_blocks(u_input)
                # routing_weights[block_idx]: (B, N_r)

                block_weights = routing_weights[block_idx]  # (B, N_r)
                all_weights.append(block_weights.cpu())

            # Average over all samples
            all_weights = torch.cat(all_weights, dim=0)  # (N_k, N_r)
            avg_distributions[dataset_name] = all_weights.mean(dim=0)  # (N_r,)

    return avg_distributions


def classify_dataset(
    routing_weights: torch.Tensor,
    avg_distributions: Dict[str, torch.Tensor],
) -> str:
    """
    Classify a single input's dataset based on its routing weights.

    Uses cross-entropy distance:
        f(I_0, Y_i) = -sum_k I_{0,k} * log(Y_{i,k})

    Args:
        routing_weights: Routing weight vector for the input (N_r,).
        avg_distributions: Dict mapping dataset name -> average distribution (N_r,).

    Returns:
        Predicted dataset name.
    """
    min_distance = float("inf")
    predicted_dataset = None

    for dataset_name, Y_i in avg_distributions.items():
        # Cross-entropy: f(I_0, Y_i) = -sum_k I_{0,k} * log(Y_{i,k})
        # Add small epsilon for numerical stability
        log_Y_i = torch.log(Y_i + 1e-8)
        distance = -(routing_weights * log_Y_i).sum().item()

        if distance < min_distance:
            min_distance = distance
            predicted_dataset = dataset_name

    return predicted_dataset


def evaluate_classification_accuracy(
    model: MoEPOT,
    dataset_loaders: Dict[str, torch.utils.data.DataLoader],
    avg_distributions: Dict[str, torch.Tensor],
    device: torch.device,
    block_idx: int = 1,
) -> Dict[str, float]:
    """
    Evaluate dataset classification accuracy using router-gating network.

    Args:
        model: Trained MoEPOT model.
        dataset_loaders: Dict mapping dataset name -> DataLoader.
        avg_distributions: Pre-computed average routing distributions.
        device: Computation device.
        block_idx: Which block to analyze.

    Returns:
        Dict with per-dataset and overall accuracy.
    """
    model.eval()
    results = {}
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for true_dataset_name, loader in dataset_loaders.items():
            correct = 0
            n_samples = 0

            for batch in loader:
                u_input = batch["input"].to(device)
                B = u_input.shape[0]

                # Get routing weights
                routing_weights = model.get_routing_weights_all_blocks(u_input)
                block_weights = routing_weights[block_idx]  # (B, N_r)

                # Classify each sample
                for b in range(B):
                    pred_dataset = classify_dataset(
                        block_weights[b].cpu(),
                        avg_distributions
                    )
                    if pred_dataset == true_dataset_name:
                        correct += 1
                    n_samples += 1

            accuracy = correct / n_samples if n_samples > 0 else 0.0
            results[true_dataset_name] = accuracy
            total_correct += correct
            total_samples += n_samples

    results["overall"] = total_correct / total_samples if total_samples > 0 else 0.0
    return results


def analyze_expert_usage(
    model: MoEPOT,
    dataset_loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    block_idx: int = 3,  # Block 4 in 1-indexed
) -> Dict[str, np.ndarray]:
    """
    Analyze expert usage ratios for different datasets.

    Computes the fraction of samples that activate each expert (top-K selection).

    Args:
        model: Trained MoEPOT model.
        dataset_loaders: Dict mapping dataset name -> DataLoader.
        device: Computation device.
        block_idx: Which block to analyze.

    Returns:
        Dict mapping dataset name -> expert usage ratio array (N_r,).
    """
    model.eval()
    usage_ratios = {}

    with torch.no_grad():
        for dataset_name, loader in dataset_loaders.items():
            num_routed = model.blocks[block_idx].moe_layer.num_routed_experts
            top_k = model.blocks[block_idx].moe_layer.top_k
            expert_counts = np.zeros(num_routed)
            total_samples = 0

            for batch in loader:
                u_input = batch["input"].to(device)
                B = u_input.shape[0]

                # Get routing weights
                routing_weights = model.get_routing_weights_all_blocks(u_input)
                block_weights = routing_weights[block_idx]  # (B, N_r)

                # Get top-K selected experts
                _, topk_indices = torch.topk(block_weights, top_k, dim=-1)
                # topk_indices: (B, K)

                for b in range(B):
                    for k in range(top_k):
                        expert_idx = topk_indices[b, k].item()
                        expert_counts[expert_idx] += 1

                total_samples += B

            # Normalize to get usage ratio
            usage_ratios[dataset_name] = expert_counts / (total_samples * top_k)

    return usage_ratios


def run_interpretability_analysis(
    model: MoEPOT,
    dataset_loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    output_dir: str = "results/interpretability",
) -> Dict:
    """
    Run the full interpretability analysis pipeline.

    1. Compute average routing distributions per dataset
    2. Evaluate classification accuracy
    3. Analyze expert usage ratios

    Args:
        model: Trained MoEPOT model.
        dataset_loaders: Dict mapping dataset name -> DataLoader.
        device: Computation device.
        output_dir: Directory to save results.

    Returns:
        Dict containing all analysis results.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Analyze each block
    num_blocks = len(model.blocks)
    for block_idx in range(num_blocks):
        print(f"\nAnalyzing Block {block_idx + 1}...")

        # Compute average routing distributions
        avg_distributions = compute_average_routing_distributions(
            model, dataset_loaders, device, block_idx=block_idx
        )

        # Evaluate classification accuracy
        accuracy = evaluate_classification_accuracy(
            model, dataset_loaders, avg_distributions, device, block_idx=block_idx
        )

        print(f"Block {block_idx + 1} Classification Accuracy:")
        for dataset_name, acc in accuracy.items():
            print(f"  {dataset_name}: {acc * 100:.1f}%")

        results[f"block_{block_idx + 1}"] = {
            "avg_distributions": {k: v.numpy() for k, v in avg_distributions.items()},
            "classification_accuracy": accuracy,
        }

    # Analyze expert usage for Block 4 (as in Figure 2 right)
    if num_blocks >= 4:
        print("\nAnalyzing expert usage in Block 4...")
        usage_ratios = analyze_expert_usage(
            model, dataset_loaders, device, block_idx=3
        )
        results["expert_usage_block_4"] = usage_ratios

        print("Expert usage ratios (Block 4):")
        for dataset_name, ratios in usage_ratios.items():
            top_experts = np.argsort(ratios)[::-1][:5]
            print(f"  {dataset_name}: top experts = {top_experts.tolist()}")

    # Save results
    import json
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        return obj

    with open(os.path.join(output_dir, "interpretability_results.json"), "w") as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return results
