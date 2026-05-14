"""
Interpretability Analysis for MoE-POT.

Implements the router-gating network analysis described in:
- Section 5.4 (Interpretable Analysis)
- Appendix B.4 (Interpretable Analysis Algorithms)
- Appendix C.6 (Extended Interpretability Analysis)

The analysis determines which PDE dataset a given input belongs to by examining
the expert selection patterns of the router-gating network.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_dataset_expert_profiles(
    model: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    block_idx: int = 1,
    device: torch.device = torch.device('cpu'),
    max_samples_per_dataset: int = 200,
) -> Dict[str, torch.Tensor]:
    """
    Compute average expert selection distribution for each dataset.
    
    For a specific block, compute the average expert selection distribution:
    Y_i = (1/N_i) * Σ_j Y_{ij}
    
    where Y_{ij} ∈ R^16 is the router-gating network output (full softmax)
    for the j-th sample in i-th dataset.
    
    Args:
        model: MoE-POT model
        dataloaders: dict mapping dataset name to dataloader
        block_idx: which MoE-POT block to analyze (0-indexed)
        device: computation device
        max_samples_per_dataset: max samples to use per dataset
        
    Returns:
        profiles: dict mapping dataset name to average expert distribution [16]
    """
    model.eval()
    profiles = {}
    
    for dataset_name, dataloader in dataloaders.items():
        all_routing = []
        num_samples = 0
        
        with torch.no_grad():
            for x, y, _ in dataloader:
                if num_samples >= max_samples_per_dataset:
                    break
                    
                x = x.to(device)
                
                # Get routing weights from the specified block
                # We need to extract the full softmax output, not just top-K
                # Do a full forward pass but intercept the routing weights
                
                # Patchify and aggregate
                B, T, C, H, W = x.shape
                patches = []
                for t in range(T):
                    patches.append(model.patch_embed(x[:, t]))
                patches = torch.stack(patches, dim=1)
                
                if model.pos_encoding is not None:
                    patches = patches + model.pos_encoding.unsqueeze(1)
                
                z = model.temporal_agg(patches)
                
                # Get routing for the target block
                block = model.blocks[block_idx]
                z = block.fourier(block.norm1(z.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
                
                # Get full routing weights (not just top-K)
                logits = block.moe.router.router(z)  # [B, 16]
                routing_weights = F.softmax(logits, dim=-1)  # [B, 16]
                
                all_routing.append(routing_weights.cpu())
                num_samples += B
        
        if all_routing:
            all_routing = torch.cat(all_routing, dim=0)[:max_samples_per_dataset]
            profiles[dataset_name] = all_routing.mean(dim=0)  # [16]
        else:
            profiles[dataset_name] = torch.zeros(16)
    
    return profiles


def classify_input_by_routing(
    model: torch.nn.Module,
    x: torch.Tensor,
    profiles: Dict[str, torch.Tensor],
    block_idx: int = 1,
    device: torch.device = torch.device('cpu'),
) -> Tuple[str, Dict[str, float]]:
    """
    Classify an input sample by comparing its expert distribution to dataset profiles.
    
    As described in Appendix B.4:
    For input X with expert distribution I_0:
    - Compute cross-entropy distance to each profile Y_i
    - f(I_0, Y_i) = -Σ_k I_{0,k} * log(Y_{i,k})
    - Classify to i_0 = argmin_i f(I_0, Y_i)
    
    Args:
        model: MoE-POT model
        x: input tensor [1, T, C, H, W]
        profiles: dataset expert profiles from compute_dataset_expert_profiles
        block_idx: which block to use for classification
        device: computation device
        
    Returns:
        predicted_class: name of predicted dataset
        distances: cross-entropy distances to each dataset
    """
    model.eval()
    
    with torch.no_grad():
        x = x.to(device)
        B, T, C, H, W = x.shape
        
        # Get routing weights
        patches = []
        for t in range(T):
            patches.append(model.patch_embed(x[:, t]))
        patches = torch.stack(patches, dim=1)
        
        if model.pos_encoding is not None:
            patches = patches + model.pos_encoding.unsqueeze(1)
        
        z = model.temporal_agg(patches)
        
        # Get full routing for target block
        block = model.blocks[block_idx]
        with torch.no_grad():
            z = block.fourier(block.norm1(z.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
            logits = block.moe.router.router(z)
            routing_weights = F.softmax(logits, dim=-1)  # [1, 16]
        
        I_0 = routing_weights[0]  # [16]
        
        # Compute cross-entropy distances
        distances = {}
        for dataset_name, profile in profiles.items():
            # f(I_0, Y_i) = -Σ_k I_{0,k} * log(Y_{i,k})
            # Add small epsilon to avoid log(0)
            ce = -torch.sum(I_0 * torch.log(profile + 1e-8))
            distances[dataset_name] = ce.item()
        
        # Find minimum distance
        predicted_class = min(distances, key=distances.get)
        
    return predicted_class, distances


def compute_classification_accuracy(
    model: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    profiles: Dict[str, torch.Tensor],
    block_idx: int = 1,
    device: torch.device = torch.device('cpu'),
    max_samples: int = 200,
) -> Dict[str, float]:
    """
    Compute classification accuracy of the router-gating network.
    
    Evaluates how well the router can identify which dataset an input belongs to.
    
    As reported in Section 5.4: Block 2 achieves 97.7% accuracy.
    
    Args:
        model: MoE-POT model
        dataloaders: dict mapping dataset name to dataloader
        profiles: pre-computed dataset profiles
        block_idx: which block to use
        device: computation device
        max_samples: max samples per dataset for evaluation
        
    Returns:
        accuracies: dict mapping dataset name to classification accuracy
    """
    model.eval()
    accuracies = {}
    total_correct = 0
    total_samples = 0
    
    for dataset_name, dataloader in dataloaders.items():
        correct = 0
        num_samples = 0
        
        for x, y, _ in dataloader:
            if num_samples >= max_samples:
                break
            
            x = x.to(device)
            B = x.shape[0]
            
            for i in range(B):
                if num_samples >= max_samples:
                    break
                    
                single_x = x[i:i+1]
                predicted, _ = classify_input_by_routing(
                    model, single_x, profiles, block_idx, device
                )
                
                if predicted == dataset_name:
                    correct += 1
                num_samples += 1
        
        accuracy = correct / num_samples if num_samples > 0 else 0.0
        accuracies[dataset_name] = accuracy
        total_correct += correct
        total_samples += num_samples
    
    accuracies['overall'] = total_correct / total_samples if total_samples > 0 else 0.0
    return accuracies


def analyze_expert_usage(
    model: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    block_idx: int = 1,
    device: torch.device = torch.device('cpu'),
    max_samples: int = 200,
) -> Dict[str, np.ndarray]:
    """
    Analyze usage ratio of each routed expert per dataset.
    
    Computes the fraction of samples for which each expert is selected (in top-K).
    This produces visualizations like Figure 2 (right).
    
    Args:
        model: MoE-POT model
        dataloaders: dict mapping dataset name to dataloader
        block_idx: which block to analyze
        device: computation device
        max_samples: max samples per dataset
        
    Returns:
        usage_ratios: dict mapping dataset name to usage array [16]
    """
    model.eval()
    usage_ratios = {}
    
    for dataset_name, dataloader in dataloaders.items():
        expert_counts = np.zeros(16)
        num_samples = 0
        
        with torch.no_grad():
            for x, y, _ in dataloader:
                if num_samples >= max_samples:
                    break
                    
                x = x.to(device)
                
                # Forward pass to collect routing info
                _, routing_info = model(x, return_routing=True)
                
                # Get indices of selected experts for this block
                indices = routing_info[block_idx]['indices'].cpu().numpy()  # [B, top_K]
                
                for b in range(indices.shape[0]):
                    for k in range(indices.shape[1]):
                        expert_counts[indices[b, k]] += 1
                    num_samples += 1
        
        # Normalize to usage ratio
        if num_samples > 0:
            usage_ratios[dataset_name] = expert_counts / (num_samples * model.blocks[block_idx].moe.top_k)
        else:
            usage_ratios[dataset_name] = np.zeros(16)
    
    return usage_ratios
