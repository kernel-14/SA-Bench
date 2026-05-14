"""
Evaluation metrics for PGR as described in the paper:

1. Dormant Ratio (DR) - measures overfitting in value-based RL
2. Generation Quality - MSE of generated dynamics vs ground truth
3. Curiosity distribution analysis
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Optional


def compute_dormant_ratio(
    policy_network: nn.Module,
    states: torch.Tensor,
    threshold: float = 0.1,
) -> float:
    """
    Compute the dormant ratio (DR) of a policy network.
    
    DR = fraction of neurons in the network whose activations
    (averaged over the batch) are below some threshold.
    
    Higher DR -> more overfitting (Sokar et al., 2023; Xu et al., 2023).
    
    Args:
        policy_network: the policy (actor) network
        states: batch of states to evaluate on, shape (B, state_dim)
        threshold: activation threshold for "dormant" classification
    
    Returns:
        dormant_ratio: float between 0 and 1
    """
    policy_network.eval()
    
    # Hook to capture activations
    activations = []
    
    def hook_fn(module, input, output):
        activations.append(output.detach())
    
    hooks = []
    for module in policy_network.modules():
        if isinstance(module, nn.ReLU) or isinstance(module, nn.SiLU):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        try:
            # Try sampling action to go through all layers
            if hasattr(policy_network, 'sample'):
                policy_network.sample(states)
            else:
                policy_network(states)
        except:
            # Fallback: just forward pass through trunk
            if hasattr(policy_network, 'trunk'):
                policy_network.trunk(states)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute dormant ratio
    total_neurons = 0
    dormant_neurons = 0
    
    for act in activations:
        # Average activation across batch
        mean_act = act.mean(dim=0)  # (hidden_dim,)
        total_neurons += mean_act.numel()
        dormant_neurons += (mean_act < threshold).sum().item()
    
    if total_neurons == 0:
        return 0.0
    
    return dormant_neurons / total_neurons


def measure_generation_quality(
    generated_states: torch.Tensor,
    generated_actions: torch.Tensor,
    generated_next_states: torch.Tensor,
    generated_rewards: torch.Tensor,
    env_step_fn,  # function(state, action) -> (next_state, reward)
    use_latent: bool = False,
    visual_decoder: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Measure faithfulness of generated transitions to environment dynamics.
    
    Following Lu et al. (2024) methodology:
    For each generated transition (s, a, s', r), roll out action a from state s
    in the environment simulator to get ground truth s' and r.
    Measure MSE between generated and ground truth.
    
    Args:
        generated_states: (N, state_dim)
        generated_actions: (N, action_dim)
        generated_next_states: (N, state_dim)
        generated_rewards: (N, 1)
        env_step_fn: function that steps the environment
        use_latent: whether states are in latent space
        visual_decoder: decoder to convert latent back to pixel space
    
    Returns:
        dict with 'next_state_mse' and 'reward_mse'
    """
    N = generated_states.shape[0]
    device = generated_states.device
    
    next_state_errors = []
    reward_errors = []
    
    # Process in batches
    batch_size = 100
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        
        s_gen = generated_states[i:end]
        a_gen = generated_actions[i:end]
        ns_gen = generated_next_states[i:end]
        r_gen = generated_rewards[i:end]
        
        # Get ground truth from environment
        if use_latent and visual_decoder is not None:
            # Decode latent to pixel space
            s_pixel = visual_decoder(s_gen)
            ns_pixel = visual_decoder(ns_gen)
            # Step env (this depends on env implementation)
            # For now, skip pixel-based quality check
            continue
        
        # Step each transition through env
        for j in range(end - i):
            state_np = s_gen[j].cpu().numpy()
            action_np = a_gen[j].cpu().numpy()
            
            try:
                true_next_state, true_reward, _, _ = env_step_fn(state_np, action_np)
                
                true_ns = torch.tensor(true_next_state, device=device)
                true_r = torch.tensor([true_reward], device=device)
                
                ns_error = torch.mean((ns_gen[j] - true_ns) ** 2).item()
                r_error = torch.mean((r_gen[j] - true_r) ** 2).item()
                
                next_state_errors.append(ns_error)
                reward_errors.append(r_error)
            except:
                continue
    
    if len(next_state_errors) == 0:
        return {'next_state_mse': float('inf'), 'reward_mse': float('inf')}
    
    return {
        'next_state_mse': np.mean(next_state_errors),
        'reward_mse': np.mean(reward_errors),
    }


def compute_curiosity_distribution(
    relevance_function: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    num_bins: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the distribution of curiosity (relevance) values.
    Used for Fig. 6b analysis.
    
    Returns:
        bin_centers, histogram_counts
    """
    with torch.no_grad():
        curiosity_values = relevance_function(states, actions, next_states, rewards)
    
    values = curiosity_values.cpu().numpy().flatten()
    
    hist, bin_edges = np.histogram(values, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return torch.tensor(bin_centers), torch.tensor(hist)


def measure_sample_diversity(
    generated_states: torch.Tensor,
) -> float:
    """
    Measure diversity of generated transitions.
    Uses average pairwise distance as a proxy.
    
    Higher value -> more diverse generations.
    """
    if generated_states.shape[0] < 2:
        return 0.0
    
    # Sample subset if too large
    n = min(generated_states.shape[0], 1000)
    indices = torch.randperm(generated_states.shape[0])[:n]
    samples = generated_states[indices]
    
    # Compute pairwise distances
    dists = torch.cdist(samples, samples, p=2)
    
    # Exclude diagonal
    mask = ~torch.eye(n, dtype=torch.bool, device=samples.device)
    avg_dist = dists[mask].mean().item()
    
    return avg_dist
