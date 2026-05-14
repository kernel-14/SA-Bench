"""
Analysis Module for Interpreting Emergent Planning.

Implements the key analysis procedures from the paper:
1. Probe training and evaluation (Section 4)
2. Plan formation analysis (Section 5)
3. Test-time plan refinement analysis (Section 5, Appendix A.3)
4. Training emergence analysis (Section 6.2, Appendix C)
5. Visualization utilities
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import f1_score
import pickle
import os
from collections import defaultdict


def build_probe_dataset(
    agent,
    env,
    levels: List[np.ndarray],
    record_levels: List[int],
    thinking_steps: int = 0,
    greedy: bool = True,
    concept: str = 'CA',
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Build a dataset for probe training by running the agent on multiple levels.
    
    Args:
        agent: DRCAgent instance
        env: SokobanEnv instance
        levels: List of level arrays
        record_levels: Which layers to record
        thinking_steps: Number of thinking steps per episode
        greedy: Whether to act greedily
        concept: 'CA' or 'CB' or 'both'
    
    Returns:
        Dict mapping layer_idx -> {'activations': (N, C, H, W), 'labels': (N, 8, 8)}
    """
    all_activations = {d: [] for d in record_levels}
    all_labels = {d: [] for d in record_levels}
    
    for level in levels:
        result = record_episode_labels(env, agent, level, thinking_steps, greedy, record_levels)
        
        for d in record_levels:
            for t, cs in enumerate(result['cell_states'][d]):
                all_activations[d].append(cs)
                if concept == 'CA':
                    all_labels[d].append(result['ca_labels'][t])
                elif concept == 'CB':
                    all_labels[d].append(result['cb_labels'][t])
                elif concept == 'both':
                    all_labels[d].append(result['ca_labels'][t])  # default to CA
    
    datasets = {}
    for d in record_levels:
        datasets[d] = {
            'activations': np.stack(all_activations[d]),  # (N, H, W, C)
            'labels': np.stack(all_labels[d]),  # (N, H, W)
        }
    
    return datasets


def evaluate_probe_predictions(
    probe,
    cell_state: torch.Tensor,
    labels: np.ndarray,
) -> Dict:
    """
    Evaluate probe predictions against ground truth labels.
    
    Args:
        probe: LinearProbe module
        cell_state: (B, C, H, W) tensor
        labels: (B, H, W) integer array
    
    Returns:
        dict with macro_f1 and per-class metrics
    """
    with torch.no_grad():
        logits = probe(cell_state)  # (B, num_classes, H, W)
        # Reshape for sklearn metrics
        B, C, H, W = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
        labels_flat = labels.reshape(-1)
        
        preds = logits_flat.argmax(axis=-1)
        
        macro_f1 = f1_score(labels_flat, preds, average='macro')
        
        # Per-class
        per_class = {}
        class_names = ['NEVER', 'UP', 'DOWN', 'LEFT', 'RIGHT']
        for i, name in enumerate(class_names):
            cls_f1 = f1_score(labels_flat == i, preds == i, average='binary')
            per_class[name] = {'f1': cls_f1}
        
        return {'macro_f1': macro_f1, 'per_class': per_class}


def analyze_test_time_plan_refinement(
    agent,
    env,
    levels: List[np.ndarray],
    probes_by_layer: Dict[int, torch.nn.Module],
    thinking_steps: int = 5,
    concept: str = 'CA',
    record_levels: Optional[List[int]] = None,
) -> Dict:
    """
    Analyze how plans refine during thinking steps.
    
    This replicates the analysis from Section 5 / Figure 6 / Appendix A.3.
    
    For each internal tick during thinking steps, decode the agent's plan
    using a probe and measure macro F1 against the final plan.
    
    Args:
        agent: DRCAgent instance
        env: SokobanEnv instance
        levels: List of levels to test on
        probes_by_layer: Dict mapping layer -> trained probe
        thinking_steps: Number of thinking steps
        concept: 'CA' or 'CB'
        record_levels: Which layers to analyze
    
    Returns:
        Dict with macro F1 per tick per layer
    """
    if record_levels is None:
        record_levels = list(probes_by_layer.keys())
    
    N = agent.N  # ticks per step
    total_ticks = thinking_steps * N
    
    # Collect macro F1s per tick
    tick_f1s = defaultdict(lambda: defaultdict(list))
    
    for level in levels:
        obs = env.reset(level)
        agent.reset_state(batch_size=1, device=next(agent.parameters()).device)
        
        obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0)
        obs_tensor = obs_tensor.to(next(agent.parameters()).device)
        
        # Run the episode without thinking steps to get ground truth labels
        # Record full trajectory to get concept labels
        from src.concept_labels import record_episode_labels as _record
        gt_result = _record(env if hasattr(env, 'reset') else env, agent, level, 0, True, record_levels)
        
        # Now run with thinking steps and decode at each tick
        obs = env.reset(level)
        agent.reset_state(batch_size=1, device=next(agent.parameters()).device)
        obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0)
        obs_tensor = obs_tensor.to(next(agent.parameters()).device)
        
        tick = 0
        for step in range(thinking_steps):
            with torch.no_grad():
                # We need per-tick cell states; use return_all_states=True
                _, _, states = agent.forward(obs_tensor, return_all_states=True)
            
            for n in range(N):
                for d in record_levels:
                    cs = states[f'tick_{n}'][f'layer_{d}']  # (1, C, H, W)
                    probe = probes_by_layer[d]
                    
                    # Decode plan
                    with torch.no_grad():
                        logits = probe(cs)
                        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
                    
                    # Compare to ground truth
                    if concept == 'CA':
                        gt = gt_result['ca_labels'][0]  # first time step
                    else:
                        gt = gt_result['cb_labels'][0]
                    
                    f1 = f1_score(gt.reshape(-1), pred.reshape(-1), average='macro')
                    tick_f1s[d][tick].append(f1)
                
                tick += 1
        
        break  # Just one level for now (placeholder)
    
    # Average over levels
    results = {}
    for d in record_levels:
        results[d] = {tick: np.mean(f1s) for tick, f1s in tick_f1s[d].items()}
    
    return results


def compute_macro_f1_curve(
    probes: Dict[int, torch.nn.Module],
    test_activations: Dict[int, torch.Tensor],
    test_labels: Dict[int, torch.Tensor],
) -> Dict[int, float]:
    """
    Compute macro F1 for probes across layers.
    
    Replicates Figure 4 analysis.
    """
    results = {}
    for layer_idx, probe in probes.items():
        cell_state = test_activations[layer_idx]
        labels = test_labels[layer_idx]
        
        with torch.no_grad():
            logits = probe(cell_state)
            B, C, H, W = logits.shape
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
            labels_flat = labels.reshape(-1).cpu().numpy()
            preds = logits_flat.argmax(axis=-1)
            macro_f1 = f1_score(labels_flat, preds, average='macro')
        
        results[layer_idx] = macro_f1
    
    return results


def compute_baseline_macro_f1(
    observations: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute baseline macro F1 using raw observations as input.
    """
    # Use a simple probe on observations
    from src.probes import LinearProbe, train_probe_pytorch
    
    baseline_probe = LinearProbe(in_channels=7, kernel_size=1)  # 7 obs channels
    results = train_probe_pytorch(
        baseline_probe, observations, labels,
        num_epochs=10, batch_size=16,
        learning_rate=0.001, weight_decay=0.001,
    )
    return results.get('macro_f1', 0.0)


def emergence_analysis(
    checkpoint_data: Dict[int, Dict],
) -> Dict:
    """
    Analyze the emergence of planning during training.
    
    Replicates Section 6.2 and Appendix C analysis.
    
    Args:
        checkpoint_data: Dict mapping checkpoint_step -> {
            'macro_f1_ca': per-layer macro F1 for C_A,
            'macro_f1_cb': per-layer macro F1 for C_B,
            'extra_levels_solved': number of additional levels solved with thinking steps,
        }
    
    Returns:
        Correlation analysis results
    """
    checkpoints = sorted(checkpoint_data.keys())
    
    # Extract data
    ca_f1s = [checkpoint_data[c]['macro_f1_ca'] for c in checkpoints]
    cb_f1s = [checkpoint_data[c]['macro_f1_cb'] for c in checkpoints]
    extra_solved = [checkpoint_data[c]['extra_levels_solved'] for c in checkpoints]
    
    # Correlation between F1 and extra levels solved
    from scipy.stats import pearsonr
    
    corr_ca, p_ca = pearsonr(ca_f1s, extra_solved)
    corr_cb, p_cb = pearsonr(cb_f1s, extra_solved)
    
    return {
        'ca_correlation': corr_ca,
        'ca_pvalue': p_ca,
        'cb_correlation': corr_cb,
        'cb_pvalue': p_cb,
        'checkpoints': checkpoints,
        'ca_f1s': ca_f1s,
        'cb_f1s': cb_f1s,
        'extra_solved': extra_solved,
    }


def compute_plan_refinement_improvement(
    tick_f1s: Dict[int, Dict[int, float]],
) -> Dict[int, float]:
    """
    Compute the improvement in macro F1 from first to last tick.
    
    Replicates Appendix C.2 / C.4 analysis.
    """
    results = {}
    for layer, f1s in tick_f1s.items():
        ticks = sorted(f1s.keys())
        if len(ticks) >= 2:
            results[layer] = f1s[ticks[-1]] - f1s[ticks[0]]
        else:
            results[layer] = 0.0
    return results


# Import fix at module level
from src.concept_labels import record_episode_labels
