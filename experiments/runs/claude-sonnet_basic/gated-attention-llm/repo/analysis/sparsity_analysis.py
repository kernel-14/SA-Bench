"""
Sparsity Analysis for Gating Scores
=====================================
Implements the sparsity analysis from Sec 4.2 of the paper.

Key findings from the paper (Table 4):
  - SDPA elementwise gating has the lowest mean gating scores (~0.1-0.2)
  - Value gating has higher mean scores (~0.3-0.4)
  - Head-shared gating has higher scores than head-specific
  - NS-sigmoid gating has scores in [0.5, 1.0] (less sparse)

The paper shows that sparsity is crucial for performance:
  - More sparse gating -> better performance
  - Query-dependent sparsity (G1) > key/value-dependent (G2)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sparsity_metrics(
    tensor: torch.Tensor,
    thresholds: List[float] = [0.01, 0.1, 0.5],
) -> Dict[str, float]:
    """
    Compute sparsity metrics for a tensor.
    
    Args:
        tensor: Input tensor (any shape)
        thresholds: List of thresholds for sparsity computation
    
    Returns:
        dict with mean, std, and fraction below each threshold
    """
    flat = tensor.detach().float().flatten()
    
    metrics = {
        "mean": flat.mean().item(),
        "std": flat.std().item(),
        "min": flat.min().item(),
        "max": flat.max().item(),
        "median": flat.median().item(),
    }
    
    for threshold in thresholds:
        metrics[f"fraction_below_{threshold}"] = (flat < threshold).float().mean().item()
    
    return metrics


def analyze_hidden_state_sparsity(
    hidden_before_gate: torch.Tensor,
    hidden_after_gate: torch.Tensor,
    gating_scores: torch.Tensor,
    thresholds: List[float] = [1e-2, 1e-3],
) -> Dict[str, Dict]:
    """
    Analyze how gating affects sparsity in hidden states.
    
    From the paper (Appendix A.2):
    - After gating, mean absolute value decreases from 0.71 to 0.05
    - Sparsity (fraction below threshold) significantly increases
    
    Args:
        hidden_before_gate: SDPA output before gating (batch, seq, heads, head_dim)
        hidden_after_gate: SDPA output after gating
        gating_scores: Gating scores applied
        thresholds: Thresholds for sparsity computation
    
    Returns:
        dict with sparsity statistics before and after gating
    """
    results = {}
    
    # Before gating
    abs_before = hidden_before_gate.abs()
    results["before_gate"] = {
        "mean_abs": abs_before.mean().item(),
        "max_abs": abs_before.max().item(),
    }
    for t in thresholds:
        results["before_gate"][f"fraction_below_{t}"] = (abs_before < t).float().mean().item()
    
    # After gating
    abs_after = hidden_after_gate.abs()
    results["after_gate"] = {
        "mean_abs": abs_after.mean().item(),
        "max_abs": abs_after.max().item(),
    }
    for t in thresholds:
        results["after_gate"][f"fraction_below_{t}"] = (abs_after < t).float().mean().item()
    
    # Gating scores
    results["gating_scores"] = compute_sparsity_metrics(gating_scores)
    
    # Counterfactual: multiply by average gating score (not actual sparse gating)
    avg_gate_score = gating_scores.mean()
    hidden_avg_gated = hidden_before_gate * avg_gate_score
    abs_avg_gated = hidden_avg_gated.abs()
    results["avg_gated"] = {
        "mean_abs": abs_avg_gated.mean().item(),
        "max_abs": abs_avg_gated.max().item(),
    }
    for t in thresholds:
        results["avg_gated"][f"fraction_below_{t}"] = (abs_avg_gated < t).float().mean().item()
    
    return results


class GatingScoreMonitor:
    """
    Monitor gating scores during model forward passes.
    
    Captures gating scores from all layers and computes statistics
    for analysis as described in Sec 4.2.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_scores: List[Dict] = []
        self._hooks = []
    
    def register_hooks(self):
        """Register hooks to capture gating scores."""
        self.layer_scores = []
        
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'attn') and hasattr(layer.attn, 'gate'):
                gate = layer.attn.gate
                if gate is not None and gate.gate_proj is not None:
                    hook = gate.register_forward_hook(
                        self._make_gate_hook(i)
                    )
                    self._hooks.append(hook)
    
    def _make_gate_hook(self, layer_idx: int):
        """Create a hook for a specific layer."""
        def hook(module, input, output):
            # input[0] is y (tensor to be gated)
            # input[1] is x (input for computing gate scores)
            if len(input) >= 2:
                x = input[1]
                with torch.no_grad():
                    raw_scores = module.gate_proj(x)
                    
                    if module.activation.value == "sigmoid":
                        scores = torch.sigmoid(raw_scores)
                    elif module.activation.value == "ns_sigmoid":
                        scores = 0.5 + 0.5 * torch.sigmoid(raw_scores)
                    elif module.activation.value == "silu":
                        scores = F.silu(raw_scores)
                    else:
                        scores = raw_scores
                    
                    self.layer_scores.append({
                        "layer": layer_idx,
                        "stats": compute_sparsity_metrics(scores),
                    })
        return hook
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def get_summary(self) -> Dict:
        """Get summary statistics across all layers."""
        if not self.layer_scores:
            return {}
        
        means = [s["stats"]["mean"] for s in self.layer_scores]
        
        return {
            "num_layers": len(self.layer_scores),
            "mean_gating_score": sum(means) / len(means),
            "min_mean_score": min(means),
            "max_mean_score": max(means),
            "layer_scores": self.layer_scores,
        }


# Expected gating score statistics from the paper (Table 4)
# These are approximate values based on the paper's figures and text
EXPECTED_GATING_STATS = {
    "sdpa_elementwise_head_specific": {
        "mean": 0.15,  # Very sparse, most scores near 0
        "description": "SDPA elementwise head-specific (best variant)",
    },
    "sdpa_headwise_head_specific": {
        "mean": 0.18,
        "description": "SDPA headwise head-specific",
    },
    "value_elementwise_head_specific": {
        "mean": 0.35,  # Less sparse than SDPA
        "description": "Value elementwise head-specific",
    },
    "sdpa_elementwise_head_shared": {
        "mean": 0.45,  # Much less sparse when head-shared
        "description": "SDPA elementwise head-shared (worse performance)",
    },
    "sdpa_ns_sigmoid": {
        "mean": 0.75,  # Constrained to [0.5, 1.0]
        "description": "SDPA NS-sigmoid (non-sparse ablation)",
    },
}
