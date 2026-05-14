"""
Analysis tools for gated attention models.

Implements the analysis described in Sections 4.2-4.4 and Appendices A.2-A.3:
  - Attention sink ratio (first-token attention proportion)
  - Massive activation detection
  - Gating score statistics (mean, distribution, sparsity)
  - SDPA output sparsity measurements
  - Attention map visualization data
  - Long-context evaluation support
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_attention_sink_ratio(
    attention_weights: torch.Tensor,
    sink_token_idx: int = 0,
) -> float:
    """Compute the proportion of attention allocated to the sink token.

    Following Sec 4.3 and Fig. 2: measures the fraction of attention
    scores directed to the first (or specified) token, averaged over
    all heads and tokens.

    Args:
        attention_weights: (batch, num_heads, seq_len, seq_len) or
                          (num_heads, seq_len, seq_len)
        sink_token_idx: Index of the token to consider as sink (default: 0)

    Returns:
        Float between 0 and 1 representing the sink ratio
    """
    # Average over batch if present
    if attention_weights.dim() == 4:
        attn = attention_weights.mean(dim=0)  # (num_heads, seq_len, seq_len)
    else:
        attn = attention_weights

    num_heads, seq_len, _ = attn.shape
    # Attention to the sink token (e.g., first token)
    sink_attention = attn[:, :, sink_token_idx]  # (num_heads, seq_len)

    # Average over heads and tokens
    return sink_attention.mean().item()


def compute_layerwise_attention_sink(
    all_attention_weights: List[torch.Tensor],
    sink_token_idx: int = 0,
) -> List[float]:
    """Compute attention sink ratio for each layer.

    Args:
        all_attention_weights: List of attention weight tensors per layer
        sink_token_idx: Index of the sink token

    Returns:
        List of sink ratios per layer
    """
    return [
        compute_attention_sink_ratio(attn, sink_token_idx)
        for attn in all_attention_weights
    ]


def compute_massive_activation(
    hidden_states: torch.Tensor,
    activation_threshold: Optional[float] = None,
    return_max: bool = True,
) -> Dict:
    """Detect massive activations in hidden states.

    Following the analysis in Sec 4.3 and Appendix A.3, computes
    the maximum absolute activation values and optionally the
    fraction of values exceeding a threshold.

    Args:
        hidden_states: (batch, seq_len, d_model) or (seq_len, d_model)
        activation_threshold: Optional threshold for massive activation detection
        return_max: Whether to return the max activation value

    Returns:
        Dict with 'max_activation' and optionally 'massive_fraction'
    """
    if hidden_states.dim() == 3:
        hs = hidden_states.view(-1, hidden_states.shape[-1])
    else:
        hs = hidden_states

    result = {}
    if return_max:
        result["max_activation"] = hs.abs().max().item()

    if activation_threshold is not None:
        massive_mask = hs.abs() > activation_threshold
        result["massive_fraction"] = massive_mask.float().mean().item()

    return result


def compute_layerwise_massive_activations(
    all_hidden_states: List[torch.Tensor],
    activation_threshold: float = 100.0,
) -> List[Dict]:
    """Compute massive activation statistics for each layer."""
    return [
        compute_massive_activation(hs, activation_threshold)
        for hs in all_hidden_states
    ]


def compute_gate_score_statistics(
    gate_scores: torch.Tensor,
) -> Dict[str, float]:
    """Compute statistics of gating scores.

    Following Sec 4.2 and Table 4 analysis:
      - Mean gate score
      - Fraction of scores below threshold (sparsity)
      - Standard deviation

    Args:
        gate_scores: Tensor of gating scores of any shape

    Returns:
        Dict with 'mean', 'std', 'sparsity_0_1', 'sparsity_0_5', etc.
    """
    scores = gate_scores.float()

    stats = {
        "mean": scores.mean().item(),
        "std": scores.std().item(),
        "min": scores.min().item(),
        "max": scores.max().item(),
    }

    # Sparsity at various thresholds
    for threshold in [0.01, 0.05, 0.1, 0.2, 0.5]:
        stats[f"sparsity_{threshold}"] = (scores < threshold).float().mean().item()

    return stats


def compute_layerwise_gate_statistics(
    all_gate_scores: List[torch.Tensor],
) -> List[Dict]:
    """Compute gate score statistics per layer.

    Corresponds to the layer-wise analysis shown in Appendix A.4.
    """
    return [compute_gate_score_statistics(gs) for gs in all_gate_scores]


def compute_sparsity_ratio(
    tensor: torch.Tensor,
    threshold: float = 0.01,
) -> float:
    """Compute the fraction of values below a threshold (sparsity).

    Used in Appendix A.2 (Fig. 5) to measure SDPA output sparsity
    before and after gating.

    Args:
        tensor: Any tensor
        threshold: Values with absolute value below this are considered sparse

    Returns:
        Fraction of values below threshold
    """
    return (tensor.abs() < threshold).float().mean().item()


def compute_sparsity_comparison(
    pre_gate: torch.Tensor,
    post_gate: torch.Tensor,
    gate_scores: torch.Tensor,
    thresholds: List[float] = [0.01, 0.001],
) -> Dict[str, float]:
    """Compare sparsity before and after gating.

    Replicates the analysis in Appendix A.2 (Fig. 5):
      - Sparsity of pre-gating hidden states
      - Sparsity of post-gating hidden states
      - Sparsity when multiplying pre-gating by mean gate score
    """
    results = {}
    mean_gate = gate_scores.mean().item()

    for t in thresholds:
        results[f"pre_gate_sparsity_{t}"] = compute_sparsity_ratio(pre_gate, t)
        results[f"post_gate_sparsity_{t}"] = compute_sparsity_ratio(post_gate, t)
        # Control: multiply pre-gate by constant mean gate score
        constant_gated = pre_gate * mean_gate
        results[f"constant_gated_sparsity_{t}"] = compute_sparsity_ratio(constant_gated, t)

    return results


def compute_attention_maps(
    attention_weights: torch.Tensor,
    average_heads: bool = True,
) -> torch.Tensor:
    """Compute attention maps for visualization.

    Following Fig. 2 (right), computes average attention weights
    for attention pattern visualization.

    Args:
        attention_weights: (batch, num_heads, seq_len, seq_len)
        average_heads: If True, average over heads

    Returns:
        Attention map: (seq_len, seq_len) if average_heads,
                      (num_heads, seq_len, seq_len) otherwise
    """
    if attention_weights.dim() == 4:
        attn = attention_weights.mean(dim=0)  # average batch
    else:
        attn = attention_weights

    if average_heads:
        attn = attn.mean(dim=0)  # average heads

    return attn


class AttentionAnalyzer:
    """Comprehensive attention analysis for gated LLMs.

    Orchestrates the collection and analysis of:
      - Attention sink patterns (Sec 4.3, Fig. 2)
      - Massive activations (Sec 4.3, Appendix A.3)
      - Gate score statistics (Sec 4.2, Table 4, Appendix A.4)
      - SDPA output sparsity (Appendix A.2, Fig. 4-5)
    """

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def analyze(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sparsity_thresholds: List[float] = [0.01, 0.001, 0.1],
        massive_activation_threshold: float = 100.0,
    ) -> Dict:
        """Run full analysis on a batch of inputs.

        Returns a comprehensive analysis dict with:
          - layerwise_attention_sink: sink ratio per layer
          - layerwise_massive_activations: max activation per layer
          - layerwise_gate_stats: gate score statistics per layer
          - overall_summary: aggregated metrics
        """
        # Forward pass with all intermediate outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )

        attentions = outputs["attentions"]
        hidden_states = outputs["hidden_states"]

        # Attention sink analysis
        sink_ratios = compute_layerwise_attention_sink(attentions)
        avg_sink = sum(sink_ratios) / len(sink_ratios) if sink_ratios else 0.0

        # Massive activation analysis
        massive_stats = compute_layerwise_massive_activations(
            hidden_states, massive_activation_threshold
        )

        # Gate score statistics (extracted from model intermediate outputs)
        gate_stats = self._collect_gate_statistics(input_ids)

        return {
            "layerwise_attention_sink": sink_ratios,
            "average_sink_ratio": avg_sink,
            "layerwise_massive_activations": massive_stats,
            "layerwise_gate_stats": gate_stats,
        }

    def _collect_gate_statistics(
        self, input_ids: torch.Tensor
    ) -> List[Dict]:
        """Collect gate score statistics by hooking into the model."""
        gate_scores_list = []

        # Register hooks to capture gate scores
        hooks = []

        def make_hook(collector_list):
            def hook(module, input, output):
                # For gate projections, capture the output (gating scores)
                if hasattr(module, "_gate_scores"):
                    collector_list.append(module._gate_scores.detach().cpu())
            return hook

        for layer in self.model.layers:
            if hasattr(layer.attn, 'gate_proj') and layer.attn.gate_proj is not None:
                hooks.append(
                    layer.attn.gate_proj.register_forward_hook(
                        make_hook(gate_scores_list)
                    )
                )

        # Run forward pass
        self.model(input_ids)

        # Remove hooks
        for h in hooks:
            h.remove()

        if gate_scores_list:
            return compute_layerwise_gate_statistics(gate_scores_list)
        return []


# ---------------------------------------------------------------------------
# Long-context evaluation utilities (Sec 4.4)
# ---------------------------------------------------------------------------

def extend_rope_base(model, new_base: float = 1_000_000.0):
    """Extend RoPE base frequency for long-context adaptation.

    Following Sec 4.4: increase RoPE base from 10k to 1M.
    """
    model.config.rope_base = new_base
    if hasattr(model, 'rotary_emb'):
        model.rotary_emb.base = new_base
        inv_freq = 1.0 / (
            new_base ** (torch.arange(0, model.rotary_emb.dim, 2).float() / model.rotary_emb.dim)
        )
        model.rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)


def apply_yarn_scaling(
    model,
    original_max_len: int = 32000,
    extended_max_len: int = 128000,
    scale: float = None,
):
    """Apply YaRN (Peng et al., 2023) for context extension.

    Following Sec 4.4: context length extended from 32k to 128k using YaRN.
    """
    if scale is None:
        scale = extended_max_len / original_max_len

    # Adjust RoPE frequencies for YaRN
    if hasattr(model, 'rotary_emb'):
        dim = model.rotary_emb.dim
        base = model.rotary_emb.base
        # YaRN scaling factor
        yarn_scale = scale ** (dim / (dim - 2))
        new_base = base * yarn_scale
        model.rotary_emb.base = new_base

        inv_freq = 1.0 / (
            new_base ** (torch.arange(0, dim, 2).float() / dim)
        )
        model.rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)


# ---------------------------------------------------------------------------
# RULER benchmark evaluation helper (Sec 4.4, Table 5)
# ---------------------------------------------------------------------------

def evaluate_ruler_format(
    model,
    tokenizer,
    seq_lengths: List[int] = [4096, 8192, 16384, 32768, 65536, 131072],
    num_samples: int = 100,
) -> Dict[int, float]:
    """Evaluate model on RULER-like long-context tasks.

    Following Table 5: measures model performance across
    varying sequence lengths.
    """
    results = {}
    # RULER-style evaluation would go here
    # This is a placeholder for the evaluation framework
    return results
