"""
Visualization for OLMoE Analysis

Creates plots corresponding to Figures 20-23 in the paper:
- Figure 20: Router saturation during pretraining
- Figure 21: Expert co-activation heatmap
- Figure 22: Domain specialization
- Figure 23: Vocabulary specialization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional, Tuple


def plot_router_saturation(
    saturation_results: Dict[str, Dict[str, float]],
    num_layers: int = 16,
    save_path: Optional[str] = None,
):
    """
    Plot router saturation during pretraining (Figure 20).

    Shows saturation at different checkpoints (1%, 10%, 20%, 40% of pretraining)
    compared to the final checkpoint.

    Args:
        saturation_results: Dict mapping checkpoint_name -> {layer_k8: float, layer_k1: float}
        num_layers: Number of transformer layers
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    checkpoint_names = list(saturation_results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(checkpoint_names)))

    for ax_idx, k_suffix in enumerate(["k1", "k8"]):
        ax = axes[ax_idx]

        for ckpt_name, color in zip(checkpoint_names, colors):
            saturation = saturation_results[ckpt_name]
            layer_scores = [
                saturation.get(f"layer_{i}_{k_suffix}", 0)
                for i in range(num_layers)
            ]
            ax.plot(
                range(num_layers),
                layer_scores,
                label=ckpt_name,
                color=color,
                marker='o',
                markersize=4,
            )

        k = 1 if k_suffix == "k1" else 8
        random_baseline = 1/64 if k == 1 else 8/64
        ax.axhline(
            y=random_baseline,
            color='gray',
            linestyle='--',
            alpha=0.5,
            label=f'Random ({random_baseline:.3f})'
        )

        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Router Saturation')
        ax.set_title(f'Router Saturation (k={k})')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Router Saturation During Pretraining\n(Measured on 0.5% of C4 validation)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    return fig


def plot_expert_coactivation(
    coactivation_matrix: np.ndarray,
    layer_idx: int,
    top_n: int = 32,
    save_path: Optional[str] = None,
):
    """
    Plot expert co-activation heatmap (Figure 21).

    Shows the 32 experts with highest maximum co-activation score.

    Args:
        coactivation_matrix: [num_experts, num_experts] co-activation matrix
        layer_idx: Layer index being visualized
        top_n: Number of experts to display
        save_path: Path to save the figure
    """
    # Select top-N experts by maximum co-activation
    max_coact = coactivation_matrix.max(axis=1)
    top_expert_indices = np.argsort(max_coact)[-top_n:][::-1]

    # Extract submatrix
    sub_matrix = coactivation_matrix[np.ix_(top_expert_indices, top_expert_indices)]

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(sub_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Co-activation Rate')

    ax.set_xticks(range(top_n))
    ax.set_yticks(range(top_n))
    ax.set_xticklabels(top_expert_indices, rotation=90, fontsize=8)
    ax.set_yticklabels(top_expert_indices, fontsize=8)

    ax.set_xlabel('Expert ID')
    ax.set_ylabel('Expert ID')
    ax.set_title(f'Expert Co-activation (Layer {layer_idx})\n'
                 f'Top {top_n} experts by max co-activation')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    return fig


def plot_domain_specialization(
    domain_specialization: Dict[str, np.ndarray],
    num_layers: int = 16,
    num_experts: int = 64,
    k: int = 8,
    save_path: Optional[str] = None,
):
    """
    Plot domain specialization heatmap (Figure 22).

    Shows how often tokens from different domains get routed to each expert.
    Horizontal gray lines correspond to random chance (k/num_experts).

    Args:
        domain_specialization: Dict mapping domain -> [num_layers, num_experts]
        num_layers: Number of transformer layers
        num_experts: Total number of experts
        k: Number of activated experts
        save_path: Path to save the figure
    """
    domains = list(domain_specialization.keys())
    n_domains = len(domains)

    # Select a few representative layers to display
    display_layers = [0, 7, 15]  # First, middle, last

    fig, axes = plt.subplots(
        n_domains, len(display_layers),
        figsize=(4 * len(display_layers), 3 * n_domains)
    )

    if n_domains == 1:
        axes = axes[np.newaxis, :]
    if len(display_layers) == 1:
        axes = axes[:, np.newaxis]

    random_baseline = k / num_experts

    for domain_idx, domain in enumerate(domains):
        spec = domain_specialization[domain]  # [num_layers, num_experts]

        for layer_plot_idx, layer_idx in enumerate(display_layers):
            ax = axes[domain_idx, layer_plot_idx]

            layer_spec = spec[layer_idx]  # [num_experts]

            # Sort experts by specialization for better visualization
            sorted_indices = np.argsort(layer_spec)[::-1]
            sorted_spec = layer_spec[sorted_indices]

            ax.bar(range(num_experts), sorted_spec, color='steelblue', alpha=0.7)
            ax.axhline(
                y=random_baseline,
                color='gray',
                linestyle='--',
                alpha=0.7,
                label=f'Random ({random_baseline:.3f})'
            )

            ax.set_xlabel('Expert (sorted by specialization)')
            ax.set_ylabel('Specialization')
            ax.set_title(f'{domain} - Layer {layer_idx}')
            ax.set_ylim(0, 1)

    plt.suptitle(f'Domain Specialization (k={k})\n'
                 f'Gray line = random chance ({random_baseline:.3f})')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    return fig


def plot_vocabulary_specialization(
    vocab_specialization: Dict[int, Dict[int, np.ndarray]],
    num_layers: int = 16,
    num_experts: int = 64,
    save_path: Optional[str] = None,
):
    """
    Plot vocabulary specialization (Figure 23).

    Shows vocabulary specialization per layer (averaged over experts) and
    per expert for a specific layer.

    Args:
        vocab_specialization: Dict mapping layer_idx -> {token_id: expert_scores}
        num_layers: Number of transformer layers
        num_experts: Total number of experts
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Average specialization per layer
    ax_left = axes[0]
    layer_avg_spec = []

    for layer_idx in range(num_layers):
        if layer_idx not in vocab_specialization:
            layer_avg_spec.append(0)
            continue

        layer_spec = vocab_specialization[layer_idx]
        if not layer_spec:
            layer_avg_spec.append(0)
            continue

        # Average max specialization across all tokens
        max_specs = [scores.max() for scores in layer_spec.values()]
        layer_avg_spec.append(np.mean(max_specs))

    ax_left.plot(range(num_layers), layer_avg_spec, 'b-o', markersize=6)
    ax_left.set_xlabel('Layer Index')
    ax_left.set_ylabel('Average Max Vocabulary Specialization')
    ax_left.set_title('Vocabulary Specialization per Layer\n(averaged over experts)')
    ax_left.set_ylim(0, 1)
    ax_left.grid(True, alpha=0.3)

    # Right: Per-expert specialization for a specific layer (layer 7 as in paper)
    ax_right = axes[1]
    target_layer = 7

    if target_layer in vocab_specialization:
        layer_spec = vocab_specialization[target_layer]

        # Compute max specialization per expert
        expert_max_spec = np.zeros(num_experts)
        for token_id, scores in layer_spec.items():
            for expert_idx in range(num_experts):
                if scores[expert_idx] > expert_max_spec[expert_idx]:
                    expert_max_spec[expert_idx] = scores[expert_idx]

        # Display first 32 experts
        display_experts = min(32, num_experts)
        ax_right.bar(
            range(display_experts),
            expert_max_spec[:display_experts],
            color='steelblue',
            alpha=0.7
        )

        # Add average line
        avg_spec = np.mean(expert_max_spec[:display_experts])
        ax_right.axhline(
            y=avg_spec,
            color='red',
            linestyle='--',
            alpha=0.7,
            label=f'Average ({avg_spec:.3f})'
        )

        ax_right.set_xlabel('Expert ID')
        ax_right.set_ylabel('Max Vocabulary Specialization')
        ax_right.set_title(f'Vocabulary Specialization per Expert (Layer {target_layer})')
        ax_right.set_ylim(0, 1)
        ax_right.legend()
        ax_right.grid(True, alpha=0.3)

    plt.suptitle('Vocabulary Specialization of OLMoE-1B-7B')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    return fig


def plot_load_balancing_comparison(
    with_lbl_expert_counts: np.ndarray,
    without_lbl_expert_counts: np.ndarray,
    layer_idx: int = 0,
    save_path: Optional[str] = None,
):
    """
    Plot expert assignment with and without load balancing loss (Figure 10).

    Args:
        with_lbl_expert_counts: [num_steps, num_experts] expert activation counts with LBL
        without_lbl_expert_counts: [num_steps, num_experts] without LBL
        layer_idx: Layer being visualized
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, counts, title in zip(
        axes,
        [without_lbl_expert_counts, with_lbl_expert_counts],
        ['Without Load Balancing Loss', 'With Load Balancing Loss']
    ):
        num_steps, num_experts = counts.shape
        steps = np.arange(num_steps)

        # Normalize to get fractions
        total_per_step = counts.sum(axis=1, keepdims=True)
        fractions = counts / (total_per_step + 1e-8)

        # Plot stacked area chart
        colors = plt.cm.tab20(np.linspace(0, 1, num_experts))
        ax.stackplot(steps, fractions.T, colors=colors, alpha=0.8)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Fraction of Tokens')
        ax.set_title(f'{title}\n(Layer {layer_idx})')
        ax.set_ylim(0, 1)

    plt.suptitle('Expert Assignment During Training')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    return fig
