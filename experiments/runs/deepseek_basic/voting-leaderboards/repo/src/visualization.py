"""
Visualization utilities for reproducing paper figures.

Figures:
- Figure 2: PCA of BoW features for model responses
- Figure 3: Detection accuracy heatmap across prompt categories
- Figure 4: Scenario 1 - Likelihood-based detection
- Figure 5: Scenario 2 - Detection rate vs noise
- Figure 6: Utility loss vs noise
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import matplotlib.patches as mpatches


def plot_pca_visualization(
    projections: np.ndarray,
    labels: np.ndarray,
    model_names: List[str],
    prompt_title: str = "",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
):
    """
    Reproduce Figure 2: PCA visualization of BoW features.
    
    Shows the first two principal components of BoW features for
    model responses, demonstrating clear model-specific clustering.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_models = len(model_names)
    colors = plt.cm.tab20(np.linspace(0, 1, n_models))
    
    for i in range(n_models):
        mask = labels == i
        ax.scatter(
            projections[mask, 0],
            projections[mask, 1],
            c=[colors[i]],
            label=model_names[i],
            alpha=0.6,
            s=30,
            edgecolors='none',
        )
    
    ax.set_xlabel("First Principal Component")
    ax.set_ylabel("Second Principal Component")
    ax.set_title(f"PCA of BoW Features - {prompt_title}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_detection_accuracy_heatmap(
    accuracy_matrix: np.ndarray,
    model_names: List[str],
    prompt_categories: List[str],
    title: str = "Detection Accuracy (%) by Model and Prompt Category",
    figsize: Tuple[int, int] = (12, 8),
    vmin: float = 85,
    vmax: float = 100,
    save_path: Optional[str] = None,
):
    """
    Reproduce Figure 3: Detection accuracy heatmap.
    
    Shows test accuracy (%) of detectors across different 
    prompt categories and target models.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(accuracy_matrix, cmap='YlOrRd', aspect='auto', vmin=vmin, vmax=vmax)
    
    ax.set_xticks(range(len(prompt_categories)))
    ax.set_xticklabels(prompt_categories, rotation=45, ha='right')
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    
    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(prompt_categories)):
            text = ax.text(
                j, i, f"{accuracy_matrix[i, j]:.1f}",
                ha="center", va="center", color="black", fontsize=8
            )
    
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Accuracy (%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_malicious_detection_likelihood(
    benign_pvalues: List[float],
    malicious_pvalues: List[float],
    n_observations: int,
    alpha: float = 0.01,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
):
    """
    Reproduce Figure 4: Scenario 1 detection.
    
    Shows how likelihood-based detection identifies malicious users
    by comparing p-value distributions of benign vs malicious users.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram of p-values
    bins = np.linspace(0, 1, 30)
    ax1.hist(benign_pvalues, bins=bins, alpha=0.6, label='Benign users', color='green')
    ax1.hist(malicious_pvalues, bins=bins, alpha=0.6, label='Malicious users', color='red')
    ax1.axvline(x=alpha, color='black', linestyle='--', label=f'α={alpha}')
    ax1.set_xlabel('p-value')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of p-values')
    ax1.legend()
    
    # Cumulative distribution
    sorted_benign = np.sort(benign_pvalues)
    sorted_malicious = np.sort(malicious_pvalues)
    
    ax2.plot(sorted_benign, np.linspace(0, 1, len(sorted_benign)), 
             label='Benign users', color='green')
    ax2.plot(sorted_malicious, np.linspace(0, 1, len(sorted_malicious)),
             label='Malicious users', color='red')
    ax2.axvline(x=alpha, color='black', linestyle='--', label=f'α={alpha}')
    ax2.set_xlabel('p-value')
    ax2.set_ylabel('CDF')
    ax2.set_title('Cumulative Distribution of p-values')
    ax2.legend()
    
    fig.suptitle(f'Malicious User Detection (n={n_observations} observations)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, (ax1, ax2)


def plot_detection_vs_noise(
    noise_scales: List[float],
    detection_rates: List[float],
    utility_losses: List[float],
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
):
    """
    Reproduce Figures 5 and 6: Effect of noise on detection and utility.
    
    Shows how increasing perturbation noise affects:
    - Detection rate of malicious users (Fig 5)
    - Utility loss (Fig 6)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Figure 5: Detection rate vs noise
    ax1.plot(noise_scales, detection_rates, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Noise Scale (σ)')
    ax1.set_ylabel('Detection Rate')
    ax1.set_title('Detection Rate vs Noise (Fig 5)')
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Figure 6: Utility loss vs noise
    ax2.plot(noise_scales, utility_losses, 'r-s', linewidth=2, markersize=8)
    ax2.set_xlabel('Noise Scale (σ)')
    ax2.set_ylabel('Average Absolute Rank Change')
    ax2.set_title('Utility Loss vs Noise (Fig 6)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, (ax1, ax2)


def plot_rank_trajectory(
    rank_history: np.ndarray,
    model_names: List[str],
    target_model: str,
    title: str = "Rank Trajectory During Attack",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Plot rank trajectories of models during attack simulation.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_models = len(model_names)
    target_idx = model_names.index(target_model)
    
    for i in range(n_models):
        if i == target_idx:
            ax.plot(rank_history[:, i], linewidth=2.5, label=model_names[i], color='red')
        else:
            ax.plot(rank_history[:, i], alpha=0.3, linewidth=1, color='gray')
    
    ax.set_xlabel('Simulation Step (x1000 interactions)')
    ax.set_ylabel('Rank')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.invert_yaxis()  # Rank 1 at top
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_detector_accuracy_comparison(
    model_names: List[str],
    accuracy_by_feature: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
):
    """
    Reproduce Table 3 as a bar chart: detector accuracy by feature type.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    feature_types = list(accuracy_by_feature.keys())
    n_models = len(model_names)
    n_features = len(feature_types)
    
    x = np.arange(n_models)
    width = 0.8 / n_features
    
    for i, ft in enumerate(feature_types):
        offset = (i - n_features / 2 + 0.5) * width
        ax.bar(x + offset, accuracy_by_feature[ft], width, label=ft)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Detection Accuracy (%)')
    ax.set_title('Detector Performance by Feature Type (Table 3)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_cost_analysis(
    cost_breakdowns: List[Dict],
    defense_names: List[str],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Visualize attack cost under different defense configurations (Section 4.1).
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_defenses = len(defense_names)
    
    detector_costs = [c['detector_cost'] for c in cost_breakdowns]
    account_costs = [c['account_cost'] for c in cost_breakdowns]
    action_costs = [c['action_cost'] for c in cost_breakdowns]
    
    x = np.arange(n_defenses)
    width = 0.6
    
    p1 = ax.bar(x, detector_costs, width, label='Detector Cost', color='#1f77b4')
    p2 = ax.bar(x, account_costs, width, bottom=detector_costs, 
                label='Account Cost', color='#ff7f0e')
    p3 = ax.bar(x, action_costs, width, 
                bottom=[a+b for a, b in zip(detector_costs, account_costs)],
                label='Action Cost', color='#2ca02c')
    
    ax.set_xlabel('Defense Configuration')
    ax.set_ylabel('Cost (USD)')
    ax.set_title('Attack Cost Under Different Defense Configurations')
    ax.set_xticks(x)
    ax.set_xticklabels(defense_names, rotation=30, ha='right')
    ax.legend()
    
    # Add total cost labels
    for i, c in enumerate(cost_breakdowns):
        total = c['total_cost']
        ax.text(i, total + max(c['total_cost'] for c in cost_breakdowns) * 0.02,
                f'${total:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax
