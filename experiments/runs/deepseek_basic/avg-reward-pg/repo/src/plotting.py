"""
Plotting utilities for reproducing the figures from the paper.

Generates plots corresponding to:
- Figure 1: Convergence with MDP complexity (reward vs iterations)
- Figure 2: Convergence as a function of C_p (transition kernel impact)

Uses matplotlib for visualization. This module produces publication-quality plots
similar to those in the paper.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import os

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def setup_plotting_style():
    """Set up a clean plotting style."""
    if not HAS_MATPLOTLIB:
        return
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'lines.linewidth': 2,
        'grid.alpha': 0.3,
    })


def plot_experiment_1(results: Dict, save_path: Optional[str] = None):
    """
    Plot Experiment 1: Convergence with different state/action space sizes.
    
    Corresponds to Figure 1(a) from the paper.
    
    Args:
        results: Output from experiment_1_state_action_size()
        save_path: If provided, save figure to this path
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return
    
    setup_plotting_style()
    fig, ax = plt.subplots()
    
    for (n_states, n_actions), data in results.items():
        rewards = data['rewards']
        label = f'|S|={n_states}, |A|={n_actions}'
        ax.plot(rewards, label=label, alpha=0.8)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Reward ρ^π')
    ax.set_title('Convergence of PG with Different State/Action Space Sizes')
    ax.legend()
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_experiment_2(results: Dict, save_path: Optional[str] = None):
    """
    Plot Experiment 2: Convergence with different reward variances.
    
    Corresponds to Figure 1(b) from the paper.
    
    Args:
        results: Output from experiment_2_reward_variance()
        save_path: If provided, save figure to this path
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return
    
    setup_plotting_style()
    fig, ax = plt.subplots()
    
    variance_labels = {
        'none': 'No variance',
        'low': 'Low variance',
        'high': 'High variance',
        'max': 'Max variance',
    }
    
    for var_level, data in results.items():
        rewards = data['rewards']
        label = variance_labels.get(var_level, var_level)
        ax.plot(rewards, label=label, alpha=0.8)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Reward ρ^π')
    ax.set_title('Convergence of PG with Different Reward Variances')
    ax.legend()
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_experiment_3(results: Dict, save_path: Optional[str] = None):
    """
    Plot Experiment 3: Convergence with different transition kernels.
    
    Corresponds to Figure 2 from the paper.
    
    Args:
        results: Output from experiment_3_transition_kernel()
        save_path: If provided, save figure to this path
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return
    
    setup_plotting_style()
    fig, ax = plt.subplots()
    
    for ktype, data in results.items():
        rewards = data['rewards']
        ax.plot(rewards, label=ktype, alpha=0.8)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Reward ρ^π')
    ax.set_title('Convergence of PG with Different Transition Kernels')
    ax.legend()
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_optimality_gap(results: Dict, theoretical_bounds: Optional[List[float]] = None,
                        save_path: Optional[str] = None):
    """
    Plot the empirical optimality gap against theoretical bounds.
    
    Useful for validating Theorem 1.
    
    Args:
        results: Output from run_projected_policy_gradient()
        theoretical_bounds: Optional list of theoretical bounds
        save_path: If provided, save figure to this path
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return
    
    setup_plotting_style()
    fig, ax = plt.subplots()
    
    gaps = results['optimality_gaps']
    iterations = range(len(gaps))
    
    ax.semilogy(iterations, gaps, label='Empirical', linewidth=2)
    
    if theoretical_bounds is not None:
        ax.semilogy(iterations, theoretical_bounds, '--', 
                   label='Theoretical bound (Theorem 1)', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Iteration k')
    ax.set_ylabel('Optimality Gap ρ* - ρ^{π_k}')
    ax.set_title('Convergence of Projected Policy Gradient')
    ax.legend()
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_all_figures(results_dir: str = 'figures'):
    """
    Generate and save all figures from the paper's experiments.
    
    Args:
        results_dir: Directory to save figures
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, cannot save figures")
        return
    
    os.makedirs(results_dir, exist_ok=True)
    
    from .simulations import (
        experiment_1_state_action_size,
        experiment_2_reward_variance,
        experiment_3_transition_kernel,
    )
    
    print("Running simulations for figures...")
    
    # Experiment 1
    print("  Experiment 1...")
    exp1 = experiment_1_state_action_size(
        sizes=[(3, 3), (9, 9), (81, 81)],
        n_iterations=500,
        eta=0.1,
        seed=42
    )
    plot_experiment_1(exp1, os.path.join(results_dir, 'fig1a_state_action_size.pdf'))
    
    # Experiment 2
    print("  Experiment 2...")
    exp2 = experiment_2_reward_variance(
        n_states=16, n_actions=16,
        n_iterations=1000,
        eta=0.1,
        seed=42
    )
    plot_experiment_2(exp2, os.path.join(results_dir, 'fig1b_reward_variance.pdf'))
    
    # Experiment 3
    print("  Experiment 3...")
    exp3 = experiment_3_transition_kernel(
        n_states=16, n_actions=16,
        n_iterations=1000,
        eta=0.1,
        seed=42
    )
    plot_experiment_3(exp3, os.path.join(results_dir, 'fig2_transition_kernel.pdf'))
    
    print(f"All figures saved to {results_dir}/")
