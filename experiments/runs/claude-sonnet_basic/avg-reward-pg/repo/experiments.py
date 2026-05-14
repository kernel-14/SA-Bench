"""
Experiments for reproducing the simulations in:
"Global Convergence of Policy Gradient in Average Reward MDPs"

Reproduces:
- Figure 1(a): Convergence with different state/action space sizes (S,A) in {(3,3),(9,9),(81,81)}
- Figure 1(b): Convergence with different reward variances, fixed (S,A)=(16,16)
- Figure 2: Convergence with different transition kernels, fixed (S,A)=(16,16)

Paper details:
- Figure 1(a): 2000 iterations, plots average reward vs iteration
- Figure 1(b): 2000 iterations, plots average reward vs iteration
- Figure 2: 3000 iterations, plots change in average reward vs iteration
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from typing import List, Dict

from mdp import AverageRewardMDP
from policy_gradient import projected_policy_gradient, compute_suboptimality_gap
from mdp_construction import (
    make_mdp_varying_size,
    make_mdp_varying_reward_variance,
    make_mdp_varying_transition,
)


def run_experiment_1a(
    num_iterations: int = 2000,
    eta: float = 0.01,
    seed: int = 42,
) -> Dict:
    """
    Experiment 1a (Figure 1a): Convergence with different state/action space sizes.
    MDPs: (S, A) in {(3,3), (9,9), (81,81)}
    """
    print("Running Experiment 1a: Varying state/action space sizes...")
    sizes = [(3, 3), (9, 9), (81, 81)]
    results = {}
    
    for S, A in sizes:
        print(f"  (S, A) = ({S}, {A})...")
        mdp = make_mdp_varying_size(S, A, seed=seed)
        _, rho_star = mdp.get_optimal_policy()
        rewards, _ = projected_policy_gradient(
            mdp, eta=eta, num_iterations=num_iterations, seed=seed
        )
        gaps = compute_suboptimality_gap(mdp, rewards, rho_star=rho_star)
        results[(S, A)] = {
            'rewards': rewards,
            'gaps': gaps,
            'rho_star': rho_star,
        }
        print(f"    rho* = {rho_star:.4f}, initial gap = {gaps[0]:.4f}, final gap = {gaps[-1]:.6f}")
    
    return results


def run_experiment_1b(
    num_iterations: int = 2000,
    eta: float = 0.01,
    seed: int = 42,
) -> Dict:
    """
    Experiment 1b (Figure 1b): Convergence with different reward variances.
    Fixed (S, A) = (16, 16). Four reward variance levels.
    """
    print("Running Experiment 1b: Varying reward variances...")
    S, A = 16, 16
    variance_types = ['no_variance', 'low_variance', 'high_variance', 'max_variance']
    results = {}
    
    for vtype in variance_types:
        print(f"  Variance type: {vtype}...")
        mdp = make_mdp_varying_reward_variance(S, A, vtype, seed=seed)
        _, rho_star = mdp.get_optimal_policy()
        rewards, _ = projected_policy_gradient(
            mdp, eta=eta, num_iterations=num_iterations, seed=seed
        )
        gaps = compute_suboptimality_gap(mdp, rewards, rho_star=rho_star)
        results[vtype] = {
            'rewards': rewards,
            'gaps': gaps,
            'rho_star': rho_star,
        }
        print(f"    rho* = {rho_star:.4f}, initial gap = {gaps[0]:.4f}, final gap = {gaps[-1]:.6f}")
    
    return results


def run_experiment_2(
    num_iterations: int = 3000,
    eta: float = 0.01,
    seed: int = 42,
) -> Dict:
    """
    Experiment 2 (Figure 2): Convergence with different transition kernels.
    Fixed (S, A) = (16, 16). Three transition kernel types.
    """
    print("Running Experiment 2: Varying transition kernels...")
    S, A = 16, 16
    kernel_types = ['uniform', 'non_uniform', 'deterministic']
    results = {}
    
    for ktype in kernel_types:
        print(f"  Kernel type: {ktype}...")
        mdp = make_mdp_varying_transition(S, A, ktype, seed=seed)
        _, rho_star = mdp.get_optimal_policy()
        rewards, _ = projected_policy_gradient(
            mdp, eta=eta, num_iterations=num_iterations, seed=seed
        )
        gaps = compute_suboptimality_gap(mdp, rewards, rho_star=rho_star)
        results[ktype] = {
            'rewards': rewards,
            'gaps': gaps,
            'rho_star': rho_star,
        }
        print(f"    rho* = {rho_star:.4f}, initial gap = {gaps[0]:.4f}, final gap = {gaps[-1]:.6f}")
    
    return results


def plot_figure_1(results_1a: Dict, results_1b: Dict, save_dir: str = 'figures'):
    """
    Plot Figure 1: two subplots (a) and (b) side by side.
    (a) Average reward vs iteration for different (S, A) sizes.
    (b) Average reward vs iteration for different reward variances.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Figure 1(a) ---
    ax = axes[0]
    colors_a = ['tab:blue', 'tab:orange', 'tab:green']
    labels_a = {
        (3, 3): '(S, A) = (3, 3)',
        (9, 9): '(S, A) = (9, 9)',
        (81, 81): '(S, A) = (81, 81)',
    }
    for (S, A), color in zip([(3, 3), (9, 9), (81, 81)], colors_a):
        if (S, A) not in results_1a:
            continue
        rewards = results_1a[(S, A)]['rewards']
        ax.plot(np.arange(len(rewards)), rewards, color=color,
                label=labels_a[(S, A)], linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('(a) Different State/Action Space Sizes', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # --- Figure 1(b) ---
    ax = axes[1]
    colors_b = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    labels_b = {
        'no_variance': 'No Variance',
        'low_variance': 'Low Variance',
        'high_variance': 'High Variance',
        'max_variance': 'Max Variance',
    }
    for vtype, color in zip(['no_variance', 'low_variance', 'high_variance', 'max_variance'], colors_b):
        if vtype not in results_1b:
            continue
        rewards = results_1b[vtype]['rewards']
        ax.plot(np.arange(len(rewards)), rewards, color=color,
                label=labels_b[vtype], linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('(b) Different Reward Variances (S=A=16)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Figure 1: Improvement in Average Reward as a Function of MDP Complexity',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    
    path = os.path.join(save_dir, 'figure_1.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved Figure 1 to {path}")
    
    # Also save individual subplots
    for suffix, results, colors, labels, title in [
        ('1a', results_1a,
         {(3,3): 'tab:blue', (9,9): 'tab:orange', (81,81): 'tab:green'},
         labels_a, 'Different State/Action Space Sizes'),
        ('1b', results_1b,
         {'no_variance': 'tab:blue', 'low_variance': 'tab:orange',
          'high_variance': 'tab:green', 'max_variance': 'tab:red'},
         labels_b, 'Different Reward Variances (S=A=16)'),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for key, color in colors.items():
            if key not in results:
                continue
            rewards = results[key]['rewards']
            ax.plot(np.arange(len(rewards)), rewards, color=color,
                    label=labels[key], linewidth=2)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'figure_{suffix}.png'), dpi=150, bbox_inches='tight')
        plt.close()


def plot_figure_2(results: Dict, save_dir: str = 'figures'):
    """
    Plot Figure 2: Change in average reward vs iteration for different transition kernels.
    The paper plots the "overall change in average reward" (i.e., |rho_{k+1} - rho_k|).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = {'uniform': 'tab:blue', 'non_uniform': 'tab:orange', 'deterministic': 'tab:green'}
    labels = {'uniform': 'Uniform', 'non_uniform': 'Non-Uniform', 'deterministic': 'Deterministic'}
    
    for ktype in ['uniform', 'non_uniform', 'deterministic']:
        if ktype not in results:
            continue
        rewards = np.array(results[ktype]['rewards'])
        delta = np.abs(np.diff(rewards))
        # Smooth slightly for visibility
        ax.semilogy(np.arange(1, len(delta) + 1), delta + 1e-10,
                    color=colors[ktype], label=labels[ktype], linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('|Δ Average Reward|', fontsize=12)
    ax.set_title('Figure 2: Convergence as a Function of $C_p$ (S=A=16)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'figure_2.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved Figure 2 to {path}")


def plot_suboptimality_gaps(results_1a: Dict, results_1b: Dict, results_2: Dict,
                             save_dir: str = 'figures'):
    """
    Additional plots: suboptimality gaps (rho* - rho^{pi_k}) for all experiments.
    These show the convergence rate more clearly.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Suboptimality for Experiment 1a
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for (S, A), color in zip([(3, 3), (9, 9), (81, 81)], colors):
        if (S, A) not in results_1a:
            continue
        gaps = np.array(results_1a[(S, A)]['gaps'])
        ax.semilogy(np.arange(len(gaps)), np.maximum(gaps, 1e-10),
                    color=color, label=f'(S, A) = ({S}, {A})', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Suboptimality Gap $\\rho^* - \\rho^{\\pi_k}$', fontsize=12)
    ax.set_title('Suboptimality Gap: Different Sizes', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'suboptimality_1a.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Suboptimality for Experiment 1b
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    labels = {'no_variance': 'No Variance', 'low_variance': 'Low Variance',
              'high_variance': 'High Variance', 'max_variance': 'Max Variance'}
    for vtype, color in zip(['no_variance', 'low_variance', 'high_variance', 'max_variance'], colors):
        if vtype not in results_1b:
            continue
        gaps = np.array(results_1b[vtype]['gaps'])
        ax.semilogy(np.arange(len(gaps)), np.maximum(gaps, 1e-10),
                    color=color, label=labels[vtype], linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Suboptimality Gap $\\rho^* - \\rho^{\\pi_k}$', fontsize=12)
    ax.set_title('Suboptimality Gap: Different Reward Variances', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'suboptimality_1b.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Suboptimality for Experiment 2
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {'uniform': 'tab:blue', 'non_uniform': 'tab:orange', 'deterministic': 'tab:green'}
    labels = {'uniform': 'Uniform', 'non_uniform': 'Non-Uniform', 'deterministic': 'Deterministic'}
    for ktype in ['uniform', 'non_uniform', 'deterministic']:
        if ktype not in results_2:
            continue
        gaps = np.array(results_2[ktype]['gaps'])
        ax.semilogy(np.arange(len(gaps)), np.maximum(gaps, 1e-10),
                    color=colors[ktype], label=labels[ktype], linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Suboptimality Gap $\\rho^* - \\rho^{\\pi_k}$', fontsize=12)
    ax.set_title('Suboptimality Gap: Different Transition Kernels', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'suboptimality_2.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved suboptimality gap plots to {save_dir}/")


def print_mdp_constants_summary(results_1a, results_1b, results_2):
    """Print MDP complexity constants for all experiments."""
    print("\n" + "=" * 70)
    print("MDP Complexity Constants Summary")
    print("=" * 70)
    
    print("\nExperiment 1a: Varying sizes")
    print(f"  {'(S,A)':<12} {'C_m':>8} {'C_p':>8} {'C_r':>8} {'kappa_r':>10} {'L2':>10}")
    for S, A in [(3, 3), (9, 9)]:
        mdp = make_mdp_varying_size(S, A)
        c = mdp.compute_mdp_constants()
        print(f"  ({S},{A}){'':<8} {c['C_m']:>8.3f} {c['C_p']:>8.3f} {c['C_r']:>8.3f} "
              f"{c['kappa_r']:>10.3f} {c['L2']:>10.3f}")
    
    print("\nExperiment 1b: Varying reward variances (S=A=16)")
    print(f"  {'Variance':<20} {'C_r':>8} {'kappa_r':>10} {'L2':>10}")
    for vtype in ['no_variance', 'low_variance', 'high_variance', 'max_variance']:
        mdp = make_mdp_varying_reward_variance(16, 16, vtype)
        c = mdp.compute_mdp_constants()
        print(f"  {vtype:<20} {c['C_r']:>8.3f} {c['kappa_r']:>10.3f} {c['L2']:>10.3f}")
    
    print("\nExperiment 2: Varying transition kernels (S=A=16)")
    print(f"  {'Kernel':<20} {'C_m':>8} {'C_p':>8} {'L2':>10}")
    for ktype in ['uniform', 'non_uniform', 'deterministic']:
        mdp = make_mdp_varying_transition(16, 16, ktype)
        c = mdp.compute_mdp_constants()
        print(f"  {ktype:<20} {c['C_m']:>8.3f} {c['C_p']:>8.3f} {c['L2']:>10.3f}")


def main():
    """Run all experiments and generate figures."""
    print("=" * 70)
    print("Reproducing experiments from:")
    print("'Global Convergence of Policy Gradient in Average Reward MDPs'")
    print("=" * 70)
    
    os.makedirs('figures', exist_ok=True)
    
    # Run experiments
    results_1a = run_experiment_1a(num_iterations=2000, eta=0.01)
    results_1b = run_experiment_1b(num_iterations=2000, eta=0.01)
    results_2 = run_experiment_2(num_iterations=3000, eta=0.01)
    
    # Generate figures
    print("\nGenerating figures...")
    plot_figure_1(results_1a, results_1b)
    plot_figure_2(results_2)
    plot_suboptimality_gaps(results_1a, results_1b, results_2)
    
    # Print MDP complexity constants
    print_mdp_constants_summary(results_1a, results_1b, results_2)
    
    print("\nAll experiments completed. Figures saved to 'figures/' directory.")


if __name__ == '__main__':
    main()
