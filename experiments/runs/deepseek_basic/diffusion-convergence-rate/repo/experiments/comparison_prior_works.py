"""Comparison with prior works (Section 1.1, Appendix B, Figure 1, Figure 3).

Computes and plots the iteration complexity / TV distance achieved
by various theoretical results as functions of L, d, and epsilon.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.lipschitz_analysis import compare_with_prior_works, theoretical_convergence_rate


def plot_iteration_complexity_vs_L(d=100, epsilon=0.1, save_path=None):
    """Plot iteration complexity as a function of L (Figure 1, left).

    Compares:
    - Benton et al. (2023): T ~ d / epsilon^2
    - Li and Yan (2024a): T ~ d / epsilon
    - Li and Cai (2024): T ~ d^{5/4} / sqrt(epsilon)
    - Li and Jiao (2024): T ~ d^{1/3} L / epsilon^{2/3}
    - This work: T ~ min{d, d^{2/3} L^{1/3}, d^{1/3} L} / epsilon^{2/3}
    """
    L_values = np.logspace(0, 3, 100) * np.sqrt(d)

    # Compute iteration complexity for each method
    T_benton = d / (epsilon ** 2) * np.ones_like(L_values)
    T_li_yan = d / epsilon * np.ones_like(L_values)
    T_li_cai = (d ** 1.25) / np.sqrt(epsilon) * np.ones_like(L_values)
    T_li_jiao = (d ** (1/3)) * L_values / (epsilon ** (2/3))
    T_ours = np.minimum(d, np.minimum(d ** (2/3) * L_values ** (1/3),
                                       d ** (1/3) * L_values)) / (epsilon ** (2/3))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(L_values / np.sqrt(d), T_benton, 'b--', label='Benton et al. (2023)')
    ax.loglog(L_values / np.sqrt(d), T_li_yan, 'g--', label='Li & Yan (2024a)')
    ax.loglog(L_values / np.sqrt(d), T_li_cai, 'm--', label='Li & Cai (2024)')
    ax.loglog(L_values / np.sqrt(d), T_li_jiao, 'c--', label='Li & Jiao (2024)')
    ax.loglog(L_values / np.sqrt(d), T_ours, 'r-', linewidth=2, label='This Work')

    ax.set_xlabel('L / sqrt(d)', fontsize=12)
    ax.set_ylabel('Iteration Complexity T', fontsize=12)
    ax.set_title(f'Iteration Complexity vs L (d={d}, epsilon={epsilon})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'figures',
                                  'iteration_complexity_vs_L.png'),
                    dpi=150, bbox_inches='tight')

    plt.close()


def plot_tv_vs_epsilon(d=100, L=np.inf, save_path=None):
    """Plot TV distance as a function of epsilon for L=inf (Figure 1, right)."""
    T_values = np.logspace(1, 6, 100)

    # Compute TV distance for each method
    # Benton: TV ~ sqrt(d / T)
    tv_benton = np.sqrt(d / T_values)
    # Li & Yan: TV ~ d / T
    tv_li_yan = d / T_values
    # Li & Cai: TV ~ d^{5/4} / sqrt(T)... reversed: T ~ d^{5/4}/eps^{1/2} => eps ~ d^{5/2}/T^2
    tv_li_cai = (d ** 2.5) / (T_values ** 2)
    # This work (L=inf): TV ~ min{d^{3/2}} * log^4(T) / T^{3/2}
    # For L=inf: min{d^{3/2}, d L^{1/2}, d^{1/2} L^{3/2}} = d^{3/2}
    tv_ours = (d ** 1.5) * (np.log(T_values)) ** 4 / (T_values ** 1.5)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(T_values, np.minimum(tv_benton, 1), 'b--', label='Benton et al. (2023)')
    ax.loglog(T_values, np.minimum(tv_li_yan, 1), 'g--', label='Li & Yan (2024a)')
    ax.loglog(T_values, np.minimum(tv_li_cai, 1), 'm--', label='Li & Cai (2024)')
    ax.loglog(T_values, np.minimum(tv_ours, 1), 'r-', linewidth=2, label='This Work (L=inf)')

    ax.set_xlabel('Iterations T', fontsize=12)
    ax.set_ylabel('TV Distance', fontsize=12)
    ax.set_title(f'TV Distance vs T (d={d}, L=inf)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'figures',
                                  'tv_vs_epsilon.png'),
                    dpi=150, bbox_inches='tight')

    plt.close()


def plot_fixed_T_comparison(d=100, save_path=None):
    """Plot TV distance for fixed T as function of L (Appendix B, Figure 3)."""
    L_values = np.logspace(0, 3, 100) * np.sqrt(d)

    # Three values of T
    T_values_list = [d, d ** 1.5, d ** 2]
    labels_list = ['T = O(d)', 'T = O(d^{3/2})', 'T = O(d^2)']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, T_val, label in zip(axes, T_values_list, labels_list):
        # Compute TV for each method
        tv_benton = np.minimum(np.sqrt(d / T_val), 1) * np.ones_like(L_values)
        tv_li_yan = np.minimum(d / T_val, 1) * np.ones_like(L_values)
        tv_li_cai = np.minimum((d ** 2.5) / (T_val ** 2), 1) * np.ones_like(L_values)

        # Li & Jiao: depends on L
        T_needed_lj = (d ** (1/3)) * L_values / (0.1 ** (2/3))
        tv_li_jiao = np.minimum((d ** (1/3)) * L_values * (np.log(T_val)) ** 4 / (T_val ** 1.5), 1)

        # This work
        min_factor = np.minimum(d ** 1.5,
                                np.minimum(d * np.sqrt(L_values),
                                          np.sqrt(d) * L_values ** 1.5))
        tv_ours = np.minimum(min_factor * (np.log(T_val)) ** 4 / (T_val ** 1.5), 1)

        ax.loglog(L_values / np.sqrt(d), tv_benton, 'b--', label='Benton et al.')
        ax.loglog(L_values / np.sqrt(d), tv_li_yan, 'g--', label='Li & Yan')
        ax.loglog(L_values / np.sqrt(d), tv_li_cai, 'm--', label='Li & Cai')
        ax.loglog(L_values / np.sqrt(d), tv_li_jiao, 'c--', label='Li & Jiao')
        ax.loglog(L_values / np.sqrt(d), tv_ours, 'r-', linewidth=2, label='This Work')

        ax.set_xlabel('L / sqrt(d)', fontsize=11)
        ax.set_ylabel('TV Distance', fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'TV Distance Comparison (d={d})', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'figures',
                                  'fixed_T_comparison.png'),
                    dpi=150, bbox_inches='tight')

    plt.close()


def print_comparison_table():
    """Print numerical comparison table for different (d, L) configurations."""
    print("\n" + "=" * 80)
    print("Iteration Complexity Comparison (epsilon = 0.1)")
    print("=" * 80)

    configs = [
        {'d': 100, 'L': 10, 'label': 'Small L'},
        {'d': 100, 'L': np.sqrt(100), 'label': 'L = sqrt(d)'},
        {'d': 100, 'L': 100, 'label': 'L = d'},
        {'d': 100, 'L': 1000, 'label': 'Large L'},
        {'d': 100, 'L': np.inf, 'label': 'L = inf'},
    ]

    print(f"{'Config':>20s}  {'Benton':>12s}  {'Li&Yan':>12s}  {'Li&Cai':>12s}  {'Li&Jiao':>12s}  {'This Work':>12s}")
    print("-" * 92)

    for cfg in configs:
        rates = compare_with_prior_works(cfg['d'], cfg['L'], 1000, epsilon=0.1)
        print(f"{cfg['label']:>20s}  {rates['Benton2023_d_eps^{-2}']:12.1f}  "
              f"{rates['LiYan2024_d_eps^{-1}']:12.1f}  "
              f"{rates['LiCai2024_d^{5/4}_eps^{-1/2}']:12.1f}  "
              f"{rates['LiJiao2024_d^{1/3}L_eps^{-2/3}']:12.1f}  "
              f"{rates['ThisWork']:12.1f}")


def main():
    """Generate all comparison figures."""
    print("Generating comparison figures...")

    # Create figures directory
    figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Figure 1 (left): Iteration complexity vs L
    plot_iteration_complexity_vs_L(
        d=100, epsilon=0.1,
        save_path=os.path.join(figures_dir, 'fig1_left_complexity_vs_L.png')
    )

    # Figure 1 (right): TV distance vs T for L=inf
    plot_tv_vs_epsilon(
        d=100, L=np.inf,
        save_path=os.path.join(figures_dir, 'fig1_right_tv_vs_T.png')
    )

    # Figure 3: Fixed T comparisons
    plot_fixed_T_comparison(
        d=100,
        save_path=os.path.join(figures_dir, 'fig3_fixed_T.png')
    )

    # Print comparison table
    print_comparison_table()


if __name__ == '__main__':
    main()
