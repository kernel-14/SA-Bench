"""Visualize the L^+ distribution (Figure 4 in the paper).

Reproduces Figure 4: Probability density for L^+ with lambda in {0.7, 0.8, 0.9}
estimated using 100,000 Dirichlet samples.

This figure demonstrates how the Bayesian quadrature approach captures
uncertainty in the expected loss through the L^+ random variable, and how
this distribution changes as lambda varies.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bayesian_quadrature import L_plus_random_variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_synthetic_binomial_losses(n, K, lam, rng):
    """Generate losses for the synthetic binomial experiment (Section 5.1)."""
    V = rng.uniform(0, 1, size=(n, K))
    return np.mean(V > lam, axis=1)


def main():
    # Parameters matching Section 5.1
    n = 10
    K = 4
    B = 1.0
    n_dirichlet_samples = 100000  # as stated in paper for Figure 4
    seed = 42

    rng = np.random.default_rng(seed)

    # Generate calibration data once
    V = rng.uniform(0, 1, size=(n, K))

    lambda_values = [0.7, 0.8, 0.9]
    colors = ['#2196F3', '#4CAF50', '#FF5722']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (lam, color) in enumerate(zip(lambda_values, colors)):
        losses = np.mean(V > lam, axis=1)
        sorted_losses = np.sort(losses)
        L_plus_samples = L_plus_random_variable(
            sorted_losses=sorted_losses,
            B=B,
            n_dirichlet_samples=n_dirichlet_samples,
            rng=rng,
        )

        ax = axes[idx]
        ax.hist(L_plus_samples, bins=80, density=True, alpha=0.7, color=color,
                edgecolor='white', linewidth=0.5)
        ax.axvline(np.mean(L_plus_samples), color='black', linestyle='--',
                   linewidth=2, label=f'E[L+] = {np.mean(L_plus_samples):.3f}')
        ax.set_title(f'$\\lambda = {lam}$', fontsize=14)
        ax.set_xlabel('$L^+$', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Density', fontsize=12)
        ax.legend(fontsize=10)

    plt.suptitle('Figure 4: Probability density for $L^+$ with varying $\\lambda$',
                 fontsize=16, y=1.02)
    plt.tight_layout()

    # Save figure
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, 'figure4_L_plus_distribution.pdf')
    plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.savefig(fig_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    print(f"Figure 4 saved to {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
