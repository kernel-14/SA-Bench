"""
Convergence analysis for:
"Global Convergence of Policy Gradient in Average Reward MDPs"

This script:
1. Verifies the smoothness of the average reward (Lemmas 3, 4)
2. Validates the convergence bound from Theorem 1
3. Computes and displays MDP complexity constants (Table 1)
4. Demonstrates the O(1/T) convergence rate
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple

from mdp import AverageRewardMDP
from policy_gradient import (
    projected_policy_gradient,
    compute_suboptimality_gap,
    theoretical_bound,
)
from mdp_construction import (
    make_mdp_varying_size,
    make_mdp_varying_reward_variance,
    make_mdp_varying_transition,
)


def verify_smoothness(mdp: AverageRewardMDP, num_samples: int = 200, seed: int = 42) -> Dict:
    """
    Empirically verify the smoothness of the average reward (Lemma 4).
    
    Checks:
    - Lipschitz: |<grad rho^pi, pi' - pi>| <= L1 * ||pi' - pi||_2
    - Smoothness: ||grad rho^{pi'} - grad rho^pi||_2 <= L2 * ||pi' - pi||_2
    
    Returns empirical constants and compares with theoretical bounds.
    """
    rng = np.random.RandomState(seed)
    S, A = mdp.S, mdp.A
    
    empirical_L1 = 0.0
    empirical_L2 = 0.0
    
    for _ in range(num_samples):
        pi = rng.dirichlet(np.ones(A), size=S)
        pi_prime = rng.dirichlet(np.ones(A), size=S)
        
        diff = pi_prime - pi
        norm_diff = np.linalg.norm(diff)
        if norm_diff < 1e-10:
            continue
        
        grad_pi = mdp.get_policy_gradient(pi)
        
        # Lipschitz: |<grad, pi' - pi>| / ||pi' - pi||
        dir_deriv = np.sum(grad_pi * diff)
        L1_sample = abs(dir_deriv) / norm_diff
        empirical_L1 = max(empirical_L1, L1_sample)
        
        # Smoothness: ||grad(pi') - grad(pi)|| / ||pi' - pi||
        grad_pi_prime = mdp.get_policy_gradient(pi_prime)
        grad_diff = grad_pi_prime - grad_pi
        L2_sample = np.linalg.norm(grad_diff) / norm_diff
        empirical_L2 = max(empirical_L2, L2_sample)
    
    return {
        'empirical_L1': empirical_L1,
        'empirical_L2': empirical_L2,
    }


def validate_theorem_1(
    mdp: AverageRewardMDP,
    num_iterations: int = 500,
    eta: float = 0.01,
    seed: int = 42,
) -> Dict:
    """
    Validate Theorem 1: the O(1/T) convergence bound.
    
    Theorem 1 states:
        rho* - rho^{pi_k} <= 1 / (1/(rho* - rho^{pi_0}) + nu * k)
    where nu = c * (1 + 4c)^{-3/2} and c = 1/(32 * C_PL^2 * |S| * L2)
    """
    constants = mdp.compute_mdp_constants()
    L2 = constants['L2']
    C_PL = constants['C_PL']
    S = mdp.S
    
    rewards, _ = projected_policy_gradient(
        mdp, eta=eta, num_iterations=num_iterations, seed=seed
    )
    _, rho_star = mdp.get_optimal_policy()
    gaps = compute_suboptimality_gap(mdp, rewards, rho_star=rho_star)
    
    rho_0 = rewards[0]
    bounds = theoretical_bound(rho_star, rho_0, L2, C_PL, S, num_iterations)
    
    return {
        'gaps': gaps,
        'bounds': bounds,
        'constants': constants,
        'rho_star': rho_star,
        'rho_0': rho_0,
    }


def plot_theorem_validation(
    results: Dict,
    title: str = 'Theorem 1 Validation',
    save_path: str = 'figures/theorem_validation.png',
):
    """Plot empirical convergence vs theoretical bound from Theorem 1."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    gaps = np.array(results['gaps'])
    bounds = np.array(results['bounds'])
    iterations = np.arange(len(gaps))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale
    ax = axes[0]
    ax.plot(iterations, gaps, 'b-', linewidth=2, label='Empirical Gap')
    ax.plot(iterations, bounds, 'r--', linewidth=2, label='Theorem 1 Bound')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Suboptimality Gap', fontsize=12)
    ax.set_title(f'{title} (linear scale)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Log scale
    ax = axes[1]
    ax.semilogy(iterations, np.maximum(gaps, 1e-12), 'b-', linewidth=2, label='Empirical Gap')
    ax.semilogy(iterations, np.maximum(bounds, 1e-12), 'r--', linewidth=2, label='Theorem 1 Bound')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Suboptimality Gap (log scale)', fontsize=12)
    ax.set_title(f'{title} (log scale)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add constants info
    c = results['constants']
    info = (f"C_m={c['C_m']:.2f}, C_p={c['C_p']:.2f}, C_r={c['C_r']:.2f}, "
            f"L2={c['L2']:.2f}, C_PL={c['C_PL']:.2f}")
    fig.text(0.5, -0.02, info, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {save_path}")


def plot_convergence_rate(
    mdp: AverageRewardMDP,
    num_iterations: int = 1000,
    eta: float = 0.01,
    title: str = 'Convergence Rate',
    save_path: str = 'figures/convergence_rate.png',
):
    """
    Plot convergence rate and compare with O(1/k) reference.
    The paper proves O(1/T) convergence rate (Theorem 1).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    rewards, _ = projected_policy_gradient(mdp, eta=eta, num_iterations=num_iterations)
    _, rho_star = mdp.get_optimal_policy()
    gaps = np.array(compute_suboptimality_gap(mdp, rewards, rho_star=rho_star))
    
    iterations = np.arange(1, len(gaps))
    gaps_from_1 = gaps[1:]
    
    # O(1/k) reference
    ref_1_over_k = gaps[1] / iterations
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(iterations, np.maximum(gaps_from_1, 1e-12), 'b-', linewidth=2, label='Empirical Gap')
    ax.loglog(iterations, ref_1_over_k, 'r--', linewidth=2, label='O(1/k) reference')
    ax.set_xlabel('Iteration (log scale)', fontsize=12)
    ax.set_ylabel('Suboptimality Gap (log scale)', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {save_path}")


def print_constants_table(mdp_configs: List[Tuple]) -> None:
    """Print MDP complexity constants in a formatted table."""
    print("\nMDP Complexity Constants (Table 1 from paper)")
    print("-" * 85)
    print(f"{'Description':<28} {'C_m':>7} {'C_p':>7} {'C_r':>7} {'kappa_r':>9} "
          f"{'L1':>9} {'L2':>9} {'C_PL':>7}")
    print("-" * 85)
    
    for desc, mdp in mdp_configs:
        c = mdp.compute_mdp_constants()
        print(f"{desc:<28} {c['C_m']:>7.3f} {c['C_p']:>7.3f} {c['C_r']:>7.3f} "
              f"{c['kappa_r']:>9.3f} {c['L1']:>9.3f} {c['L2']:>9.3f} {c['C_PL']:>7.3f}")
    
    print("-" * 85)


def main():
    """Run convergence analysis."""
    print("=" * 65)
    print("Convergence Analysis")
    print("'Global Convergence of Policy Gradient in Average Reward MDPs'")
    print("=" * 65)
    
    os.makedirs('figures', exist_ok=True)
    
    # 1. Verify smoothness
    print("\n1. Verifying smoothness of average reward (Lemmas 3, 4)...")
    mdp_3x3 = make_mdp_varying_size(3, 3)
    smooth_results = verify_smoothness(mdp_3x3, num_samples=300)
    constants_3x3 = mdp_3x3.compute_mdp_constants()
    
    print(f"   Empirical L1 (Lipschitz): {smooth_results['empirical_L1']:.4f}")
    print(f"   Theoretical L1 bound:     {constants_3x3['L1']:.4f}")
    print(f"   Empirical L2 (Smooth):    {smooth_results['empirical_L2']:.4f}")
    print(f"   Theoretical L2 bound:     {constants_3x3['L2']:.4f}")
    
    if smooth_results['empirical_L1'] <= constants_3x3['L1'] + 1e-6:
        print("   ✓ Lipschitz bound verified")
    else:
        print("   ✗ Lipschitz bound NOT verified (may need more samples)")
    
    if smooth_results['empirical_L2'] <= constants_3x3['L2'] + 1e-6:
        print("   ✓ Smoothness bound verified")
    else:
        print("   ✗ Smoothness bound NOT verified (may need more samples)")
    
    # 2. Validate Theorem 1
    print("\n2. Validating Theorem 1 convergence bound...")
    
    for S, A in [(3, 3), (9, 9)]:
        print(f"   (S, A) = ({S}, {A})...")
        mdp = make_mdp_varying_size(S, A)
        results = validate_theorem_1(mdp, num_iterations=500, eta=0.01)
        plot_theorem_validation(
            results,
            title=f'Theorem 1 Validation (S=A={S})',
            save_path=f'figures/theorem_validation_{S}x{A}.png',
        )
        c = results['constants']
        print(f"     L2={c['L2']:.3f}, C_PL={c['C_PL']:.3f}")
        print(f"     Initial gap: {results['gaps'][0]:.4f}, Final gap: {results['gaps'][-1]:.6f}")
    
    # 3. Convergence rate (O(1/k))
    print("\n3. Demonstrating O(1/k) convergence rate...")
    mdp = make_mdp_varying_size(9, 9)
    plot_convergence_rate(
        mdp, num_iterations=1000, eta=0.01,
        title='Convergence Rate: O(1/k) (S=A=9)',
        save_path='figures/convergence_rate.png',
    )
    
    # 4. MDP complexity constants table
    print("\n4. MDP Complexity Constants:")
    mdp_configs = [
        ('Size (3,3)', make_mdp_varying_size(3, 3)),
        ('Size (9,9)', make_mdp_varying_size(9, 9)),
        ('Reward: no var (16x16)', make_mdp_varying_reward_variance(16, 16, 'no_variance')),
        ('Reward: low var (16x16)', make_mdp_varying_reward_variance(16, 16, 'low_variance')),
        ('Reward: high var (16x16)', make_mdp_varying_reward_variance(16, 16, 'high_variance')),
        ('Reward: max var (16x16)', make_mdp_varying_reward_variance(16, 16, 'max_variance')),
        ('Kernel: uniform (16x16)', make_mdp_varying_transition(16, 16, 'uniform')),
        ('Kernel: non-uniform (16x16)', make_mdp_varying_transition(16, 16, 'non_uniform')),
        ('Kernel: deterministic (16x16)', make_mdp_varying_transition(16, 16, 'deterministic')),
    ]
    print_constants_table(mdp_configs)
    
    print("\nConvergence analysis completed. Figures saved to 'figures/'.")


if __name__ == '__main__':
    main()
