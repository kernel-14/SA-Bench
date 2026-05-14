#!/usr/bin/env python3
"""
Main entry point for reproducing the paper:
"Global Convergence of Policy Gradient in Average Reward MDPs"

This script runs the key demonstrations:
1. Projected policy gradient on a sample MDP
2. Computation of MDP complexity constants
3. Comparison with theoretical convergence bounds
4. All three simulation experiments from Section 4
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.mdp import AverageRewardMDP, make_random_mdp
from src.projection import make_projection_matrix, verify_projection_properties
from src.constants import compute_all_constants
from src.policy_gradient import (
    run_projected_policy_gradient, compute_theoretical_bound
)
from src.smoothness import (
    verify_smoothness_constant, verify_lipschitz_constant
)
from src.simulations import (
    run_all_simulations,
    experiment_1_state_action_size,
    experiment_2_reward_variance,
    experiment_3_transition_kernel,
)


def demo_projection():
    """Demonstrate the projection matrix properties (Lemma 1)."""
    print("\n" + "=" * 60)
    print("DEMO: Projection Matrix Φ (Lemma 1)")
    print("=" * 60)
    
    for n in [3, 5, 10]:
        valid = verify_projection_properties(n)
        print(f"  |S| = {n}: Properties hold = {valid}")
    
    # Show the projection matrix for a small case
    Phi = make_projection_matrix(4)
    print(f"\n  Φ (|S|=4):\n{Phi}")
    print(f"  Φ·1 = {Phi @ np.ones(4)}")


def demo_policy_gradient():
    """Demonstrate the policy gradient algorithm on a small MDP."""
    print("\n" + "=" * 60)
    print("DEMO: Projected Policy Gradient on Small MDP")
    print("=" * 60)
    
    # Create a small random MDP
    n_states, n_actions = 5, 4
    mdp = make_random_mdp(n_states, n_actions, seed=42)
    
    # Run policy gradient
    n_iters = 200
    eta = 0.1
    result = run_projected_policy_gradient(mdp, n_iterations=n_iters, eta=eta, seed=0)
    
    rewards = result['average_rewards']
    gaps = result['optimality_gaps']
    
    print(f"  MDP: |S|={n_states}, |A|={n_actions}")
    print(f"  Initial reward: {rewards[0]:.6f}")
    print(f"  Final reward:   {rewards[-1]:.6f}")
    print(f"  Improvement:    {rewards[-1] - rewards[0]:.6f}")
    print(f"  Initial gap:    {gaps[0]:.6f}")
    print(f"  Final gap:      {gaps[-1]:.6f}")
    
    # Sample some intermediate rewards
    print(f"\n  Reward trajectory (every {n_iters//5} iters):")
    for i in range(0, n_iters + 1, n_iters // 5):
        print(f"    iter {i:4d}: ρ = {rewards[i]:.6f}")


def demo_constants():
    """Demonstrate the computation of MDP complexity constants."""
    print("\n" + "=" * 60)
    print("DEMO: MDP Complexity Constants (Table 1)")
    print("=" * 60)
    
    n_states, n_actions = 5, 3
    mdp = make_random_mdp(n_states, n_actions, seed=123)
    
    # Generate some sample policies for constant computation
    rng = np.random.RandomState(0)
    pi_list = []
    for _ in range(10):
        pi = rng.rand(n_states, n_actions)
        pi = pi / pi.sum(axis=1, keepdims=True)
        pi_list.append(pi)
    
    # Generate policy pairs
    policy_pairs = []
    for i in range(len(pi_list)):
        for j in range(i + 1, len(pi_list)):
            policy_pairs.append((pi_list[i], pi_list[j]))
            if len(policy_pairs) >= 20:
                break
        if len(policy_pairs) >= 20:
            break
    
    # Use the first policy as proxy for optimal
    constants = compute_all_constants(mdp, pi_list, pi_list[0], policy_pairs)
    
    print(f"  C_m     = {constants['C_m']:.4f}  (mixing rate)")
    print(f"  C_p     = {constants['C_p']:.4f}  (transition kernel diameter)")
    print(f"  C_r     = {constants['C_r']:.4f}  (reward function diameter)")
    print(f"  κ_r     = {constants['kappa_r']:.4f}  (reward variance)")
    print(f"  L_1^Π   = {constants['L1_Pi']:.4f}  (restricted Lipschitz)")
    print(f"  L_2^Π   = {constants['L2_Pi']:.4f}  (restricted smoothness)")
    print(f"  C_PL    = {constants['C_PL']:.4f}  (distribution mismatch)")
    
    # Theoretical bounds from Lemma 18
    print(f"\n  Theoretical bounds (Lemma 18):")
    print(f"    C_m ≤ 2Ce|S|/(1-λ)")
    print(f"    C_p ≤ √|A| = {np.sqrt(n_actions):.4f}")
    print(f"    C_r ≤ √|A| = {np.sqrt(n_actions):.4f}")
    print(f"    κ_r ≤ 2")
    print(f"    C_m (computed) = {constants['C_m']:.4f}")


def demo_theoretical_bounds():
    """Demonstrate the theoretical convergence bounds from Theorem 1."""
    print("\n" + "=" * 60)
    print("DEMO: Theoretical Convergence Bounds (Theorem 1)")
    print("=" * 60)
    
    n_states, n_actions = 5, 3
    mdp = make_random_mdp(n_states, n_actions, seed=123)
    
    # Run policy gradient to get empirical data
    result = run_projected_policy_gradient(mdp, n_iterations=200, eta=0.1, seed=0)
    
    # Get constants
    pi_0 = result['policies'][0]
    pi_star = result['policies'][-1]  # using final as proxy for optimal
    
    rng = np.random.RandomState(0)
    pi_list = [pi_0, pi_star]
    for _ in range(8):
        pi = rng.rand(n_states, n_actions)
        pi = pi / pi.sum(axis=1, keepdims=True)
        pi_list.append(pi)
    
    policy_pairs = [(pi_list[i], pi_list[j]) 
                    for i in range(len(pi_list)) 
                    for j in range(i+1, len(pi_list))]
    
    constants = compute_all_constants(mdp, pi_list, pi_star, policy_pairs)
    
    L2 = constants['L2_Pi']
    C_PL = constants['C_PL']
    
    bounds = compute_theoretical_bound(mdp, pi_0, pi_star, L2, C_PL, 200)
    
    print(f"  L_2^Π = {L2:.6f}")
    print(f"  C_PL  = {C_PL:.6f}")
    print(f"  c     = {bounds['c']:.6f}")
    print(f"  ν     = {bounds['nu']:.6f}")
    print(f"  Is simple MDP: {bounds['is_simple']}")
    
    # Compare empirical vs theoretical
    print(f"\n  Empirical vs Theoretical suboptimality:")
    for k in [0, 10, 50, 100, 200]:
        emp_gap = result['optimality_gaps'][k]
        thm_gap = bounds['bounds_sublinear'][k]
        print(f"    k={k:3d}: empirical={emp_gap:.6f}, bound={thm_gap:.6f}")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Global Convergence of Policy Gradient in Average Reward MDPs")
    print("Paper Reproduction")
    print("=" * 60)
    
    # Core theory demonstrations
    demo_projection()
    demo_constants()
    demo_policy_gradient()
    demo_theoretical_bounds()
    
    # Simulation experiments
    print("\n" + "=" * 60)
    print("SIMULATIONS (Section 4)")
    print("=" * 60)
    
    # Run simplified versions of all three experiments
    print("\n--- Experiment 1: State/Action Space Size ---")
    exp1 = experiment_1_state_action_size(
        sizes=[(3, 3), (9, 9)],  # Reduced for speed; paper uses up to (81,81)
        n_iterations=300,
        eta=0.1,
        seed=42
    )
    
    print("\n--- Experiment 2: Reward Variance ---")
    exp2 = experiment_2_reward_variance(
        n_states=8, n_actions=8,  # Reduced for speed; paper uses (16,16)
        n_iterations=500,
        eta=0.1,
        seed=42
    )
    
    print("\n--- Experiment 3: Transition Kernel ---")
    exp3 = experiment_3_transition_kernel(
        n_states=8, n_actions=8,  # Reduced for speed; paper uses (16,16)
        n_iterations=500,
        eta=0.1,
        seed=42
    )
    
    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)


if __name__ == '__main__':
    main()
