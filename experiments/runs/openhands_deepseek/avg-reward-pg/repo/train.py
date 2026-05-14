"""Main entry point for reproducing all experiments from the paper.

Run: python train.py
"""

import os
import sys
import numpy as np
from experiments import (
    experiment_varying_states_actions,
    experiment_varying_reward_variance,
    experiment_varying_transitions,
    plot_experiment_results,
)
from constants import compute_all_constants
from policy_gradient import compute_exact_optimal_policy
from config import (
    ETA, NUM_ITERS_EXP1, NUM_ITERS_EXP2, NUM_ITERS_EXP3,
    SIZES_EXP1, S_EXP23, A_EXP23, N_RESTARTS_OPT, SEED,
)
from utils import compute_nu, compute_diameter_pi


def print_constants(results: dict, exp_name: str):
    """Print MDP complexity constants for each experiment setting."""
    print(f"\n{'='*60}")
    print(f"MDP Complexity Constants: {exp_name}")
    print(f"{'='*60}")
    for label, data in results.items():
        mdp = data['mdp']
        pi_star, _ = compute_exact_optimal_policy(mdp, n_restarts=N_RESTARTS_OPT)
        consts = compute_all_constants(mdp, pi_star)
        print(f"\n  {label}:")
        for k, v in consts.items():
            print(f"    {k}: {v:.6f}")
        nu = compute_nu(consts['L2_Pi'], consts['C_PL'], mdp.S)
        print(f"    nu (convergence rate): {nu:.8f}")
        if 32.0 * mdp.S * consts['L2_Pi'] * consts['C_PL']**2 < 1.0:
            print(f"    >> Simple MDP: exponential convergence expected")


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figs_dir, exist_ok=True)

    print("=" * 60)
    print("Reproducing: Global Convergence of Policy Gradient")
    print("            in Average Reward MDPs")
    print("=" * 60)

    # Experiment 1: Varying State/Action Sizes
    print("\n[Experiment 1] Varying State and Action Space Sizes")
    print(f"  Sizes: {SIZES_EXP1}, Eta: {ETA}, Iters: {NUM_ITERS_EXP1}")
    results1 = experiment_varying_states_actions(
        sizes=SIZES_EXP1,
        num_iters=NUM_ITERS_EXP1,
        eta=ETA,
        seed=SEED,
    )
    plot_experiment_results(
        results1,
        'Convergence with Different State/Action Sizes',
        os.path.join(figs_dir, 'fig1a_varying_sizes.png'),
        ylabel='Average Reward',
    )
    print_constants(results1, "Exp 1 (Varying S, A)")
    for label, data in results1.items():
        print(f"  {label}: final rho = {data['rho_history'][-1]:.6f}, "
              f"opt gap = {data['opt_gap_history'][-1]:.6f}")

    # Experiment 2: Varying Reward Variance
    print(f"\n[Experiment 2] Varying Reward Variance")
    print(f"  (S,A) = ({S_EXP23},{A_EXP23}), Eta: {ETA}, Iters: {NUM_ITERS_EXP2}")
    results2 = experiment_varying_reward_variance(
        S=S_EXP23, A=A_EXP23,
        num_iters=NUM_ITERS_EXP2,
        eta=ETA,
        seed=SEED,
    )
    plot_experiment_results(
        results2,
        'Convergence with Different Reward Variances',
        os.path.join(figs_dir, 'fig1b_varying_reward.png'),
        ylabel='Average Reward',
    )
    print_constants(results2, "Exp 2 (Varying Reward)")
    for label, data in results2.items():
        print(f"  {label}: final rho = {data['rho_history'][-1]:.6f}, "
              f"opt gap = {data['opt_gap_history'][-1]:.6f}")

    # Experiment 3: Varying Transition Kernels
    print(f"\n[Experiment 3] Varying Transition Kernels")
    print(f"  (S,A) = ({S_EXP23},{A_EXP23}), Eta: {ETA}, Iters: {NUM_ITERS_EXP3}")
    results3 = experiment_varying_transitions(
        S=S_EXP23, A=A_EXP23,
        num_iters=NUM_ITERS_EXP3,
        eta=ETA,
        seed=SEED,
    )
    plot_experiment_results(
        results3,
        'Convergence with Different Transition Kernels (C_p)',
        os.path.join(figs_dir, 'fig2_varying_transitions.png'),
        ylabel='Average Reward',
    )
    print_constants(results3, "Exp 3 (Varying Transitions)")
    for label, data in results3.items():
        print(f"  {label}: final rho = {data['rho_history'][-1]:.6f}, "
              f"opt gap = {data['opt_gap_history'][-1]:.6f}")

    print(f"\nAll figures saved to: {figs_dir}/")
    print("Reproduction complete.")


if __name__ == '__main__':
    main()
