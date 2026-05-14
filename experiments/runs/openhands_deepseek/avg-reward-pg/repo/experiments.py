import numpy as np
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mdp import TabularMDP, generate_random_mdp
from policy_gradient import run_ppg, compute_exact_optimal_policy, set_opt_gaps


def experiment_varying_states_actions(
    sizes: List[Tuple[int, int]] = [(3, 3), (9, 9), (81, 81)],
    num_iters: int = 2000,
    eta: float = 0.01,
    seed: int = 42,
) -> Dict[str, dict]:
    """Experiment 1: Varying state and action space sizes.
    Figure 1(a) in the paper.

    MDPs are constructed with:
    - Transition kernel: P(i|s,i) = (1 + 1/S)/2, P(i|s,j) = 1/(2S) for i != j
    - Reward: half of actions get reward 1, half get -1, for every state.
    """
    rng = np.random.RandomState(seed)
    results = {}

    for S, A in sizes:
        # Build transition kernel as specified in paper Appendix C.1
        P = np.zeros((S, A, S))
        for s in range(S):
            for a in range(A):
                for ns in range(S):
                    if ns == a % S:
                        P[s, a, ns] = (1.0 + 1.0 / S) / 2.0
                    else:
                        P[s, a, ns] = 1.0 / (2.0 * S)
        # Normalize to ensure row sums = 1
        P = P / P.sum(axis=2, keepdims=True)

        # Build reward: half actions 1, half -1 (maximal variance)
        r = np.ones((S, A))
        n_neg = max(1, A // 2)
        r[:, :n_neg] = -1.0

        mdp = TabularMDP(S, A, P, r)

        # Initial policy
        pi0 = rng.rand(S, A)
        pi0 = pi0 / pi0.sum(axis=1, keepdims=True)

        rho_hist, pi_hist, opt_gaps = run_ppg(mdp, pi0, eta=eta, num_iters=num_iters, track_every=50)
        pi_star, rho_star = compute_exact_optimal_policy(mdp, n_restarts=10)
        set_opt_gaps(mdp, rho_star, opt_gaps, rho_hist)

        results[f"S{S}_A{A}"] = {
            'rho_history': rho_hist,
            'opt_gap_history': opt_gaps,
            'rho_star': rho_star,
            'mdp': mdp,
        }

    return results


def experiment_varying_reward_variance(
    S: int = 16,
    A: int = 16,
    num_iters: int = 3000,
    eta: float = 0.01,
    seed: int = 42,
) -> Dict[str, dict]:
    """Experiment 2: Varying reward variance.
    Figure 1(b) in the paper.

    Fixed (S=16, A=16). Randomly generated transition kernel (fixed).
    Varying reward variance: no, low, high, maximal.

    Reward construction:
    - No variance: r(s,a) = 1 for all (s,a)
    - Low variance: 1/8 of actions at s0 get -1, rest 1
    - High variance: 1/4 of actions at s0 get -1, rest 1
    - Max variance: 1/2 of actions at s0 get -1, rest 1
    """
    rng = np.random.RandomState(seed)
    # Generate random but fixed transition kernel
    P_fixed = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            probs = rng.dirichlet(np.ones(S))
            P_fixed[s, a, :] = probs

    reward_configs = {
        'no_variance': 'none',
        'low_variance': 'low',
        'high_variance': 'high',
        'max_variance': 'max',
    }

    results = {}

    for label, rvar in reward_configs.items():
        r = np.ones((S, A))
        if rvar == 'low':
            n_neg = max(1, A // 8)
            r[:, :n_neg] = -1.0
        elif rvar == 'high':
            n_neg = max(1, A // 4)
            r[:, :n_neg] = -1.0
        elif rvar == 'max':
            n_neg = max(1, A // 2)
            r[:, :n_neg] = -1.0
        # 'none': all stay at 1

        mdp = TabularMDP(S, A, P_fixed.copy(), r)

        pi0 = rng.rand(S, A)
        pi0 = pi0 / pi0.sum(axis=1, keepdims=True)

        rho_hist, pi_hist, opt_gaps = run_ppg(mdp, pi0, eta=eta, num_iters=num_iters, track_every=50)
        pi_star, rho_star = compute_exact_optimal_policy(mdp, n_restarts=10)
        set_opt_gaps(mdp, rho_star, opt_gaps, rho_hist)

        results[label] = {
            'rho_history': rho_hist,
            'opt_gap_history': opt_gaps,
            'rho_star': rho_star,
            'mdp': mdp,
        }

    return results


def experiment_varying_transitions(
    S: int = 16,
    A: int = 16,
    num_iters: int = 3000,
    eta: float = 0.01,
    seed: int = 42,
) -> Dict[str, dict]:
    """Experiment 3: Varying transition kernels.
    Figure 2 in the paper.

    Fixed (S=16, A=16). Three transition kernels:
    - Uniform: P(s'|s,a) = 1/S
    - Non-uniform: P(i|s,i) = 1/(2S) + 1/2, P(i|s,j) = 1/(2S) for i != j
    - Deterministic: random permutation of identity per state-action

    Reward is high variance.
    """
    rng = np.random.RandomState(seed)

    # High variance reward
    r = np.ones((S, A))
    n_neg = max(1, A // 4)
    r[:, :n_neg] = -1.0

    transition_configs = {}

    # Uniform
    P_uniform = np.ones((S, A, S)) / S
    transition_configs['uniform'] = P_uniform

    # Non-uniform
    P_nonuni = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            for ns in range(S):
                if ns == a % S:
                    P_nonuni[s, a, ns] = 1.0 / (2.0 * S) + 0.5
                else:
                    P_nonuni[s, a, ns] = 1.0 / (2.0 * S)
    P_nonuni = P_nonuni / P_nonuni.sum(axis=2, keepdims=True)
    transition_configs['non_uniform'] = P_nonuni

    # Deterministic
    P_det = np.zeros((S, A, S))
    for s in range(S):
        perm = rng.permutation(S)
        for a in range(A):
            next_s = perm[a % S]
            P_det[s, a, next_s] = 1.0
    transition_configs['deterministic'] = P_det

    results = {}

    for label, P in transition_configs.items():
        mdp = TabularMDP(S, A, P, r.copy())

        pi0 = rng.rand(S, A)
        pi0 = pi0 / pi0.sum(axis=1, keepdims=True)

        rho_hist, pi_hist, opt_gaps = run_ppg(mdp, pi0, eta=eta, num_iters=num_iters, track_every=50)
        pi_star, rho_star = compute_exact_optimal_policy(mdp, n_restarts=10)
        set_opt_gaps(mdp, rho_star, opt_gaps, rho_hist)

        results[label] = {
            'rho_history': rho_hist,
            'opt_gap_history': opt_gaps,
            'rho_star': rho_star,
            'mdp': mdp,
        }

    return results


def plot_experiment_results(results: dict, title: str, save_path: str,
                             ylabel: str = 'Average Reward',
                             use_opt_gap: bool = False):
    """Plot results from an experiment."""
    plt.figure(figsize=(10, 6))
    for label, data in results.items():
        iterations = np.arange(len(data['rho_history'])) * 50
        if use_opt_gap:
            values = data['opt_gap_history']
        else:
            values = data['rho_history']
        plt.plot(iterations, values, label=label.replace('_', ' ').title(), linewidth=2)

    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    import os
    output_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figs_dir, exist_ok=True)

    print("Running Experiment 1: Varying State and Action Space Sizes...")
    results1 = experiment_varying_states_actions(
        sizes=[(3, 3), (9, 9), (81, 81)],
        num_iters=2000,
        eta=0.01
    )
    plot_experiment_results(
        results1,
        'Convergence with Different State/Action Sizes',
        os.path.join(figs_dir, 'fig1a_varying_sizes.png'),
        ylabel='Average Reward'
    )
    print("  -> Saved fig1a_varying_sizes.png")

    print("Running Experiment 2: Varying Reward Variance...")
    results2 = experiment_varying_reward_variance(
        S=16, A=16, num_iters=3000, eta=0.01
    )
    plot_experiment_results(
        results2,
        'Convergence with Different Reward Variances',
        os.path.join(figs_dir, 'fig1b_varying_reward.png'),
        ylabel='Average Reward'
    )
    print("  -> Saved fig1b_varying_reward.png")

    print("Running Experiment 3: Varying Transition Kernels...")
    results3 = experiment_varying_transitions(
        S=16, A=16, num_iters=3000, eta=0.01
    )
    plot_experiment_results(
        results3,
        'Convergence with Different Transition Kernels (C_p)',
        os.path.join(figs_dir, 'fig2_varying_transitions.png'),
        ylabel='Average Reward'
    )
    print("  -> Saved fig2_varying_transitions.png")

    # Print summary statistics
    print("\n=== Summary ===")
    for exp_name, results in [('Exp 1 (Varying S,A)', results1),
                               ('Exp 2 (Varying Reward)', results2),
                               ('Exp 3 (Varying Transitions)', results3)]:
        print(f"\n{exp_name}:")
        for label, data in results.items():
            print(f"  {label}: final rho = {data['rho_history'][-1]:.4f}, "
                  f"opt gap = {data['opt_gap_history'][-1]:.6f}")


if __name__ == '__main__':
    main()
