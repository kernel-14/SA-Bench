```python
## experiments.py
"""Experiment orchestration for reproducing Figures 1(a), 1(b), and 2 from:
    Murthy et al., "Global Convergence of Policy Gradient in Average Reward
    MDPs", ICLR 2024.

This module constructs the three experimental MDP configurations described in
Section 4 and Appendices C.1–C.3 of the paper, runs the Projected Policy
Gradient (PPG) algorithm on each, and returns structured results for plotting.

Experiments:
    1. Figure 1(a): Convergence vs state/action space size.
       MDPs: (S,A) ∈ {(3,3), (9,9), (81,81)}, non-uniform kernel, max-variance
       reward, 2000 iterations. (Appendix C.1)

    2. Figure 1(b): Convergence vs reward variance (C_r).
       MDP: (16,16), shared random Dirichlet kernel, four reward variance
       levels, 2000 iterations. (Appendix C.2)

    3. Figure 2: Convergence vs transition kernel type (C_p).
       MDP: (16,16), shared high-variance reward, three kernel types
       (uniform, non-uniform, deterministic), 3000 iterations. (Appendix C.3)

Step size strategy (config.yaml policy_gradient section):
    eta = step_size_multiplier / L2  if L2 > 1e-10
    eta = step_size_fallback          otherwise
where L2 = L_2^Π is the restricted smoothness constant from Lemma 4.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
from numpy import ndarray

from complexity import ComplexityMetrics
from config import Config
from mdp import MDP
from policy_gradient import PolicyGradient


class Experiments:
    """Constructs MDP configurations and runs PPG for all three experiments.

    This class is the primary consumer of Config and the primary producer of
    structured result dictionaries consumed by Plotter. It faithfully
    implements the MDP constructions from Appendices C.1–C.3 of the paper.

    Attributes:
        config: Centralized configuration instance. All hyperparameters,
            MDP sizes, iteration counts, and step size settings are read
            from this object.
    """

    def __init__(self, config: Config) -> None:
        """Initialize Experiments with the global configuration.

        Sets the numpy random seed to ensure reproducibility of all random
        MDP constructions (Dirichlet kernels, deterministic permutations).
        The seed is set here in addition to Config.__post_init__ because
        experiments.py is where all random constructions actually occur.

        Args:
            config: Fully initialized Config instance. All fields must be
                set before passing to this constructor. The config's
                random_seed is used to seed numpy immediately.
        """
        self.config: Config = config
        # Re-seed numpy here to ensure reproducibility regardless of
        # what other code ran between Config() and Experiments() construction.
        np.random.seed(config.random_seed)

    # =========================================================================
    # Private: Transition Kernel Construction Methods
    # =========================================================================

    def _make_nonuniform_kernel(self, S: int, A: int) -> ndarray:
        """Construct the non-uniform stochastic transition kernel.

        Implements the kernel from Appendix C.1 of the paper:
            P(i | s, i) = (1 + 1/S) / 2   — action i increases prob of state i
            P(i | s, j) = 1 / (2*S)        — for i ≠ j (uniform background)

        The kernel is the same for all current states s — transition
        probabilities depend only on the action a and the target state s'.
        This is consistent with the paper's description in Appendix C.1.

        Verification (S=A=3):
            sum_{s'} P[s,a,s'] = (S-1)*(1/(2S)) + (1+1/S)/2
                                = (S-1)/(2S) + (S+1)/(2S)
                                = (S-1+S+1)/(2S) = 2S/(2S) = 1 ✓

        Used in: Experiment 1 (all sizes), Experiment 3 (non-uniform variant).

        Args:
            S: Number of states. Must be a positive integer.
            A: Number of actions. Must equal S for the kernel to be
                well-defined (action a maps to state a). In the paper's
                experiments, S = A always.

        Returns:
            Transition kernel of shape (S, A, S). P[s, a, s'] is the
            probability of transitioning to state s' from state s under
            action a. Each slice P[s, a, :] sums to 1.0.
        """
        # Initialize all entries to the background probability 1/(2S)
        # This handles the P(i|s,j) = 1/(2S) for i ≠ j case.
        P: ndarray = np.ones((S, A, S), dtype=np.float64) / (2.0 * S)

        # Override the diagonal: P(a|s,a) = (1 + 1/S) / 2
        # For each action a, the probability of transitioning to state a
        # (the "matching" state) is elevated.
        diagonal_prob: float = (1.0 + 1.0 / S) / 2.0
        for a in range(A):
            # P[:, a, a] sets P[s, a, a] for all states s simultaneously.
            # The target state index is a % S to handle A > S edge cases,
            # though in the paper's experiments A = S always.
            target_state: int = a % S
            P[:, a, target_state] = diagonal_prob

        # Numerical safety: clip and renormalize to ensure exact stochasticity
        P = np.clip(P, 0.0, 1.0)
        row_sums: ndarray = P.sum(axis=2, keepdims=True)
        P = P / row_sums

        return P

    def _make_uniform_kernel(self, S: int, A: int) -> ndarray:
        """Construct the uniform transition kernel.

        Implements the kernel from Appendix C.3 (uniform variant):
            P(s' | s, a) = 1/S  for all s, a, s'

        This kernel is independent of both the current state s and the
        action a. Consequently, C_p = 0 exactly because P^{π'} = P^π
        for all policies π, π' (the induced transition matrix is always
        the same uniform matrix regardless of policy).

        This is the "trivial MDP" case mentioned in Section 3.2 of the
        paper, where every policy is optimal and convergence is immediate.

        Used in: Experiment 3 (uniform variant).

        Args:
            S: Number of states. Must be a positive integer.
            A: Number of actions. Must be a positive integer.

        Returns:
            Transition kernel of shape (S, A, S) with all entries equal
            to 1/S. Each slice P[s, a, :] sums to 1.0 exactly.
        """
        P: ndarray = np.ones((S, A, S), dtype=np.float64) / S
        return P

    def _make_deterministic_kernel(self, S: int, A: int) -> ndarray:
        """Construct a deterministic transition kernel via random permutation.

        Implements the kernel from Appendix C.3 (deterministic variant):
        "P(·|s,·) viewed as S×A matrix, assigned a random permutation of
        the identity matrix. Every state leads to a different one."

        Interpretation: Generate a random permutation sigma of {0,...,S-1}.
        For action a, the target state is sigma[a % S] for all current
        states s. This gives a deterministic MDP where:
            P[s, a, sigma[a % S]] = 1.0  for all s
            P[s, a, s'] = 0.0            for s' ≠ sigma[a % S]

        The permutation is shared across all states (target depends only
        on action, not current state), which is the most natural reading
        of "random permutation of the identity matrix" applied to the
        S×A matrix P(·|s,·).

        This gives the highest C_p among the three kernel types because
        small changes in policy (mixing actions) cause large changes in
        the transition distribution (from one deterministic target to
        another).

        Used in: Experiment 3 (deterministic variant).

        Args:
            S: Number of states. Must be a positive integer.
            A: Number of actions. Should equal S for the permutation to
                cover all states. In the paper's Experiment 3, S = A = 16.

        Returns:
            Transition kernel of shape (S, A, S). Each slice P[s, a, :]
            is a one-hot vector with exactly one entry equal to 1.0.
            Each slice P[s, a, :] sums to 1.0 exactly.
        """
        P: ndarray = np.zeros((S, A, S), dtype=np.float64)

        # Generate a single random permutation of {0, ..., S-1}
        # This permutation maps action index a to target state sigma[a % S].
        sigma: ndarray = np.random.permutation(S)

        # For each action a, set P[s, a, sigma[a % S]] = 1 for all states s
        for a in range(A):
            target_state: int = int(sigma[a % S])
            P[:, a, target_state] = 1.0

        return P

    def _random_dirichlet_kernel(self, S: int, A: int) -> ndarray:
        """Construct a random stochastic transition kernel via Dirichlet sampling.

        Implements the "randomly generated transition kernel" from Appendix C.2.
        Each row P[s, a, :] is independently drawn from Dirichlet(alpha * 1_S)
        where alpha = config.complexity.dirichlet_alpha = 1.0 (from config.yaml).

        With alpha=1.0, this is equivalent to sampling uniformly from the
        probability simplex for each (s, a) pair, giving a fully random
        stochastic kernel with diverse transition probabilities.

        The global numpy random seed (set in __init__) ensures this kernel
        is reproducible. The kernel is generated once and shared across all
        reward variants in Experiment 2.

        Used in: Experiment 2 (shared kernel across all reward variants).

        Args:
            S: Number of states. Must be a positive integer.
            A: Number of actions. Must be a positive integer.

        Returns:
            Transition kernel of shape (S, A, S). Each slice P[s, a, :]
            is a valid probability vector drawn from Dirichlet(alpha * 1_S).
            Each slice P[s, a, :] sums to 1.0 (guaranteed by Dirichlet).
        """
        alpha: float = self.config.dirichlet_alpha
        alpha_vec: ndarray = alpha * np.ones(S, dtype=np.float64)

        P: ndarray = np.zeros((S, A, S), dtype=np.float64)
        for s in range(S):
            for a in range(A):
                P[s, a, :] = np.random.dirichlet(alpha_vec)

        return P

    # =========================================================================
    # Private: Reward Construction Methods
    # =========================================================================

    def _make_maxvar_reward(self, S: int, A: int) -> ndarray:
        """Construct the maximal variance reward function.

        Implements the reward from Appendix C.1:
        "rewards of half the actions to 1 and the rest to -1, for every state."

        Concretely:
            r[s, a] = +1  if a < A//2
            r[s, a] = -1  if a >= A//2

        This is the "maximal variance" reward used in Experiment 1 (all MDP
        sizes) and as the base for Experiment 3's reward function.

        The C_r constant is maximized for this reward because the reward
        changes maximally (from +1 to -1) with small changes in policy
        that shift probability mass between the two halves of actions.

        Used in: Experiment 1 (all sizes).

        Args:
            S: Number of states. Must be a positive integer.
            A: Number of actions. Must be a positive integer. Should be
                even for a clean 50/50 split; if odd, A//2 actions get -1.

        Returns:
            Reward array of shape (S, A). Entry [s, a] is +1 for a < A//2
            and -1 for a >= A//2. All states have the same reward structure.
        """
        r: ndarray = np.ones((S, A), dtype=np.float64)
        # Second half of actions get reward -1
        r[:, A // 2:] = -1.0
        return r

    def _make_reward_by_variance(self, S: int, A: int, level: str) -> ndarray:
        """Construct a reward function with specified variance level.

        Implements the reward construction from Appendix C.2:
        "r(s,a) = 0 for any s and a except for one state s_0."
        For state s_0 (= config.exp2_special_state = 0):
            - 'no_variance':  all actions get +1 (fraction_negative = 0.0)
            - 'low_variance': 1/8 of actions get -1 (fraction_negative = 0.125)
            - 'high_variance': 1/4 of actions get -1 (fraction_negative = 0.25)
            - 'max_variance': 1/2 of actions get -1 (fraction_negative = 0.5)

        The first n_negative = int(fraction_negative * A) actions at s_0
        are assigned reward -1; the remaining actions get +1.

        Higher fraction_negative → higher reward variance → larger C_r →
        slower convergence per Theorem 1 of the paper.

        Used in: Experiment 2 (all variance levels), Experiment 3 (high
        variance as the shared reward function).

        Args:
            S: Number of states. Must be a positive integer.
            A: Number of actions. Must be a positive integer.
            level: Variance level string. Must be one of:
                'no_variance', 'low_variance', 'high_variance', 'max_variance'.
                These match the keys in config.exp2_reward_variants.

        Returns:
            Reward array of shape (S, A). All entries are 0.0 except for
            state s_0 (index config.exp2_special_state), where:
                - First n_negative actions have reward -1.0
                - Remaining actions have reward +1.0

        Raises:
            KeyError: If level is not a valid key in config.exp2_reward_variants.
        """
        # Look up fraction_negative from config
        variant_config: Dict = self.config.exp2_reward_variants[level]
        fraction_negative: float = float(variant_config['fraction_negative'])

        # Initialize all rewards to zero
        r: ndarray = np.zeros((S, A), dtype=np.float64)

        # Special state index from config (= 0 per config.yaml)
        s0: int = self.config.exp2_special_state

        # Set all actions at s0 to +1 first
        r[s0, :] = 1.0

        # Override first n_negative actions at s0 with -1
        n_negative: int = int(fraction_negative * A)
        if n_negative > 0:
            r[s0, :n_negative] = -1.0

        return r

    # =========================================================================
    # Public: MDP Builder Methods
    # =========================================================================

    def build_exp1_mdp(self, S: int, A: int) -> MDP:
        """Build the MDP for Experiment 1 (convergence vs state/action size).

        Constructs an MDP with:
        - Non-uniform stochastic transition kernel (Appendix C.1)
        - Maximal variance reward function (Appendix C.1)

        The same construction is used for all three MDP sizes in Experiment 1:
        (3,3), (9,9), and (81,81). The structural similarity across sizes
        allows isolating the effect of |S| and |A| on convergence.

        Args:
            S: Number of states. One of {3, 9, 81} for Experiment 1.
            A: Number of actions. One of {3, 9, 81} for Experiment 1.
                Should equal S for the non-uniform kernel to be well-defined.

        Returns:
            MDP instance with non-uniform kernel and max-variance reward.
            Passes mdp.validate() (stochastic kernel, finite rewards).
        """
        P: ndarray = self._make_nonuniform_kernel(S, A)
        r: ndarray = self._make_maxvar_reward(S, A)
        mdp: MDP = MDP(S, A, P, r)
        mdp.validate()
        return mdp

    def build_exp2_mdp(self, level: str, P_fixed: ndarray) -> MDP:
        """Build the MDP for Experiment 2 (convergence vs reward variance).

        Constructs an MDP with:
        - Pre-generated shared random Dirichlet kernel P_fixed
        - Reward function with specified variance level (Appendix C.2)

        The shared kernel ensures that all four reward variants use the
        same transition dynamics, isolating the effect of reward variance
        (C_r) on convergence.

        Args:
            level: Variance level string. One of:
                'no_variance', 'low_variance', 'high_variance', 'max_variance'.
                Maps to fraction_negative in config.exp2_reward_variants.
            P_fixed: Pre-generated transition kernel of shape (S, A, S)
                where S = A = config.exp2_size[0] = 16. Generated once
                in run_exp2() and shared across all variance levels.

        Returns:
            MDP instance with the shared kernel and variance-specific reward.
            Passes mdp.validate() (stochastic kernel, finite rewards).
        """
        S: int = self.config.exp2_size[0]
        A: int = self.config.exp2_size[1]
        r: ndarray = self._make_reward_by_variance(S, A, level)
        mdp: MDP = MDP(S, A, P_fixed, r)
        mdp.validate()
        return mdp

    def build_exp3_mdp(self, kernel_type: str, r_fixed: ndarray) -> MDP:
        """Build the MDP for Experiment 3 (convergence vs transition kernel).

        Constructs an MDP with:
        - Transition kernel of the specified type (Appendix C.3)
        - Pre-generated shared high-variance reward r_fixed

        The shared reward ensures that all three kernel variants use the
        same reward structure, isolating the effect of the transition kernel
        (C_p) on convergence.

        Args:
            kernel_type: Kernel type string. One of:
                'uniform', 'nonuniform', 'deterministic'.
                Maps to the kernel construction methods.
            r_fixed: Pre-generated reward array of shape (S, A) where
                S = A = config.exp3_size[0] = 16. Generated once in
                run_exp3() using the high-variance construction.

        Returns:
            MDP instance with the specified kernel and shared reward.
            Passes mdp.validate() (stochastic kernel, finite rewards).

        Raises:
            ValueError: If kernel_type is not one of the three valid types.
        """
        S: int = self.config.exp3_size[0]
        A: int = self.config.exp3_size[1]

        # Dispatch to the appropriate kernel builder
        if kernel_type == 'uniform':
            P: ndarray = self._make_uniform_kernel(S, A)
        elif kernel_type == 'nonuniform':
            P = self._make_nonuniform_kernel(S, A)
        elif kernel_type == 'deterministic':
            P = self._make_deterministic_kernel(S, A)
        else:
            raise ValueError(
                f"Unknown kernel_type '{kernel_type}'. "
                "Must be one of: 'uniform', 'nonuniform', 'deterministic'."
            )

        mdp: MDP = MDP(S, A, P, r_fixed)
        mdp.validate()
        return mdp

    # =========================================================================
    # Private: Step Size Computation
    # =========================================================================

    def _compute_eta(self, L2: float) -> float:
        """Compute the PPG step size from the restricted smoothness constant.

        Implements the step size strategy from config.yaml:
            eta = step_size_multiplier / L2  if L2 > 1e-10
            eta = step_size_fallback          otherwise

        The paper's Theorem 1 requires eta < 1/L_2^Π. Using
        step_size_multiplier = 0.5 gives eta = 0.5/L2, which satisfies
        this condition with a safety factor of 2.

        The fallback step_size_fallback = 0.01 is used when L2 ≈ 0,
        which occurs for trivial MDPs (e.g., uniform kernel where C_p = 0
        and C_r = 0). In such cases, any positive step size is valid.

        Args:
            L2: Estimated restricted smoothness constant L_2^Π from
                ComplexityMetrics.compute_all(). Non-negative float.
                Returns 0.0 for trivial MDPs.

        Returns:
            Step size eta as a positive float. Satisfies eta < 1/L2
            when L2 > 1e-10 (by construction with multiplier 0.5 < 1).
        """
        if L2 > 1.0e-10:
            eta: float = self.config.step_size_multiplier / L2
        else:
            eta = self.config.step_size_fallback
        return float(eta)

    # =========================================================================
    # Private: Monotonicity Validation
    # =========================================================================

    def _check_monotonicity(
        self,
        reward_history: List[float],
        label: str,
    ) -> None:
        """Check that the reward history is approximately monotonically non-decreasing.

        Lemma 5 of the paper guarantees that when eta < 1/L_2^Π:
            rho^{pi_{k+1}} >= rho^{pi_k}  for all k >= 0

        Violations indicate numerical issues (e.g., ill-conditioned linear
        system in projected_value_function, or step size too large).
        Issues are reported as warnings rather than errors to allow the
        experiment to complete and produce diagnostic plots.

        Args:
            reward_history: List of average rewards from PPG run.
                reward_history[k] = rho^{pi_k}.
            label: Descriptive label for the MDP (used in warning messages).
        """
        rewards: ndarray = np.array(reward_history)
        diffs: ndarray = np.diff(rewards)

        # Allow small numerical violations (tolerance: 1e-6)
        n_violations: int = int(np.sum(diffs < -1.0e-6))
        if n_violations > 0:
            max_violation: float = float(np.min(diffs))
            warnings.warn(
                f"[{label}] Reward history has {n_violations} monotonicity "
                f"violations (max decrease: {max_violation:.2e}). "
                "This may indicate numerical issues or a step size that is "
                "too large. Consider reducing step_size_multiplier in config.",
                RuntimeWarning,
                stacklevel=3,
            )

    # =========================================================================
    # Public: Run Methods
    # =========================================================================

    def run_exp1(self) -> Dict[Tuple[int, int], Dict]:
        """Run Experiment 1: Convergence vs State/Action Space Size (Figure 1a).

        Reproduces Figure 1(a) from Section 4 / Appendix C.1 of the paper.
        Tests three MDP sizes: (3,3), (9,9), (81,81) with the same structural
        construction (non-uniform kernel, max-variance reward) to isolate the
        effect of |S| and |A| on convergence speed.

        Expected result: Larger (|S|, |A|) → larger L_2^Π → slower convergence.
        The (3,3) MDP converges fastest; the (81,81) MDP converges slowest.

        Algorithm:
            For each (S, A) in config.exp1_sizes:
                1. Build MDP with non-uniform kernel and max-variance reward
                2. Estimate L_2^Π via ComplexityMetrics.compute_all()
                3. Set eta = 0.5 / L_2^Π (or fallback)
                4. Run PPG for config.exp1_iterations = 2000 iterations
                5. Store reward_history and complexity metrics

        Returns:
            Dictionary keyed by (S, A) tuples. Each value is a dict with:
                'reward_history': List[float] of length exp1_iterations (2000).
                    reward_history[k] = rho^{pi_k} (average reward at iteration k).
                'policy_history': List[ndarray] of sparse policy snapshots
                    (every 100th iterate). Each array has shape (S, A).
                'complexity': Dict with keys 'C_m', 'C_p', 'C_r', 'kappa_r',
                    'L2', 'eta_max' from ComplexityMetrics.compute_all().
                'eta': float — step size used for this MDP.

        Example:
            >>> results = experiments.run_exp1()
            >>> rewards_3x3 = results[(3, 3)]['reward_history']
            >>> rewards_81x81 = results[(81, 81)]['reward_history']
        """
        # Re-seed for reproducibility (ensures same MDP constructions
        # regardless of what ran before this method).
        np.random.seed(self.config.random_seed)

        results: Dict[Tuple[int, int], Dict] = {}

        mdp_sizes: List[Tuple[int, int]] = self.config.exp1_sizes
        n_iterations: int = self.config.exp1_iterations
        n_samples: int = self.config.complexity_n_samples

        for size_pair in mdp_sizes:
            S: int = int(size_pair[0])
            A: int = int(size_pair[1])
            label: str = f"|S|={S},|A|={A}"

            print(f"  [Exp1] Building MDP {label}...")

            # Step 1: Build MDP
            mdp: MDP = self.build_exp1_mdp(S, A)

            # Step 2: Estimate complexity constants and step size
            print(f"  [Exp1] Computing complexity metrics for {label}...")
            cm: ComplexityMetrics = ComplexityMetrics(
                mdp,
                dirichlet_alpha=self.config.dirichlet_alpha,
            )
            metrics: Dict[str, float] = cm.compute_all(n_samples=n_samples)
            eta: float = self._compute_eta(metrics['L2'])

            print(
                f"  [Exp1] {label}: C_m={metrics['C_m']:.4f}, "
                f"C_p={metrics['C_p']:.4f}, C_r={metrics['C_r']:.4f}, "
                f"kappa_r={metrics['kappa_r']:.4f}, L2={metrics['L2']:.4f}, "
                f"eta={eta:.6f}"
            )

            # Step 3: Run PPG
            print(
                f"  [Exp1] Running PPG for {label} "
                f"({n_iterations} iterations, eta={eta:.6f})..."
            )
            pg: PolicyGradient = PolicyGradient(mdp, eta)
            pi_init: ndarray = pg.uniform_policy()
            reward_history: List[float]
            policy_history: List[ndarray]
            reward_history, policy_history = pg.run(
                n_iterations=n_iterations,
                pi_init=pi_init,
            )

            # Step 4: Validate monotonicity (Lemma 5 check)
            self._check_monotonicity(reward_history, label)

            # Step 5: Log final reward
            final_reward: float = reward_history[-1]
            initial_reward: float = reward_history[0]
            print(
                f"  [Exp1] {label}: initial_rho={initial_reward:.6f}, "
                f"final_rho={final_reward:.6f}, "
                f"improvement={final_reward - initial_reward:.6f}"
            )

            # Step 6: Store results
            results[(S, A)] = {
                'reward_history': reward_history,
                'policy_history': policy_history,
                'complexity': metrics,
                'eta': eta,
            }

        return results

    def run_exp2(self) -> Dict[str, Dict]:
        """Run Experiment 2: Convergence vs Reward Variance / C_r (Figure 1b).

        Reproduces Figure 1(b) from Section 4 / Appendix C.2 of the paper.
        Tests four reward variance levels on a fixed (16,16) MDP with a
        shared randomly-generated transition kernel to isolate the effect
        of reward variance (C_r) on convergence speed.

        The shared kernel P_fixed is generated once before the loop using
        the fixed random seed. This ensures all four reward variants use
        identical transition dynamics, making C_r the only varying factor.

        Expected result: Higher reward variance → larger C_r → larger L_2^Π
        → slower convergence. 'no_variance' converges fastest; 'max_variance'
        converges slowest.

        Algorithm:
            1. Generate shared P_fixed via Dirichlet sampling (once)
            2. For each variance level in ['no_variance', 'low_variance',
               'high_variance', 'max_variance']:
                a. Build MDP with P_fixed and variance-specific reward
                b. Estimate L_2^Π and set eta
                c. Run PPG for config.exp2_iterations = 2000 iterations
                d. Store reward_history, complexity metrics, and label

        Returns:
            Dictionary keyed by variance level strings. Each value is a dict:
                'reward_history': List[float] of length exp2_iterations (2000).
                    reward_history[k] = rho^{pi_k}.
                'complexity': Dict with keys 'C_m', 'C_p', 'C_r', 'kappa_r',
                    'L2', 'eta_max' from ComplexityMetrics.compute_all().
                'eta': float — step size used for this variance level.
                'label': str — human-readable label (e.g., 'No Variance').

        Example:
            >>> results = experiments.run_exp2()
            >>> rewards_novar = results['no_variance']['reward_history']
            >>> rewards_maxvar = results['max_variance']['reward_history']
        """
        # Re-seed for reproducibility before generating the shared kernel
        np.random.seed(self.config.random_seed)

        S: int = self.config.exp2_size[0]  # 16
        A: int = self.config.exp2_size[1]  # 16
        n_iterations: int = self.config.exp2_iterations  # 2000
        n_samples: int = self.config.complexity_n_samples  # 200

        # Step 1: Generate shared random Dirichlet kernel (once)
        # This kernel is shared across all four reward variants