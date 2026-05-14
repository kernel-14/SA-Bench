## complexity.py
"""MDP complexity constants estimation for average reward policy gradient analysis.

This module implements the ComplexityMetrics class, which numerically estimates
the MDP complexity constants defined in Table 1/2 of:
    Murthy et al., "Global Convergence of Policy Gradient in Average Reward
    MDPs", ICLR 2024.

The constants C_m, C_p, C_r, κ_r, and L_2^Π characterize the "hardness" of
an MDP and appear directly in the convergence bound of Theorem 1:
    ρ* - ρ^{π_k} ≤ 1 / (1/(ρ* - ρ^{π_0}) + ν·k)

where ν depends on L_2^Π and C_PL. Since exact computation requires maximizing
over all policies (intractable for large MDPs), all constants are estimated via
Monte Carlo sampling of random Dirichlet policies.

Configuration values from config.yaml:
    complexity.n_samples: 200       — number of random policies sampled
    complexity.dirichlet_alpha: 1.0 — Dirichlet concentration parameter

Mathematical definitions (Table 1/2 of the paper):
    C_m   = max_π ||(I - Φ P^π)^{-1}||_∞         (lowest mixing rate)
    C_p   = max_{π,π'} ||P^{π'} - P^π||_∞ / ||π' - π||_2  (kernel diameter)
    C_r   = max_{π,π'} ||r^{π'} - r^π||_∞ / ||π' - π||_2  (reward diameter)
    κ_r   = max_π ||Φ r^π||_∞                     (reward variance)
    L_2^Π = 4*(C_p²C_m²κ_r + C_pC_mC_r + (C_p+1)(C_m²C_pκ_r + C_mC_r)
              + 4*(C_m³C_p²κ_r + C_m²C_pC_r))     (restricted smoothness)

Sanity bounds from Lemma 18:
    ||Φ||_∞ ≤ 2,  ||P^π||_∞ ≤ 1,  κ_r ≤ 2,  C_p ≤ √|A|,  C_r ≤ √|A|
"""

from __future__ import annotations

import warnings
from typing import Dict, List

import numpy as np
import scipy.linalg
from numpy import ndarray

from mdp import MDP
from value_functions import ValueFunctions
import utils


class ComplexityMetrics:
    """Estimates MDP complexity constants from Table 1/2 of the paper.

    All constants are estimated via Monte Carlo sampling of random Dirichlet
    policies. The number of samples and Dirichlet concentration parameter
    are taken from config.yaml (complexity.n_samples=200,
    complexity.dirichlet_alpha=1.0).

    The class is designed to be instantiated once per MDP and then queried
    for individual constants or all constants at once via compute_all().

    Attributes:
        mdp: The MDP instance providing transition kernel P and reward r.
        vf: ValueFunctions instance for projected value function computations.
            Shared across all metric computations to avoid redundant Phi
            precomputation.
        Phi: Projection matrix of shape (S, S). Phi = I - 11^T/|S|.
            Precomputed once for reuse in compute_C_m and compute_kappa_r.
        S: Number of states (convenience alias for mdp.S).
        A: Number of actions (convenience alias for mdp.A).
        I: Identity matrix of shape (S, S). Precomputed for reuse in
            compute_C_m when forming (I - Phi @ P_pi).
        _dirichlet_alpha: Dirichlet concentration parameter for policy
            sampling. Default 1.0 from config.yaml complexity.dirichlet_alpha.
    """

    def __init__(self, mdp: MDP, dirichlet_alpha: float = 1.0) -> None:
        """Initialize ComplexityMetrics for a given MDP.

        Precomputes the projection matrix Phi = I - 11^T/|S| and the
        identity matrix I, both of shape (S, S), to avoid redundant
        allocation across multiple method calls.

        Also instantiates a ValueFunctions object for the MDP, which
        itself precomputes Phi. While this creates a second copy of Phi,
        it is necessary because ValueFunctions is used for its
        projected_value_function method in potential extensions.

        Args:
            mdp: The MDP instance. Must have valid S, A, P, r attributes.
                Used to compute all complexity constants.
            dirichlet_alpha: Concentration parameter for Dirichlet policy
                sampling. Default 1.0 from config.yaml
                complexity.dirichlet_alpha. With alpha=1.0, each row of
                the sampled policy is drawn uniformly from the simplex.
                Larger values produce policies closer to uniform; smaller
                values produce more peaked (near-deterministic) policies.
        """
        self.mdp: MDP = mdp
        self.S: int = mdp.S
        self.A: int = mdp.A

        # Instantiate ValueFunctions — precomputes Phi internally and
        # provides projected_value_function for potential extensions.
        self.vf: ValueFunctions = ValueFunctions(mdp)

        # Precompute projection matrix Phi = I - 11^T/|S|, shape (S, S)
        # Used directly in compute_C_m (forming I - Phi @ P_pi) and
        # compute_kappa_r (computing Phi @ r_pi).
        self.Phi: ndarray = utils.make_projection_matrix(mdp.S)

        # Precompute identity matrix for reuse in compute_C_m
        self.I: ndarray = np.eye(mdp.S, dtype=np.float64)

        # Store Dirichlet concentration parameter
        self._dirichlet_alpha: float = float(dirichlet_alpha)

    def _sample_random_policies(self, n: int) -> List[ndarray]:
        """Sample n random stochastic policies using the Dirichlet distribution.

        Each policy is sampled by drawing each row independently from
        Dirichlet(alpha * ones(A)), where alpha = self._dirichlet_alpha
        (default 1.0 from config.yaml). With alpha=1.0, this samples
        uniformly over the probability simplex for each state.

        The global numpy random seed set in Config.__post_init__ ensures
        reproducibility across all calls to this method.

        Args:
            n: Number of random policies to sample. Each policy has shape
                (S, A) with each row being a valid probability vector.

        Returns:
            List of n policy arrays, each of shape (S, A). Every row of
            every policy satisfies: pi[s, :] >= 0 and sum(pi[s, :]) == 1.
        """
        policies: List[ndarray] = [
            utils.sample_dirichlet_policy(self.S, self.A, alpha=self._dirichlet_alpha)
            for _ in range(n)
        ]
        return policies

    def _operator_norm_inf(self, A: ndarray) -> float:
        """Compute the L_infinity operator norm of a matrix.

        Thin wrapper around utils.operator_norm_inf. The L_∞ operator norm
        is the maximum absolute row sum:
            ||A||_∞ = max_i sum_j |A[i,j]|

        This matches the paper's convention for C_m (Lemma 18, item 4)
        and C_p (Table 1/2).

        Args:
            A: Matrix of shape (m, n). Can be any real-valued matrix.

        Returns:
            The L_infinity operator norm as a non-negative float.
        """
        return utils.operator_norm_inf(A)

    def compute_C_m(self, n_samples: int = 200) -> float:
        """Estimate C_m = max_π ||(I - Φ P^π)^{-1}||_∞.

        C_m captures the lowest rate of mixing across all policies. It is
        the maximum L_∞ operator norm of the matrix (I - Φ P^π)^{-1} over
        all policies π ∈ Π. Lemma 12 guarantees this inverse exists for
        all irreducible, aperiodic chains (Assumption 1).

        Paper bound (Lemma 18, item 4): C_m ≤ 2·C_e·|S| / (1-λ)
        where C_e and λ are the geometric ergodicity constants.

        Algorithm:
            1. Sample n_samples random policies via _sample_random_policies
            2. For each policy π:
               a. Compute P^π = mdp.get_policy_transition(π)
               b. Form A_mat = I - Φ @ P^π
               c. Invert A_mat via scipy.linalg.inv
               d. Compute ||A_mat^{-1}||_∞ via _operator_norm_inf
            3. Return max over all computed norms

        Numerical stability: scipy.linalg.inv may fail for near-singular
        matrices (e.g., near-deterministic policies). Such samples are
        skipped with a warning.

        Args:
            n_samples: Number of random policies to sample. Default 200
                from config.yaml complexity.n_samples.

        Returns:
            Estimated C_m as a non-negative float. Returns 1.0 as a safe
            fallback if all samples fail (should not occur for valid MDPs
            satisfying Assumption 1).
        """
        policies: List[ndarray] = self._sample_random_policies(n_samples)

        max_norm: float = 0.0
        n_failed: int = 0

        for pi in policies:
            # Compute policy-induced transition matrix P^π, shape (S, S)
            P_pi: ndarray = self.mdp.get_policy_transition(pi)

            # Form coefficient matrix A_mat = I - Φ @ P^π, shape (S, S)
            # Lemma 12 guarantees this is invertible for irreducible/aperiodic chains.
            A_mat: ndarray = self.I - self.Phi @ P_pi

            try:
                # Invert A_mat using scipy.linalg.inv
                M: ndarray = scipy.linalg.inv(A_mat)

                # Compute L_∞ operator norm of the inverse
                norm: float = self._operator_norm_inf(M)

                # Update running maximum
                if norm > max_norm:
                    max_norm = norm

            except (scipy.linalg.LinAlgError, np.linalg.LinAlgError):
                # Near-singular matrix — skip this sample
                n_failed += 1

        if n_failed > 0:
            warnings.warn(
                f"compute_C_m: {n_failed}/{n_samples} samples failed due to "
                "near-singular (I - Φ P^π). These were skipped.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Fallback: return 1.0 if all samples failed (safe lower bound)
        if max_norm == 0.0 and n_failed == n_samples:
            warnings.warn(
                "compute_C_m: All samples failed. Returning fallback value 1.0.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 1.0

        return float(max(max_norm, 0.0))

    def compute_C_p(self, n_samples: int = 200) -> float:
        """Estimate C_p = max_{π,π'} ||(P^{π'} - P^π)||_∞ / ||π' - π||_2.

        C_p captures the diameter of the transition kernel as a function of
        the policy class. It measures how much the induced transition matrix
        changes per unit change in policy (in L_2 norm).

        Paper bound (Lemma 18, item 5): C_p ≤ √|A|.

        Algorithm:
            1. Sample 2*n_samples random policies to form n_samples pairs
               (pi, pi_prime) — each pair is independent
            2. For each pair (π, π'):
               a. Compute P^π and P^{π'} via get_policy_transition
               b. Compute diff_P = P^{π'} - P^π, shape (S, S)
               c. Compute op_norm = ||(diff_P)||_∞ (L_∞ operator norm)
               d. Compute pi_diff_norm = ||π' - π||_2 (Frobenius/L_2 norm
                  of the flattened policy difference, shape (S*A,))
               e. If pi_diff_norm > 1e-10: ratio = op_norm / pi_diff_norm
            3. Return max over all valid ratios

        Norm convention: ||π' - π||_2 is the Euclidean norm of the
        flattened (S*A)-dimensional vector, consistent with the paper's
        Table 1 definition.

        Args:
            n_samples: Number of policy pairs to evaluate. Default 200
                from config.yaml complexity.n_samples. Internally samples
                2*n_samples policies to form n_samples independent pairs.

        Returns:
            Estimated C_p as a non-negative float. Returns 0.0 if all
            policy pairs are too similar (||π' - π||_2 < 1e-10), which
            occurs for trivial MDPs where all policies are equivalent.
        """
        # Sample 2*n_samples policies to form n_samples independent pairs
        all_policies: List[ndarray] = self._sample_random_policies(2 * n_samples)

        max_ratio: float = 0.0

        for i in range(n_samples):
            pi: ndarray = all_policies[2 * i]
            pi_prime: ndarray = all_policies[2 * i + 1]

            # Compute policy difference norm ||π' - π||_2 (flattened L_2 norm)
            pi_diff: ndarray = (pi_prime - pi).ravel()
            pi_diff_norm: float = float(np.linalg.norm(pi_diff))

            # Skip pairs that are too similar to avoid numerical instability
            if pi_diff_norm < 1e-10:
                continue

            # Compute policy-induced transition matrices
            P_pi: ndarray = self.mdp.get_policy_transition(pi)
            P_pi_prime: ndarray = self.mdp.get_policy_transition(pi_prime)

            # Compute difference matrix and its L_∞ operator norm
            diff_P: ndarray = P_pi_prime - P_pi
            op_norm: float = self._operator_norm_inf(diff_P)

            # Compute ratio and update maximum
            ratio: float = op_norm / pi_diff_norm
            if ratio > max_ratio:
                max_ratio = ratio

        return float(max(max_ratio, 0.0))

    def compute_C_r(self, n_samples: int = 200) -> float:
        """Estimate C_r = max_{π,π'} ||r^{π'} - r^π||_∞ / ||π' - π||_2.

        C_r captures the diameter of the single-step reward function as a
        function of the policy class. It measures how much the expected
        reward vector changes per unit change in policy.

        Paper bound (Lemma 18, item 6): C_r ≤ √|A|.

        For the "no variance" reward in Experiment 2 (all rewards equal),
        C_r = 0 exactly since r^{π'} = r^π for all π, π'.

        Algorithm:
            1. Sample 2*n_samples random policies to form n_samples pairs
            2. For each pair (π, π'):
               a. Compute r^π and r^{π'} via get_policy_reward, shape (S,)
               b. Compute diff_r = r^{π'} - r^π, shape (S,)
               c. Compute inf_norm = ||diff_r||_∞ = max|diff_r[s]|
               d. Compute pi_diff_norm = ||π' - π||_2 (flattened L_2 norm)
               e. If pi_diff_norm > 1e-10: ratio = inf_norm / pi_diff_norm
            3. Return max over all valid ratios

        Args:
            n_samples: Number of policy pairs to evaluate. Default 200
                from config.yaml complexity.n_samples. Internally samples
                2*n_samples policies to form n_samples independent pairs.

        Returns:
            Estimated C_r as a non-negative float. Returns 0.0 for trivial
            reward functions where all policies yield the same reward vector
            (e.g., "no variance" case in Experiment 2).
        """
        # Sample 2*n_samples policies to form n_samples independent pairs
        all_policies: List[ndarray] = self._sample_random_policies(2 * n_samples)

        max_ratio: float = 0.0

        for i in range(n_samples):
            pi: ndarray = all_policies[2 * i]
            pi_prime: ndarray = all_policies[2 * i + 1]

            # Compute policy difference norm ||π' - π||_2 (flattened L_2 norm)
            pi_diff: ndarray = (pi_prime - pi).ravel()
            pi_diff_norm: float = float(np.linalg.norm(pi_diff))

            # Skip pairs that are too similar to avoid numerical instability
            if pi_diff_norm < 1e-10:
                continue

            # Compute policy-induced reward vectors, shape (S,)
            r_pi: ndarray = self.mdp.get_policy_reward(pi)
            r_pi_prime: ndarray = self.mdp.get_policy_reward(pi_prime)

            # Compute reward difference and its L_∞ (vector) norm
            diff_r: ndarray = r_pi_prime - r_pi
            inf_norm: float = float(np.max(np.abs(diff_r)))

            # Compute ratio and update maximum
            ratio: float = inf_norm / pi_diff_norm
            if ratio > max_ratio:
                max_ratio = ratio

        return float(max(max_ratio, 0.0))

    def compute_kappa_r(self, n_samples: int = 200) -> float:
        """Estimate κ_r = max_π ||Φ r^π||_∞.

        κ_r captures the variance of the single-step reward function across
        the class of policies. It measures the maximum deviation of the
        projected (zero-mean) reward vector from zero.

        Interpretation: Φ r^π is the centered reward vector (subtract mean).
        Large κ_r means the reward varies significantly across states for
        some policy, indicating high reward variance.

        Paper bound (Lemma 18, item 3): κ_r ≤ 2.
        This follows from ||Φ||_∞ ≤ 2 and ||r^π||_∞ ≤ 1 (rewards in [-1,1]).

        Algorithm:
            1. Sample n_samples random policies
            2. For each policy π:
               a. Compute r^π = mdp.get_policy_reward(π), shape (S,)
               b. Compute Φ r^π = self.Phi @ r_pi, shape (S,)
               c. Compute inf_norm = ||Φ r^π||_∞ = max|Φ r^π[s]|
            3. Return max over all computed inf_norms

        Args:
            n_samples: Number of random policies to sample. Default 200
                from config.yaml complexity.n_samples.

        Returns:
            Estimated κ_r as a non-negative float in [0, 2]. Returns 0.0
            for constant reward functions (e.g., "no variance" case where
            r[s,a] = 1 for all s,a, giving r^π = ones(S) and Φ r^π = 0).
        """
        policies: List[ndarray] = self._sample_random_policies(n_samples)

        max_norm: float = 0.0

        for pi in policies:
            # Compute policy-induced reward vector r^π, shape (S,)
            r_pi: ndarray = self.mdp.get_policy_reward(pi)

            # Apply projection: Φ r^π = (I - 11^T/S) r^π, shape (S,)
            # This centers the reward vector (subtracts its mean).
            Phi_r: ndarray = self.Phi @ r_pi

            # Compute L_∞ (vector) norm: max absolute value
            inf_norm: float = float(np.max(np.abs(Phi_r)))

            # Update running maximum
            if inf_norm > max_norm:
                max_norm = inf_norm

        return float(max(max_norm, 0.0))

    def compute_L2(self, n_samples: int = 200) -> float:
        """Compute L_2^Π using the formula from Lemma 4 of the paper.

        L_2^Π is the restricted smoothness constant of the average reward
        ρ^π with respect to the policy π. It appears in:
        - Theorem 1: step size condition η < 1/L_2^Π
        - Theorem 1: convergence rate ν = (1/(32·C_PL²·|S|·L_2^Π)) · (...)
        - Lemma 5: monotone improvement bound

        Formula from Lemma 4 (Table 1/2):
            L_2^Π = 4 * (C_p²·C_m²·κ_r + C_p·C_m·C_r
                        + (C_p+1)·(C_m²·C_p·κ_r + C_m·C_r)
                        + 4·(C_m³·C_p²·κ_r + C_m²·C_p·C_r))

        Note: This method independently samples n_samples policies for each
        sub-constant (C_m, C_p, C_r, κ_r), resulting in approximately
        4*n_samples total policy evaluations. For efficiency when all
        constants are needed, use compute_all() which computes each
        constant once and reuses the values.

        If L_2^Π = 0 (trivial MDP with C_p = 0 and C_r = 0), the step
        size η = 0.5/L_2^Π would be infinite. The caller (experiments.py)
        handles this via step_size_fallback. This method returns 0.0 in
        that case without special handling.

        Args:
            n_samples: Number of random policies to sample per sub-constant.
                Default 200 from config.yaml complexity.n_samples.

        Returns:
            Estimated L_2^Π as a non-negative float. Returns 0.0 for
            trivial MDPs where C_p = 0 and C_r = 0 (e.g., transition
            kernel independent of action).
        """
        # Compute all four base constants independently
        Cm: float = self.compute_C_m(n_samples)
        Cp: float = self.compute_C_p(n_samples)
        Cr: float = self.compute_C_r(n_samples)
        kr: float = self.compute_kappa_r(n_samples)

        # Apply the formula from Lemma 4 exactly
        # L_2^Π = 4*(C_p²C_m²κ_r + C_pC_mC_r + (C_p+1)(C_m²C_pκ_r + C_mC_r)
        #           + 4*(C_m³C_p²κ_r + C_m²C_pC_r))
        term1: float = Cp**2 * Cm**2 * kr
        term2: float = Cp * Cm * Cr
        term3: float = (Cp + 1.0) * (Cm**2 * Cp * kr + Cm * Cr)
        term4: float = 4.0 * (Cm**3 * Cp**2 * kr + Cm**2 * Cp * Cr)

        L2: float = 4.0 * (term1 + term2 + term3 + term4)

        return float(max(L2, 0.0))

    def compute_all(self, n_samples: int = 200) -> Dict[str, float]:
        """Compute all MDP complexity constants and return as a dictionary.

        Computes C_m, C_p, C_r, κ_r, L_2^Π, and η_max = 1/L_2^Π in a
        single call. Unlike compute_L2(), this method computes each base
        constant exactly once and reuses the values to compute L_2^Π,
        avoiding redundant policy sampling.

        The returned dictionary is used by:
        - experiments.py: to set the step size η = step_size_multiplier / L2
        - plotter.py: to display complexity metrics in the complexity table
        - main.py: to log complexity metrics for each MDP configuration

        Args:
            n_samples: Number of random policies to sample per base constant.
                Default 200 from config.yaml complexity.n_samples.
                Total policy evaluations: approximately 4*n_samples
                (n_samples for C_m and κ_r each, 2*n_samples pairs for
                C_p and C_r each).

        Returns:
            Dictionary with the following keys and values:
                'C_m':     float — estimated max_π ||(I - Φ P^π)^{-1}||_∞
                'C_p':     float — estimated max_{π,π'} ||P^{π'}-P^π||_∞/||π'-π||_2
                'C_r':     float — estimated max_{π,π'} ||r^{π'}-r^π||_∞/||π'-π||_2
                'kappa_r': float — estimated max_π ||Φ r^π||_∞
                'L2':      float — L_2^Π from Lemma 4 formula
                'eta_max': float — 1/L_2^Π (maximum valid step size);
                           float('inf') if L_2^Π = 0 (trivial MDP)

        Example:
            >>> metrics = ComplexityMetrics(mdp).compute_all(n_samples=200)
            >>> print(f"L2={metrics['L2']:.4f}, eta_max={metrics['eta_max']:.4f}")
        """
        # Compute each base constant once (no redundant sampling)
        Cm: float = self.compute_C_m(n_samples)
        Cp: float = self.compute_C_p(n_samples)
        Cr: float = self.compute_C_r(n_samples)
        kr: float = self.compute_kappa_r(n_samples)

        # Compute L_2^Π from the formula in Lemma 4 using already-computed values
        # This avoids the 4x redundant sampling that compute_L2() would incur.
        term1: float = Cp**2 * Cm**2 * kr
        term2: float = Cp * Cm * Cr
        term3: float = (Cp + 1.0) * (Cm**2 * Cp * kr + Cm * Cr)
        term4: float = 4.0 * (Cm**3 * Cp**2 * kr + Cm**2 * Cp * Cr)
        L2: float = float(max(4.0 * (term1 + term2 + term3 + term4), 0.0))

        # Compute maximum valid step size η_max = 1/L_2^Π
        # Returns float('inf') for trivial MDPs where L_2^Π = 0
        eta_max: float = (1.0 / L2) if L2 > 0.0 else float('inf')

        return {
            'C_m': Cm,
            'C_p': Cp,
            'C_r': Cr,
            'kappa_r': kr,
            'L2': L2,
            'eta_max': eta_max,
        }
