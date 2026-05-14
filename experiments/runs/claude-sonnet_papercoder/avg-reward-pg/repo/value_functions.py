## value_functions.py
"""Value function computations for finite average-reward MDPs.

This module implements the core mathematical operations from Section 2.1 and
Appendix A of:
    Murthy et al., "Global Convergence of Policy Gradient in Average Reward
    MDPs", ICLR 2024.

Key equations implemented:
    - Stationary distribution: d^pi P^pi = d^pi  (Section 2.1)
    - Projection matrix: Phi = I - 11^T/|S|  (Lemma 1)
    - Projected value function: v_phi^pi = (I - Phi P^pi)^{-1} Phi r^pi
      (Lemma 1, Equation 15)
    - Q-function: Q^pi[s,a] = r[s,a] + sum_{s'} P[s,a,s'] v_phi^pi[s'] - rho^pi
      (Section 2.1)
    - Average reward: rho^pi = sum_s d^pi(s) r^pi(s)  (Equation 3)
    - Policy iteration for optimal policy computation  (Theorem 1)
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from numpy import ndarray

from mdp import MDP
from utils import make_projection_matrix, power_iteration


class ValueFunctions:
    """Computes all value-related quantities for a given MDP.

    This class is the mathematical core of the PPG implementation. Every
    method maps directly to a specific equation or lemma in the paper.
    The class is stateless with respect to the policy — all methods accept
    a policy as an argument and return the corresponding value quantity.

    The only persistent state is the MDP reference and the precomputed
    projection matrix Phi, which depends only on |S| and is fixed for the
    lifetime of the object.

    Attributes:
        mdp: The MDP instance providing transition kernel P and reward r.
        Phi: Projection matrix of shape (S, S). Phi = I - 11^T/|S|.
            Precomputed once in __init__ for efficiency.
        I: Identity matrix of shape (S, S). Precomputed for reuse in
            projected_value_function.
        _power_iter_tol: Convergence tolerance for power iteration.
        _power_iter_max: Maximum iterations for power iteration.
    """

    def __init__(
        self,
        mdp: MDP,
        power_iter_tol: float = 1.0e-10,
        power_iter_max: int = 10000,
    ) -> None:
        """Initialize ValueFunctions with an MDP and solver settings.

        Precomputes the projection matrix Phi = I - 11^T/|S| once, since
        it depends only on |S| which is fixed for the lifetime of this object.
        This avoids redundant allocation of an (S, S) matrix at every PPG
        iteration.

        Args:
            mdp: The MDP instance. Must have valid S, A, P, r attributes.
            power_iter_tol: Convergence tolerance for power iteration used
                in stationary_distribution. Default 1e-10 matches
                config.yaml value_functions.power_iter_tol.
            power_iter_max: Maximum iterations for power iteration.
                Default 10000 matches config.yaml
                value_functions.power_iter_max_iter.
        """
        self.mdp: MDP = mdp

        # Precompute projection matrix Phi = I - 11^T/|S| (Lemma 1)
        # Shape: (S, S). Satisfies Phi @ ones(S) = zeros(S) and Phi @ Phi = Phi.
        self.Phi: ndarray = make_projection_matrix(mdp.S)

        # Precompute identity matrix for reuse in projected_value_function
        self.I: ndarray = np.eye(mdp.S, dtype=np.float64)

        # Store solver settings (matching config.yaml defaults)
        self._power_iter_tol: float = power_iter_tol
        self._power_iter_max: int = power_iter_max

    def stationary_distribution(self, pi: ndarray) -> ndarray:
        """Compute the stationary distribution d^pi of the induced Markov chain.

        Finds d^pi such that d^pi @ P^pi = d^pi, sum(d^pi) = 1, d^pi >= 0,
        using power iteration on the policy-induced transition matrix P^pi.

        Under Assumption 1 of the paper (irreducible, aperiodic P^pi for all
        pi in Pi), convergence of power iteration is guaranteed.

        Paper reference: Section 2.1 — "d^pi satisfies d^pi P^pi = d^pi".

        Args:
            pi: Policy array of shape (S, A). Each row pi[s, :] must be a
                valid probability vector.

        Returns:
            Stationary distribution of shape (S,). Satisfies:
                - d >= 0 (clipped to handle floating point drift)
                - sum(d) == 1 (normalized after iteration)
                - d @ P_pi ≈ d (stationarity, up to _power_iter_tol)
        """
        # Compute policy-induced transition matrix P^pi, shape (S, S)
        P_pi: ndarray = self.mdp.get_policy_transition(pi)

        # Run power iteration to find left eigenvector for eigenvalue 1
        d: ndarray = power_iteration(
            P_pi,
            tol=self._power_iter_tol,
            max_iter=self._power_iter_max,
        )
        return d

    def projection_matrix(self) -> ndarray:
        """Return the precomputed projection matrix Phi.

        Phi = I - 11^T/|S| projects any vector onto the subspace orthogonal
        to the all-ones vector. This is used to obtain a unique representation
        of the value function by removing the additive constant ambiguity
        inherent in average-reward Bellman equations.

        Paper reference: Lemma 1 — "Phi = (I - 11^T/|S|)".

        Returns:
            Projection matrix of shape (S, S). Satisfies:
                - Phi @ ones(S) = zeros(S)  (null space contains 1)
                - Phi @ Phi = Phi            (idempotent)
                - Phi^T = Phi                (symmetric)
        """
        return self.Phi

    def projected_value_function(self, pi: ndarray) -> ndarray:
        """Compute the unique projected value function v_phi^pi.

        Implements Equation 15 from Lemma 1 of the paper:
            v_phi^pi = (I - Phi P^pi)^{-1} Phi r^pi

        The projection Phi removes the additive constant ambiguity in the
        average-reward Bellman equation, yielding a unique value function
        satisfying v_phi^pi^T 1 = 0.

        Lemma 12 of the paper guarantees that (I - Phi P^pi) is invertible
        for all irreducible, aperiodic P^pi (Assumption 1). In practice,
        scipy.linalg.solve is used as the primary solver, with
        scipy.linalg.lstsq as a fallback for near-singular cases
        (e.g., deterministic transition kernels in Experiment 3).

        Paper reference: Lemma 1, Equation 15.

        Args:
            pi: Policy array of shape (S, A). Each row pi[s, :] must be a
                valid probability vector.

        Returns:
            Projected value function of shape (S,). Satisfies:
                - v_phi^T 1 ≈ 0  (zero mean under uniform distribution)
                - rho * ones(S) + v_phi ≈ r_pi + P_pi @ v_phi  (Bellman eq.)
        """
        S: int = self.mdp.S

        # Step 1: Compute policy-induced transition matrix P^pi, shape (S, S)
        P_pi: ndarray = self.mdp.get_policy_transition(pi)

        # Step 2: Compute policy-induced reward vector r^pi, shape (S,)
        r_pi: ndarray = self.mdp.get_policy_reward(pi)

        # Step 3: Form the coefficient matrix A_mat = I - Phi @ P^pi, shape (S, S)
        # Lemma 12 guarantees this is invertible for irreducible, aperiodic chains.
        A_mat: ndarray = self.I - self.Phi @ P_pi

        # Step 4: Form the right-hand side b = Phi @ r^pi, shape (S,)
        # This is the projected reward vector (zero-mean component of r^pi).
        b: ndarray = self.Phi @ r_pi

        # Step 5: Solve A_mat @ v = b for v using scipy.linalg.solve
        # Fall back to least-squares if the matrix is (near-)singular.
        try:
            v_phi: ndarray = scipy.linalg.solve(
                A_mat,
                b,
                assume_a='gen',  # general matrix, no symmetry assumption
                check_finite=True,
            )
        except scipy.linalg.LinAlgError:
            # Fallback: least-squares solution for near-singular A_mat
            # This can occur for deterministic transition kernels (Experiment 3)
            # where P^pi has entries in {0, 1} and A_mat may be ill-conditioned.
            v_phi, _, _, _ = scipy.linalg.lstsq(A_mat, b)

        return v_phi

    def q_function(self, pi: ndarray) -> ndarray:
        """Compute the relative Q-function Q^pi[s,a] for all (s,a) pairs.

        Implements the Q-function definition from Section 2.1 of the paper:
            Q^pi[s,a] = r[s,a] + sum_{s'} P[s,a,s'] v_phi^pi[s'] - rho^pi

        This is the relative state-action value function, which measures the
        advantage of taking action a in state s relative to the average reward.
        It satisfies sum_a pi[s,a] Q^pi[s,a] = v_phi^pi[s] for all s.

        Paper reference: Section 2.1, definition of Q^pi(s,a).

        Args:
            pi: Policy array of shape (S, A). Each row pi[s, :] must be a
                valid probability vector.

        Returns:
            Q-function array of shape (S, A). Entry [s, a] equals:
                r[s,a] + sum_{s'} P[s,a,s'] v_phi^pi[s'] - rho^pi
            Satisfies: np.einsum('sa,sa->s', pi, Q) ≈ v_phi^pi for all s.
        """
        # Step 1: Compute projected value function v_phi^pi, shape (S,)
        v_phi: ndarray = self.projected_value_function(pi)

        # Step 2: Compute average reward rho^pi (scalar)
        # Note: this internally calls stationary_distribution and
        # get_policy_reward. The get_policy_transition call is redundant
        # with projected_value_function above, but is accepted for clarity
        # per the design specification.
        rho: float = self.average_reward(pi)

        # Step 3: Compute expected future value for each (s,a) pair
        # future_val[s,a] = sum_{s'} P[s,a,s'] v_phi^pi[s']
        # einsum: 'sab,b->sa' where s=state, a=action, b=next state (contracted)
        future_val: ndarray = np.einsum('sab,b->sa', self.mdp.P, v_phi)

        # Step 4: Assemble Q-function
        # Q[s,a] = r[s,a] + future_val[s,a] - rho
        # Broadcasting: rho is a scalar subtracted from every entry.
        Q: ndarray = self.mdp.r + future_val - rho

        return Q

    def average_reward(self, pi: ndarray) -> float:
        """Compute the average reward rho^pi = sum_s d^pi(s) r^pi(s).

        Implements Equation 3 from Section 2.1 of the paper:
            rho^pi = sum_s d^pi(s) r^pi(s)

        where d^pi is the stationary distribution and r^pi[s] = sum_a pi[s,a] r[s,a]
        is the expected single-step reward at state s under policy pi.

        This is the primary performance metric tracked during PPG iterations.
        The paper's Theorem 1 bounds the convergence of rho^{pi_k} to rho*.

        Paper reference: Section 2.1, Equation 3.

        Args:
            pi: Policy array of shape (S, A). Each row pi[s, :] must be a
                valid probability vector.

        Returns:
            Average reward as a Python float. For the paper's reward functions
            (values in {-1, 0, +1}), this lies in [-1, 1].
        """
        # Compute stationary distribution d^pi, shape (S,)
        d: ndarray = self.stationary_distribution(pi)

        # Compute policy-induced reward vector r^pi, shape (S,)
        r_pi: ndarray = self.mdp.get_policy_reward(pi)

        # Compute rho^pi = d^pi . r^pi (dot product)
        rho: float = float(np.dot(d, r_pi))

        return rho

    def optimal_average_reward(self) -> float:
        """Compute the optimal average reward rho* = max_pi rho^pi.

        Finds the optimal policy via exact policy iteration and returns its
        average reward. This is used to compute the suboptimality gap
        rho* - rho^{pi_k} for validation and plotting.

        Paper reference: Theorem 1 — rho* is the globally optimal average
        reward that PPG converges to.

        Returns:
            Optimal average reward as a Python float. This is the maximum
            achievable average reward over all stationary policies.
        """
        pi_star: ndarray = self._policy_iteration()
        return self.average_reward(pi_star)

    def _policy_iteration(self) -> ndarray:
        """Find the optimal policy via exact policy iteration.

        Implements the standard policy iteration algorithm for average-reward
        MDPs. Starting from a uniform policy, alternates between:
        1. Policy evaluation: compute Q^pi for the current policy
        2. Policy improvement: greedily update to argmax_a Q^pi[s,a]

        Convergence is guaranteed for finite MDPs with irreducible, aperiodic
        chains under all policies (Assumption 1 of the paper). The algorithm
        terminates when the greedy policy is unchanged between iterations,
        which occurs in at most |A|^|S| steps (finite policy space). In
        practice, convergence is much faster (typically O(|S|) iterations).

        The returned policy is deterministic (one-hot rows), which is always
        optimal for finite MDPs.

        Paper reference: Used implicitly to compute rho* for Theorem 1
        validation and suboptimality gap plots.

        Returns:
            Optimal deterministic policy of shape (S, A). Each row is a
            one-hot vector: pi_star[s, argmax_a Q*[s,a]] = 1.0, all other
            entries are 0.0.
        """
        S: int = self.mdp.S
        A: int = self.mdp.A

        # Initialize with uniform policy (standard starting point)
        pi: ndarray = np.ones((S, A), dtype=np.float64) / A

        max_policy_iter: int = 1000  # Upper bound on iterations

        for iteration in range(max_policy_iter):
            # Policy evaluation: compute Q-function for current policy
            # q_function internally calls projected_value_function and
            # average_reward, providing a complete policy evaluation step.
            Q: ndarray = self.q_function(pi)

            # Policy improvement: greedy update
            # best_actions[s] = argmax_a Q^pi[s, a], shape (S,)
            best_actions: ndarray = np.argmax(Q, axis=1)

            # Construct one-hot deterministic policy
            pi_new: ndarray = np.zeros((S, A), dtype=np.float64)
            pi_new[np.arange(S), best_actions] = 1.0

            # Check convergence: policy unchanged between iterations
            # Compare argmax of old and new policy (both deterministic after
            # first improvement step)
            old_actions: ndarray = np.argmax(pi, axis=1)
            if np.array_equal(best_actions, old_actions):
                # Policy has stabilized — we have found the optimal policy
                pi = pi_new
                break

            pi = pi_new

        return pi
