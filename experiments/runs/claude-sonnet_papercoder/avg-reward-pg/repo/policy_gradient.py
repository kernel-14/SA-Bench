## policy_gradient.py
"""Projected Policy Gradient (PPG) algorithm for average reward MDPs.

This module implements the core optimization algorithm from:
    Murthy et al., "Global Convergence of Policy Gradient in Average Reward
    MDPs", ICLR 2024.

The PPG update rule (Equation 6 of the paper) is:
    π_{k+1} = Proj_Π [ π_k + η * ∂ρ^π/∂π |_{π=π_k} ]

where:
    - Proj_Π is the Euclidean projection onto the policy space (product of
      probability simplices, one per state)
    - η is the step size satisfying η < 1/L_2^Π (Theorem 1)
    - ∂ρ^π/∂π[s,a] = d^π(s) * Q^π(s,a) (average reward policy gradient
      theorem, Equation 5)

The module provides:
    - PolicyGradient: main class implementing the PPG loop
    - Simplex projection (per-row) via utils.project_simplex
    - Gradient computation via ValueFunctions
    - Memory-efficient policy history storage (every 100th iterate)

References:
    Paper Section 2.1 (update rule, policy gradient theorem)
    Paper Theorem 1 (step size condition, convergence guarantee)
    Paper Lemma 5 (monotone improvement)
    Paper Lemma 19 (projection properties)
    config.yaml policy_gradient section (step_size_multiplier, init_policy)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy import ndarray

import utils
from mdp import MDP
from value_functions import ValueFunctions


class PolicyGradient:
    """Projected Policy Gradient algorithm for average reward MDPs.

    Implements the PPG update from Equation 6 of the paper:
        π_{k+1} = Proj_Π [ π_k + η * ∇_π ρ^π |_{π=π_k} ]

    The policy space Π = ×_{s∈S} Δ(A) is a Cartesian product of |S|
    probability simplices. The Euclidean projection onto Π decomposes into
    independent per-row simplex projections (one per state), which is the
    Proj_Π operator in the paper.

    The gradient ∂ρ^π/∂π[s,a] = d^π(s) * Q^π(s,a) follows from the
    average reward policy gradient theorem (Equation 5 of the paper) under
    tabular parameterization where θ = π.

    Attributes:
        mdp: The MDP instance providing transition kernel P and reward r.
        vf: ValueFunctions instance for computing stationary distributions,
            Q-functions, and average rewards. Shared across all calls to
            avoid redundant Phi precomputation.
        eta: Step size for the gradient ascent update. Must satisfy
            η < 1/L_2^Π per Theorem 1 of the paper. Set by experiments.py
            as 0.5 / L2 where L2 is computed by ComplexityMetrics.
    """

    def __init__(self, mdp: MDP, eta: float) -> None:
        """Initialize the PolicyGradient optimizer.

        Creates a shared ValueFunctions instance for the given MDP, which
        precomputes the projection matrix Phi = I - 11^T/|S| once. This
        avoids redundant (S, S) matrix allocation at every PPG iteration.

        The step size eta is fixed for the entire run, consistent with the
        paper's constant step size analysis in Theorem 1. The paper's
        convergence bound requires η < 1/L_2^Π; the caller (experiments.py)
        is responsible for setting eta appropriately using ComplexityMetrics.

        Args:
            mdp: The MDP instance. Must have valid S, A, P, r attributes.
                Used to instantiate ValueFunctions and to determine the
                policy shape (S, A) in uniform_policy().
            eta: Step size for gradient ascent. Must be a positive float
                satisfying η < 1/L_2^Π (Theorem 1). Typical values set by
                experiments.py: 0.5 / L2 where L2 is the restricted
                smoothness constant from ComplexityMetrics.compute_L2().
                Fallback value from config.yaml: step_size_fallback = 0.01.
        """
        self.mdp: MDP = mdp

        # Instantiate ValueFunctions once — precomputes Phi = I - 11^T/|S|
        # and stores it for reuse across all gradient/value computations.
        self.vf: ValueFunctions = ValueFunctions(mdp)

        # Fixed step size for the entire run (constant step size analysis)
        self.eta: float = float(eta)

    def uniform_policy(self) -> ndarray:
        """Return the uniform policy over all actions at every state.

        Constructs the initial policy π_0 where every action is equally
        likely at every state:
            π_0[s, a] = 1/|A|  for all s ∈ S, a ∈ A

        This is the standard initialization used in all three experiments.
        The paper does not specify the initial policy explicitly; the config
        specifies init_policy: "uniform" (config.yaml policy_gradient section).

        Returns:
            Uniform policy array of shape (S, A). Each row sums to 1.0 and
            all entries equal 1/A. Satisfies all validity constraints for
            a stochastic policy.
        """
        return np.ones((self.mdp.S, self.mdp.A), dtype=np.float64) / self.mdp.A

    def project_simplex_row(self, v: ndarray) -> ndarray:
        """Project a single row vector onto the probability simplex.

        Thin wrapper around utils.project_simplex for a single 1D vector.
        This method exists to match the design interface and to make
        project_policy() readable.

        The probability simplex is:
            Δ(A) = {x ∈ R^A : x ≥ 0, sum(x) = 1}

        The projection is the Euclidean (L2) projection, which is the
        Proj_Π operator from Equation 6 of the paper applied to a single
        state's action distribution.

        Args:
            v: 1D numpy array of shape (A,). Represents the updated action
                distribution for a single state before projection. May have
                negative entries or sum ≠ 1 after the gradient step.

        Returns:
            1D numpy array of shape (A,) lying on the probability simplex.
            Satisfies: result >= 0 and sum(result) == 1 (up to floating
            point precision ~1e-15).
        """
        return utils.project_simplex(v)

    def project_policy(self, pi: ndarray) -> ndarray:
        """Project a policy matrix onto the policy space Π.

        Applies the Euclidean projection onto the policy space:
            Π = ×_{s∈S} Δ(A)  (Cartesian product of |S| probability simplices)

        The projection onto a Cartesian product decomposes into independent
        per-component projections (Lemma 19 of the paper). Each row s of pi
        is projected independently onto Δ(A) via project_simplex_row.

        This implements the Proj_Π operator in Equation 6 of the paper:
            π_{k+1} = Proj_Π [ π_k + η * ∇ρ^{π_k} ]

        Args:
            pi: Policy array of shape (S, A). May have negative entries or
                rows not summing to 1 after the gradient ascent step
                π_k + η * ∇ρ^{π_k}. This is the pre-projection iterate.

        Returns:
            Projected policy of shape (S, A). Every row is a valid
            probability vector: pi[s, :] >= 0 and sum(pi[s, :]) == 1
            for all s ∈ {0, ..., S-1}.

        Note:
            The per-row projection is applied using a vectorized loop over
            states. For the MDP sizes in the paper (up to 81×81), this is
            not a performance bottleneck. The result is a new array (not
            an in-place modification) to avoid aliasing issues.
        """
        S: int = self.mdp.S
        projected: ndarray = np.zeros_like(pi, dtype=np.float64)

        for s in range(S):
            projected[s] = self.project_simplex_row(pi[s])

        return projected

    def compute_gradient(self, pi: ndarray) -> ndarray:
        """Compute the policy gradient ∂ρ^π/∂π[s,a] = d^π(s) * Q^π(s,a).

        Implements the average reward policy gradient theorem (Equation 5
        of the paper) under tabular parameterization (θ = π):

            ∂ρ^π/∂π[s,a] = d^π(s) * Q^π(s,a)

        where:
            - d^π(s) is the stationary distribution at state s under π
            - Q^π(s,a) is the relative Q-function (advantage of action a
              at state s relative to the average reward ρ^π)

        The gradient has shape (S, A) and is computed by broadcasting the
        stationary distribution vector d^π (shape (S,)) across the action
        dimension of Q^π (shape (S, A)).

        Note on Q-function uniqueness: Q^π is unique up to an additive
        constant, but the gradient direction is unaffected by this constant
        because sum_a π(a|s) * Q^π(s,a) = v_φ^π(s) and the additive
        constant cancels in the gradient computation. The projected value
        function v_φ^π used in q_function() fixes the constant uniquely.

        Args:
            pi: Policy array of shape (S, A). Each row pi[s, :] must be a
                valid probability vector (non-negative, sums to 1).

        Returns:
            Policy gradient array of shape (S, A). Entry [s, a] equals
            d^π(s) * Q^π(s,a). This is the direction of steepest ascent
            for the average reward ρ^π within the policy space.
        """
        # Step 1: Compute stationary distribution d^π, shape (S,)
        # d[s] = probability of being in state s under the stationary
        # distribution of the Markov chain induced by policy π.
        d: ndarray = self.vf.stationary_distribution(pi)

        # Step 2: Compute relative Q-function Q^π, shape (S, A)
        # Q[s,a] = r[s,a] + sum_{s'} P[s,a,s'] v_φ^π[s'] - ρ^π
        # This internally calls projected_value_function and average_reward.
        Q: ndarray = self.vf.q_function(pi)

        # Step 3: Compute gradient via broadcasting
        # g[s,a] = d[s] * Q[s,a]
        # d[:, None] reshapes d from (S,) to (S, 1) for broadcasting with
        # Q of shape (S, A), yielding g of shape (S, A).
        g: ndarray = d[:, None] * Q

        return g

    def run(
        self,
        n_iterations: int,
        pi_init: ndarray,
    ) -> Tuple[List[float], List[ndarray]]:
        """Run the Projected Policy Gradient algorithm for n_iterations steps.

        Implements the PPG loop from Equation 6 of the paper:
            π_{k+1} = Proj_Π [ π_k + η * ∇_π ρ^π |_{π=π_k} ]

        for k = 0, 1, ..., n_iterations - 1.

        The average reward ρ^{π_k} is logged before each update, so
        reward_history[k] corresponds to ρ^{π_k} (the reward at the
        k-th iterate, before the (k+1)-th update). This matches the
        paper's notation in Theorem 1.

        By Lemma 5 of the paper, when η < 1/L_2^Π, the reward sequence
        is monotonically non-decreasing:
            ρ^{π_{k+1}} ≥ ρ^{π_k}  for all k ≥ 0

        This is observable in the returned reward_history.

        Memory efficiency: Storing every policy iterate would require
        n_iterations × S × A floats. For the 81×81 MDP with 2000 iterations,
        that is ~13M floats (~100MB). To avoid this, only every 100th policy
        iterate is stored in policy_history. The reward history (a list of
        scalars) is always stored in full.

        Args:
            n_iterations: Number of PPG iterations to run. From config.yaml:
                - Experiments 1 & 2: exp1_iterations = 2000,
                  exp2_iterations = 2000
                - Experiment 3: exp3_iterations = 3000
            pi_init: Initial policy array of shape (S, A). Each row must be
                a valid probability vector. Typically the uniform policy
                from uniform_policy(). The input array is not modified
                (a copy is made at the start of the loop).

        Returns:
            A tuple (reward_history, policy_history) where:
                - reward_history: List of n_iterations floats. Entry k is
                  ρ^{π_k} = average reward at the k-th iterate (before the
                  (k+1)-th update). Used directly for plotting Figures 1a,
                  1b, and 2.
                - policy_history: List of policy arrays, one stored every
                  100 iterations. Entry j is a copy of π_{100*j}. Used for
                  optional analysis; not required for the paper's figures.

        Note:
            The pi_init array is copied at the start to avoid mutating the
            caller's array. This is important when experiments.py reuses
            the same pi_init across multiple PPG runs (e.g., different
            reward variants in Experiment 2 share the same initial policy).
        """
        # Copy initial policy to avoid mutating the caller's array
        pi: ndarray = pi_init.copy()

        # Pre-allocate reward history list with known size for efficiency
        reward_history: List[float] = []

        # Policy history stores every 100th iterate to manage memory
        # For n_iterations=3000, this stores at most 30 policy arrays.
        policy_history: List[ndarray] = []

        for k in range(n_iterations):
            # ----------------------------------------------------------------
            # Step 1: Log current average reward ρ^{π_k} BEFORE the update
            # reward_history[k] = ρ^{π_k} (k-th iterate's performance)
            # ----------------------------------------------------------------
            rho_k: float = self.vf.average_reward(pi)
            reward_history.append(rho_k)

            # ----------------------------------------------------------------
            # Step 2: Store policy snapshot every 100 iterations
            # pi.copy() is essential — without it, all entries would point
            # to the same (final) array after the loop completes.
            # ----------------------------------------------------------------
            if k % 100 == 0:
                policy_history.append(pi.copy())

            # ----------------------------------------------------------------
            # Step 3: Compute policy gradient ∂ρ^π/∂π |_{π=π_k}
            # g[s,a] = d^{π_k}(s) * Q^{π_k}(s,a), shape (S, A)
            # ----------------------------------------------------------------
            g: ndarray = self.compute_gradient(pi)

            # ----------------------------------------------------------------
            # Step 4: Gradient ascent step + projection (Equation 6)
            # π_{k+1} = Proj_Π [ π_k + η * g ]
            # The gradient ascent step π_k + η * g may produce a matrix
            # with negative entries or rows not summing to 1. The projection
            # maps it back to the valid policy space Π.
            # ----------------------------------------------------------------
            pi = self.project_policy(pi + self.eta * g)

        return reward_history, policy_history
