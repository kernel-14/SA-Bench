## pg_solver.py

"""
Core implementation of the Projected Policy Gradient (PPG) algorithm for average‑reward
Markov Decision Processes as described in the paper:

    "Global Convergence of Policy Gradient in Average Reward MDPs"
    by Yashaswini Murthy et al.

This module provides the :class:`PolicyGradientSolver` class that encapsulates
the exact (model‑based) gradient updates for a tabular MDP.  The computation of
the stationary distribution, projected value function and action‑value function
follows the derivations in Sections 2, 3 and Appendix A of the paper.

Usage example::

    from mdp import MDP
    from pg_solver import PolicyGradientSolver

    mdp = MDP.build_exp1(S=3, A=3, seed=42)
    solver = PolicyGradientSolver(mdp, eta=0.1, max_iter=2000)
    hist = solver.run()
    print(hist[-1])
"""

import numpy as np
from scipy import linalg as spla
from typing import Optional, List, Tuple

# Import the utility function for simplex projection.
from utils import project_simplex

# Optional: import the MDP class for type hinting.
# This is safe because mdp.py does not import pg_solver.py.
from mdp import MDP


class PolicyGradientSolver:
    """
    Exact projected policy gradient solver for average‑reward MDPs.

    Attributes
    ----------
    mdp : MDP
        The Markov Decision Process (transition tensor ``P``, reward matrix ``R``,
        number of states ``S`` and actions ``A``).
    eta : float
        Step size used in the gradient ascent update.  Must satisfy
        ``η < 1 / L₂^Π`` (the paper's smoothness constant), which is ensured by
        manual tuning in the experiment scripts.
    max_iter : int
        Maximum number of gradient steps to perform.
    pi : numpy.ndarray
        Current policy matrix of shape ``(S, A)``.  Rows sum to 1.
    history : List[float]
        Accumulated average rewards at each iteration (including the initial
        policy at index 0).
    """

    def __init__(
        self,
        mdp: MDP,
        eta: float,
        max_iter: int,
        init_policy: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialise the solver.

        Parameters
        ----------
        mdp : MDP
            An instance of the MDP class providing ``S``, ``A``, ``P`` and ``R``.
        eta : float
            Step size for projected gradient ascent.
        max_iter : int
            Number of iterations to perform.
        init_policy : numpy.ndarray, optional
            If ``None`` (default), the policy is initialised uniformly over actions:
            ``π(s, a) = 1/A`` for all ``(s, a)``.  Otherwise it must be an array
            of shape ``(S, A)`` with rows summing to 1.

        Raises
        ------
        ValueError
            If ``init_policy`` does not have shape ``(S, A)`` or has non‑stochastic rows.
        """
        self.mdp = mdp
        self.eta = eta
        self.max_iter = max_iter
        self.S = mdp.S
        self.A = mdp.A

        # Projection matrix Φ = I - (1/S) * 1 1^T  (Lemma 1)
        self.Phi = np.eye(self.S) - np.ones((self.S, self.S)) / self.S

        # Initialize policy
        if init_policy is None:
            self.pi = np.ones((self.S, self.A)) / self.A
        else:
            if init_policy.shape != (self.S, self.A):
                raise ValueError(
                    f"init_policy must have shape ({self.S}, {self.A}), "
                    f"got {init_policy.shape}"
                )
            # Verify row stochasticity (allowing small numerical errors)
            row_sums = init_policy.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                raise ValueError(
                    "Rows of init_policy must sum to 1. "
                    f"Max deviation: {np.max(np.abs(row_sums - 1.0))}"
                )
            self.pi = init_policy.copy()

        # Will be populated by run()
        self.history: List[float] = []

    def compute_stationary(self, policy: np.ndarray) -> np.ndarray:
        """
        Compute the stationary distribution ``d^π`` induced by the given policy.

        The stationary distribution is the unique probability vector satisfying::

            (d^π)^T P^π = (d^π)^T,   d^π ≥ 0,  Σ_s d^π(s) = 1

        where ``P^π`` is the state‑to‑state transition matrix under ``policy``.

        Parameters
        ----------
        policy : numpy.ndarray
            Policy matrix of shape ``(S, A)``.

        Returns
        -------
        numpy.ndarray
            Stationary distribution vector of shape ``(S,)``.
        """
        # Compute the Markov chain transition matrix P^π
        P_pi = np.einsum("sa,san->sn", policy, self.mdp.P)

        # Eigen decomposition of the transpose
        eigvals, eigvecs = np.linalg.eig(P_pi.T)

        # Find the eigenvector corresponding to the eigenvalue closest to 1
        idx = np.argmin(np.abs(eigvals - 1.0))
        d = np.real(eigvecs[:, idx])

        # Ensure non‑negativity (small negative entries can appear due to numerical noise)
        d = np.maximum(d, 0.0)

        # Normalise to a probability vector
        d /= d.sum()

        return d

    def compute_q_and_rho(
        self, policy: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Compute the projected value function, average reward, and action‑value function.

        The method follows the steps described in Lemma 1, Lemma 15 and Lemma 16
        of the paper:

        1. Compute ``r^π(s) = Σ_a π(a|s) R(s,a)``.
        2. Solve the linear system::

               (I - Φ P^π) v_φ^π = Φ r_π

           for the unique projected value function ``v_φ^π``.
        3. Compute the average reward::

               ρ^π = Σ_s d^π(s) r^π(s)

           where ``d^π`` is the stationary distribution of ``π``.
        4. Compute the action‑value function::

               Q^π(s,a) = R(s,a) + Σ_{s'} P(s'|s,a) v_φ^π(s') - ρ^π

        Parameters
        ----------
        policy : numpy.ndarray
            Policy matrix of shape ``(S, A)``.

        Returns
        -------
        v_phi : numpy.ndarray
            Projected value function, shape ``(S,)``.
        rho : float
            Average reward under the current policy.
        Q : numpy.ndarray
            Action‑value function, shape ``(S, A)``.
        """
        # 1. Expected reward vector r^π
        r_pi = np.einsum("sa,sa->s", policy, self.mdp.R)

        # 2. Stationary distribution
        d = self.compute_stationary(policy)

        # 3. Average reward
        rho = np.dot(d, r_pi)

        # 4. Transition matrix P^π
        P_pi = np.einsum("sa,san->sn", policy, self.mdp.P)

        # 5. Solve (I - Φ P^π) v = Φ r_pi
        A = np.eye(self.S) - self.Phi @ P_pi
        b = self.Phi @ r_pi

        try:
            # Use scipy.linalg.solve for a general (non‑symmetric) matrix
            v_phi = spla.solve(A, b, assume_a="gen")
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                "Failed to solve the linear system for the projected value function. "
                "This may indicate that the Markov chain is periodic or ill‑conditioned. "
                f"Original error: {e}"
            )

        # 6. Action‑value function Q^π
        # Compute P v_phi: for each (s,a), expectation over next states
        Pv = np.einsum("ijk,k->ij", self.mdp.P, v_phi)
        Q = self.mdp.R + Pv - rho

        return v_phi, rho, Q

    def compute_gradient(
        self, policy: np.ndarray, d: np.ndarray, Q: np.ndarray
    ) -> np.ndarray:
        """
        Compute the policy gradient of the average reward with respect to the
        tabular policy.

        According to the Average Reward Policy Gradient Theorem (Equation 5)::

            ∂ρ^π / ∂π(s,a) = d^π(s) · Q^π(s,a)

        where ``d^π`` is the stationary distribution and ``Q^π`` the action‑value
        function.

        Parameters
        ----------
        policy : numpy.ndarray
            (Unused, kept for interface consistency.)  Policy matrix of shape
            ``(S, A)``.
        d : numpy.ndarray
            Stationary distribution vector of shape ``(S,)``.
        Q : numpy.ndarray
            Action‑value function matrix of shape ``(S, A)``.

        Returns
        -------
        numpy.ndarray
            Policy gradient array of shape ``(S, A)``.
        """
        # Broadcasting: multiply each row of Q by the corresponding d(s)
        return d[:, np.newaxis] * Q

    def project_policy(self, policy: np.ndarray) -> np.ndarray:
        """
        Project each state's action probability vector onto the simplex.

        This implements the orthogonal Euclidean projection onto the space of
        randomised policies (product of simplices).  Row‑wise projection uses
        the efficient Duchi et al. algorithm from ``utils.project_simplex``.

        Parameters
        ----------
        policy : numpy.ndarray
            Unprojected policy matrix of shape ``(S, A)``.

        Returns
        -------
        numpy.ndarray
            Projected policy matrix of the same shape, where each row lies on
            the probability simplex.
        """
        proj = np.empty_like(policy)
        for s in range(self.S):
            proj[s] = project_simplex(policy[s])
        return proj

    def step(self) -> float:
        """
        Execute one iteration of the projected policy gradient update.

        The steps performed are:

        1. Evaluate the current policy (stationary distribution, Q‑values).
        2. Compute the gradient using the policy gradient theorem.
        3. Apply the gradient ascent step.
        4. Project the new policy onto the simplex.
        5. Evaluate the new policy's average reward and return it.

        Returns
        -------
        float
            The average reward of the policy *after* the update.
        """
        # ---- Evaluate current policy ----
        d = self.compute_stationary(self.pi)
        v_phi, rho_current, Q = self.compute_q_and_rho(self.pi)
        # The gradient only needs d and Q
        grad = self.compute_gradient(self.pi, d, Q)

        # ---- Gradient ascent step ----
        pi_new = self.pi + self.eta * grad

        # ---- Project onto the simplex ----
        pi_new = self.project_policy(pi_new)

        # ---- Evaluate the new policy's average reward ----
        # Compute new stationary distribution and expected rewards
        d_new = self.compute_stationary(pi_new)
        r_new = np.einsum("sa,sa->s", pi_new, self.mdp.R)
        rho_new = np.dot(d_new, r_new)

        # Update stored policy
        self.pi = pi_new

        return rho_new

    def run(self) -> List[float]:
        """
        Run the projected policy gradient algorithm for the configured number of
        iterations.

        The sequence of average rewards is recorded as ``self.history``, with
        ``history[0]`` corresponding to the initial policy and ``history[k]``
        to the reward after the ``k``‑th gradient step.

        Returns
        -------
        List[float]
            The list of average rewards, one per iteration (length ``max_iter+1``).
        """
        # Evaluate the initial policy
        v_phi_init, rho_init, Q_init = self.compute_q_and_rho(self.pi)
        self.history = [rho_init]

        for _ in range(self.max_iter):
            rho_new = self.step()
            self.history.append(rho_new)

        return self.history
