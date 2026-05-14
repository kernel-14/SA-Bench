"""
Average Reward MDP implementation for reproducing:
"Global Convergence of Policy Gradient in Average Reward MDPs"

Key formulas from the paper:
- Projected value function: v_phi^pi = (I - Phi P^pi)^{-1} Phi r^pi  (Lemma 1)
- Average reward: rho^pi = d^pi^T r^pi
- Q-function: Q^pi(s,a) = r(s,a) + sum_{s'} P(s'|s,a) v_phi^pi(s') - rho^pi
- Policy gradient theorem: d rho / d pi(a|s) = d^pi(s) * Q^pi(s, a)
- Projected gradient update: pi_{k+1} = Proj_Pi[pi_k + eta * grad]
"""

import numpy as np
from typing import Tuple, Optional


class AverageRewardMDP:
    """
    Tabular average reward MDP with finite state and action spaces.
    
    Supports computing:
    - Stationary distribution d^pi
    - Average reward rho^pi
    - Relative value function v_phi^pi (projected, unique)
    - Q-function Q^pi
    - Policy gradient
    - MDP complexity constants (C_m, C_p, C_r, kappa_r, L1, L2, C_PL)
    """

    def __init__(self, S: int, A: int, P: np.ndarray, R: np.ndarray):
        """
        Args:
            S: Number of states
            A: Number of actions
            P: Transition kernel, shape (S, A, S), P[s, a, s'] = P(s'|s,a)
            R: Reward function, shape (S, A), R[s, a] = r(s, a)
        """
        self.S = S
        self.A = A
        self.P = P  # (S, A, S)
        self.R = R  # (S, A)

        # Projection matrix Phi = I - 11^T / S  (Lemma 1)
        self.Phi = np.eye(S) - np.ones((S, S)) / S

    def get_transition_matrix(self, pi: np.ndarray) -> np.ndarray:
        """
        Compute transition matrix P^pi of shape (S, S).
        P^pi[s, s'] = sum_a pi(a|s) * P(s'|s,a)
        """
        return np.einsum('sa,sab->sb', pi, self.P)

    def get_reward_vector(self, pi: np.ndarray) -> np.ndarray:
        """
        Compute expected reward vector r^pi of shape (S,).
        r^pi[s] = sum_a pi(a|s) * R(s, a)
        """
        return np.einsum('sa,sa->s', pi, self.R)

    def get_stationary_distribution(self, pi: np.ndarray) -> np.ndarray:
        """
        Compute stationary distribution d^pi satisfying d^T P^pi = d^T, sum(d) = 1.
        
        Uses a robust approach: solve the linear system (P^pi - I)^T d = 0
        with normalization constraint.
        """
        P_pi = self.get_transition_matrix(pi)
        n = self.S
        
        # Build system: (P^pi^T - I) d = 0, with last row replaced by sum = 1
        A_mat = P_pi.T - np.eye(n)
        A_mat[-1, :] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0
        
        try:
            d_pi = np.linalg.solve(A_mat, b)
        except np.linalg.LinAlgError:
            # Fallback: use power iteration
            d_pi = np.ones(n) / n
            for _ in range(10000):
                d_new = d_pi @ P_pi
                if np.max(np.abs(d_new - d_pi)) < 1e-12:
                    break
                d_pi = d_new
        
        # Ensure non-negative and normalized
        d_pi = np.maximum(d_pi, 0)
        total = d_pi.sum()
        if total > 0:
            d_pi /= total
        else:
            d_pi = np.ones(n) / n
        return d_pi

    def get_average_reward(self, pi: np.ndarray) -> float:
        """
        Compute average reward rho^pi = sum_s d^pi(s) r^pi(s).
        """
        d_pi = self.get_stationary_distribution(pi)
        r_pi = self.get_reward_vector(pi)
        return float(np.dot(d_pi, r_pi))

    def get_projected_value_function(self, pi: np.ndarray) -> np.ndarray:
        """
        Compute the projected value function (Lemma 1):
            v_phi^pi = (I - Phi P^pi)^{-1} Phi r^pi
        
        This is the unique value function satisfying 1^T v = 0.
        """
        P_pi = self.get_transition_matrix(pi)
        r_pi = self.get_reward_vector(pi)
        
        Phi_r = self.Phi @ r_pi
        I_minus_PhiP = np.eye(self.S) - self.Phi @ P_pi
        
        v_phi = np.linalg.solve(I_minus_PhiP, Phi_r)
        return v_phi

    def get_q_function(self, pi: np.ndarray) -> np.ndarray:
        """
        Compute Q^pi(s, a) = r(s,a) + sum_{s'} P(s'|s,a) v_phi^pi(s') - rho^pi.
        """
        rho = self.get_average_reward(pi)
        v_phi = self.get_projected_value_function(pi)
        Q = self.R + np.einsum('sab,b->sa', self.P, v_phi) - rho
        return Q

    def get_policy_gradient(self, pi: np.ndarray) -> np.ndarray:
        """
        Compute policy gradient using the average reward policy gradient theorem:
            d rho / d pi(a|s) = d^pi(s) * Q^pi(s, a)
        """
        d_pi = self.get_stationary_distribution(pi)
        Q = self.get_q_function(pi)
        return d_pi[:, np.newaxis] * Q

    def get_optimal_policy(self) -> Tuple[np.ndarray, float]:
        """
        Compute the optimal policy using policy iteration.
        
        Returns:
            pi_star: Optimal policy, shape (S, A)
            rho_star: Optimal average reward
        """
        # Start with uniform policy
        pi = np.ones((self.S, self.A)) / self.A
        
        for _ in range(1000):
            pi_old = pi.copy()
            Q = self.get_q_function(pi)
            best_actions = np.argmax(Q, axis=1)
            pi_new = np.zeros((self.S, self.A))
            pi_new[np.arange(self.S), best_actions] = 1.0
            
            if np.allclose(pi_new, pi_old, atol=1e-10):
                break
            pi = pi_new
        
        rho_star = self.get_average_reward(pi)
        return pi, rho_star

    def compute_mdp_constants(self, n_random_policies: int = 20) -> dict:
        """
        Compute MDP complexity constants from Table 1 of the paper:
        
        C_m = max_pi ||(I - Phi P^pi)^{-1}||_inf  (mixing rate)
        C_p = max_{pi,pi'} ||P^{pi'} - P^pi||_inf / ||pi' - pi||_2  (transition diameter)
        C_r = max_{pi,pi'} ||r^{pi'} - r^pi||_inf / ||pi' - pi||_2  (reward diameter)
        kappa_r = max_pi ||Phi r^pi||_inf  (reward variance)
        L1 = 2(C_r + C_p C_m kappa_r + 2(C_m^2 C_p kappa_r + C_m C_r))  (Lipschitz)
        L2 = 4(C_p^2 C_m^2 kappa_r + C_p C_m C_r + (C_p+1)(C_m^2 C_p kappa_r + C_m C_r)
               + 4(C_m^3 C_p^2 kappa_r + C_m^2 C_p C_r))  (smoothness)
        C_PL = max_{pi,s} d^{pi*}(s) / d^pi(s)  (gradient domination)
        
        Returns:
            constants: Dictionary of MDP complexity constants
        """
        S, A = self.S, self.A
        
        # Build a representative set of policies
        policies = []
        policies.append(np.ones((S, A)) / A)  # uniform
        
        # Deterministic policies (one per action)
        for a in range(min(A, 10)):  # cap at 10 to avoid slowness for large A
            pi = np.zeros((S, A))
            pi[:, a] = 1.0
            policies.append(pi)
        
        # Random policies
        rng = np.random.RandomState(42)
        for _ in range(n_random_policies):
            pi = rng.dirichlet(np.ones(A), size=S)
            policies.append(pi)
        
        # --- C_m: max_pi ||(I - Phi P^pi)^{-1}||_inf ---
        C_m = 0.0
        for pi in policies:
            P_pi = self.get_transition_matrix(pi)
            I_minus_PhiP = np.eye(S) - self.Phi @ P_pi
            try:
                M = np.linalg.inv(I_minus_PhiP)
                norm_M = np.max(np.sum(np.abs(M), axis=1))
                C_m = max(C_m, norm_M)
            except np.linalg.LinAlgError:
                pass
        
        # --- kappa_r: max_pi ||Phi r^pi||_inf ---
        kappa_r = 0.0
        for pi in policies:
            r_pi = self.get_reward_vector(pi)
            Phi_r = self.Phi @ r_pi
            kappa_r = max(kappa_r, np.max(np.abs(Phi_r)))
        
        # --- C_p and C_r: pairwise over policies ---
        C_p = 0.0
        C_r = 0.0
        n_pol = len(policies)
        for i in range(n_pol):
            for j in range(i + 1, n_pol):
                pi = policies[i]
                pi_prime = policies[j]
                diff_pi = pi_prime - pi
                norm_diff = np.linalg.norm(diff_pi)
                if norm_diff < 1e-10:
                    continue
                
                # C_p
                P_pi = self.get_transition_matrix(pi)
                P_pi_prime = self.get_transition_matrix(pi_prime)
                diff_P = P_pi_prime - P_pi
                norm_diff_P = np.max(np.sum(np.abs(diff_P), axis=1))
                C_p = max(C_p, norm_diff_P / norm_diff)
                
                # C_r
                r_pi = self.get_reward_vector(pi)
                r_pi_prime = self.get_reward_vector(pi_prime)
                diff_r = r_pi_prime - r_pi
                norm_diff_r = np.max(np.abs(diff_r))
                C_r = max(C_r, norm_diff_r / norm_diff)
        
        # --- L1: Restricted Lipschitz constant (Lemma 3) ---
        # L1 = 2(C_r + C_p C_m kappa_r + 2(C_m^2 C_p kappa_r + C_m C_r))
        L1 = 2 * (C_r + C_p * C_m * kappa_r + 2 * (C_m**2 * C_p * kappa_r + C_m * C_r))
        
        # --- L2: Restricted smoothness constant (Lemma 4 / Lemma 17) ---
        # L2 = 4(C_p^2 C_m^2 kappa_r + C_p C_m C_r
        #        + (C_p+1)(C_m^2 C_p kappa_r + C_m C_r)
        #        + 4(C_m^3 C_p^2 kappa_r + C_m^2 C_p C_r))
        L2 = 4 * (
            C_p**2 * C_m**2 * kappa_r
            + C_p * C_m * C_r
            + (C_p + 1) * (C_m**2 * C_p * kappa_r + C_m * C_r)
            + 4 * (C_m**3 * C_p**2 * kappa_r + C_m**2 * C_p * C_r)
        )
        
        # --- C_PL: gradient domination constant (Lemma 7) ---
        # C_PL = max_{pi, s} d^{pi*}(s) / d^pi(s)
        pi_star, _ = self.get_optimal_policy()
        d_star = self.get_stationary_distribution(pi_star)
        C_PL = 0.0
        for pi in policies:
            d_pi = self.get_stationary_distribution(pi)
            ratio = d_star / np.maximum(d_pi, 1e-10)
            C_PL = max(C_PL, np.max(ratio))
        
        return {
            'C_m': C_m,
            'C_p': C_p,
            'C_r': C_r,
            'kappa_r': kappa_r,
            'L1': L1,
            'L2': L2,
            'C_PL': C_PL,
        }
