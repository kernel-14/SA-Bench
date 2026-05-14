import numpy as np
from mdp import AverageRewardMDP # Assuming mdp.py is in the same directory

class Analysis:
    def __init__(self, mdp: AverageRewardMDP):
        self.mdp = mdp
        self.num_states = mdp.num_states
        self.num_actions = mdp.num_actions
        self.ones_vec = np.ones(self.num_states)
        self.Phi = np.identity(self.num_states) - np.outer(self.ones_vec, self.ones_vec) / self.num_states

    def calculate_l_inf_op_norm(self, matrix):
        """Calculates the L_infinity operator norm (maximum absolute row sum)."""
        if matrix.ndim == 1: # For a vector, it's the max absolute value
            return np.max(np.abs(matrix))
        return np.max(np.sum(np.abs(matrix), axis=1))

    def get_Cm_constant(self, policy_matrix):
        """
        Calculates C_m for a given policy pi.
        The paper defines C_m = max_pi ||(I - Phi P^pi)^-1||_inf.
        For a practical implementation, we compute it for the *current* policy.
        The global maximum over all policies is generally intractable to compute.
        """
        P_pi = self.mdp.get_policy_transition_kernel(policy_matrix)
        matrix_to_invert = np.identity(self.num_states) - self.Phi @ P_pi
        
        # In a real scenario, should check for singularity, but for ergodic MDPs this should be invertible.
        # Lemma 12 proves invertibility.
        inverse_matrix = np.linalg.inv(matrix_to_invert)
        return self.calculate_l_inf_op_norm(inverse_matrix)

    def get_Cp_constant(self, policy_matrix_k, policy_matrix_kp1):
        """
        Calculates C_p, which is the Lipschitz constant for the transition kernel w.r.t. policy.
        Defined as: max_{pi,pi'} max_{||v||_inf<=1} (||(P^pi' - P^pi) v||_inf / ||pi' - pi||_2).
        This is highly complex to compute exactly over all policies and vectors v.
        As an approximation for a given policy update, we use ||P^pi_kp1 - P^pi_k||_inf / ||pi_kp1 - pi_k||_2.
        This provides a local estimate of the Lipschitz constant for the current policy change.
        """
        P_pi_k = self.mdp.get_policy_transition_kernel(policy_matrix_k)
        P_pi_kp1 = self.mdp.get_policy_transition_kernel(policy_matrix_kp1)
        
        delta_P = P_pi_kp1 - P_pi_k
        delta_pi_norm = np.linalg.norm(policy_matrix_kp1 - policy_matrix_k, ord=2)

        if delta_pi_norm == 0: # Avoid division by zero if policies are identical
            return 0.0 
        
        return self.calculate_l_inf_op_norm(delta_P) / delta_pi_norm
    
    def get_Cr_constant(self, policy_matrix_k, policy_matrix_kp1):
        """
        Calculates C_r, which is the Lipschitz constant for the reward function w.r.t. policy.
        Defined as: max_{pi,pi'} (||r^pi' - r^pi||_inf / ||pi' - pi||_2).
        Similar to C_p, this is difficult to compute exactly.
        We provide a local estimate for two given policies.
        """
        r_pi_k = self.mdp.get_policy_reward_function(policy_matrix_k)
        r_pi_kp1 = self.mdp.get_policy_reward_function(policy_matrix_kp1)

        delta_r = r_pi_kp1 - r_pi_k
        delta_pi_norm = np.linalg.norm(policy_matrix_kp1 - policy_matrix_k, ord=2)

        if delta_pi_norm == 0: # Avoid division by zero if policies are identical
            return 0.0
        
        # ||delta_r||_inf = max_i |delta_r_i| (vector L_infinity norm)
        return self.calculate_l_inf_op_norm(delta_r) / delta_pi_norm

    def get_kappa_r_constant(self, policy_matrix):
        """
        Calculates kappa_r = max_pi ||Phi r^pi||_inf.
        We compute it for the current policy and acknowledge the 'max_pi' limitation.
        """
        r_pi = self.mdp.get_policy_reward_function(policy_matrix)
        phi_r_pi = self.Phi @ r_pi
        return self.calculate_l_inf_op_norm(phi_r_pi)

    def get_L1_Pi_constant(self, Cm, Cp, Cr, kappa_r):
        """
        Calculates the restricted Lipschitz constant L1^Pi (Lemma 3).
        L1^Pi = 2(Cr + Cp*Cm*kappa_r + 2*(Cm^2*Cp*kappa_r + Cm*Cr))
        """
        return 2 * (Cr + Cp * Cm * kappa_r + 2 * (Cm**2 * Cp * kappa_r + Cm * Cr))

    def get_L2_Pi_constant(self, Cm, Cp, Cr, kappa_r):
        """
        Calculates the restricted smoothness constant L2^Pi (Lemma 4).
        L2^Pi = 4(Cp^2*Cm^2*kappa_r + Cp*Cm*Cr + (Cp + 1)*(Cm^2*Cp*kappa_r + Cm*Cr) + 4*(Cm**3*Cp**2*kappa_r + Cm^2*Cp*Cr))
        Note: The markdown version had a minor difference, used the PDF version here for accuracy.
        """
        term1 = Cp**2 * Cm**2 * kappa_r
        term2 = Cp * Cm * Cr
        term3 = (Cp + 1) * (Cm**2 * Cp * kappa_r + Cm * Cr)
        term4 = 4 * (Cm**3 * Cp**2 * kappa_r + Cm**2 * Cp * Cr)
        return 4 * (term1 + term2 + term3 + term4)

    def get_CPL_constant(self, optimal_stationary_distribution, current_stationary_distribution):
        """
        Calculates C_PL = max_s (d^pi*(s) / d^pi(s)).
        This constant is involved in the Polyak-Lojasiewicz condition related bound (Lemma 7).
        It is a max over all states 's' for the ratio of the optimal stationary distribution to the
        current policy's stationary distribution. The 'max_pi' part is hard to compute globally.
        Here, we compute it for a given optimal stationary distribution and the current one.
        """
        # To avoid division by zero or very small numbers, add a small epsilon
        epsilon = 1e-10
        ratio = optimal_stationary_distribution / (current_stationary_distribution + epsilon)
        return np.max(ratio)

    def calculate_optimality_gap_bound(self, rho_star, rho_pi_0, L2_Pi, C_PL, k):
        """
        Calculates the upper bound for the optimality gap as per Theorem 1.
        rho* - rho_pi_k <= 1 / (1 / (rho* - rho_pi_0) + nu * k)
        where nu = (1 / (32 * C_PL^2 * |S| * L2_Pi)) * (1 + 4 * (1 / (32 * C_PL^2 * |S| * L2_Pi)))^(-3/2)
        """
        term_in_nu = 1.0 / (32 * C_PL**2 * self.num_states * L2_Pi)
        nu = term_in_nu * (1 + 4 * term_in_nu)**(-3/2)
        
        # Ensure rho_star - rho_pi_0 is not zero for the initial term
        initial_term_inverse = 0.0
        # Add a small epsilon for robustness to handle cases where rho_star - rho_pi_0 is very small or zero
        if (rho_star - rho_pi_0) > 1e-9: # Check for non-zero to avoid division by zero
            initial_term_inverse = 1.0 / (rho_star - rho_pi_0)
        else:
            # If initial gap is zero or very small, the bound can be problematic.
            # For k=0, if gap is 0, bound should be 0. If gap is small, bound should be small.
            # If the initial policy is optimal, then rho* - rho_pi_0 = 0, leading to division by zero.
            # In such a case, the optimality gap is 0, and the bound should reflect that.
            if k == 0 and (rho_star - rho_pi_0) <= 1e-9: return 0.0
            # For k > 0, if initial gap is ~0, then 1/nu*k would be effectively the bound.
            # Let's handle this carefully: if rho_star - rho_pi_0 is very small, we treat 1/(rho_star - rho_pi_0) as a large number.
            # But if it's truly zero, initial_term_inverse can't be computed this way. 
            # The formula expects a positive initial gap. If it's zero, we're already optimal.
            pass # This case will rely on nu*k making the denominator large. This might not be robust.

        denominator = initial_term_inverse + nu * k
        if denominator <= 1e-9: # Avoid division by near-zero denominator
            return np.inf # Or some very large number indicating no meaningful bound
            
        bound = 1.0 / denominator
        return bound

