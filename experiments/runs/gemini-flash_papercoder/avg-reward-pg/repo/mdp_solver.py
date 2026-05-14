import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import linprog # Policy iteration can solve LP, but direct method is usually preferred
import warnings

# Assuming mdp_definitions.py is in the same directory or importable path
from mdp_definitions import MDP


class MDPSolver:
    """
    This class provides methods for solving core MDP equations in the context of
    Average Reward MDPs, including computing stationary distributions, projected
    value functions, state-action value functions, and finding the globally
    optimal average reward.
    """

    def __init__(self, mdp: MDP):
        """
        Initializes the MDPSolver with an MDP instance.

        Args:
            mdp (MDP): An instance of the MDP class.
        """
        if not isinstance(mdp, MDP):
            raise TypeError("mdp must be an instance of mdp_definitions.MDP")
        self.mdp: MDP = mdp
        self.PHI: np.ndarray = self._compute_phi() # Pre-compute projection matrix

    def _compute_phi(self) -> np.ndarray:
        """
        Computes the orthogonal projection matrix Φ = I - (1 1^T)/|S|.
        This matrix projects any vector onto the subspace orthogonal to the all-ones vector.

        Returns:
            np.ndarray: The projection matrix Φ of shape (S, S).
        """
        s_count: int = self.mdp.S
        identity_matrix: np.ndarray = np.eye(s_count)
        ones_vector: np.ndarray = np.ones((s_count, 1))
        ones_outer_product: np.ndarray = ones_vector @ ones_vector.T
        phi_matrix: np.ndarray = identity_matrix - (ones_outer_product / s_count)
        return phi_matrix

    def _solve_linear_system(self, a_matrix: np.ndarray, b_vector: np.ndarray) -> np.ndarray:
        """
        A private helper method to solve linear equations of the form A @ x = b.

        Args:
            a_matrix (np.ndarray): The coefficient matrix A.
            b_vector (np.ndarray): The right-hand side vector b.

        Returns:
            np.ndarray: The solution vector x.

        Raises:
            np.linalg.LinAlgError: If the matrix A is singular or ill-conditioned.
        """
        try:
            solution: np.ndarray = np.linalg.solve(a_matrix, b_vector)
            return solution
        except np.linalg.LinAlgError as e:
            # For theoretical guarantees (e.g., Lemma 12), the matrices are invertible.
            # If this error occurs, it might be due to extreme numerical precision issues
            # or an unexpected MDP structure that violates assumptions.
            # np.linalg.lstsq can provide a least-squares solution if needed,
            # but for exact methods, solve is preferred.
            warnings.warn(f"Linear system solving failed: {e}. Attempting with lstsq (might indicate problem in MDP or numerical instability).")
            # Fallback to least squares solution, although for exact solutions this might not be ideal
            solution, residuals, rank, singular_values = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
            return solution


    def _project_onto_simplex(self, prob_vector: np.ndarray) -> np.ndarray:
        """
        Projects a given vector onto the probability simplex.
        This ensures the resulting probabilities are non-negative and sum to 1.
        (Algorithm by John Duchi, et al. 2008)

        Args:
            prob_vector (np.ndarray): A 1D NumPy array representing probabilities for actions
                                      in a single state (or any vector to be projected).

        Returns:
            np.ndarray: The projected vector, where elements are non-negative and sum to 1.
        """
        num_elements: int = len(prob_vector)
        # Handle edge case where num_elements is 0 or 1
        if num_elements == 0:
            return np.array([])
        if num_elements == 1:
            return np.array([1.0])

        sorted_vector: np.ndarray = np.sort(prob_vector)[::-1] # Descending order
        
        # Calculate candidates for lambda
        # t_j = (sum(u[:j+1]) - 1) / (j+1)
        # Using cumulative sum for efficiency
        cum_sum_u: np.ndarray = np.cumsum(sorted_vector)
        
        # Calculate t_j for all j
        t_values: np.ndarray = (cum_sum_u - 1) / np.arange(1, num_elements + 1)
        
        # Find rho: the largest j such that u[j] - t_j > 0
        valid_rhos = sorted_vector - t_values > 0
        
        if not np.any(valid_rhos): # All u[j] - t_j <= 0, implies all t_j >= u[j].
                                   # This means lambda should be t_0 if u is already sorted correctly.
                                   # This happens when the original vector is already close to or on the simplex.
            lambda_val = t_values[-1] # Fallback to the largest average if no explicit rho is found
                                      # This typically shouldn't be reached if formulation is exact.
        else:
            rho: int = np.where(valid_rhos)[0][-1] # Get the largest index j
            lambda_val: float = t_values[rho]

        projected_vector: np.ndarray = np.maximum(0, prob_vector - lambda_val)
        
        # Ensure exact sum to 1 due to floating point inaccuracies, if needed.
        # This can sometimes cause issues in very tight numerical comparisons,
        # but often helps with maintaining the simplex property.
        sum_projected = np.sum(projected_vector)
        if sum_projected > 0:
            projected_vector /= sum_projected
        else: # Should not happen if prob_vector is reasonable
            projected_vector = np.ones_like(prob_vector) / num_elements
            warnings.warn("Projection resulted in all zeros, returning uniform distribution.")

        return projected_vector

    def compute_stationary_distribution(self, policy_p: np.ndarray) -> np.ndarray:
        """
        Computes the stationary distribution d^π for a given policy-dependent transition kernel P^π.
        Solves d^π @ P^π = d^π and sum(d^π) = 1.

        Args:
            policy_p (np.ndarray): The policy-dependent transition kernel P^π of shape (S, S).

        Returns:
            np.ndarray: The stationary distribution d^π of shape (S,).
        """
        s_count: int = self.mdp.S

        # The equation d^π P^π = d^π can be rewritten as d^π (P^π - I) = 0.
        # In column vector form, this is (P^π - I)^T d^π = 0.
        # The matrix (P^π - I)^T is singular. To get a unique solution,
        # one equation is replaced by the normalization constraint sum(d^π) = 1.

        # System matrix A_system: (P^π - I)^T with last row replaced by ones.
        a_system: np.ndarray = (policy_p - np.eye(s_count)).T
        
        # Replace the last row with the sum constraint (sum d_i = 1)
        a_system[-1, :] = np.ones(s_count)

        # Right-hand side vector b_system
        b_system: np.ndarray = np.zeros(s_count)
        # Set the last element to 1 for the sum constraint
        b_system[-1] = 1.0

        d_pi: np.ndarray = self._solve_linear_system(a_system, b_system)
        
        # Ensure non-negativity and normalization due to potential numerical errors
        d_pi = np.maximum(0, d_pi)
        if np.sum(d_pi) > 0:
            d_pi /= np.sum(d_pi) # Ensure it sums to exactly 1
        else: # Should ideally not happen for irreducible MDPs.
            d_pi = np.ones(s_count) / s_count
            warnings.warn("Stationary distribution sum is zero, returning uniform distribution.")

        return d_pi

    def compute_projected_value_function(self, policy_p: np.ndarray, policy_r_vec: np.ndarray) -> np.ndarray:
        """
        Computes the unique projected relative value function v_φ^π.
        This is solved from (I - Φ P^π) v_φ^π = Φ r^π.

        Args:
            policy_p (np.ndarray): The policy-dependent transition kernel P^π of shape (S, S).
            policy_r_vec (np.ndarray): The policy-dependent reward vector r^π of shape (S,).

        Returns:
            np.ndarray: The projected relative value function v_φ^π of shape (S,).
        """
        s_count: int = self.mdp.S

        # Left-hand side matrix: A_matrix = I - Φ P^π
        a_matrix: np.ndarray = np.eye(s_count) - self.PHI @ policy_p

        # Right-hand side vector: b_vector = Φ r^π
        b_vector: np.ndarray = self.PHI @ policy_r_vec

        v_phi: np.ndarray = self._solve_linear_system(a_matrix, b_vector)
        return v_phi

    def compute_action_value_function(
        self,
        current_policy: np.ndarray,
        projected_value_function: np.ndarray,
        avg_reward: float
    ) -> np.ndarray:
        """
        Computes the state-action value function Q^π.
        Q^π(s,a) = r(s,a) + sum_s' P(s'|s,a) v_φ^π(s') - ρ^π.

        Args:
            current_policy (np.ndarray): The current policy π of shape (S, A).
            projected_value_function (np.ndarray): The projected relative value function v_φ^π of shape (S,).
            avg_reward (float): The current average reward ρ^π.

        Returns:
            np.ndarray: The state-action value function Q^π of shape (S, A).
        """
        s_count: int = self.mdp.S
        a_count: int = self.mdp.A
        transitions: np.ndarray = self.mdp.P # P[s,a,s']
        rewards: np.ndarray = self.mdp.R # R[s,a]

        q_pi: np.ndarray = np.zeros((s_count, a_count))

        for s in range(s_count):
            for a in range(a_count):
                # Expected value of the next state: sum_s' P(s'|s,a) v_φ^π(s')
                expected_next_value: float = np.dot(transitions[s, a, :], projected_value_function)
                q_pi[s, a] = rewards[s, a] + expected_next_value - avg_reward
        return q_pi

    def find_optimal_average_reward(self, convergence_threshold: float = 1e-9, max_iterations: int = 1000) -> float:
        """
        Calculates the globally optimal average reward ρ* for the MDP using Average Reward Policy Iteration.

        Args:
            convergence_threshold (float): The threshold for policy improvement convergence.
            max_iterations (int): Maximum number of policy iteration loops to prevent infinite loops.

        Returns:
            float: The optimal average reward ρ*.
        """
        s_count: int = self.mdp.S
        a_count: int = self.mdp.A

        # Initialize with an arbitrary policy (e.g., uniform random)
        current_policy: np.ndarray = self.mdp.generate_uniform_random_policy()
        
        last_avg_reward: float = -np.inf # Initialize with a very small number

        for iter_pi in range(max_iterations):
            # --- Policy Evaluation ---
            policy_p: np.ndarray = self.mdp.generate_policy_P(current_policy)
            policy_r_vec: np.ndarray = self.mdp.generate_policy_R(current_policy)

            d_pi: np.ndarray = self.compute_stationary_distribution(policy_p)
            v_phi: np.ndarray = self.compute_projected_value_function(policy_p, policy_r_vec)
            
            avg_reward: float = np.dot(d_pi, policy_r_vec) # ρ^π = sum_s d^π(s) r^π(s)

            # --- Policy Improvement ---
            q_pi: np.ndarray = self.compute_action_value_function(current_policy, v_phi, avg_reward)
            
            new_policy: np.ndarray = np.zeros_like(current_policy)
            policy_changed: bool = False

            for s in range(s_count):
                # Find action that maximizes Q(s,a)
                best_action: int = np.argmax(q_pi[s, :])
                
                # If the greedy action is different from the current policy's dominant action
                # (or if current policy is not purely greedy for this state)
                if not np.isclose(current_policy[s, best_action], 1.0):
                    policy_changed = True
                
                new_policy[s, best_action] = 1.0 # Set greedy action probability to 1

            # Check for convergence
            if not policy_changed:
                break
            
            current_policy = new_policy # Update policy for next iteration

            # Check convergence of average reward (alternative convergence check)
            if np.abs(avg_reward - last_avg_reward) < convergence_threshold:
                break
            last_avg_reward = avg_reward
        else:
            warnings.warn(f"Policy Iteration did not converge within {max_iterations} iterations. Current average reward might not be optimal.")

        return avg_reward

