
import numpy as np
from scipy.linalg import inv
from typing import Tuple, Optional

class MDP:
    def __init__(self, S: int, A: int, P: np.ndarray, R: np.ndarray):
        """
        Initializes a Markov Decision Process (MDP).

        Args:
            S (int): Number of states.
            A (int): Number of actions.
            P (np.ndarray): Transition probability kernel P(s'|s,a). Shape (S, A, S).
            R (np.ndarray): Reward function R(s,a). Shape (S, A).
        """
        self.S = S
        self.A = A
        self.P = P
        self.R = R

    def get_reward_for_policy(self, policy: np.ndarray) -> np.ndarray:
        """
        Calculates the expected reward for each state under a given policy.

        Args:
            policy (np.ndarray): Policy pi(a|s). Shape (S, A).

        Returns:
            np.ndarray: Expected reward r^pi(s). Shape (S,).
        """
        return np.sum(self.R * policy, axis=1)

    def get_transition_matrix_for_policy(self, policy: np.ndarray) -> np.ndarray:
        """
        Calculates the transition matrix for a given policy.

        Args:
            policy (np.ndarray): Policy pi(a|s). Shape (S, A).

        Returns:
            np.ndarray: Transition matrix P^pi(s'|s). Shape (S, S).
        """
        return np.sum(self.P * policy[:, np.newaxis, :], axis=2) # P(s'|s,a) * pi(a|s) -> sum over a

    def get_stationary_distribution(self, P_pi: np.ndarray) -> np.ndarray:
        """
        Calculates the stationary distribution for a given transition matrix P^pi.

        Args:
            P_pi (np.ndarray): Transition matrix P^pi(s'|s). Shape (S, S).

        Returns:
            np.ndarray: Stationary distribution d^pi(s). Shape (S,).
        """
        # (I - P_pi.T) d_pi = 0 subject to sum(d_pi) = 1
        # A_matrix = (np.eye(self.S) - P_pi.T)
        # A_matrix = np.vstack([A_matrix, np.ones(self.S)])
        # b_vector = np.zeros(self.S + 1)
        # b_vector[-1] = 1
        # d_pi = np.linalg.lstsq(A_matrix, b_vector, rcond=None)[0]
        # This is a more numerically stable way:
        # P_pi_ext = np.vstack([P_pi.T - np.eye(self.S), np.ones(self.S)])
        # b_ext = np.zeros(self.S + 1)
        # b_ext[-1] = 1.
        # d_pi = np.linalg.lstsq(P_pi_ext, b_ext)[0]
        # return d_pi

        # The eigenvalue method should be more robust
        eigenvalues, eigenvectors = np.linalg.eig(P_pi.T)
        stationary_dist = np.real(eigenvectors[:, np.isclose(eigenvalues, 1)][:, 0])
        stationary_dist /= np.sum(stationary_dist)
        return stationary_dist

    def get_average_reward(self, policy: np.ndarray, d_pi: np.ndarray, r_pi: np.ndarray) -> float:
        """
        Calculates the average reward for a given policy.

        Args:
            policy (np.ndarray): Policy pi(a|s). Shape (S, A).
            d_pi (np.ndarray): Stationary distribution d^pi(s). Shape (S,).
            r_pi (np.ndarray): Expected reward r^pi(s). Shape (S,).

        Returns:
            float: Average reward rho^pi.
        """
        return np.sum(d_pi * r_pi)

    def get_projected_value_function(self, policy: np.ndarray, rho_pi: float, r_pi: np.ndarray, P_pi: np.ndarray) -> np.ndarray:
        """
        Calculates the unique projected value function v_phi^pi.
        Equation 15 and 36 from the paper.

        Args:
            policy (np.ndarray): Policy pi(a|s). Shape (S, A).
            rho_pi (float): Average reward rho^pi.
            r_pi (np.ndarray): Expected reward r^pi(s). Shape (S,).
            P_pi (np.ndarray): Transition matrix P^pi(s'|s). Shape (S, S).

        Returns:
            np.ndarray: Projected value function v_phi^pi(s). Shape (S,).
        """
        Phi = np.eye(self.S) - np.outer(np.ones(self.S), np.ones(self.S)) / self.S
        # v_phi = (I - Phi * P_pi)^-1 * Phi * r_pi
        # v_phi^pi = Phi (r^pi + P^pi v_phi^pi - rho^pi 1) -> (I - Phi P^pi) v_phi^pi = Phi (r^pi - rho^pi 1)
        # (I - Phi @ P_pi) @ v_phi = Phi @ (r_pi - rho_pi * np.ones(self.S))
        try:
            return np.linalg.solve(np.eye(self.S) - Phi @ P_pi, Phi @ (r_pi - rho_pi * np.ones(self.S)))
        except np.linalg.LinAlgError:
            # Handle singular matrix case, e.g., by adding a small regularization term or pseudo-inverse
            # This can happen if P_pi is not well-behaved or Phi is constructed improperly.
            # For irreducible and aperiodic, (I - Phi P_pi) should be invertible.
            print("Warning: (I - Phi @ P_pi) is singular. Using pseudo-inverse.")
            return np.linalg.pinv(np.eye(self.S) - Phi @ P_pi) @ Phi @ (r_pi - rho_pi * np.ones(self.S))


    def get_q_function(self, v_phi_pi: np.ndarray, rho_pi: float) -> np.ndarray:
        """
        Calculates the state-action value function Q^pi(s,a) for a given policy.
        Based on the average reward Bellman equation and Q function definition.

        Q^pi(s,a) = r(s,a) + sum_{s'} P(s'|s,a) v^pi(s') - rho^pi
        Here we use v_phi^pi.

        Args:
            v_phi_pi (np.ndarray): Projected value function v_phi^pi(s). Shape (S,).
            rho_pi (float): Average reward rho^pi.

        Returns:
            np.ndarray: State-action value function Q^pi(s,a). Shape (S, A).
        """
        Q_pi = np.zeros((self.S, self.A))
        for s in range(self.S):
            for a in range(self.A):
                Q_pi[s, a] = self.R[s, a] + np.sum(self.P[s, a, :] * v_phi_pi) - rho_pi
        return Q_pi


class PolicyGradient:
    def __init__(self, mdp: MDP, learning_rate: float, initial_policy_type: str = "uniform"):
        """
        Initializes the Policy Gradient algorithm.

        Args:
            mdp (MDP): The Markov Decision Process.
            learning_rate (float): Step size for policy updates (eta).
            initial_policy_type (str): Type of initial policy ("uniform" or "random").
        """
        self.mdp = mdp
        self.learning_rate = learning_rate
        self.policy = self._initialize_policy(initial_policy_type)
        self.avg_rewards_history = []
        self.value_functions_history = []
        self.q_functions_history = []
        self.stationary_dists_history = []

    def _initialize_policy(self, initial_policy_type: str) -> np.ndarray:
        """
        Initializes the policy.

        Args:
            initial_policy_type (str): "uniform" or "random".

        Returns:
            np.ndarray: Initial policy pi(a|s). Shape (S, A).
        """
        if initial_policy_type == "uniform":
            policy = np.ones((self.mdp.S, self.mdp.A)) / self.mdp.A
        elif initial_policy_type == "random":
            policy = np.random.rand(self.mdp.S, self.mdp.A)
            policy = policy / np.sum(policy, axis=1, keepdims=True)
        else:
            raise ValueError("Invalid initial_policy_type. Must be 'uniform' or 'random'.")
        return policy

    def _project_policy(self, policy_unprojected: np.ndarray) -> np.ndarray:
        """
        Projects the policy onto the space of valid randomized policies (simplex constraint).
        This is Proj_Pi in equation 6.

        Args:
            policy_unprojected (np.ndarray): Unprojected policy. Shape (S, A).

        Returns:
            np.ndarray: Projected policy. Shape (S, A).
        """
        # Ensure non-negativity
        projected_policy = np.maximum(0, policy_unprojected)
        # Normalize to sum to 1 across actions for each state
        projected_policy = projected_policy / np.sum(projected_policy, axis=1, keepdims=True)
        return projected_policy

    def _calculate_policy_gradient(self, Q_pi: np.ndarray) -> np.ndarray:
        """
        Calculates the policy gradient.
        For tabular policies, the parameter theta is equivalent to pi.
        Equation for d(rho)/d(theta) in the paper for tabular case becomes:
        d(rho)/d(pi(s,a)) = d^pi(s) * Q^pi(s,a) (simplified as d^pi(s) is 1 for the state s being updated)
        Actually, the paper says: d(rho)/d(theta) = sum_{s in S} d^pi(s) sum_{a in A} (d(pi(s,a))/d(theta)) Q^pi(s,a)
        If theta is simply pi(s,a) for a specific (s,a), then the gradient for that (s,a) would be d^pi(s) * Q^pi(s,a).
        However, the update rule (Eq 6) is for pi itself, so we need the gradient of rho w.r.t pi(s,a) for each (s,a).
        The term sum_{a in A} d(pi(s,a))/d(theta) Q^pi(s,a) becomes Q^pi(s,a) for the specific pi(s,a) we are updating.
        So the gradient should be d^pi(s) * Q^pi(s,a). This is typically referred to as the policy gradient for tabular case.

        Args:
            Q_pi (np.ndarray): State-action value function Q^pi(s,a). Shape (S, A).

        Returns:
            np.ndarray: Policy gradient. Shape (S, A).
        """
        # P_pi = self.mdp.get_transition_matrix_for_policy(self.policy)
        # d_pi = self.mdp.get_stationary_distribution(P_pi)
        # grad_rho = d_pi[:, np.newaxis] * Q_pi
        #
        # For simplicity, assuming the gradient update acts on the policy at each state independently,
        # with Q_pi representing the advantage-like function.
        # This aligns with common policy gradient implementations for tabular MDPs where pi(s,a) are parameters.
        # The paper's Equation 6 implies a direct update to pi using a gradient-like term.
        # "As we focus on tabular policies in this paper, our parameterization aligns with the tabular policy, where theta is equivalent to pi ."
        # "pi_{k+1} := Proj_Pi[ pi_k + eta * d(rho^pi)/d(pi) | pi = pi_k ]"
        # The exact form of d(rho^pi)/d(pi) is more complex than just d^pi(s) * Q^pi(s,a) because of the dependency of d^pi(s) on pi.
        # However, many policy gradient derivations simplify this to Q_pi for the update,
        # often assuming average reward is independent of d_pi or other simplifications.
        # Given the update structure, let's assume the gradient is simply Q_pi itself, or a scaled version.
        # A more common form of policy gradient for average reward is d(rho)/d(log pi(s,a)) = d^pi(s) * (Q^pi(s,a) - V^pi(s))
        # but the paper specifies d(rho)/d(theta) directly as sum d^pi(s) sum d(pi(s,a))/d(theta) Q^pi(s,a)

        # Let's use the interpretation from Sutton & Barto (2018) which the paper references for the theorem
        # In the context of "episodic" or "continuing" tasks, the policy gradient theorem for average reward is
        # grad_theta J(theta) = sum_s d_pi(s) sum_a grad_theta(pi(a|s)) * Q_pi(s,a)
        # For tabular case, if theta_sa = pi(a|s), then grad_theta_s'a' (pi(a|s)) = 1 if s=s', a=a', else 0.
        # So grad_pi(s,a) J(pi) = d_pi(s) * Q_pi(s,a).
        # This requires d_pi. We compute it for the current policy.

        P_pi = self.mdp.get_transition_matrix_for_policy(self.policy)
        d_pi = self.mdp.get_stationary_distribution(P_pi)
        grad_pi = d_pi[:, np.newaxis] * Q_pi # Policy gradient with respect to pi(s,a)
        return grad_pi

    def step(self) -> Tuple[np.ndarray, float]:
        """
        Performs one step of the Policy Gradient algorithm.

        Returns:
            Tuple[np.ndarray, float]: New policy and the average reward.
        """
        # 1. Calculate R^pi, P^pi, d^pi for current policy
        r_pi = self.mdp.get_reward_for_policy(self.policy)
        P_pi = self.mdp.get_transition_matrix_for_policy(self.policy)
        d_pi = self.mdp.get_stationary_distribution(P_pi)
        rho_pi = self.mdp.get_average_reward(self.policy, d_pi, r_pi)

        # 2. Calculate v_phi^pi and Q^pi
        v_phi_pi = self.mdp.get_projected_value_function(self.policy, rho_pi, r_pi, P_pi)
        Q_pi = self.mdp.get_q_function(v_phi_pi, rho_pi)

        # 3. Calculate Policy Gradient
        grad_pi = self._calculate_policy_gradient(Q_pi)

        # 4. Update policy (pi_{k+1} := pi_k + eta * grad_pi)
        policy_unprojected = self.policy + self.learning_rate * grad_pi

        # 5. Project policy onto the simplex (Proj_Pi)
        self.policy = self._project_policy(policy_unprojected)

        # Store history
        self.avg_rewards_history.append(rho_pi)
        self.value_functions_history.append(v_phi_pi) # Store v_phi_pi for consistency
        self.q_functions_history.append(Q_pi)
        self.stationary_dists_history.append(d_pi)

        return self.policy, rho_pi

    def train(self, iterations: int) -> Tuple[list, list, list, list]:
        """
        Runs the policy gradient training loop.

        Args:
            iterations (int): Number of iterations to train.

        Returns:
            Tuple[list, list, list, list]: Histories of average rewards, value functions, Q-functions, and stationary distributions.
        """
        print(f"Starting Policy Gradient training for {iterations} iterations...")
        for i in range(iterations):
            _, current_avg_reward = self.step()
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Iteration {i+1}/{iterations}, Average Reward: {current_avg_reward:.4f}")
        print("Training complete.")
        return self.avg_rewards_history, self.value_functions_history, self.q_functions_history, self.stationary_dists_history

