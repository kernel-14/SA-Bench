import numpy as np

class AverageRewardMDP:
    def __init__(self, num_states, num_actions, transitions, rewards):
        """
        Initializes an Average Reward MDP.

        Args:
            num_states (int): Number of states in the MDP.
            num_actions (int): Number of actions in the MDP.
            transitions (np.ndarray): Transition probability kernel.
                                      Shape: (num_states, num_actions, num_states)
                                      transitions[s, a, s_prime] is P(s_prime | s, a)
            rewards (np.ndarray): Reward function.
                                  Shape: (num_states, num_actions)
                                  rewards[s, a] is r(s, a)
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.transitions = transitions
        self.rewards = rewards

    def get_policy_transition_kernel(self, policy_matrix):
        """
        Calculates the policy-dependent transition kernel P^pi.

        Args:
            policy_matrix (np.ndarray): Policy matrix. Shape: (num_states, num_actions)
                                        policy_matrix[s, a] is pi(a | s)

        Returns:
            np.ndarray: P^pi transition kernel. Shape: (num_states, num_states)
                        P_pi[s, s_prime] is P^pi(s_prime | s)
        """
        # P^pi(s'|s) = sum_{a in A} pi(a|s) P(s'|s, a)
        P_pi = np.einsum('sa,sas->ss', policy_matrix, self.transitions)
        return P_pi

    def get_policy_reward_function(self, policy_matrix):
        """
        Calculates the policy-dependent reward function r^pi.

        Args:
            policy_matrix (np.ndarray): Policy matrix. Shape: (num_states, num_actions)

        Returns:
            np.ndarray: r^pi reward function. Shape: (num_states,)
                        r_pi[s] is r^pi(s)
        """\
        # r^pi(s) = sum_{a in A} pi(a|s) r(s, a)
        r_pi = np.einsum('sa,sa->s', policy_matrix, self.rewards)
        return r_pi

    def get_stationary_distribution(self, P_pi):
        """
        Calculates the stationary distribution d^pi for a given P^pi.

        Args:
            P_pi (np.ndarray): Policy-dependent transition kernel. Shape: (num_states, num_states)

        Returns:
            np.ndarray: Stationary distribution d^pi. Shape: (num_states,)
        """\
        # d^pi P^pi = d^pi  =>  d^pi (I - P^pi) = 0
        # Also, sum(d^pi) = 1
        A = P_pi.T - np.identity(self.num_states)
        # Add the constraint sum(d^pi) = 1 by replacing the last row
        # This is a common technique to solve for the unique stationary distribution
        # when the matrix (P_pi.T - I) is rank deficient (which it is, by 1)
        A[-1, :] = 1
        b = np.zeros(self.num_states)
        b[-1] = 1 # The sum must be 1
        d_pi = np.linalg.solve(A, b)
        return d_pi

    def get_average_reward(self, r_pi, d_pi):
        """
        Calculates the average reward rho^pi.

        Args:
            r_pi (np.ndarray): Policy-dependent reward function. Shape: (num_states,)
            d_pi (np.ndarray): Stationary distribution. Shape: (num_states,)

        Returns:
            float: Average reward rho^pi.
        """\
        # rho^pi = sum_{s in S} d^pi(s) r^pi(s)
        rho_pi = np.dot(d_pi, r_pi)
        return rho_pi

    def get_relative_value_function(self, P_pi, r_pi, rho_pi):
        """
        Calculates the unique relative state value function v_phi^pi using the projection method.
        This follows Lemma 1 from the paper.

        Args:
            P_pi (np.ndarray): Policy-dependent transition kernel. Shape: (num_states, num_states)
            r_pi (np.ndarray): Policy-dependent reward function. Shape: (num_states,)
            rho_pi (float): Average reward.

        Returns:
            np.ndarray: Unique relative value function v_phi^pi. Shape: (num_states,)
        """\
        # Projection matrix Phi = I - (1 @ 1.T) / |S|
        ones_vec = np.ones(self.num_states)
        Phi = np.identity(self.num_states) - np.outer(ones_vec, ones_vec) / self.num_states

        # From the paper, Equation 15 and 16, and the derivation in 3.1.1:
        # (I - Phi P^pi) v_phi^pi = Phi (r^pi - rho^pi 1)
        A = np.identity(self.num_states) - Phi @ P_pi
        b = Phi @ (r_pi - rho_pi * ones_vec)
        
        v_phi_pi = np.linalg.solve(A, b)
        return v_phi_pi

    def get_q_function(self, policy_matrix, P_pi, r_pi, rho_pi, v_phi_pi):
        """
        Calculates the relative state-action value function Q^pi.
        Q^pi(s, a) = r(s, a) + sum_{s'} P(s'|s,a) v^pi(s') - rho^pi.
        We use v_phi^pi as the unique representation of v^pi.

        Args:
            policy_matrix (np.ndarray): Policy matrix. Shape: (num_states, num_actions)
            P_pi (np.ndarray): Policy-dependent transition kernel. Shape: (num_states, num_states) - not directly used but good for context
            r_pi (np.ndarray): Policy-dependent reward function. Shape: (num_states,) - not directly used but good for context
            rho_pi (float): Average reward.
            v_phi_pi (np.ndarray): Unique relative value function v_phi^pi. Shape: (num_states,)

        Returns:
            np.ndarray: Relative state-action value function Q^pi. Shape: (num_states, num_actions)
        """\
        Q_pi = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                # Q^pi(s,a) = r(s,a) + sum_{s'} P(s'|s,a) v_phi^pi(s') - rho^pi
                Q_pi[s, a] = self.rewards[s, a] + np.dot(self.transitions[s, a, :], v_phi_pi) - rho_pi
        return Q_pi
