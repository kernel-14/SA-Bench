import numpy as np

class Policy:
    def __init__(self, num_states, num_actions, initial_policy=None):
        self.num_states = num_states
        self.num_actions = num_actions
        if initial_policy is None:
            # Initialize with a uniform random policy
            self.policy_matrix = np.random.rand(num_states, num_actions)
            self.policy_matrix = self._project_to_simplex(self.policy_matrix)
        else:
            # Ensure the provided policy is valid
            assert initial_policy.shape == (num_states, num_actions)
            self.policy_matrix = self._project_to_simplex(initial_policy)

    def _project_to_simplex(self, prob_matrix):
        """
        Projects a matrix onto the policy simplex, ensuring each row sums to 1 and is non-negative.
        This is a common projection for probability distributions.
        It handles numerical instabilities that might lead to negative probabilities.
        """
        # Ensure non-negativity first
        projected_matrix = np.maximum(prob_matrix, 0)
        # Normalize each row to sum to 1
        row_sums = projected_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero for rows that might sum to zero (e.g., all zeros)
        # In such cases, distribute probability uniformly if row_sum is 0
        projected_matrix = np.where(row_sums == 0, 1.0 / self.num_actions, projected_matrix / row_sums)
        return projected_matrix

    def get_policy_gradient(self, stationary_distribution, q_function):
        """
        Calculates the policy gradient according to Equation 97 from the paper.

        d(rho)/d(pi(s,a)) = d^pi(s) * Q^pi(s,a)

        Args:
            stationary_distribution (np.ndarray): Stationary distribution d^pi. Shape: (num_states,)
            q_function (np.ndarray): Relative state-action value function Q^pi.
                                     Shape: (num_states, num_actions)

        Returns:
            np.ndarray: Policy gradient matrix. Shape: (num_states, num_actions)
        """
        # Element-wise product of d^pi(s) and Q^pi(s,a)
        # stationary_distribution has shape (num_states,)
        # q_function has shape (num_states, num_actions)
        # We want d^pi(s) * Q^pi(s,a) for each s,a pair.
        # This can be achieved by broadcasting stationary_distribution
        policy_gradient = stationary_distribution[:, np.newaxis] * q_function
        return policy_gradient

    def update_policy(self, policy_gradient, step_size):
        """
        Updates the policy using policy gradient ascent and projection, as per Equation 103.

        pi_{k+1} = Proj_Pi [pi_k + eta * grad_pi(rho^pi)]

        Args:
            policy_gradient (np.ndarray): The calculated policy gradient matrix.
                                          Shape: (num_states, num_actions)
            step_size (float): The learning rate (eta).
        """
        # Gradient ascent step
        new_policy_unprojected = self.policy_matrix + step_size * policy_gradient
        
        # Project back to the policy simplex
        self.policy_matrix = self._project_to_simplex(new_policy_unprojected)

    def get_policy_matrix(self):
        return self.policy_matrix
