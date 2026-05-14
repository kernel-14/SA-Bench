
class AverageRewardMDP:
    def __init__(self, states, actions, transition_kernel, rewards):
        """
        Initialize an Average Reward MDP instance.

        Args:
            states (list): List of states.
            actions (list): List of actions.
            transition_kernel (dict): Transition probabilities with policy.
            rewards (dict): Reward values for state-action pairs.
        """
        self.states = states
        self.actions = actions
        self.transition_kernel = transition_kernel
        self.rewards = rewards

    def get_reward(self, state, action):
        """
        Return the reward for a given state and action pair.

        Args:
            state (str): State identifier.
            action (str): Action identifier.

        Returns:
            float: Reward value for state-action.
        """
        return self.rewards.get((state, action), 0)

    def transition_probability(self, state, action, next_state):
        """
        Return the probability of transitioning to the next state.

        Args:
            state (str): Current state identifier.
            action (str): Action identifier.
            next_state (str): Next state identifier.

        Returns:
            float: Transition probability.
        """
        return self.transition_kernel.get((state, action, next_state), 0)

    def compute_projection_matrix(self):
        """
        Compute the projection matrix to the subspace orthogonal to the all-ones vector.
        """
        import numpy as np
        num_states = len(self.states)
        identity_matrix = np.eye(num_states)
        ones_vector = np.ones((num_states, 1))
        projection_matrix = identity_matrix - (ones_vector @ ones_vector.T) / num_states
        return projection_matrix

