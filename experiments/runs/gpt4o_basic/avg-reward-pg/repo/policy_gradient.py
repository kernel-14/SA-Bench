
import numpy as np

def projected_policy_gradient(mdp, policy, step_size):
    """
    Perform one iteration of the projected policy gradient algorithm on the MDP.

    Args:
        mdp (AverageRewardMDP): The MDP instance containing states, actions, rewards, and transition probabilities.
        policy (np.ndarray): Current policy probabilities for states and actions.
        step_size (float): Learning rate for the gradient ascent step.

    Returns:
        np.ndarray: Updated policy probabilities after projection and gradient step.
    """
    projection_matrix = mdp.compute_projection_matrix()
    # Calculate gradient (Placeholder for actual gradient computation)
    gradient = np.zeros_like(policy) # Replace with actual computation
    # Perform gradient ascent update
    updated_policy = policy + step_size * gradient
    # Project updated policy back into feasible space
    projected_policy = projection_matrix @ updated_policy
    return projected_policy

