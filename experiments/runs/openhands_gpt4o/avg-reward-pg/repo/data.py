import numpy as np
from typing import Callable, Tuple

def generate_synthetic_mdp(
    state_space: int,
    action_space: int,
    reward_variance: float = 1.0
) -> Tuple[Callable[[int, int], float], Callable[[int, int], int]]:
    """
    Generate synthetic reward and transition functions for an MDP.

    Args:
        state_space (int): Number of states.
        action_space (int): Number of actions.
        reward_variance (float): Variance of the reward function.

    Returns:
        Tuple[Callable[[int, int], float], Callable[[int, int], int]]: Reward and transition functions.
    """
    rewards = np.random.uniform(-reward_variance, reward_variance, (state_space, action_space))
    transitions = np.random.randint(0, state_space, (state_space, action_space))

    def reward_function(state: int, action: int) -> float:
        return rewards[state, action]

    def transition_function(state: int, action: int) -> int:
        return transitions[state, action]

    return reward_function, transition_function

# Example usage:
# reward_fn, transition_fn = generate_synthetic_mdp(state_space=5, action_space=3)
# print(reward_fn(0, 1))  # Reward for state 0, action 1
# print(transition_fn(0, 1))  # Next state for state 0, action 1