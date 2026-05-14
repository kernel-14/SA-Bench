import numpy as np
from model import PolicyGradient

def train_policy_gradient(
    state_space: int,
    action_space: int,
    learning_rate: float,
    num_iterations: int,
    reward_function,
    transition_function,
    initial_state: int
):
    policy_gradient = PolicyGradient(state_space, action_space, learning_rate)

    for iteration in range(num_iterations):
        state = initial_state
        total_reward = 0

        for _ in range(100):  # Assume a fixed episode length of 100 steps
            action_probabilities = policy_gradient.get_action_probabilities(state)
            action = np.random.choice(action_space, p=action_probabilities)

            reward = reward_function(state, action)
            next_state = transition_function(state, action)

            advantage = reward - total_reward / (iteration + 1)
            gradient = policy_gradient.compute_gradient(state, action, advantage)
            policy_gradient.update_policy(gradient)

            state = next_state
            total_reward += reward

        print(f"Iteration {iteration + 1}/{num_iterations}, Total Reward: {total_reward}")

    return policy_gradient

# Example usage (replace with actual reward and transition functions):
# trained_policy = train_policy_gradient(
#     state_space=5,
#     action_space=3,
#     learning_rate=0.01,
#     num_iterations=1000,
#     reward_function=lambda s, a: np.random.rand(),
#     transition_function=lambda s, a: (s + a) % 5,
#     initial_state=0
# )