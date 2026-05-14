# Configuration file for the Policy Gradient implementation

CONFIG = {
    "state_space": 5,  # Number of states in the MDP
    "action_space": 3,  # Number of actions in the MDP
    "learning_rate": 0.01,  # Learning rate for policy updates
    "num_iterations": 1000,  # Number of training iterations
    "reward_variance": 1.0,  # Variance of the reward function
    "initial_state": 0,  # Initial state for training
    "episode_length": 100,  # Length of each episode
}