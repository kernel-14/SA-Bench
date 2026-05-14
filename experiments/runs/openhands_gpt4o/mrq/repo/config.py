# Configuration file for MR.Q

# Hyperparameters
state_dim = 512
action_dim = 256
zsa_dim = 512
learning_rate = 1e-4
weight_decay = 1e-4
batch_size = 256
epochs = 100

# Environment settings
env_name = "Atari-Pong-v5"

# Training settings
encoder_horizon = 3
target_update_frequency = 250
reward_bins = 65
reward_range = [-10, 10]