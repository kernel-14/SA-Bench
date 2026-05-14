import torch
import torch.nn as nn
import torch.optim as optim

class MRQ:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        Initializes MR.Q architecture with primary components.
        # State encoder
        self.f_omega = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # State-action encoder
        self.g_omega = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value network
        self.Q_theta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Policy network
        self.pi_phi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward_state(self, state):
        Processes state embedding.
        return self.f_omega(state)

    def forward_state_action(self, state, action):
        Processes state-action embedding.
        state_action = torch.cat([state, action], dim=-1)
        return self.g_omega(state_action)


import torch.nn.functional as F

    def compute_reward_loss(self, predicted_reward, actual_reward):
        Computes the loss for reward prediction.
        two_hot = self.two_hot_encode(actual_reward)
        return F.cross_entropy(predicted_reward, two_hot)

    def two_hot_encode(self, reward):
        Two-hot encoding for sparse reward magnitudes.
        # Assuming symexp encoding (exponential intervals)
        encoded = torch.zeros(reward.size(0), dtype=torch.float32)
        # Simplified placeholder logic; actual intervals expand symmetrically
        return encoded

    def compute_dynamics_loss(self, predicted_dynamics, target_dynamics):
        Computes the loss for dynamics prediction.
        return F.mse_loss(predicted_dynamics, target_dynamics)

    def compute_terminal_loss(self, predicted_terminal, actual_terminal):
        Computes the terminal signal loss.
        return F.mse_loss(predicted_terminal, actual_terminal)

# More functions or adjustments for model training follow.
    def update_value_function(self, predicted_value, target_value):
        Updates the value function using Huber loss.
        return F.smooth_l1_loss(predicted_value, target_value)

    def update_policy(self, state_embedding, advantage):
        Updates the policy using deterministic policy gradient.
        action_predictions = self.pi_phi(state_embedding)
        policy_loss = -0.5 * torch.sum(action_predictions * advantage)
        return policy_loss + self.regularization_penalty(action_predictions)

    def regularization_penalty(self, pre_activations):
        Optional penalty term to avoid sparse rewards causing trivial solutions.
        return torch.sum(pre_activations**2) * 0.01


