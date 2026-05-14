## curiosity_module.py

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any

class CuriosityModule:
    """
    Responsible for computing relevance scores based on curiosity-driven exploration.
    Implements a feature encoder and forward dynamics model to estimate prediction error.
    """

    def __init__(self, config: dict) -> None:
        """
        Initializes the curiosity module with a feature encoder, forward dynamics model, 
        and optimizers.

        Args:
            config (dict): Configuration dictionary parsed from `config.yaml`.
        """
        # Load hyperparameters from configuration file
        encoder_layers = config["curiosity_module"]["encoder_layers"]
        latent_dim = config["curiosity_module"]["latent_dim"]
        forward_dynamics_hidden = config["curiosity_module"]["forward_dynamics_hidden"]
        learning_rate = config["training"]["learning_rate"]

        # Initialize feature encoder (3-layer CNN)
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Initialize forward dynamics model (2-layer MLP)
        self.forward_dynamics = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, forward_dynamics_hidden),
            nn.ReLU(),
            nn.Linear(forward_dynamics_hidden, latent_dim)
        )

        # Optimizers
        self.encoder_optimizer = optim.Adam(self.feature_encoder.parameters(), lr=learning_rate)
        self.dynamics_optimizer = optim.Adam(self.forward_dynamics.parameters(), lr=learning_rate)

        # Loss
        self.loss_fn = nn.MSELoss()

    def compute_relevance_scores(self, transitions: List[Dict[str, Any]]) -> List[float]:
        """
        Computes relevance scores for each transition based on curiosity (prediction error).

        Args:
            transitions (List[Dict[str, Any]]): List of transitions in the form:
                [{'s': state, 'a': action, 's_prime': next_state, 'r': reward}]

        Returns:
            List[float]: List of relevance scores for each transition.
        """
        relevance_scores = []
        for transition in transitions:
            state = torch.tensor(transition["s"], dtype=torch.float32).unsqueeze(0)
            action = torch.tensor(transition["a"], dtype=torch.float32).unsqueeze(0)
            next_state = torch.tensor(transition["s_prime"], dtype=torch.float32).unsqueeze(0)

            # Encode raw states into latent features
            state_features = self.feature_encoder(state)
            next_state_features = self.feature_encoder(next_state)

            # Predict next latent state using dynamics model
            predicted_next_state_features = self.forward_dynamics(
                torch.cat([state_features, action], dim=1)
            )

            # Compute relevance score (curiosity / prediction error)
            score = self.loss_fn(predicted_next_state_features, next_state_features).item()
            relevance_scores.append(score)

        return relevance_scores

    def train(self, transitions: List[Dict[str, Any]]) -> None:
        """
        Trains the curiosity module (`feature_encoder` and `forward_dynamics`) 
        on a batch of transitions.

        Args:
            transitions (List[Dict[str, Any]]): List of transitions in the form:
                [{'s': state, 'a': action, 's_prime': next_state, 'r': reward}]
        """
        # Reset gradients
        self.encoder_optimizer.zero_grad()
        self.dynamics_optimizer.zero_grad()

        # Accumulate loss over the batch
        total_loss = 0.0
        batch_size = len(transitions)
        for transition in transitions:
            state = torch.tensor(transition["s"], dtype=torch.float32).unsqueeze(0)
            action = torch.tensor(transition["a"], dtype=torch.float32).unsqueeze(0)
            next_state = torch.tensor(transition["s_prime"], dtype=torch.float32).unsqueeze(0)

            # Encode raw states into latent features
            state_features = self.feature_encoder(state)
            next_state_features = self.feature_encoder(next_state)

            # Predict next latent state using dynamics model
            predicted_next_state_features = self.forward_dynamics(
                torch.cat([state_features, action], dim=1)
            )

            # Compute prediction error as the loss
            loss = self.loss_fn(predicted_next_state_features, next_state_features)
            total_loss += loss

        # Average batch loss
        average_loss = total_loss / batch_size

        # Backpropagation and optimization
        average_loss.backward()
        self.encoder_optimizer.step()
        self.dynamics_optimizer.step()

    def encode(self, states: List[torch.Tensor]) -> torch.Tensor:
        """
        Encodes raw states into latent features using the feature encoder.

        Args:
            states (List[torch.Tensor]): List of raw states.

        Returns:
            torch.Tensor: Encoded latent features.
        """
        encoded_states = [self.feature_encoder(state.unsqueeze(0)) for state in states]
        return torch.cat(encoded_states, dim=0)

    def predict(self, state_features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Predicts the next latent state features given current state features and actions.

        Args:
            state_features (torch.Tensor): Current latent state features.
            actions (torch.Tensor): Actions taken from the states.

        Returns:
            torch.Tensor: Predicted next latent state features.
        """
        return self.forward_dynamics(torch.cat([state_features, actions], dim=1))
