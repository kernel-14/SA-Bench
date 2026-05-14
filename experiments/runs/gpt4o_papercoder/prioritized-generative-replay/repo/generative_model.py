# generative_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any
from replay_buffer import ReplayBuffer
import numpy as np

class GenerativeModel:
    """
    Implements a conditional diffusion model for generating synthetic transitions.
    The model is trained using transitions from the real replay buffer and can generate
    synthetic transitions conditioned on relevance scores.
    """

    def __init__(self, config: dict) -> None:
        """
        Initializes the generative diffusion model using parameters from the configuration.

        Args:
            config (dict): Configuration dictionary parsed from `config.yaml`.
        """
        # Load hyperparameters from config
        self.latent_dim = config["generative_model"]["latent_dim"]
        self.guidance_scale = config["generative_model"]["guidance_scale"]
        self.noise_schedule = config["generative_model"]["noise_schedule"]
        self.training_steps = config["generative_model"]["training_steps"]
        self.learning_rate = config["training"]["learning_rate"]
        self.p_uncond = 0.25  # Probability of dropping the condition during training

        # Define the neural network for predicting noise (ϵ_θ)
        self.noise_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.latent_dim)
        )

        # Optimizer
        self.optimizer = optim.Adam(self.noise_predictor.parameters(), lr=self.learning_rate)

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noise_predictor.to(self.device)

    def _noise_schedule(self, step: int) -> float:
        """
        Calculates the noise level according to the configured noise schedule.

        Args:
            step (int): Current step in the diffusion process.

        Returns:
            float: Noise level β_t for the current step.
        """
        if self.noise_schedule == "linear":
            return step / self.training_steps
        elif self.noise_schedule == "cosine":
            return 0.5 * (1 - np.cos(np.pi * step / self.training_steps))
        else:
            raise ValueError(f"Unsupported noise schedule: {self.noise_schedule}")

    def _forward_process(self, transitions: List[Dict[str, Any]], time_step: int) -> torch.Tensor:
        """
        Applies the forward process by adding Gaussian noise to the transitions.

        Args:
            transitions (List[Dict[str, Any]]): List of transitions in the form:
                [{'s': state, 'a': action, 's_prime': next_state, 'r': reward}]
            time_step (int): Current step in the forward process.

        Returns:
            torch.Tensor: Noisy transitions with added Gaussian noise.
        """
        beta_t = self._noise_schedule(time_step)
        noisy_transitions = []
        for transition in transitions:
            transition_tensor = self._pack_transition(transition)  # Pack transition into a tensor
            noise = torch.randn_like(transition_tensor).to(self.device) * beta_t
            noisy_transitions.append(transition_tensor + noise)
        return torch.stack(noisy_transitions, dim=0)

    def _pack_transition(self, transition: Dict[str, Any]) -> torch.Tensor:
        """
        Packs a transition dictionary into a tensor for processing.

        Args:
            transition (Dict[str, Any]): A single transition in the form:
                {'s': state, 'a': action, 's_prime': next_state, 'r': reward}

        Returns:
            torch.Tensor: Packed tensor representation of the transition.
        """
        state = torch.tensor(transition["s"], dtype=torch.float32)
        action = torch.tensor(transition["a"], dtype=torch.float32)
        next_state = torch.tensor(transition["s_prime"], dtype=torch.float32)
        reward = torch.tensor([transition["r"]], dtype=torch.float32)

        return torch.cat([state.flatten(), action.flatten(), next_state.flatten(), reward], dim=0).to(self.device)

    def train(self, real_buffer: ReplayBuffer) -> None:
        """
        Trains the diffusion model using transitions from the real replay buffer.

        Args:
            real_buffer (ReplayBuffer): Replay buffer containing real transitions.
        """
        self.noise_predictor.train()

        for step in range(self.training_steps):
            # Sample transitions from the real replay buffer
            transitions = real_buffer.sample(batch_size=self.latent_dim)
            relevance_scores = np.random.uniform(0, 1, len(transitions))  # Placeholder for relevance scores
            
            # Forward noise process
            noisy_transitions = self._forward_process(transitions, step)
            predicted_noise = self.noise_predictor(noisy_transitions)

            # Compute reconstruction loss
            actual_noise = noisy_transitions - torch.stack([self._pack_transition(t) for t in transitions]).to(self.device)
            loss = self.loss_fn(predicted_noise, actual_noise)

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def sample(self, cond: bool, transition_dim: int) -> Dict[str, Any]:
        """
        Generates a single synthetic transition using classifier-free guidance.

        Args:
            cond (bool): Whether to condition sampling on relevance scores.
            transition_dim (int): Dimensionality of the transition space.

        Returns:
            Dict[str, Any]: Synthetic transition.
        """
        noisy_transition = torch.randn(transition_dim).to(self.device)

        # Apply denoising steps with classifier-free guidance
        for step in reversed(range(self.training_steps)):
            noise_prediction = self.noise_predictor(noisy_transition)
            if cond:
                conditioned_noise = self.guidance_scale * noise_prediction
                noisy_transition = noisy_transition - conditioned_noise

        # Unpack and return the generated transition
        return self._unpack_transition(noisy_transition)

    def generate_synthetic_transitions(self, relevance_scores: List[float]) -> List[Dict[str, Any]]:
        """
        Generates a batch of synthetic transitions conditioned on relevance scores.

        Args:
            relevance_scores (List[float]): List of relevance scores for conditioning.

        Returns:
            List[Dict[str, Any]]: List of synthetic transitions.
        """
        synthetic_transitions = []
        transition_dim = self.latent_dim  # Same as latent space dimension

        for score in relevance_scores:
            synthetic_transition = self.sample(cond=True, transition_dim=transition_dim)
            synthetic_transitions.append(synthetic_transition)

        return synthetic_transitions

    def _unpack_transition(self, transition_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Unpacks a tensor representation of a transition back into its dictionary form.

        Args:
            transition_tensor (torch.Tensor): Tensor representation of the transition.

        Returns:
            Dict[str, Any]: Unpacked transition in dictionary form.
        """
        split_indices = [0, 1, 2, 3]  # Assuming fixed sizes; adjust based on task
        return {
            "s": transition_tensor[split_indices[0]:split_indices[1]].cpu().numpy(),
            "a": transition_tensor[split_indices[1]:split_indices[2]].cpu().numpy(),
            "s_prime": transition_tensor[split_indices[2]:split_indices[3]].cpu().numpy(),
            "r": transition_tensor[split_indices[3]].item()
        }
