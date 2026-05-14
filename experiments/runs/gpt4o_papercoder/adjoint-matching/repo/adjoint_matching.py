## adjoint_matching.py

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing import List, Dict
import numpy as np
from scipy.integrate import solve_ivp
from memoryless_noise_schedule import NoiseSchedule
from model import BaseModel

class AdjointMatching:
    """
    Implements the Adjoint Matching algorithm for fine-tuning models 
    according to the memoryless stochastic optimal control (SOC) formulation.
    """

    def __init__(self, model: BaseModel, config: Dict, train_loader: DataLoader):
        """
        Initializes the Adjoint Matching class.

        Args:
            model (BaseModel): Pre-trained FlowMatching or Diffusion model.
            config (Dict): Configuration dictionary loaded from `config.yaml`.
            train_loader (DataLoader): PyTorch DataLoader for training data.
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.device = config['general'].get("device", "cuda")

        # Initialize noise schedule
        self.noise_schedule = NoiseSchedule(config)
        self.t_values = np.linspace(0, 1, config['training']['timesteps'], endpoint=False)
        self.sigma_values = self.noise_schedule.get_schedule()

        # Loss clipping threshold
        self.loss_clipping_threshold = config['fine_tuning'].get("loss_clipping_threshold", 1.6 * config['fine_tuning']['scaling_factor'] ** 2)

        # Scaling factor for the reward model
        self.reward_scaling = config['fine_tuning'].get("scaling_factor", 1000)

    def solve_adjoint_ode(self, timesteps: List[float], trajectories: List[Tensor]) -> List[Tensor]:
        """
        Solves the lean adjoint ODE backwards in time to compute adjoint states.

        Args:
            timesteps (List[float]): Discretized timesteps, descending from 1 to 0.
            trajectories (List[Tensor]): Trajectories generated in the forward process.

        Returns:
            List[Tensor]: Adjoint states for all timesteps.
        """
        adjoint_states = []
        batch_size = trajectories[0].size(0)  # Assume batch size remains constant across timesteps

        # Initialize adjoint state at final timestep using terminal cost g(X1)
        terminal_states = trajectories[-1].detach()
        adjoint_t = torch.autograd.grad(
            outputs=self._compute_terminal_cost(terminal_states),
            inputs=terminal_states,
            grad_outputs=torch.ones_like(terminal_states),
            retain_graph=True,
            create_graph=True,
        )[0]
        adjoint_states.append(adjoint_t)

        # Solve backwards using Euler discretization
        for i in reversed(range(len(timesteps) - 1)):
            delta_t = timesteps[i + 1] - timesteps[i]
            current_state = trajectories[i].detach()
            next_adjoint = adjoint_states[-1]

            # Compute intermediate terms for adjoint ODE
            drift_gradient = torch.autograd.grad(
                outputs=self._compute_drift(current_state, timesteps[i]),
                inputs=current_state,
                grad_outputs=torch.ones_like(current_state),
                retain_graph=True,
                create_graph=True,
            )[0]

            running_gradient = self._compute_running_cost_gradient(current_state, timesteps[i])

            # Euler backward update for adjoint state
            adjoint_t = next_adjoint + delta_t * (drift_gradient.T @ next_adjoint + running_gradient)
            adjoint_states.append(adjoint_t)

        # Reverse adjoint states to match time progression
        return adjoint_states[::-1]

    def compute_adjoint_loss(self) -> Tensor:
        """
        Computes the Adjoint Matching loss over sampled trajectories.

        Returns:
            Tensor: Scalar loss value for optimization.
        """
        loss = 0.0
        for batch in self.train_loader:
            batch_input = batch[0].to(self.device)  # Assuming input data is at index 0

            # Generate forward trajectories
            trajectories = self._generate_forward_trajectories(batch_input)

            # Solve adjoint ODE backwards in time
            adjoint_states = self.solve_adjoint_ode(timesteps=self.t_values, trajectories=trajectories)

            # Compute regression loss at each timestep
            for t_idx, t in enumerate(self.t_values[:-1]):
                current_trajectory = trajectories[t_idx].to(self.device)
                adjoint_t = adjoint_states[t_idx].to(self.device)
                sigma_t = self.sigma_values[t_idx]

                # Match adjoint state and gradient
                u_theta = self.model(current_trajectory, t)
                base_velocity = self._compute_base_velocity(current_trajectory, t)
                matching_term = (2 / sigma_t) * (u_theta - base_velocity) + sigma_t * adjoint_t

                # Compute squared loss and apply clipping
                timestep_loss = torch.mean(matching_term.pow(2))
                loss += min(self.loss_clipping_threshold, timestep_loss.item())

        return loss / len(self.train_loader)

    def fine_tune(self):
        """
        Executes the full Adjoint Matching fine-tuning loop.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=(0.95, 0.999),
            weight_decay=self.config['training']['weight_decay']
        )

        for epoch in range(self.config['training']['epochs']):
            self.model.train()
            epoch_loss = 0.0

            for batch in self.train_loader:
                # Reset optimizer gradients
                optimizer.zero_grad()

                # Compute adjoint loss for batch
                loss = self.compute_adjoint_loss()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clipping'])

                # Perform optimization step
                optimizer.step()
                epoch_loss += loss.item()

            print(f"[INFO] Epoch {epoch + 1}/{self.config['training']['epochs']} - Loss: {epoch_loss:.4f}")

    def _generate_forward_trajectories(self, batch_input: Tensor) -> List[Tensor]:
        """
        Generates forward trajectories using the model and memoryless noise schedule.

        Args:
            batch_input (Tensor): Model input for generating trajectories.

        Returns:
            List[Tensor]: Trajectories for all timesteps.
        """
        trajectories = []
        current_state = batch_input
        for t_idx, t in enumerate(self.t_values):
            noise = self._generate_noise(batch_input)
            drift = self._compute_drift(current_state, t)
            sigma = self.sigma_values[t_idx]

            # Forward trajectory update
            next_state = current_state + drift + sigma * noise
            trajectories.append(next_state)
            current_state = next_state
        return trajectories

    def _compute_drift(self, x: Tensor, t: float) -> Tensor:
        """
        Computes drift term using the model's forward pass.

        Args:
            x (Tensor): Current state tensor.
            t (float): Timestep.

        Returns:
            Tensor: Drift velocity vector.
        """
        return self.model(x, t)

    def _compute_base_velocity(self, x: Tensor, t: float) -> Tensor:
        """
        Computes the base pre-trained model velocity.

        Args:
            x (Tensor): Current state tensor.
            t (float): Timestep.

        Returns:
            Tensor: Base model velocity vector for current state.
        """
        # This retrieves the velocity from the pre-trained model
        return self.model(x, t)

    def _compute_terminal_cost(self, x_final: Tensor) -> Tensor:
        """
        Computes the terminal cost g(X1).

        Args:
            x_final (Tensor): Final state tensor at timestep T.

        Returns:
            Tensor: Terminal cost scalar.
        """
        reward_model = self.config.get("reward_model", None)
        return reward_model(x_final) * self.reward_scaling if reward_model else torch.zeros_like(x_final)

    def _compute_running_cost_gradient(self, x: Tensor, t: float) -> Tensor:
        """
        Computes the gradient of the running state cost f(x, t).

        This is set to zero, per the configuration and paper assumptions.

        Args:
            x (Tensor): State tensor.
            t (float): Timestep.

        Returns:
            Tensor: Running cost gradient (zero tensor).
        """
        return torch.zeros_like(x)

    def _generate_noise(self, batch_input: Tensor) -> Tensor:
        """
        Generates noise for trajectory updates.

        Args:
            batch_input (Tensor): Batch data input dimensions.

        Returns:
            Tensor: Gaussian noise tensor.
        """
        return torch.randn_like(batch_input).to(self.device)

