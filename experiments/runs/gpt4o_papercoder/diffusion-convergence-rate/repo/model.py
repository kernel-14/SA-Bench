# model.py
import torch
import numpy as np
from typing import Dict, Any, Optional


class DiffusionModel:
    """
    The DiffusionModel class implements forward processes, reverse processes, and score functions
    for score-based diffusion models. It serves as the backbone for the experimental pipeline.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize the diffusion model with the configuration parameters.

        Args:
            params (Dict[str, Any]): Configuration dictionary loaded from the YAML file.
        """
        self.params = params
        self.T = params.get("sampling", {}).get("T", 1000)  # Number of diffusion steps
        self.L = params.get("sampling", {}).get("Lipschitz_constant", 0.01)  # Lipschitz constant
        self.dimensions = params.get("dataset", {}).get("dimensions", 10)  # Dataset dimensions
        self.randomized_schedule = params.get("sampling", {}).get("step_schedule", {}).get("randomized", True)
        self.c0 = params.get("sampling", {}).get("step_schedule", {}).get("c0", 0.001)
        self.c1 = params.get("sampling", {}).get("step_schedule", {}).get("c1", 0.01)
        self.pretrained_score = params.get("model", {}).get("score_function", {}).get("pretrained", False)

        # Score function attributes
        self.score_function_weights = None
        self.layers = params.get("model", {}).get("layers", 3)
        self.hidden_units = params.get("model", {}).get("hidden_units", 128)

        # Precompute alpha and cumulative alpha values
        self._initialize_alpha_schedule()

    def _initialize_alpha_schedule(self) -> None:
        """
        Precompute alpha values and cumulative alpha values for the forward process.
        These values directly affect noisification and denoising.
        """
        self.alpha_t = np.linspace(1 - self.c0, self.c1, self.T)
        self.cumulative_alpha_t = np.cumprod(self.alpha_t)

    def forward_process(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward diffusion process.

        Args:
            x (torch.Tensor): Input data sample drawn from the dataset (X_0).

        Returns:
            torch.Tensor: Noisy version of the input data (XT) at the final timestep T.
        """
        noise = torch.randn_like(x)
        cumulative_alpha_t = torch.tensor(self.cumulative_alpha_t[-1], dtype=torch.float32)
        noisy_x = torch.sqrt(cumulative_alpha_t) * x + torch.sqrt(1 - cumulative_alpha_t) * noise
        return noisy_x

    def reverse_process(self, y: torch.Tensor, score_function_override: Optional[torch.nn.Module] = None) -> torch.Tensor:
        """
        Implements the reverse diffusion process.

        Args:
            y (torch.Tensor): Input Gaussian noise (Y_0).
            score_function_override (Optional[torch.nn.Module]): If provided, use this score function model instead.

        Returns:
            torch.Tensor: Reconstructed data sample resembling the target dataset (YT).
        """
        score_fn = score_function_override or self._sample_score_function
        cumulative_alpha_t = torch.tensor(self.cumulative_alpha_t, dtype=torch.float32)

        for t in range(self.T, 0, -1):
            tau = 1 - cumulative_alpha_t[t - 1]
            score = score_fn(y, tau)
            y = y + (1 / (2 * (1 - tau))) * score
        return y

    def score_function(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Implements the score function estimated or pretrained:
        s_tau(x) = ∇ log p_(X_t).

        Args:
            x (torch.Tensor): Noisy sample X_t or Y_t at time τ.
            tau (float): Time or step-index of the noisy sample.

        Returns:
            torch.Tensor: Estimated score function values s_tau(x).
        """
        if self.pretrained_score:
            assert self.score_function_weights is not None, "Pretrained score function weights not loaded."
            return self.score_function_weights(x)

        # Approximate gradient using numerics (Monte Carlo approximation for Gaussian noise)
        mean = torch.sqrt(1 - tau) * x
        covariance_matrix = torch.tensor(tau, dtype=torch.float32)
        noise = torch.randn_like(x, dtype=torch.float32)

        # Numerical approximation: ∇ log p_(X_tau) via centered Gaussian perturbations
        score = (x - mean) / covariance_matrix
        return score

    def _sample_score_function(self, y: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Internal utility to calculate the score function:
        s_tau*(x) = ∇ log p_(X_tau).

        Args:
            y (torch.Tensor): Noisy sample Y converted during reverse traversal.
            tau (float): Discretized or continuous time τ.

        Returns:
            torch.Tensor: Estimated score function values for Y_t.
        """
        return self.score_function(y, tau)
