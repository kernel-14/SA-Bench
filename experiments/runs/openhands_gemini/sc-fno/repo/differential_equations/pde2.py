
import torch
import torch.nn.functional as F

class PDE2:
    """
    PDE2: Forced Burgers’ Equation
    Defined by: (1/pi) * du/dt + alpha * u * du/dx = gamma * d^2u/dx^2 + delta * sin(omega * t)
    Initial condition: u(x, 0) = (e^(-((x-x0)^2 / (2*sigma^2))) + sin(0.5 * pi * x))
    Parameters: alpha, gamma, delta, omega
    Domain: spatial x in [0, 1], temporal t in [0, pi]
    """
    def __init__(self):
        pass

    def initial_condition(self, x: torch.Tensor, x0: float = 0.5, sigma: float = 0.3) -> torch.Tensor:
        """
        Initial condition for Burgers' Equation.
        u(x, 0) = (e^(-((x-x0)^2 / (2*sigma^2))) + sin(0.5 * pi * x))
        """
        gaussian_pulse = torch.exp(-((x - x0)**2) / (2 * sigma**2))
        sinusoidal_component = torch.sin(0.5 * torch.pi * x)
        return gaussian_pulse + sinusoidal_component

    def rhs(self, t: torch.Tensor, u: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor,
            delta: torch.Tensor, omega: torch.Tensor, dx: float) -> torch.Tensor:
        """
        Right-hand side of the PDE for numerical solvers.
        du/dt = pi * (gamma * d^2u/dx^2 - alpha * u * du/dx + delta * sin(omega * t))
        """
        # u shape: (batch_size, spatial_x)

        # Compute first spatial derivative du/dx using central difference
        # Assuming periodic boundary conditions as stated in the paper: u(0,t) = u(1.0,t)
        u_shifted_right = torch.roll(u, shifts=1, dims=-1)
        u_shifted_left = torch.roll(u, shifts=-1, dims=-1)
        
        dudx = (u_shifted_left - u_shifted_right) / (2 * dx)

        # Compute second spatial derivative d^2u/dx^2 using central difference
        d2udx2 = (u_shifted_left - 2 * u + u_shifted_right) / (dx ** 2)

        # RHS terms
        advection_term = -alpha * u * dudx
        diffusion_term = gamma * d2udx2
        forcing_term = delta * torch.sin(omega * t)

        dudt = torch.pi * (advection_term + diffusion_term + forcing_term)
        return dudt

    def solution(self, x: torch.Tensor, t: torch.Tensor, initial_conditions: torch.Tensor, params: dict):
        """
        No analytical solution provided in the paper. This would typically be solved numerically.
        Placeholder for consistency.
        """
        raise NotImplementedError("Analytical solution for PDE2 not provided in the paper.")

    def get_sensitivities(self, x: torch.Tensor, t: torch.Tensor, initial_conditions: torch.Tensor, params: dict):
        """
        No analytical sensitivities provided in the paper. This would typically be computed via AD.
        Placeholder for consistency.
        """
        raise NotImplementedError("Analytical sensitivities for PDE2 not provided in the paper.")
