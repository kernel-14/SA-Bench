
import torch

class ODE2:
    """
    ODE2: Duffing Oscillator Equation
    Defined by: x_ddot + delta * x_dot + alpha * x + beta * x^3 = gamma * cos(omega * t)
    Initial conditions: x(0) = epsilon, x_dot(0) = zeta
    Parameters: delta, alpha, beta, gamma, omega, epsilon, zeta
    Domain: t in [0, 1]
    """
    def __init__(self):
        pass

    def rhs(self, t: torch.Tensor, u: torch.Tensor, delta: torch.Tensor, alpha: torch.Tensor,
            beta: torch.Tensor, gamma: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """
        Right-hand side of the ODE for numerical solvers.
        The state vector u is [x, x_dot].
        du[0]/dt = x_dot
        du[1]/dt = -delta * x_dot - alpha * x - beta * x^3 + gamma * cos(omega * t)
        """
        x, x_dot = u[..., 0], u[..., 1] # assuming u is (..., 2)
        
        dxdt = x_dot
        dx_dotdt = -delta * x_dot - alpha * x - beta * x**3 + gamma * torch.cos(omega * t)
        
        # Stack them back to (..., 2)
        return torch.stack([dxdt, dx_dotdt], dim=-1)

    def solution(self, t: torch.Tensor, initial_conditions: torch.Tensor, params: dict):
        """
        No analytical solution provided in the paper. This would typically be solved numerically.
        Placeholder for consistency.
        """
        raise NotImplementedError("Analytical solution for Duffing Oscillator not provided in the paper.")

    def get_sensitivities(self, t: torch.Tensor, initial_conditions: torch.Tensor, params: dict):
        """
        No analytical sensitivities provided in the paper. This would typically be computed via AD.
        Placeholder for consistency.
        """
        raise NotImplementedError("Analytical sensitivities for Duffing Oscillator not provided in the paper.")
