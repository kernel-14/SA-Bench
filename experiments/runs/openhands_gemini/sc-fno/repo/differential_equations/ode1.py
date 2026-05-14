
import torch

class ODE1:
    """
    ODE1: Composite Harmonic Oscillator
    Defined by: du/dt = alpha * sin(alpha * pi * t) + beta * cos(beta * pi * t)
    Initial condition: u(0) = sin(gamma * pi)
    Parameters: alpha, beta, gamma
    Domain: t in [0, 1]
    """
    def __init__(self):
        pass

    def solution(self, t: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        Analytical solution for ODE1.
        u(t) = - (1/pi) * cos(alpha * pi * t) + (1/pi) * sin(beta * pi * t) + sin(gamma * pi) + (1/pi)
        """
        term1 = -1/torch.pi * torch.cos(alpha * torch.pi * t)
        term2 = 1/torch.pi * torch.sin(beta * torch.pi * t)
        term3 = torch.sin(gamma * torch.pi)
        term4 = 1/torch.pi
        return term1 + term2 + term3 + term4

    def sensitivity_alpha(self, t: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        Analytical sensitivity of u with respect to alpha.
        d(u)/d(alpha) = t * sin(alpha * pi * t)
        """
        return t * torch.sin(alpha * torch.pi * t)

    def sensitivity_beta(self, t: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        Analytical sensitivity of u with respect to beta.
        d(u)/d(beta) = t * cos(beta * pi * t)
        """
        return t * torch.cos(beta * torch.pi * t)

    def sensitivity_gamma(self, t: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        Analytical sensitivity of u with respect to gamma.
        d(u)/d(gamma) = pi * cos(gamma * pi)
        """
        return torch.pi * torch.cos(gamma * torch.pi)

    def get_sensitivities(self, t: torch.Tensor, params: dict) -> dict:
        """
        Returns a dictionary of sensitivities with respect to each parameter.
        """
        alpha = params["alpha"]
        beta = params["beta"]
        gamma = params["gamma"]
        
        sensitivities = {
            "alpha": self.sensitivity_alpha(t, alpha, beta, gamma),
            "beta": self.sensitivity_beta(t, alpha, beta, gamma),
            "gamma": self.sensitivity_gamma(t, alpha, beta, gamma),
        }
        return sensitivities

    def rhs(self, t: torch.Tensor, u: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Right-hand side of the ODE for numerical solvers.
        """
        return alpha * torch.sin(alpha * torch.pi * t) + beta * torch.cos(beta * torch.pi * t)

