"""
Memoryless noise schedule for SOC fine-tuning.

From Proposition 1 and Theorem 1:
- The memoryless noise schedule sigma(t) = sqrt(2 * eta_t) is required for fine-tuning
- This ensures X_0 and X_1 are independent, removing the initial value function bias
- After fine-tuning, any noise schedule can be used for sampling

eta_t = beta_t * (alpha_dot/alpha * beta_t - beta_dot)

For Flow Matching with alpha_t = t, beta_t = 1-t:
    eta_t = (1-t) * (1/t * (1-t) + 1) = (1-t)*(1/t + 1 - 1) = (1-t)/t
    sigma(t) = sqrt(2 * (1-t) / t)

For DDIM with alpha_bar_t:
    eta_t = alpha_dot / (2 * alpha_bar)
    sigma(t) = sqrt(alpha_dot / alpha_bar)
"""
import torch
from typing import Callable, Optional


class MemorylessNoiseSchedule:
    """
    Computes the memoryless noise schedule sigma(t) = sqrt(2 * eta_t).
    
    Handles offset to avoid division by zero at t=0 and t close to 1.
    """
    def __init__(
        self,
        alpha_fn: Callable = lambda t: t,
        beta_fn: Callable = lambda t: 1.0 - t,
        dt: float = 0.025,  # h = 1/K
        use_offset: bool = True,
    ):
        self.alpha_fn = alpha_fn
        self.beta_fn = beta_fn
        self.dt = dt
        self.use_offset = use_offset

    def _derive_alpha(self, t: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            t_in = t.detach().requires_grad_(True)
            return torch.autograd.grad(
                self.alpha_fn(t_in).sum(), t_in, create_graph=False
            )[0]

    def _derive_beta(self, t: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            t_in = t.detach().requires_grad_(True)
            return torch.autograd.grad(
                self.beta_fn(t_in).sum(), t_in, create_graph=False
            )[0]

    def kappa_t(self, t: torch.Tensor) -> torch.Tensor:
        """kappa_t = alpha_dot / alpha"""
        a = self.alpha_fn(t)
        a_dot = self._derive_alpha(t)
        return a_dot / (a + 1e-8)

    def eta_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        eta_t = beta_t * (kappa_t * beta_t - beta_dot)
        """
        b = self.beta_fn(t)
        k = self.kappa_t(t)
        b_dot = self._derive_beta(t)
        return b * (k * b - b_dot)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Memoryless noise schedule sigma(t) = sqrt(2 * eta_t).
        
        With offset (for numerical stability):
            sigma(t) = sqrt(2 * (1 - t + h) / (t + h))
        when alpha_t = t, beta_t = 1 - t.
        """
        if self.use_offset:
            a = self.alpha_fn(t) + self.dt
            b = self.beta_fn(t) + self.dt
            a_dot = self._derive_alpha(t)
            b_dot = self._derive_beta(t)
            k = a_dot / (a + 1e-8)
            eta = b * (k * b - b_dot).clamp(min=1e-8)
        else:
            eta = self.eta_t(t).clamp(min=1e-8)
        
        return torch.sqrt(2.0 * eta)

    def compute_control_scaling(self, t: torch.Tensor) -> torch.Tensor:
        """
        For Flow Matching:
        u(x,t) = sqrt(2 / eta_t) * (v_finetune(x,t) - v_base(x,t))
        
        Returns: sqrt(2 / eta_t)
        """
        eta = self.eta_t(t)
        if self.use_offset:
            a = self.alpha_fn(t) + self.dt
            b = self.beta_fn(t) + self.dt
            a_dot = self._derive_alpha(t)
            b_dot = self._derive_beta(t)
            k = a_dot / (a + 1e-8)
            eta = b * (k * b - b_dot).clamp(min=1e-8)
        return torch.sqrt(2.0 / eta.clamp(min=1e-8))
