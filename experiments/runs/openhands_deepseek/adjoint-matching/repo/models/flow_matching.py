"""
Flow Matching model with ODE and SDE sampling.
Implements both:
- ODE sampling (sigma(t) = 0)
- Memoryless Flow Matching (sigma(t) = sqrt(2*eta_t))
"""
import torch
import torch.nn as nn
from typing import Optional, Callable
from .unet import UNetModel


class FlowMatchingModel(nn.Module):
    """
    Flow Matching generative model.
    
    The reference flow is:
        X_t = beta_t * X_0 + alpha_t * X_1
    with X_0 ~ N(0,I), X_1 ~ p_data.
    
    The velocity field v(x,t) predicts d/dt E[X_t | X_t = x].
    """
    def __init__(
        self,
        unet: UNetModel,
        alpha_fn: Callable[[torch.Tensor], torch.Tensor] = None,
        beta_fn: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__()
        self.unet = unet
        
        if alpha_fn is None:
            self.alpha_fn = lambda t: t
        else:
            self.alpha_fn = alpha_fn
            
        if beta_fn is None:
            self.beta_fn = lambda t: 1.0 - t
        else:
            self.beta_fn = beta_fn

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha_fn(t)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_fn(t)

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Derivative of alpha(t). For alpha(t)=t, returns 1."""
        with torch.enable_grad():
            t_in = t.detach().requires_grad_(True)
            out = self.alpha_fn(t_in)
            return torch.autograd.grad(out.sum(), t_in, create_graph=False)[0]

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Derivative of beta(t). For beta(t)=1-t, returns -1."""
        with torch.enable_grad():
            t_in = t.detach().requires_grad_(True)
            out = self.beta_fn(t_in)
            return torch.autograd.grad(out.sum(), t_in, create_graph=False)[0]

    def kappa_t(self, t: torch.Tensor) -> torch.Tensor:
        """kappa_t = alpha_dot / alpha"""
        return self.alpha_dot(t) / self.alpha(t)

    def eta_t(self, t: torch.Tensor) -> torch.Tensor:
        """eta_t = beta_t * (alpha_dot/alpha * beta_t - beta_dot)"""
        bt = self.beta(t)
        return bt * (self.kappa_t(t) * bt - self.beta_dot(t))

    def sigma_memoryless(self, t: torch.Tensor) -> torch.Tensor:
        """Memoryless noise schedule: sigma(t) = sqrt(2 * eta_t)"""
        return torch.sqrt(2.0 * self.eta_t(t))

    def get_score_from_velocity(self, v: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Convert velocity to score function.
        s(x,t) = (v(x,t) - kappa_t * x) / (beta_t * (kappa_t * beta_t - beta_dot))
        """
        bt = self.beta(t)
        et = self.eta_t(t)
        kt = self.kappa_t(t)
        return (v - kt * x) / (et + 1e-8)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Predict velocity field v(x,t)."""
        return self.unet(x, t, context)

    def sample_ode(
        self,
        context: torch.Tensor,
        num_steps: int = 40,
        x0: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Sample using ODE (sigma(t) = 0).
        dX_t = v(X_t, t) dt, X_0 ~ N(0, I)
        """
        batch_size = context.shape[0]
        device = context.device
        if x0 is None:
            x0 = torch.randn(batch_size, self.unet.in_channels, 
                           self.unet.in_channels, self.unet.in_channels, device=device)
        
        h = 1.0 / num_steps
        x = x0
        trajectory = [x0] if return_trajectory else None
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * h, device=device)
            v = self.forward(x, t * 1000, context)
            x = x + h * v
            if return_trajectory:
                trajectory.append(x.clone())
        
        if return_trajectory:
            return x, trajectory
        return x

    def sample_sde(
        self,
        context: torch.Tensor,
        num_steps: int = 40,
        sigma_fn: Optional[Callable] = None,
        x0: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Sample using SDE:
        dX_t = (v(X_t,t) + sigma(t)^2/(2*beta_t*(...)) * (v(X_t,t) - kappa_t*x)) dt + sigma(t) dB_t
        
        If sigma_fn is None, uses memoryless noise schedule.
        """
        batch_size = context.shape[0]
        device = context.device
        if x0 is None:
            x0 = torch.randn(batch_size, self.unet.in_channels,
                           self.unet.in_channels, self.unet.in_channels, device=device)
        
        h = 1.0 / num_steps
        x = x0
        trajectory = [x0] if return_trajectory else None
        
        for i in range(num_steps):
            t_val = i * h
            t = torch.full((batch_size,), t_val, device=device)
            
            if sigma_fn is None:
                sigma = self.sigma_memoryless(t)
            else:
                sigma = sigma_fn(t)
            
            v = self.forward(x, t * 1000, context)
            kt = self.kappa_t(t)
            bt = self.beta(t)
            et = self.eta_t(t)
            
            # Full drift for arbitrary sigma:
            # v + sigma^2 / (2 * beta_t * (kappa_t*beta_t - beta_dot)) * (v - kappa_t * x)
            correction = (sigma ** 2) / (2.0 * et.clamp(min=1e-8)) * (v - kt * x)
            drift = v + correction
            
            noise = torch.randn_like(x)
            x = x + h * drift + math_sqrt(h) * sigma[:, None, None, None] * noise
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        if return_trajectory:
            return x, trajectory
        return x


math_sqrt = torch.sqrt
