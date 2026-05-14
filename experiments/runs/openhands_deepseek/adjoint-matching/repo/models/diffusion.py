"""
Denoising Diffusion Models (DDIM / DDPM) in continuous time.

DDIM continuous-time SDE:
    dX_t = (alpha_dot/(2*alpha) * X_t 
            - (alpha_dot/(2*alpha) + sigma(t)^2/2) * epsilon(X_t,t)/sqrt(1-alpha_t)) dt 
            + sigma(t) dB_t

DDPM continuous-time SDE (memoryless):
    dX_t = (alpha_dot/(2*alpha) * X_t 
            - alpha_dot/alpha * epsilon(X_t,t)/sqrt(1-alpha_t)) dt 
            + sqrt(alpha_dot/alpha) dB_t
"""
import torch
import torch.nn as nn
from typing import Optional, Callable
from .unet import UNetModel


class DiffusionModel(nn.Module):
    """
    Diffusion model with noise predictor epsilon(x,t).
    """
    def __init__(
        self,
        unet: UNetModel,
        alpha_bar_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        self.unet = unet
        
        # Default alpha_bar schedule (cosine-like, approximated linearly for simplicity)
        if alpha_bar_fn is None:
            self.alpha_bar_fn = lambda t: t
        else:
            self.alpha_bar_fn = alpha_bar_fn

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha_bar_fn(t)

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Derivative of alpha_bar(t)."""
        with torch.enable_grad():
            t_in = t.detach().requires_grad_(True)
            out = self.alpha_bar_fn(t_in)
            return torch.autograd.grad(out.sum(), t_in, create_graph=False)[0]

    def kappa_t(self, t: torch.Tensor) -> torch.Tensor:
        """kappa_t = alpha_dot / (2 * alpha_bar)"""
        return self.alpha_dot(t) / (2.0 * self.alpha_bar(t))

    def eta_t(self, t: torch.Tensor) -> torch.Tensor:
        """eta_t = kappa_t"""
        return self.kappa_t(t)

    def sigma_memoryless(self, t: torch.Tensor) -> torch.Tensor:
        """Memoryless noise schedule for DDIM: sigma = sqrt(2 * eta_t) = sqrt(alpha_dot/alpha_bar)"""
        return torch.sqrt(2.0 * self.eta_t(t))

    def get_score_from_epsilon(self, epsilon: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Score function from noise predictor:
        s(x,t) = -epsilon(x,t) / sqrt(1 - alpha_bar(t))
        """
        return -epsilon / torch.sqrt(1.0 - self.alpha_bar(t))

    def get_epsilon_from_score(self, score: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Inverse: epsilon = -score * sqrt(1 - alpha_bar)"""
        return -score * torch.sqrt(1.0 - self.alpha_bar(t))

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise epsilon(x,t)."""
        return self.unet(x, t, context)

    def sample_ddim(
        self,
        context: torch.Tensor,
        num_steps: int = 40,
        sigma_fn: Optional[Callable] = None,
        x0: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        DDIM sampling.
        If sigma_fn is None, uses ODE (sigma=0).
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
            
            eps = self.forward(x, t * 1000, context)
            kt = self.kappa_t(t)
            at = self.alpha_bar(t)
            
            if sigma_fn is None:
                sigma = torch.zeros_like(t)
            else:
                sigma = sigma_fn(t)
            
            drift = kt * x - (kt + sigma**2 / 2.0) * eps / torch.sqrt(1.0 - at)
            
            noise = torch.randn_like(x)
            x = x + h * drift + torch.sqrt(h) * sigma[:, None, None, None] * noise
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        if return_trajectory:
            return x, trajectory
        return x

    def sample_ddpm(
        self,
        context: torch.Tensor,
        num_steps: int = 40,
        x0: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        DDPM sampling (memoryless).
        sigma(t) = sqrt(alpha_dot/alpha_bar)
        """
        sigma_fn = lambda t: torch.sqrt(self.alpha_dot(t) / self.alpha_bar(t))
        return self.sample_ddim(context, num_steps, sigma_fn, x0, return_trajectory)
