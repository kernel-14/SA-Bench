"""
Fine-tuning through Stochastic Optimal Control.

Implements:
- FineTuner: base class for Flow Matching fine-tuning with memoryless noise schedule
- FineTunerDDIM: base class for DDIM fine-tuning with memoryless noise schedule
- ContinuousAdjoint: continuous adjoint method (Pontryagin)
- DiscreteAdjoint: discrete adjoint method (discretize-then-differentiate)
"""
import torch
import torch.nn as nn
from typing import Optional, Callable, List
from .memoryless_schedule import MemorylessNoiseSchedule


class FineTuner(nn.Module):
    """
    Base fine-tuner for Flow Matching models.
    
    Uses memoryless noise schedule sigma(t) = sqrt(2 * eta_t) during fine-tuning.
    After fine-tuning, can sample with any noise schedule.
    
    Controlled process (during fine-tuning):
    X_{t+h} = X_t + h * (2*v_finetune(X_t, t) - kappa_t * X_t) + sqrt(h) * sigma(t) * eps
    
    Control: u(x,t) = sqrt(2/eta_t) * (v_finetune(x,t) - v_base(x,t))
    """
    def __init__(
        self,
        base_model: nn.Module,
        finetune_model: nn.Module,
        schedule: MemorylessNoiseSchedule,
        num_steps: int = 40,
        lambda_reward: float = 12500.0,
    ):
        super().__init__()
        self.base_model = base_model  # pre-trained, frozen
        self.finetune_model = finetune_model  # to be trained
        self.schedule = schedule
        self.num_steps = num_steps
        self.lambda_reward = lambda_reward
        self.dt = 1.0 / num_steps

    def sample_trajectory(
        self,
        context: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        detach: bool = False,
    ) -> List[torch.Tensor]:
        """
        Sample trajectory using memoryless noise schedule.
        
        X_{t+h} = X_t + h * (2*v_finetune(X_t, t) - kappa_t * X_t) + sqrt(h) * sigma(t) * eps
        
        Returns list of states [X_0, X_h, X_2h, ..., X_1].
        """
        batch_size = context.shape[0]
        device = context.device
        if x0 is None:
            x0 = torch.randn(batch_size, self.finetune_model.unet.in_channels,
                           self.finetune_model.unet.in_channels, 
                           self.finetune_model.unet.in_channels, device=device)
        
        trajectory = [x0]
        x = x0
        
        for i in range(self.num_steps):
            t_val = i * self.dt
            t = torch.full((batch_size,), t_val, device=device)
            
            sigma = self.schedule.sigma(t)
            kappa = self.schedule.kappa_t(t)
            
            with torch.set_grad_enabled(not detach):
                v_finetune = self.finetune_model(x, t * 1000, context)
            
            drift = 2.0 * v_finetune - kappa * x
            
            noise = torch.randn_like(x)
            x = x + self.dt * drift + math_sqrt(self.dt) * sigma[:, None, None, None] * noise
            
            if detach:
                x = x.detach()
            
            trajectory.append(x.clone() if not detach else x)
        
        return trajectory

    def get_control(
        self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute control: u(x,t) = sqrt(2/eta_t) * (v_finetune(x,t) - v_base(x,t))
        """
        v_f = self.finetune_model(x, t * 1000, context)
        v_b = self.base_model(x, t * 1000, context)
        scale = self.schedule.compute_control_scaling(t)
        return scale[:, None, None, None] * (v_f - v_b)

    def compute_base_drift(
        self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Base drift for lean adjoint:
        b(x,t) = kappa_t * x + (sigma(t)^2/2 + eta_t) * s(x,t)
        
        Or equivalently for Flow Matching:
        b(x,t) = 2*v_base(x,t) - kappa_t * x
        """
        v_b = self.base_model(x, t * 1000, context)
        kappa = self.schedule.kappa_t(t)
        return 2.0 * v_b - kappa * x

    def compute_reward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reward r(x) on generated sample. To be overridden."""
        raise NotImplementedError

    def compute_reward_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient of reward w.r.t. x: nabla_x r(x)."""
        raise NotImplementedError


class FineTunerDDIM(nn.Module):
    """
    Base fine-tuner for DDIM/DDPM models.
    
    DDPM sampling during fine-tuning (memoryless):
    X_{k+1} = sqrt(alpha_{k+1}/alpha_k) * (X_k - (1-alpha_k/alpha_{k+1})/sqrt(1-alpha_k) * eps_f(X_k, k))
              + sqrt((1-alpha_{k+1})/(1-alpha_k) * (1 - alpha_k/alpha_{k+1})) * noise
    
    Control: u(x,t) = -sqrt(alpha_dot/(alpha*(1-alpha))) * (eps_f(x,t) - eps_b(x,t))
    """
    def __init__(
        self,
        base_model: nn.Module,
        finetune_model: nn.Module,
        alpha_bar_fn: Callable,
        num_steps: int = 40,
        lambda_reward: float = 12500.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.finetune_model = finetune_model
        self.alpha_bar_fn = alpha_bar_fn
        self.num_steps = num_steps
        self.lambda_reward = lambda_reward

    def sample_trajectory_ddpm(
        self, context: torch.Tensor, x0: Optional[torch.Tensor] = None, detach: bool = False
    ) -> List[torch.Tensor]:
        """DDPM sampling (memoryless)."""
        batch_size = context.shape[0]
        device = context.device
        if x0 is None:
            x0 = torch.randn(batch_size, self.finetune_model.unet.in_channels,
                           self.finetune_model.unet.in_channels,
                           self.finetune_model.unet.in_channels, device=device)
        
        K = self.num_steps
        trajectory = [x0]
        x = x0
        
        for k in range(K):
            t_k = k / K
            t_kp1 = (k + 1) / K
            
            a_k = self.alpha_bar_fn(torch.tensor(t_k, device=device))
            a_kp1 = self.alpha_bar_fn(torch.tensor(t_kp1, device=device))
            
            with torch.set_grad_enabled(not detach):
                eps = self.finetune_model(x, torch.full((batch_size,), t_k * 1000, device=device), context)
            
            # DDPM update
            coeff1 = torch.sqrt(a_kp1 / a_k)
            coeff2 = (1.0 - a_k / a_kp1) / torch.sqrt(1.0 - a_k)
            std = torch.sqrt((1.0 - a_kp1) / (1.0 - a_k) * (1.0 - a_k / a_kp1))
            
            x = coeff1 * (x - coeff2 * eps) + std * torch.randn_like(x)
            
            if detach:
                x = x.detach()
            trajectory.append(x.clone() if not detach else x)
        
        return trajectory


math_sqrt = torch.sqrt
