"""
SDE simulation for Flow Matching and Diffusion models.

Implements the forward (sampling) SDEs for:
1. Flow Matching ODE (sigma=0)
2. Memoryless Flow Matching SDE (sigma = sqrt(2*eta_t))
3. DDIM/DDPM

The unified SDE form (Eq. 10-11 in paper):
  dX_t = b(X_t, t) dt + sigma(t) dB_t
  b(x, t) = kappa_t * x + (sigma(t)^2/2 + eta_t) * s(x, t)
  kappa_t = alpha_dot_t / alpha_t
  eta_t = beta_t * (alpha_dot_t/alpha_t * beta_t - beta_dot_t)
"""

import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple
from .noise_schedules import get_sigma_memoryless_fm, get_eta_fm, get_kappa_fm


def euler_maruyama_step(
    x: torch.Tensor,
    t: float,
    h: float,
    drift: torch.Tensor,
    sigma: float,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Single Euler-Maruyama step.
    
    x_{t+h} = x_t + h * drift + sqrt(h) * sigma * eps
    
    Args:
        x: Current state [batch, ...]
        t: Current time
        h: Step size
        drift: Drift vector [batch, ...]
        sigma: Diffusion coefficient (scalar)
        noise: Optional pre-sampled noise [batch, ...]
    
    Returns:
        Next state [batch, ...]
    """
    if noise is None:
        noise = torch.randn_like(x)
    return x + h * drift + (h ** 0.5) * sigma * noise


def simulate_fm_sde(
    velocity_fn: Callable,
    x0: torch.Tensor,
    num_steps: int = 40,
    sigma_schedule: str = "memoryless",
    sigma_value: float = 0.0,
    return_trajectory: bool = False,
    condition: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    """
    Simulate the Flow Matching SDE forward in time.
    
    For Memoryless Flow Matching (Algorithm 1, Eq. 40):
    X_{t+h} = X_t + h * (2 * v_finetune(X_t, t) - kappa_t * X_t) + sqrt(h) * sigma(t) * eps
    
    For ODE (sigma=0):
    X_{t+h} = X_t + h * v(X_t, t)
    
    Args:
        velocity_fn: Function (x, t, condition) -> velocity
        x0: Initial noise [batch, ...]
        num_steps: Number of discretization steps
        sigma_schedule: "memoryless", "ode", or "constant"
        sigma_value: Value for constant schedule
        return_trajectory: Whether to return full trajectory
        condition: Optional conditioning [batch, ...]
    
    Returns:
        (x1, trajectory) where trajectory is list of states if return_trajectory=True
    """
    h = 1.0 / num_steps
    x = x0.clone()
    trajectory = [x.detach().clone()] if return_trajectory else None
    
    t_tensor = torch.zeros(x.shape[0], device=x.device)
    
    for k in range(num_steps):
        t = k * h
        t_tensor.fill_(t)
        
        # Compute sigma(t)
        if sigma_schedule == "memoryless":
            sigma_t = get_sigma_memoryless_fm(
                torch.tensor(t, device=x.device), h=h
            ).item()
        elif sigma_schedule == "ode":
            sigma_t = 0.0
        elif sigma_schedule == "constant":
            sigma_t = sigma_value
        else:
            raise ValueError(f"Unknown sigma_schedule: {sigma_schedule}")
        
        # Compute velocity
        with torch.no_grad() if not velocity_fn.training else torch.enable_grad():
            if condition is not None:
                v = velocity_fn(x, t_tensor, condition)
            else:
                v = velocity_fn(x, t_tensor)
        
        # Compute drift for memoryless SDE (Eq. 40):
        # drift = 2 * v_finetune(X_t, t) - kappa_t * X_t
        # where kappa_t = 1/t for linear interpolant
        if sigma_schedule == "memoryless":
            kappa_t = 1.0 / (t + h)  # Use offset to avoid division by zero
            drift = 2.0 * v - kappa_t * x
        else:
            # For ODE: drift = v(X_t, t)
            drift = v
        
        # Euler-Maruyama step
        if sigma_t > 0:
            noise = torch.randn_like(x)
            x = x + h * drift + (h ** 0.5) * sigma_t * noise
        else:
            x = x + h * drift
        
        if return_trajectory:
            trajectory.append(x.detach().clone())
    
    return x, trajectory


def simulate_fm_sde_with_grad(
    velocity_fn: Callable,
    x0: torch.Tensor,
    num_steps: int = 40,
    sigma_schedule: str = "memoryless",
    condition: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[float]]:
    """
    Simulate the Flow Matching SDE and store trajectory for adjoint computation.
    
    Returns trajectory with gradients enabled for adjoint matching.
    
    Args:
        velocity_fn: Fine-tuned velocity function
        x0: Initial noise [batch, ...]
        num_steps: Number of steps
        sigma_schedule: Noise schedule type
        condition: Optional conditioning
    
    Returns:
        (x1, states, noises, sigmas) for use in adjoint computation
    """
    h = 1.0 / num_steps
    x = x0.clone()
    states = [x.detach().clone()]
    noises = []
    sigmas = []
    
    t_tensor = torch.zeros(x.shape[0], device=x.device)
    
    for k in range(num_steps):
        t = k * h
        t_tensor.fill_(t)
        
        # Compute sigma(t)
        if sigma_schedule == "memoryless":
            sigma_t = get_sigma_memoryless_fm(
                torch.tensor(t, device=x.device), h=h
            ).item()
        elif sigma_schedule == "ode":
            sigma_t = 0.0
        else:
            sigma_t = 0.0
        
        sigmas.append(sigma_t)
        
        # Compute velocity (with gradients for fine-tuned model)
        if condition is not None:
            v = velocity_fn(x, t_tensor, condition)
        else:
            v = velocity_fn(x, t_tensor)
        
        # Compute drift
        if sigma_schedule == "memoryless":
            kappa_t = 1.0 / (t + h)
            drift = 2.0 * v - kappa_t * x
        else:
            drift = v
        
        # Sample noise and step
        if sigma_t > 0:
            noise = torch.randn_like(x)
            x = x + h * drift + (h ** 0.5) * sigma_t * noise
        else:
            noise = torch.zeros_like(x)
            x = x + h * drift
        
        noises.append(noise.detach().clone())
        states.append(x.detach().clone())
    
    return x, states, noises, sigmas


def simulate_fm_ode(
    velocity_fn: Callable,
    x0: torch.Tensor,
    num_steps: int = 40,
    condition: Optional[torch.Tensor] = None,
    return_trajectory: bool = False,
) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    """
    Simulate the Flow Matching ODE (noiseless generation).
    
    dX_t = v(X_t, t) dt
    
    Args:
        velocity_fn: Velocity function
        x0: Initial noise [batch, ...]
        num_steps: Number of steps
        condition: Optional conditioning
        return_trajectory: Whether to return full trajectory
    
    Returns:
        (x1, trajectory)
    """
    return simulate_fm_sde(
        velocity_fn=velocity_fn,
        x0=x0,
        num_steps=num_steps,
        sigma_schedule="ode",
        return_trajectory=return_trajectory,
        condition=condition,
    )
