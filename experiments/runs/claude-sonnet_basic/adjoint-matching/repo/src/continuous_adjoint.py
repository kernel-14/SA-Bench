"""
Continuous Adjoint method for SOC.

This implements the "differentiate-then-discretize" approach for solving
the SOC problem. The key difference from Adjoint Matching is that the
continuous adjoint includes the gradient of the control cost term
nabla_x(1/2 * ||u(x,t)||^2) in the adjoint ODE.

From Eq. 30-31 in the paper:
d/dt a(t; X, u) = -[a(t; X, u)^T * nabla_{X_t}(b(X_t,t) + sigma(t)*u(X_t,t)) 
                    + nabla_{X_t}(f(X_t,t) + 1/2*||u(X_t,t)||^2)]
a(1; X, u) = nabla g(X_1)

For reward fine-tuning (f=0, g=-r):
d/dt a(t; X, u) = -[a(t; X, u)^T * nabla_{X_t}(2*v_ft(X_t,t) - kappa_t*X_t) 
                    + nabla_{X_t}(1/2*||u(X_t,t)||^2)]
a(1; X, u) = -nabla r(X_1)

The gradient of the loss w.r.t. theta is (Eq. 32):
dL/dtheta = 1/2 * integral_0^1 partial/partial_theta ||u(X_t,t)||^2 dt
           + integral_0^1 (partial u/partial theta)^T * sigma(t)^T * a(t; X, u) dt
"""

import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple
from .noise_schedules import get_sigma_memoryless_fm


def compute_full_adjoint(
    states: List[torch.Tensor],
    velocity_fn: Callable,
    base_velocity_fn: Callable,
    reward_fn: Callable,
    num_steps: int,
    condition: Optional[torch.Tensor] = None,
    use_noiseless_final: bool = True,
) -> List[torch.Tensor]:
    """
    Compute the full adjoint state for the continuous adjoint method.
    
    This includes the control cost gradient term, unlike the lean adjoint.
    
    Full adjoint ODE (Eq. 30-31):
    d/dt a = -[a^T * nabla_x(b + sigma*u) + nabla_x(1/2*||u||^2)]
    a(1) = -nabla r(X_1)
    
    For memoryless FM:
    b + sigma*u = 2*v_ft - kappa_t*x
    u = (2/sigma) * (v_ft - v_base)
    
    So:
    nabla_x(b + sigma*u) = nabla_x(2*v_ft - kappa_t*x)
    nabla_x(1/2*||u||^2) = (2/sigma)^2 * nabla_x(v_ft - v_base) * (v_ft - v_base)
                         ≈ (2/sigma)^2 * nabla_x(v_ft) * (v_ft - v_base)  [treating v_base as constant]
    
    Args:
        states: Trajectory states [X_0, ..., X_K]
        velocity_fn: Fine-tuned velocity function
        base_velocity_fn: Base velocity function
        reward_fn: Reward function
        num_steps: Number of timesteps
        condition: Optional conditioning
        use_noiseless_final: Use noiseless final step for terminal condition
    
    Returns:
        List of adjoint states
    """
    h = 1.0 / num_steps
    device = states[0].device
    batch_size = states[0].shape[0]
    
    # Compute terminal condition
    if use_noiseless_final:
        x_last = states[-2].detach()
        t_last = torch.full((batch_size,), 1.0 - h, device=device)
        with torch.no_grad():
            if condition is not None:
                v_last = base_velocity_fn(x_last, t_last, condition)
            else:
                v_last = base_velocity_fn(x_last, t_last)
        x_hat_1 = x_last + h * v_last
    else:
        x_hat_1 = states[-1].detach()
    
    x_hat_1_req = x_hat_1.detach().requires_grad_(True)
    if condition is not None:
        reward = reward_fn(x_hat_1_req, condition)
    else:
        reward = reward_fn(x_hat_1_req)
    
    reward_grad = torch.autograd.grad(reward.sum(), x_hat_1_req)[0]
    a = -reward_grad.detach()
    
    adjoint_states = [None] * (num_steps + 1)
    adjoint_states[num_steps] = a
    
    # Solve full adjoint ODE backwards
    for k in range(num_steps - 1, -1, -1):
        t = k * h
        t_tensor = torch.full((batch_size,), t, device=device)
        
        x_k = states[k].detach()
        x_k_req = x_k.requires_grad_(True)
        
        sigma_t = get_sigma_memoryless_fm(torch.tensor(t, device=device), h=h)
        kappa_t = 1.0 / (t + h)
        
        with torch.enable_grad():
            if condition is not None:
                v_ft = velocity_fn(x_k_req, t_tensor, condition)
                v_base = base_velocity_fn(x_k_req, t_tensor, condition)
            else:
                v_ft = velocity_fn(x_k_req, t_tensor)
                v_base = base_velocity_fn(x_k_req, t_tensor)
            
            # Full drift: b + sigma*u = 2*v_ft - kappa_t*x
            full_drift = 2.0 * v_ft - kappa_t * x_k_req
            
            # Control: u = (2/sigma) * (v_ft - v_base)
            u = (2.0 / sigma_t) * (v_ft - v_base)
            
            # Term 1: a^T * nabla_x(b + sigma*u)
            vjp_drift = torch.autograd.grad(
                full_drift,
                x_k_req,
                grad_outputs=a,
                create_graph=False,
                retain_graph=True,
            )[0]
            
            # Term 2: nabla_x(1/2 * ||u||^2)
            # = (2/sigma)^2 * nabla_x(v_ft - v_base) * (v_ft - v_base)
            # We compute this as nabla_x(||u||^2/2) directly
            u_sq_half = 0.5 * (u ** 2).sum(dim=list(range(1, u.dim())))
            grad_u_sq = torch.autograd.grad(
                u_sq_half.sum(),
                x_k_req,
                create_graph=False,
                retain_graph=False,
            )[0]
        
        # Full adjoint update:
        # a_{t-h} = a_t + h * (-[a_t^T * nabla_x(b+sigma*u) + nabla_x(1/2*||u||^2)])
        # = a_t + h * (-vjp_drift - grad_u_sq)
        a = (a + h * (-vjp_drift - grad_u_sq)).detach()
        adjoint_states[k] = a
    
    return adjoint_states


def continuous_adjoint_loss_fm(
    finetune_velocity_fn: Callable,
    base_velocity_fn: Callable,
    states: List[torch.Tensor],
    adjoint_states: List[torch.Tensor],
    num_steps: int,
    lct: Optional[float] = None,
    gradient_timesteps: Optional[List[int]] = None,
    condition: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the continuous adjoint loss for Flow Matching.
    
    The gradient of the loss is (Eq. 32):
    dL/dtheta = 1/2 * sum_t partial/partial_theta ||u(X_t,t)||^2
               + sum_t (partial u/partial theta)^T * sigma(t)^T * a(t; X, u)
    
    This is equivalent to the gradient of:
    L = sum_t [1/2 * ||u(X_t,t)||^2 + u(X_t,t)^T * sigma(t)^T * a(t; X, u)]
    
    For memoryless FM:
    u = (2/sigma) * (v_ft - v_base)
    sigma^T * a = sigma * a (scalar sigma)
    
    So:
    L = sum_t [1/2 * (2/sigma)^2 * ||v_ft - v_base||^2 
              + (2/sigma) * (v_ft - v_base)^T * sigma * a]
      = sum_t [2/sigma^2 * ||v_ft - v_base||^2 + 2 * (v_ft - v_base)^T * a]
    
    Args:
        finetune_velocity_fn: Fine-tuned velocity function
        base_velocity_fn: Base velocity function
        states: Trajectory states
        adjoint_states: Full adjoint states
        num_steps: Number of timesteps
        lct: Loss clipping threshold
        gradient_timesteps: Subset of timesteps
        condition: Optional conditioning
    
    Returns:
        Scalar loss
    """
    h = 1.0 / num_steps
    device = states[0].device
    batch_size = states[0].shape[0]
    
    if gradient_timesteps is None:
        gradient_timesteps = list(range(num_steps))
    
    total_loss = torch.tensor(0.0, device=device)
    
    for k in gradient_timesteps:
        t = k * h
        t_tensor = torch.full((batch_size,), t, device=device)
        
        x_k = states[k].detach()
        a_k = adjoint_states[k].detach()
        
        sigma_t = get_sigma_memoryless_fm(torch.tensor(t, device=device), h=h)
        
        if condition is not None:
            v_ft = finetune_velocity_fn(x_k, t_tensor, condition)
        else:
            v_ft = finetune_velocity_fn(x_k, t_tensor)
        
        with torch.no_grad():
            if condition is not None:
                v_base = base_velocity_fn(x_k, t_tensor, condition)
            else:
                v_base = base_velocity_fn(x_k, t_tensor)
        
        # Control: u = (2/sigma) * (v_ft - v_base)
        u = (2.0 / sigma_t) * (v_ft - v_base)
        
        # Loss term: 1/2 * ||u||^2 + u^T * sigma * a
        # = 1/2 * ||u||^2 + sigma * u^T * a
        u_sq = 0.5 * (u ** 2).sum(dim=list(range(1, u.dim())))
        u_dot_a = (u * sigma_t * a_k).sum(dim=list(range(1, u.dim())))
        
        loss_k = u_sq + u_dot_a
        
        if lct is not None:
            loss_k = torch.clamp(loss_k, max=lct)
        
        total_loss = total_loss + loss_k.mean()
    
    return total_loss / len(gradient_timesteps)
