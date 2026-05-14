"""
Baseline fine-tuning methods for comparison with Adjoint Matching.

Implements:
1. DRaFT-K (Clark et al., 2024): Direct reward fine-tuning with gradient stopping
2. ReFL (Xu et al., 2023): Reward Feedback Learning
3. DPO (Wallace et al., 2023): Direct Preference Optimization for Flow Matching
4. Continuous Adjoint method
5. Discrete Adjoint method
"""

import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple, Dict
from .noise_schedules import get_sigma_memoryless_fm


# ============================================================================
# DRaFT-K: Direct Reward Fine-Tuning
# ============================================================================

def draft_loss(
    velocity_fn: Callable,
    x0: torch.Tensor,
    reward_fn: Callable,
    num_steps: int = 40,
    K: int = 1,
    sigma_schedule: str = "memoryless",
    condition: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    DRaFT-K loss: backpropagate reward through last K steps of trajectory.
    
    DRaFT-1 (K=1): Only backpropagate through the last step.
    DRaFT-40 (K=40): Backpropagate through all steps.
    
    From Clark et al. (2024): "Directly fine-tuning diffusion models on differentiable rewards"
    
    Args:
        velocity_fn: Velocity model to fine-tune
        x0: Initial noise [batch, ...]
        reward_fn: Differentiable reward function
        num_steps: Total number of steps
        K: Number of steps to backpropagate through
        sigma_schedule: Noise schedule for sampling
        condition: Optional conditioning
    
    Returns:
        Negative reward (to minimize)
    """
    h = 1.0 / num_steps
    batch_size = x0.shape[0]
    device = x0.device
    
    # Run first (num_steps - K) steps without gradients
    x = x0.detach()
    
    with torch.no_grad():
        for k in range(num_steps - K):
            t = k * h
            t_tensor = torch.full((batch_size,), t, device=device)
            
            if sigma_schedule == "memoryless":
                sigma_t = get_sigma_memoryless_fm(
                    torch.tensor(t, device=device), h=h
                ).item()
                if condition is not None:
                    v = velocity_fn(x, t_tensor, condition)
                else:
                    v = velocity_fn(x, t_tensor)
                kappa_t = 1.0 / (t + h)
                drift = 2.0 * v - kappa_t * x
                noise = torch.randn_like(x)
                x = x + h * drift + (h ** 0.5) * sigma_t * noise
            else:
                if condition is not None:
                    v = velocity_fn(x, t_tensor, condition)
                else:
                    v = velocity_fn(x, t_tensor)
                x = x + h * v
    
    # Run last K steps with gradients
    for k in range(num_steps - K, num_steps):
        t = k * h
        t_tensor = torch.full((batch_size,), t, device=device)
        
        if sigma_schedule == "memoryless":
            sigma_t = get_sigma_memoryless_fm(
                torch.tensor(t, device=device), h=h
            ).item()
            if condition is not None:
                v = velocity_fn(x, t_tensor, condition)
            else:
                v = velocity_fn(x, t_tensor)
            kappa_t = 1.0 / (t + h)
            drift = 2.0 * v - kappa_t * x
            noise = torch.randn_like(x)
            x = x + h * drift + (h ** 0.5) * sigma_t * noise
        else:
            if condition is not None:
                v = velocity_fn(x, t_tensor, condition)
            else:
                v = velocity_fn(x, t_tensor)
            x = x + h * v
    
    # Compute reward
    if condition is not None:
        reward = reward_fn(x, condition)
    else:
        reward = reward_fn(x)
    
    return -reward.mean()


# ============================================================================
# ReFL: Reward Feedback Learning
# ============================================================================

def refl_loss(
    velocity_fn: Callable,
    x0: torch.Tensor,
    reward_fn: Callable,
    num_steps: int = 40,
    sigma_schedule: str = "memoryless",
    condition: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    ReFL loss: maximize reward on denoised samples at random timesteps.
    
    From Xu et al. (2023): "ImageReward: Learning and evaluating human preferences"
    
    The denoiser map for Flow Matching (Appendix F.1):
    X_hat_1(x, t) = (v(x,t) - (beta_dot_t/beta_t)*x) / (alpha_dot_t - (beta_dot_t/beta_t)*alpha_t)
    
    For linear interpolant (alpha_t=t, beta_t=1-t):
    X_hat_1(x, t) = (v(x,t) + x/(1-t)) / (1 + 1/(1-t)) = (v(x,t)*(1-t) + x) / (2-t)
    
    Args:
        velocity_fn: Velocity model to fine-tune
        x0: Initial noise [batch, ...]
        reward_fn: Differentiable reward function
        num_steps: Total number of steps
        sigma_schedule: Noise schedule for sampling
        condition: Optional conditioning
    
    Returns:
        Negative reward (to minimize)
    """
    h = 1.0 / num_steps
    batch_size = x0.shape[0]
    device = x0.device
    
    # Sample a random timestep
    k = torch.randint(0, num_steps, (1,)).item()
    t = k * h
    
    # Run trajectory up to timestep k without gradients
    x = x0.detach()
    
    with torch.no_grad():
        for step in range(k):
            t_step = step * h
            t_tensor = torch.full((batch_size,), t_step, device=device)
            
            if sigma_schedule == "memoryless":
                sigma_t = get_sigma_memoryless_fm(
                    torch.tensor(t_step, device=device), h=h
                ).item()
                if condition is not None:
                    v = velocity_fn(x, t_tensor, condition)
                else:
                    v = velocity_fn(x, t_tensor)
                kappa_t = 1.0 / (t_step + h)
                drift = 2.0 * v - kappa_t * x
                noise = torch.randn_like(x)
                x = x + h * drift + (h ** 0.5) * sigma_t * noise
            else:
                if condition is not None:
                    v = velocity_fn(x, t_tensor, condition)
                else:
                    v = velocity_fn(x, t_tensor)
                x = x + h * v
    
    # Compute denoised sample at timestep k (with gradients)
    t_tensor = torch.full((batch_size,), t, device=device)
    if condition is not None:
        v = velocity_fn(x, t_tensor, condition)
    else:
        v = velocity_fn(x, t_tensor)
    
    # Denoiser map for linear interpolant:
    # X_hat_1 = (v(x,t) - (beta_dot/beta)*x) / (alpha_dot - (beta_dot/beta)*alpha)
    # For alpha_t=t, beta_t=1-t: beta_dot=-1, alpha_dot=1
    # X_hat_1 = (v + x/(1-t)) / (1 + t/(1-t)) = (v*(1-t) + x) / (2-t)
    beta_t = 1.0 - t
    alpha_t = t
    beta_dot = -1.0
    alpha_dot = 1.0
    
    if beta_t > 1e-6:
        beta_dot_over_beta = beta_dot / beta_t
        x_hat_1 = (v - beta_dot_over_beta * x) / (alpha_dot - beta_dot_over_beta * alpha_t)
    else:
        x_hat_1 = v  # At t=1, v ≈ x_1
    
    # Compute reward on denoised sample
    if condition is not None:
        reward = reward_fn(x_hat_1, condition)
    else:
        reward = reward_fn(x_hat_1)
    
    return -reward.mean()


# ============================================================================
# DPO: Direct Preference Optimization for Flow Matching
# ============================================================================

def dpo_loss_fm(
    velocity_fn: Callable,
    ref_velocity_fn: Callable,
    x1_a: torch.Tensor,
    x1_b: torch.Tensor,
    reward_fn: Callable,
    beta_dpo: float = 5000.0,
    num_steps: int = 40,
    condition: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    DPO loss for Flow Matching (Appendix F.2).
    
    Adapted from Wallace et al. (2023a) to Flow Matching.
    
    Uses on-policy samples from the current model.
    
    Args:
        velocity_fn: Current velocity model
        ref_velocity_fn: Reference (base) velocity model
        x1_a, x1_b: Pair of generated samples
        reward_fn: Reward function
        beta_dpo: DPO temperature parameter
        num_steps: Number of timesteps
        condition: Optional conditioning
    
    Returns:
        DPO loss
    """
    h = 1.0 / num_steps
    batch_size = x1_a.shape[0]
    device = x1_a.device
    
    # Sample random timestep
    k = torch.randint(0, num_steps, (1,)).item()
    t = k * h
    t_tensor = torch.full((batch_size,), t, device=device)
    
    # Sample noisy versions of x1_a and x1_b using forward process
    # For linear interpolant: x_t = t * x1 + (1-t) * eps
    eps_a = torch.randn_like(x1_a)
    eps_b = torch.randn_like(x1_b)
    
    alpha_t = t
    beta_t = 1.0 - t
    
    x_t_a = alpha_t * x1_a + beta_t * eps_a
    x_t_b = alpha_t * x1_b + beta_t * eps_b
    
    # Compute denoiser maps
    def denoiser_map(x, v, t):
        """X_hat_1(x, t) = (v*(1-t) + x) / (2-t) for linear interpolant"""
        beta_t = 1.0 - t
        alpha_t = t
        beta_dot = -1.0
        alpha_dot = 1.0
        if beta_t > 1e-6:
            beta_dot_over_beta = beta_dot / beta_t
            return (v - beta_dot_over_beta * x) / (alpha_dot - beta_dot_over_beta * alpha_t)
        return v
    
    # Current model predictions
    if condition is not None:
        v_a = velocity_fn(x_t_a, t_tensor, condition)
        v_b = velocity_fn(x_t_b, t_tensor, condition)
        v_ref_a = ref_velocity_fn(x_t_a, t_tensor, condition)
        v_ref_b = ref_velocity_fn(x_t_b, t_tensor, condition)
    else:
        v_a = velocity_fn(x_t_a, t_tensor)
        v_b = velocity_fn(x_t_b, t_tensor)
        with torch.no_grad():
            v_ref_a = ref_velocity_fn(x_t_a, t_tensor)
            v_ref_b = ref_velocity_fn(x_t_b, t_tensor)
    
    # Compute denoised samples
    x_hat_a = denoiser_map(x_t_a, v_a, t)
    x_hat_b = denoiser_map(x_t_b, v_b, t)
    x_hat_ref_a = denoiser_map(x_t_a, v_ref_a, t)
    x_hat_ref_b = denoiser_map(x_t_b, v_ref_b, t)
    
    # Compute rewards
    with torch.no_grad():
        if condition is not None:
            r_a = reward_fn(x1_a, condition)
            r_b = reward_fn(x1_b, condition)
        else:
            r_a = reward_fn(x1_a)
            r_b = reward_fn(x1_b)
    
    # DPO loss (Eq. 235 in paper)
    # For each pair (a, b), compute the DPO loss
    alpha_t_sq = alpha_t ** 2
    beta_t_sq = beta_t ** 2
    
    # Compute ||x_hat - x1||^2 terms
    def sq_norm(x):
        return (x ** 2).sum(dim=list(range(1, x.dim())))
    
    # Scale factor from Eq. 232
    scale = beta_t_sq / alpha_t_sq if alpha_t > 1e-6 else 1.0
    
    diff_a = sq_norm(x_hat_a - x1_a) / scale - sq_norm(x_hat_ref_a - x1_a) / scale
    diff_b = sq_norm(x_hat_b - x1_b) / scale - sq_norm(x_hat_ref_b - x1_b) / scale
    
    # DPO loss with soft labels based on reward difference
    reward_diff = r_a - r_b
    
    loss = 0.0
    for s in [1, -1]:
        weight = torch.sigmoid(s * reward_diff)
        logit = -s * beta_dpo / 2.0 * (diff_a - diff_b)
        loss = loss - (weight * torch.log(torch.sigmoid(logit) + 1e-8)).mean()
    
    return loss


# ============================================================================
# Continuous Adjoint Method
# ============================================================================

def continuous_adjoint_loss(
    velocity_fn: Callable,
    x0: torch.Tensor,
    reward_fn: Callable,
    num_steps: int = 40,
    sigma_schedule: str = "memoryless",
    lct: Optional[float] = None,
    condition: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Continuous adjoint method for SOC.
    
    Directly differentiates through the SDE simulation using the adjoint method.
    This is the "differentiate-then-discretize" approach.
    
    The gradient is computed via the adjoint ODE (Eq. 30-31):
    d/dt a(t; X, u) = -[a(t; X, u)^T * nabla_{X_t}(b + sigma*u) + nabla_{X_t}(f + 1/2*||u||^2)]
    a(1; X, u) = nabla g(X_1)
    
    For reward fine-tuning (f=0, g=-r):
    a(1; X, u) = -nabla_{X_1} r(X_1)
    
    Args:
        velocity_fn: Velocity model to fine-tune
        x0: Initial noise [batch, ...]
        reward_fn: Differentiable reward function
        num_steps: Number of steps
        sigma_schedule: Noise schedule
        lct: Loss clipping threshold
        condition: Optional conditioning
    
    Returns:
        SOC objective value
    """
    h = 1.0 / num_steps
    batch_size = x0.shape[0]
    device = x0.device
    
    # Forward pass: simulate trajectory
    states = [x0.detach()]
    x = x0.detach()
    
    with torch.no_grad():
        for k in range(num_steps):
            t = k * h
            t_tensor = torch.full((batch_size,), t, device=device)
            
            if sigma_schedule == "memoryless":
                sigma_t = get_sigma_memoryless_fm(
                    torch.tensor(t, device=device), h=h
                ).item()
                if condition is not None:
                    v = velocity_fn(x, t_tensor, condition)
                else:
                    v = velocity_fn(x, t_tensor)
                kappa_t = 1.0 / (t + h)
                drift = 2.0 * v - kappa_t * x
                noise = torch.randn_like(x)
                x = x + h * drift + (h ** 0.5) * sigma_t * noise
            else:
                if condition is not None:
                    v = velocity_fn(x, t_tensor, condition)
                else:
                    v = velocity_fn(x, t_tensor)
                x = x + h * v
            
            states.append(x.detach())
    
    # Terminal condition
    x_last = states[-2].detach()
    t_last = torch.full((batch_size,), 1.0 - h, device=device)
    
    with torch.no_grad():
        if condition is not None:
            v_last = velocity_fn(x_last, t_last, condition)
        else:
            v_last = velocity_fn(x_last, t_last)
    
    x_hat_1 = x_last + h * v_last
    
    x_hat_1_req = x_hat_1.detach().requires_grad_(True)
    if condition is not None:
        reward = reward_fn(x_hat_1_req, condition)
    else:
        reward = reward_fn(x_hat_1_req)
    
    reward_grad = torch.autograd.grad(reward.sum(), x_hat_1_req)[0]
    a = -reward_grad.detach()
    
    # Backward pass: compute adjoint with full Jacobian (including control terms)
    adjoint_states = [None] * (num_steps + 1)
    adjoint_states[num_steps] = a
    
    for k in range(num_steps - 1, -1, -1):
        t = k * h
        t_tensor = torch.full((batch_size,), t, device=device)
        
        x_k = states[k].detach()
        x_k_req = x_k.requires_grad_(True)
        
        with torch.enable_grad():
            if condition is not None:
                v = velocity_fn(x_k_req, t_tensor, condition)
            else:
                v = velocity_fn(x_k_req, t_tensor)
            
            if sigma_schedule == "memoryless":
                sigma_t = get_sigma_memoryless_fm(
                    torch.tensor(t, device=device), h=h
                ).item()
                kappa_t = 1.0 / (t + h)
                # Full drift including control
                full_drift = 2.0 * v - kappa_t * x_k_req
                
                # Control u = (2/sigma) * (v_ft - v_base)
                # For continuous adjoint, we need nabla_x(b + sigma*u)
                # which equals nabla_x(2*v_ft - kappa*x)
                vjp = torch.autograd.grad(
                    full_drift,
                    x_k_req,
                    grad_outputs=a,
                    create_graph=False,
                )[0]
                
                # Also add gradient of control cost: nabla_x(1/2 * ||u||^2)
                # u = (2/sigma) * (v_ft - v_base), but we approximate v_base as constant
                # so nabla_x(||u||^2) ≈ (4/sigma^2) * nabla_x(v_ft) * (v_ft - v_base)
                # This is the key difference from lean adjoint
            else:
                vjp = torch.autograd.grad(
                    v,
                    x_k_req,
                    grad_outputs=a,
                    create_graph=False,
                )[0]
        
        a = (a + h * vjp).detach()
        adjoint_states[k] = a
    
    # Compute loss using adjoint states
    total_loss = torch.tensor(0.0, device=device)
    
    for k in range(num_steps):
        t = k * h
        t_tensor = torch.full((batch_size,), t, device=device)
        
        x_k = states[k].detach()
        a_k = adjoint_states[k].detach()
        
        if sigma_schedule == "memoryless":
            sigma_t = get_sigma_memoryless_fm(
                torch.tensor(t, device=device), h=h
            )
            
            if condition is not None:
                v_ft = velocity_fn(x_k, t_tensor, condition)
            else:
                v_ft = velocity_fn(x_k, t_tensor)
            
            control = (2.0 / sigma_t) * v_ft
            target = sigma_t * a_k
            
            residual = control + target
            loss_k = (residual ** 2).sum(dim=list(range(1, residual.dim())))
            
            if lct is not None:
                loss_k = torch.clamp(loss_k, max=lct)
            
            total_loss = total_loss + loss_k.mean()
    
    return total_loss / num_steps


# ============================================================================
# Discrete Adjoint Method
# ============================================================================

def discrete_adjoint_loss(
    velocity_fn: Callable,
    x0: torch.Tensor,
    reward_fn: Callable,
    num_steps: int = 40,
    sigma_schedule: str = "memoryless",
    lct: Optional[float] = None,
    condition: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Discrete adjoint method: differentiate through the discretized SDE.
    
    This is the "discretize-then-differentiate" approach.
    Stores the full computational graph and backpropagates through it.
    
    Args:
        velocity_fn: Velocity model to fine-tune
        x0: Initial noise [batch, ...]
        reward_fn: Differentiable reward function
        num_steps: Number of steps
        sigma_schedule: Noise schedule
        lct: Loss clipping threshold
        condition: Optional conditioning
    
    Returns:
        Negative reward (to minimize)
    """
    h = 1.0 / num_steps
    batch_size = x0.shape[0]
    device = x0.device
    
    # Forward pass with gradient tracking
    x = x0.clone()
    
    for k in range(num_steps):
        t = k * h
        t_tensor = torch.full((batch_size,), t, device=device)
        
        if sigma_schedule == "memoryless":
            sigma_t = get_sigma_memoryless_fm(
                torch.tensor(t, device=device), h=h
            ).item()
            if condition is not None:
                v = velocity_fn(x, t_tensor, condition)
            else:
                v = velocity_fn(x, t_tensor)
            kappa_t = 1.0 / (t + h)
            drift = 2.0 * v - kappa_t * x
            noise = torch.randn_like(x)
            x = x + h * drift + (h ** 0.5) * sigma_t * noise
        else:
            if condition is not None:
                v = velocity_fn(x, t_tensor, condition)
            else:
                v = velocity_fn(x, t_tensor)
            x = x + h * v
    
    # Compute reward
    if condition is not None:
        reward = reward_fn(x, condition)
    else:
        reward = reward_fn(x)
    
    return -reward.mean()
