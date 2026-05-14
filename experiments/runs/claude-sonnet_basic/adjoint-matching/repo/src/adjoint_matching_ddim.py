"""
Adjoint Matching for DDIM/DDPM fine-tuning.

Implements Algorithm 2 from Appendix E.4 of the paper:
"Adjoint Matching: Fine-tuning Flow and Diffusion Models with Memoryless SOC"

For DDPM, the memoryless noise schedule is already satisfied (sigma(t) = sqrt(2*eta_t)).
The algorithm is analogous to Algorithm 1 but uses the DDPM/DDIM parameterization.

Key equations for DDPM:
- Forward SDE (Eq. 219-220):
  X_{k+1} = sqrt(alpha_bar_{k+1}/alpha_bar_k) * (X_k - (1 - alpha_bar_k/alpha_bar_{k+1})/sqrt(1-alpha_bar_k) * eps_ft(X_k, k))
           + sqrt((1-alpha_bar_{k+1})/(1-alpha_bar_k) * (1 - alpha_bar_k/alpha_bar_{k+1})) * eps_k

- Lean adjoint (Eq. 221-222):
  a_k = a_{k+1} + a_{k+1}^T * nabla_{X_k}(sqrt(alpha_bar_{k+1}/alpha_bar_k) * (X_k - (1-alpha_bar_k/alpha_bar_{k+1})/sqrt(1-alpha_bar_k) * eps_base(X_k, k)) - X_k)
  a_K = nabla_{X_K} r(X_K)

- Loss (Eq. 223-224):
  L = sum_k || sqrt((alpha_bar_{k+1}-alpha_bar_k)/(alpha_bar_k*(1-alpha_bar_k))) * (eps_ft(X_k,k) - eps_base(X_k,k)) - sqrt((alpha_bar_{k+1}-alpha_bar_k)/alpha_bar_k) * a_k ||^2
"""

import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple, Dict


def get_ddpm_sigma(alpha_bar_k: float, alpha_bar_k1: float) -> float:
    """
    DDPM noise schedule at step k.
    
    sigma_k = sqrt((1-alpha_bar_{k+1})/(1-alpha_bar_k) * (1 - alpha_bar_k/alpha_bar_{k+1}))
    """
    ratio = (1 - alpha_bar_k1) / (1 - alpha_bar_k) * (1 - alpha_bar_k / alpha_bar_k1)
    return ratio ** 0.5


def ddpm_step(
    x: torch.Tensor,
    eps_fn: Callable,
    k: int,
    alpha_bars: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single DDPM step.
    
    X_{k+1} = sqrt(alpha_bar_{k+1}/alpha_bar_k) * (X_k - (1-alpha_bar_k/alpha_bar_{k+1})/sqrt(1-alpha_bar_k) * eps(X_k, k))
            + sigma_k * eps_k
    
    Args:
        x: Current state [batch, ...]
        eps_fn: Noise predictor function
        k: Current step
        alpha_bars: Cumulative alpha products [K+1]
        batch_size: Batch size
        device: Device
    
    Returns:
        (next_state, noise)
    """
    alpha_bar_k = alpha_bars[k].item()
    alpha_bar_k1 = alpha_bars[k + 1].item()
    
    k_tensor = torch.full((batch_size,), k, dtype=torch.float32, device=device)
    eps = eps_fn(x, k_tensor)
    
    # DDPM update
    coeff1 = (alpha_bar_k1 / alpha_bar_k) ** 0.5
    coeff2 = (1 - alpha_bar_k / alpha_bar_k1) / (1 - alpha_bar_k) ** 0.5
    
    x_denoised = coeff1 * (x - coeff2 * eps)
    
    sigma_k = get_ddpm_sigma(alpha_bar_k, alpha_bar_k1)
    noise = torch.randn_like(x)
    x_next = x_denoised + sigma_k * noise
    
    return x_next, noise


def compute_lean_adjoint_ddpm(
    states: List[torch.Tensor],
    base_eps_fn: Callable,
    reward_fn: Callable,
    alpha_bars: torch.Tensor,
    num_steps: int,
) -> List[torch.Tensor]:
    """
    Solve the lean adjoint ODE backwards for DDPM.
    
    Lean adjoint update (Eq. 221-222):
    a_k = a_{k+1} + a_{k+1}^T * nabla_{X_k}(DDPM_step(X_k, eps_base, k) - X_k)
    a_K = nabla_{X_K} r(X_K)
    
    Args:
        states: Trajectory states [X_0, ..., X_K]
        base_eps_fn: Base noise predictor
        reward_fn: Reward function
        alpha_bars: Cumulative alpha products
        num_steps: Number of steps K
    
    Returns:
        List of adjoint states
    """
    device = states[0].device
    batch_size = states[0].shape[0]
    
    # Terminal condition: a_K = nabla_{X_K} r(X_K)
    x_K = states[-1].detach().requires_grad_(True)
    reward = reward_fn(x_K)
    reward_grad = torch.autograd.grad(reward.sum(), x_K)[0]
    a = -reward_grad.detach()  # g = -r, so a_K = -nabla r(X_K)
    
    adjoint_states = [None] * (num_steps + 1)
    adjoint_states[num_steps] = a
    
    # Solve backwards
    for k in range(num_steps - 1, -1, -1):
        alpha_bar_k = alpha_bars[k].item()
        alpha_bar_k1 = alpha_bars[k + 1].item()
        
        x_k = states[k].detach()
        x_k_req = x_k.requires_grad_(True)
        
        k_tensor = torch.full((batch_size,), k, dtype=torch.float32, device=device)
        
        with torch.enable_grad():
            eps_base = base_eps_fn(x_k_req, k_tensor)
            
            # DDPM step with base model
            coeff1 = (alpha_bar_k1 / alpha_bar_k) ** 0.5
            coeff2 = (1 - alpha_bar_k / alpha_bar_k1) / (1 - alpha_bar_k) ** 0.5
            
            x_next_base = coeff1 * (x_k_req - coeff2 * eps_base)
            
            # Lean adjoint: a^T * nabla_x(x_next_base - x_k)
            # = a^T * nabla_x(x_next_base) - a^T * I
            # = vjp(x_next_base, a) - a
            vjp = torch.autograd.grad(
                x_next_base,
                x_k_req,
                grad_outputs=a,
                create_graph=False,
            )[0]
        
        # a_{k} = a_{k+1} + a_{k+1}^T * nabla_{X_k}(x_next_base - X_k)
        # = a_{k+1} + vjp - a_{k+1}
        # = vjp
        # Wait, let me re-read the paper...
        # From Eq. 41: a_{t-h} = a_t + h * a_t^T * nabla_{X_t}(b(X_t, t))
        # For DDPM, the "drift" is the DDPM update minus X_k
        # So: a_k = a_{k+1} + a_{k+1}^T * nabla_{X_k}(x_next_base - X_k)
        # = a_{k+1} + vjp - a_{k+1}  (since nabla_x(x_next_base - x_k) = nabla_x(x_next_base) - I)
        # = vjp
        # But this doesn't seem right. Let me reconsider.
        # 
        # Actually from Eq. 221 in the paper:
        # a_k = a_{k+1} + a_{k+1}^T * nabla_{X_k}(sqrt(alpha_bar_{k+1}/alpha_bar_k) * (X_k - ...) - X_k)
        # The term inside is x_next_base - X_k (without the noise term)
        # So: a_k = a_{k+1} + vjp(x_next_base - x_k, a_{k+1})
        # = a_{k+1} + vjp(x_next_base, a_{k+1}) - a_{k+1}
        # = vjp(x_next_base, a_{k+1})
        
        a = vjp.detach()
        adjoint_states[k] = a
    
    return adjoint_states


def adjoint_matching_loss_ddpm(
    finetune_eps_fn: Callable,
    base_eps_fn: Callable,
    states: List[torch.Tensor],
    adjoint_states: List[torch.Tensor],
    alpha_bars: torch.Tensor,
    num_steps: int,
    lct: Optional[float] = None,
    gradient_timesteps: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Compute the Adjoint Matching loss for DDPM.
    
    Loss (Eq. 223-224):
    L = sum_k || sqrt((alpha_bar_{k+1}-alpha_bar_k)/(alpha_bar_k*(1-alpha_bar_k))) * (eps_ft - eps_base)
              - sqrt((alpha_bar_{k+1}-alpha_bar_k)/alpha_bar_k) * a_k ||^2
    
    Args:
        finetune_eps_fn: Fine-tuned noise predictor
        base_eps_fn: Base noise predictor
        states: Trajectory states
        adjoint_states: Lean adjoint states
        alpha_bars: Cumulative alpha products
        num_steps: Number of steps
        lct: Loss clipping threshold
        gradient_timesteps: Subset of timesteps
    
    Returns:
        Scalar loss
    """
    device = states[0].device
    batch_size = states[0].shape[0]
    
    if gradient_timesteps is None:
        gradient_timesteps = list(range(num_steps))
    
    total_loss = torch.tensor(0.0, device=device)
    
    for k in gradient_timesteps:
        alpha_bar_k = alpha_bars[k].item()
        alpha_bar_k1 = alpha_bars[k + 1].item()
        
        x_k = states[k].detach()
        a_k = adjoint_states[k].detach()
        
        k_tensor = torch.full((batch_size,), k, dtype=torch.float32, device=device)
        
        # Fine-tuned prediction
        eps_ft = finetune_eps_fn(x_k, k_tensor)
        
        # Base prediction (no gradients)
        with torch.no_grad():
            eps_base = base_eps_fn(x_k, k_tensor)
        
        # Compute control and target
        delta_alpha = alpha_bar_k1 - alpha_bar_k
        
        # Control: sqrt((delta_alpha)/(alpha_bar_k*(1-alpha_bar_k))) * (eps_ft - eps_base)
        control_coeff = (delta_alpha / (alpha_bar_k * (1 - alpha_bar_k))) ** 0.5
        control = control_coeff * (eps_ft - eps_base)
        
        # Target: -sqrt(delta_alpha/alpha_bar_k) * a_k
        target_coeff = (delta_alpha / alpha_bar_k) ** 0.5
        target = -target_coeff * a_k
        
        residual = control + target
        loss_k = (residual ** 2).sum(dim=list(range(1, residual.dim())))
        
        if lct is not None:
            loss_k = torch.clamp(loss_k, max=lct)
        
        total_loss = total_loss + loss_k.mean()
    
    return total_loss / len(gradient_timesteps)
