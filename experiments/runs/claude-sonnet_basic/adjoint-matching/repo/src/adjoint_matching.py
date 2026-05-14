"""
Adjoint Matching algorithm for Flow Matching fine-tuning.

Implements Algorithm 1 from the paper:
"Adjoint Matching: Fine-tuning Flow and Diffusion Models with Memoryless SOC"

The algorithm:
1. Sample trajectories using the memoryless noise schedule
2. Solve the lean adjoint ODE backwards in time
3. Compute the Adjoint Matching loss
4. Update the fine-tuned velocity field

Key equations:
- Forward SDE (Eq. 40): X_{t+h} = X_t + h*(2*v_ft(X_t,t) - kappa_t*X_t) + sqrt(h)*sigma(t)*eps
- Lean adjoint (Eq. 41): a_{t-h} = a_t + h * a_t^T * nabla_{X_t}(2*v_base(X_t,t) - kappa_t*X_t)
  with a_1 = -nabla_{X_1} r(X_1)
- Loss (Eq. 42): L = sum_t || (2/sigma(t)) * (v_ft(X_t,t) - v_base(X_t,t)) + sigma(t)*a_t ||^2
"""

import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple, Dict
from .noise_schedules import get_sigma_memoryless_fm


def select_gradient_timesteps(
    num_steps: int,
    num_early: int = 10,
    num_late: int = 10,
    device: torch.device = None,
) -> List[int]:
    """
    Select timestep indices for gradient computation.
    
    From Appendix G.2: sample 10 timesteps uniformly from [0, 0.725]
    and always include the last 10 timesteps [0.75, ..., 0.975].
    
    Args:
        num_steps: Total number of timesteps (K=40)
        num_early: Number of early timesteps to sample
        num_late: Number of late timesteps to always include
        device: Torch device
    
    Returns:
        List of selected timestep indices
    """
    # Last num_late steps are always included
    late_start = num_steps - num_late
    late_indices = list(range(late_start, num_steps))
    
    # Sample num_early from the early steps
    early_pool = list(range(0, late_start))
    if len(early_pool) <= num_early:
        early_indices = early_pool
    else:
        perm = torch.randperm(len(early_pool))[:num_early].tolist()
        early_indices = [early_pool[i] for i in sorted(perm)]
    
    return early_indices + late_indices


def compute_lean_adjoint(
    states: List[torch.Tensor],
    base_velocity_fn: Callable,
    reward_fn: Callable,
    num_steps: int,
    condition: Optional[torch.Tensor] = None,
    use_noiseless_final: bool = True,
) -> List[torch.Tensor]:
    """
    Solve the lean adjoint ODE backwards in time.
    
    Lean adjoint ODE (Eq. 38-39):
    d/dt a_tilde(t; X) = -(a_tilde(t; X)^T * nabla_x b(X_t, t) + nabla_x f(X_t, t))
    a_tilde(1; X) = nabla_x g(X_1)
    
    For reward fine-tuning (f=0, g=-r):
    a_tilde(1; X) = -nabla_{X_1} r(X_1)
    
    Euler discretization (Eq. 41):
    a_{t-h} = a_t + h * a_t^T * nabla_{X_t}(2*v_base(X_t,t) - kappa_t*X_t)
    
    Note: The base drift b(x,t) for memoryless FM is:
    b(x,t) = 2*v_base(x,t) - kappa_t*x
    
    Args:
        states: List of trajectory states [X_0, X_h, ..., X_1]
        base_velocity_fn: Base (pre-trained) velocity function
        reward_fn: Reward function r(x) -> scalar
        num_steps: Number of timesteps
        condition: Optional conditioning
        use_noiseless_final: Use noiseless final step for terminal condition
    
    Returns:
        List of adjoint states [a_0, a_h, ..., a_1]
    """
    h = 1.0 / num_steps
    device = states[0].device
    batch_size = states[0].shape[0]
    
    # Compute terminal condition: a_1 = -nabla_{X_1} r(X_1)
    if use_noiseless_final:
        # Use noiseless final step (Appendix G.1)
        # X_hat_1 = X_{1-h} + h * v_base(X_{1-h}, 1-h)
        x_last = states[-2].detach().requires_grad_(False)
        t_last = torch.full((batch_size,), 1.0 - h, device=device)
        
        with torch.no_grad():
            if condition is not None:
                v_last = base_velocity_fn(x_last, t_last, condition)
            else:
                v_last = base_velocity_fn(x_last, t_last)
        
        x_hat_1 = x_last + h * v_last
    else:
        x_hat_1 = states[-1].detach()
    
    # Compute reward gradient at terminal state
    x_hat_1_req = x_hat_1.detach().requires_grad_(True)
    if condition is not None:
        reward = reward_fn(x_hat_1_req, condition)
    else:
        reward = reward_fn(x_hat_1_req)
    
    # a_1 = -nabla_{X_1} r(X_1) (negative because g = -r)
    reward_grad = torch.autograd.grad(
        reward.sum(), x_hat_1_req, create_graph=False
    )[0]
    a = -reward_grad.detach()
    
    # Store adjoint states (backwards)
    adjoint_states = [None] * (num_steps + 1)
    adjoint_states[num_steps] = a
    
    # Solve lean adjoint ODE backwards
    for k in range(num_steps - 1, -1, -1):
        t = k * h
        t_tensor = torch.full((batch_size,), t, device=device)
        
        x_k = states[k].detach()
        
        # Compute base drift: b(x,t) = 2*v_base(x,t) - kappa_t*x
        # We need nabla_x b(x,t) for the adjoint update
        x_k_req = x_k.requires_grad_(True)
        
        with torch.enable_grad():
            if condition is not None:
                v_base = base_velocity_fn(x_k_req, t_tensor, condition)
            else:
                v_base = base_velocity_fn(x_k_req, t_tensor)
            
            kappa_t = 1.0 / (t + h)  # offset to avoid division by zero
            base_drift = 2.0 * v_base - kappa_t * x_k_req
            
            # Compute a^T * nabla_x b(x,t) via vector-Jacobian product
            # This is equivalent to (nabla_x b)^T * a
            vjp = torch.autograd.grad(
                base_drift,
                x_k_req,
                grad_outputs=a,
                create_graph=False,
                retain_graph=False,
            )[0]
        
        # Lean adjoint update (Eq. 41):
        # a_{t-h} = a_t + h * a_t^T * nabla_{X_t}(b(X_t, t))
        # = a_t + h * vjp
        a = (a + h * vjp).detach()
        adjoint_states[k] = a
    
    return adjoint_states


def adjoint_matching_loss_fm(
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
    Compute the Adjoint Matching loss for Flow Matching.
    
    Loss (Eq. 42):
    L = sum_{t in kappa} || (2/sigma(t)) * (v_ft(X_t,t) - v_base(X_t,t)) + sigma(t)*a_t ||^2
    
    With loss clipping (Appendix G.3):
    L_hat = sum_{t in kappa} min{LCT, || ... ||^2}
    
    Args:
        finetune_velocity_fn: Fine-tuned velocity function (with gradients)
        base_velocity_fn: Base velocity function (no gradients needed)
        states: Trajectory states [X_0, ..., X_K]
        adjoint_states: Lean adjoint states [a_0, ..., a_K]
        num_steps: Number of timesteps
        lct: Loss clipping threshold (None = no clipping)
        gradient_timesteps: Subset of timesteps to compute loss at
        condition: Optional conditioning
    
    Returns:
        Scalar loss value
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
        
        # Compute sigma(t)
        sigma_t = get_sigma_memoryless_fm(
            torch.tensor(t, device=device), h=h
        )
        
        # Compute fine-tuned velocity (with gradients)
        if condition is not None:
            v_ft = finetune_velocity_fn(x_k, t_tensor, condition)
        else:
            v_ft = finetune_velocity_fn(x_k, t_tensor)
        
        # Compute base velocity (no gradients needed)
        with torch.no_grad():
            if condition is not None:
                v_base = base_velocity_fn(x_k, t_tensor, condition)
            else:
                v_base = base_velocity_fn(x_k, t_tensor)
        
        # Compute the control u(x,t) = (2/sigma(t)) * (v_ft - v_base)
        # and the target: -sigma(t) * a_t
        # Loss: || u(x,t) + sigma(t) * a_t ||^2
        # = || (2/sigma(t)) * (v_ft - v_base) + sigma(t) * a_t ||^2
        
        control = (2.0 / sigma_t) * (v_ft - v_base)
        target = sigma_t * a_k
        
        residual = control + target  # shape: [batch, ...]
        
        # Compute squared norm per sample
        loss_k = (residual ** 2).sum(dim=list(range(1, residual.dim())))  # [batch]
        
        # Apply loss clipping if specified
        if lct is not None:
            loss_k = torch.clamp(loss_k, max=lct)
        
        total_loss = total_loss + loss_k.mean()
    
    return total_loss / len(gradient_timesteps)


class AdjointMatchingTrainer:
    """
    Trainer for Adjoint Matching fine-tuning of Flow Matching models.
    
    Implements Algorithm 1 from the paper.
    """
    
    def __init__(
        self,
        finetune_model: nn.Module,
        base_model: nn.Module,
        reward_fn: Callable,
        optimizer: torch.optim.Optimizer,
        num_steps: int = 40,
        lambda_reward: float = 12500.0,
        lct_factor: float = 1.6,
        num_gradient_early: int = 10,
        num_gradient_late: int = 10,
        device: torch.device = None,
    ):
        """
        Args:
            finetune_model: Fine-tuned velocity model (initialized from base)
            base_model: Pre-trained base velocity model (frozen)
            reward_fn: Reward function r(x) -> scalar
            optimizer: Optimizer for finetune_model
            num_steps: Number of discretization steps (K=40)
            lambda_reward: Reward scaling factor
            lct_factor: Loss clipping threshold factor (LCT = lct_factor * lambda^2)
            num_gradient_early: Number of early timesteps for gradient
            num_gradient_late: Number of late timesteps for gradient
            device: Torch device
        """
        self.finetune_model = finetune_model
        self.base_model = base_model
        self.reward_fn = reward_fn
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.lambda_reward = lambda_reward
        self.lct = lct_factor * (lambda_reward ** 2)
        self.num_gradient_early = num_gradient_early
        self.num_gradient_late = num_gradient_late
        self.device = device or next(finetune_model.parameters()).device
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad_(False)
        self.base_model.eval()
    
    def train_step(
        self,
        x0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single training step of Adjoint Matching.
        
        Args:
            x0: Initial noise [batch, ...]
            condition: Optional conditioning [batch, ...]
        
        Returns:
            Dictionary of metrics
        """
        self.finetune_model.train()
        self.optimizer.zero_grad()
        
        h = 1.0 / self.num_steps
        batch_size = x0.shape[0]
        
        # Step 1: Sample trajectories with memoryless noise schedule
        # Using stopgrad on the trajectory (no gradients through simulation)
        states = [x0.detach()]
        
        with torch.no_grad():
            x = x0.clone()
            for k in range(self.num_steps):
                t = k * h
                t_tensor = torch.full((batch_size,), t, device=self.device)
                
                sigma_t = get_sigma_memoryless_fm(
                    torch.tensor(t, device=self.device), h=h
                ).item()
                
                if condition is not None:
                    v_ft = self.finetune_model(x, t_tensor, condition)
                else:
                    v_ft = self.finetune_model(x, t_tensor)
                
                kappa_t = 1.0 / (t + h)
                drift = 2.0 * v_ft - kappa_t * x
                
                noise = torch.randn_like(x)
                x = x + h * drift + (h ** 0.5) * sigma_t * noise
                states.append(x.detach())
        
        # Step 2: Compute lean adjoint backwards
        adjoint_states = compute_lean_adjoint(
            states=states,
            base_velocity_fn=self.base_model,
            reward_fn=lambda x, *args: self.lambda_reward * self.reward_fn(x, *args),
            num_steps=self.num_steps,
            condition=condition,
            use_noiseless_final=True,
        )
        
        # Step 3: Select gradient timesteps
        gradient_timesteps = select_gradient_timesteps(
            num_steps=self.num_steps,
            num_early=self.num_gradient_early,
            num_late=self.num_gradient_late,
        )
        
        # Step 4: Compute Adjoint Matching loss
        loss = adjoint_matching_loss_fm(
            finetune_velocity_fn=self.finetune_model,
            base_velocity_fn=self.base_model,
            states=states,
            adjoint_states=adjoint_states,
            num_steps=self.num_steps,
            lct=self.lct,
            gradient_timesteps=gradient_timesteps,
            condition=condition,
        )
        
        # Step 5: Backpropagate and update
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.finetune_model.parameters(), 1.0)
        self.optimizer.step()
        
        # Compute reward for monitoring
        with torch.no_grad():
            x1 = states[-1]
            if condition is not None:
                reward = self.reward_fn(x1, condition).mean().item()
            else:
                reward = self.reward_fn(x1).mean().item()
        
        return {
            "loss": loss.item(),
            "reward": reward,
        }
    
    @torch.no_grad()
    def generate(
        self,
        x0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        sigma_schedule: str = "ode",
    ) -> torch.Tensor:
        """
        Generate samples using the fine-tuned model.
        
        Args:
            x0: Initial noise [batch, ...]
            condition: Optional conditioning
            sigma_schedule: "ode" or "memoryless"
        
        Returns:
            Generated samples [batch, ...]
        """
        self.finetune_model.eval()
        h = 1.0 / self.num_steps
        batch_size = x0.shape[0]
        x = x0.clone()
        
        for k in range(self.num_steps):
            t = k * h
            t_tensor = torch.full((batch_size,), t, device=self.device)
            
            if condition is not None:
                v = self.finetune_model(x, t_tensor, condition)
            else:
                v = self.finetune_model(x, t_tensor)
            
            if sigma_schedule == "memoryless":
                sigma_t = get_sigma_memoryless_fm(
                    torch.tensor(t, device=self.device), h=h
                ).item()
                kappa_t = 1.0 / (t + h)
                drift = 2.0 * v - kappa_t * x
                noise = torch.randn_like(x)
                x = x + h * drift + (h ** 0.5) * sigma_t * noise
            else:
                # ODE
                x = x + h * v
        
        return x
