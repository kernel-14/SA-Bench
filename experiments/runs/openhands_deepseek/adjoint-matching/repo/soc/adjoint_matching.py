"""
Adjoint Matching algorithm for solving Stochastic Optimal Control problems.

Based on Section 5.2 of the paper:
- Combines continuous adjoint method with least-squares regression
- Uses "lean" adjoint state (removes terms that have expectation zero at optimum)
- No importance weighting needed (unlike SOCM)

Algorithm 1 (Flow Matching) and Algorithm 2 (DDIM) from the paper.
"""
import torch
import torch.nn as nn
from typing import Optional, Callable, List, Tuple
from torch.autograd import grad


def stop_grad(x: torch.Tensor) -> torch.Tensor:
    """Equivalent to stop_gradient."""
    return x.detach()


class LeanAdjointSolver:
    """
    Solves the lean adjoint ODE backwards in time.
    
    d/dt a_tilde(t) = -a_tilde(t)^T * nabla_x b(X_t, t)
    a_tilde(1) = nabla_x g(X_1)
    
    For reward fine-tuning: g = -r, so a_tilde(1) = -nabla_x r(X_1).
    f = 0.
    """
    def __init__(self):
        pass

    def solve_backward(
        self,
        trajectory: List[torch.Tensor],
        times: List[float],
        base_drift_fn: Callable,
        reward_grad_fn: Callable,
        dt: float,
    ) -> List[torch.Tensor]:
        """
        Solve lean adjoint backwards in time.
        
        Args:
            trajectory: list of states [X_0, X_h, ..., X_1]
            times: list of times [0, h, ..., 1]
            base_drift_fn: function (x, t, context) -> b(x,t)
            reward_grad_fn: function (x) -> nabla_x r(x)
            dt: step size
        
        Returns:
            a_tilde: list of adjoint states [a_0, a_h, ..., a_1]
        """
        K = len(trajectory) - 1
        device = trajectory[0].device
        
        # Terminal condition: a_tilde(1) = -nabla_x r(X_1)
        X_1 = trajectory[-1]
        X_1_with_grad = X_1.detach().requires_grad_(True)
        reward = reward_grad_fn(X_1_with_grad)  # This should return r(X_1) scalar
        # Actually we need: a_tilde(1) = -nabla_x r(X_1)
        # The reward function returns scalar, we need its gradient
        reward_sum = reward.sum()
        a_tilde_1 = -torch.autograd.grad(reward_sum, X_1_with_grad, create_graph=False)[0]
        
        a_tilde_list = [a_tilde_1]
        a_curr = a_tilde_1
        
        # Solve backwards: t = 1-h, 1-2h, ..., 0
        for k in range(K - 1, -1, -1):
            X_t = trajectory[k].detach()
            t_val = times[k]
            t_plus_h = times[k + 1]
            
            # Compute nabla_x b(X_t, t)
            # b(x,t) = base_drift_fn(x, t, context), but we only need the jacobian-vector product
            # a_tilde_{t-h} = a_tilde_t + h * a_tilde_t^T * nabla_x b(X_t, t)
            # This is: a_tilde_t + h * vjp(a_tilde_t, X_t, lambda x: b(x, t))
            
            X_t_grad = X_t.detach().requires_grad_(True)
            b_val = base_drift_fn(X_t_grad, t_val)  # shape: (B, C, H, W)
            
            # vjp: compute a_curr^T * nabla_x b
            # We need to compute sum(a_curr * b_val) then grad w.r.t. X_t_grad
            vjp_out = torch.autograd.grad(
                outputs=(b_val * a_curr.detach()).sum(),
                inputs=X_t_grad,
                create_graph=False,
                retain_graph=False,
            )[0]
            
            # Backward Euler: a_tilde_{t-h} = a_tilde_t + h * vjp_out
            a_prev = a_curr + dt * vjp_out
            a_tilde_list.insert(0, a_prev)
            a_curr = a_prev.detach()
        
        return a_tilde_list


class FullAdjointSolver:
    """
    Solves the full adjoint ODE backwards in time (for Continuous Adjoint).
    
    d/dt a(t) = -[a(t)^T * nabla_x(b(X_t,t) + sigma(t)*u(X_t,t)) + nabla_x(1/2 ||u(X_t,t)||^2)]
    a(1) = nabla_x g(X_1)
    """
    def __init__(self):
        pass

    def solve_backward(
        self,
        trajectory: List[torch.Tensor],
        times: List[float],
        base_drift_fn: Callable,
        control_fn: Callable,
        sigma_fn: Callable,
        reward_grad_fn: Callable,
        dt: float,
    ) -> List[torch.Tensor]:
        """
        Solve full adjoint backwards in time.
        
        Returns:
            a: list of adjoint states [a_0, a_h, ..., a_1]
        """
        K = len(trajectory) - 1
        device = trajectory[0].device
        
        # Terminal condition
        X_1 = trajectory[-1]
        X_1_with_grad = X_1.detach().requires_grad_(True)
        reward = reward_grad_fn(X_1_with_grad)
        reward_sum = reward.sum()
        a_1 = -torch.autograd.grad(reward_sum, X_1_with_grad, create_graph=False)[0]
        
        a_list = [a_1]
        a_curr = a_1
        
        for k in range(K - 1, -1, -1):
            X_t = trajectory[k].detach()
            t_val = times[k]
            
            X_t_grad = X_t.detach().requires_grad_(True)
            
            b_val = base_drift_fn(X_t_grad, t_val)
            sigma_val = sigma_fn(t_val)
            u_val = control_fn(X_t_grad, t_val)
            
            # Full drift = b + sigma * u
            full_drift = b_val + sigma_val[:, None, None, None] * u_val
            
            # Control cost: 1/2 ||u||^2
            control_cost = 0.5 * (u_val ** 2).sum(dim=(1, 2, 3))
            
            # vjp for full_drift
            vjp_drift = torch.autograd.grad(
                outputs=(full_drift * a_curr.detach()).sum(),
                inputs=X_t_grad,
                create_graph=False,
                retain_graph=False,
            )[0]
            
            # vjp for control_cost
            vjp_control = torch.autograd.grad(
                outputs=control_cost.sum(),
                inputs=X_t_grad,
                create_graph=False,
                retain_graph=False,
            )[0]
            
            a_prev = a_curr + dt * (vjp_drift + vjp_control)
            a_list.insert(0, a_prev)
            a_curr = a_prev.detach()
        
        return a_list


class AdjointMatchingLoss(nn.Module):
    """
    Adjoint Matching objective (Algorithm 1, equation 37 in the paper).
    
    For Flow Matching:
    L_AdjMatch(theta) = sum_t || 2/sigma(t) * (v_theta^finetune(X_t, t) - v^base(X_t, t)) + sigma(t) * a_tilde_t ||^2
    
    For DDIM:
    L_AdjMatch(theta) = sum_k || sqrt(alpha_dot/(alpha*(1-alpha))) * (eps_theta^finetune - eps^base) - sqrt(alpha_dot/alpha) * a_tilde_k ||^2
    """
    def __init__(
        self,
        base_model: nn.Module,
        schedule,
        model_type: str = "flow_matching",  # "flow_matching" or "ddim"
        lct: Optional[float] = None,  # loss clipping threshold
        lambda_reward: float = 12500.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.schedule = schedule
        self.model_type = model_type
        self.lct = lct
        self.lambda_reward = lambda_reward

    def forward(
        self,
        finetune_model: nn.Module,
        trajectory: List[torch.Tensor],
        a_tilde_list: List[torch.Tensor],
        times: List[float],
        context: torch.Tensor,
        timestep_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Compute Adjoint Matching loss.
        
        Args:
            finetune_model: the fine-tuned model
            trajectory: list of states [X_0, ..., X_1]
            a_tilde_list: list of lean adjoint states
            times: list of times
            context: text conditioning
            timestep_indices: subset of timesteps to compute loss on (None = all)
        
        Returns:
            scalar loss
        """
        K = len(trajectory) - 1
        batch_size = trajectory[0].shape[0]
        device = trajectory[0].device
        
        if timestep_indices is None:
            timestep_indices = list(range(K))
        
        total_loss = 0.0
        count = 0
        
        for k in timestep_indices:
            X_t = trajectory[k].detach()
            t_val = times[k]
            t = torch.full((batch_size,), t_val, device=device)
            
            a_tilde = a_tilde_list[k].detach()
            
            sigma = self.schedule.sigma(t)
            
            v_finetune = finetune_model(X_t, t * 1000, context)
            v_base = self.base_model(X_t, t * 1000, context)
            
            if self.model_type == "flow_matching":
                # u(x,t) = sqrt(2/eta_t) * (v_finetune - v_base)
                # 2/sigma(t) * (v_finetune - v_base) + sigma(t) * a_tilde
                scale = 2.0 / (sigma[:, None, None, None] + 1e-8)
                diff = v_finetune - v_base
                target = -sigma[:, None, None, None] * a_tilde
                pred = scale * diff
                loss_per_step = ((pred - target) ** 2).sum(dim=(1, 2, 3))
            else:
                raise NotImplementedError("DDIM Adjoint Matching not yet implemented")
            
            if self.lct is not None:
                loss_per_step = torch.clamp(loss_per_step, max=self.lct)
            
            total_loss += loss_per_step.mean()
            count += 1
        
        return total_loss / max(count, 1)


class ContinuousAdjointLoss(nn.Module):
    """
    Continuous Adjoint method.
    
    Gradient of objective equals:
    dL/dtheta = 1/2 * int d/dtheta ||u(X_t,t)||^2 dt + int (du/dtheta)^T * sigma(t)^T * a(t) dt
    """
    def __init__(
        self,
        base_model: nn.Module,
        schedule,
        model_type: str = "flow_matching",
        lct: Optional[float] = None,
        lambda_reward: float = 12500.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.schedule = schedule
        self.model_type = model_type
        self.lct = lct
        self.lambda_reward = lambda_reward

    def compute_gradient(
        self,
        finetune_model: nn.Module,
        trajectory: List[torch.Tensor],
        a_list: List[torch.Tensor],
        times: List[float],
        context: torch.Tensor,
        timestep_indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the continuous adjoint objective and return gradients.
        
        Returns:
            loss value, but gradient is accumulated in finetune_model.parameters()
        """
        K = len(trajectory) - 1
        batch_size = trajectory[0].shape[0]
        device = trajectory[0].device
        
        if timestep_indices is None:
            timestep_indices = list(range(K))
        
        total_loss = 0.0
        
        for k in timestep_indices:
            X_t = trajectory[k]  # keep grad for vjp
            t_val = times[k]
            t = torch.full((batch_size,), t_val, device=device)
            
            a = a_list[k].detach()
            sigma = self.schedule.sigma(t)
            
            v_finetune = finetune_model(X_t, t * 1000, context)
            v_base = self.base_model(stop_grad(X_t), t * 1000, context)
            
            scale = self.schedule.compute_control_scaling(t)
            u = scale[:, None, None, None] * (v_finetune - v_base)
            
            # Control cost term: 1/2 ||u||^2
            control_cost = 0.5 * (u ** 2).sum(dim=(1, 2, 3))
            
            # Jacobian term: u^T * sigma * a
            jacobian_term = (u * sigma[:, None, None, None] * a).sum(dim=(1, 2, 3))
            
            step_loss = control_cost + jacobian_term
            
            if self.lct is not None:
                step_loss = torch.clamp(step_loss, max=self.lct)
            
            total_loss += step_loss.mean()
        
        return total_loss
