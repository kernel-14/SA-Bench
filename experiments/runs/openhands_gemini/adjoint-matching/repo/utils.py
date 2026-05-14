
import torch
import numpy as np
from typing import Callable, Tuple

from adjoint_matching.config import Config

class SDE:
    """
    Base class for Stochastic Differential Equations.
    Represents dX_t = b(X_t, t)dt + sigma(t)dB_t
    """
    def __init__(self, config: Config):
        self.config = config
        self.h = config.H
        
    def b(self, x: torch.Tensor, t: float, score_fn: Callable) -> torch.Tensor:
        """Drift coefficient."""
        raise NotImplementedError

    def sigma(self, t: float) -> torch.Tensor:
        """Diffusion coefficient."""
        raise NotImplementedError

    def sde_step(self, x: torch.Tensor, t: float, dt: float, score_fn: Callable) -> torch.Tensor:
        """
        Euler-Maruyama integration step.
        dX_t = b(X_t, t)dt + sigma(t)dB_t
        X_{t+dt} = X_t + b(X_t, t)dt + sigma(t)sqrt(dt)Z, where Z ~ N(0, I)
        """
        # Ensure sigma(t) is a tensor for element-wise multiplication with noise
        sigma_val = self.sigma(t)
        
        # Reshape sigma_val to be broadcastable with x (e.g., [1, D] for x of shape [N, D])
        if sigma_val.dim() == 0: # Scalar
            sigma_val = sigma_val.view(1) 
        noise = torch.randn_like(x) * torch.sqrt(torch.tensor(dt, device=x.device, dtype=x.dtype))
        
        drift = self.b(x, t, score_fn) * dt
        diffusion = sigma_val * noise
        
        return x + drift + diffusion

class MemorylessFlowMatchingSDE(SDE):
    """
    SDE for Flow Matching with Memoryless Noise Schedule (Eq. 10, 11, and Proposition 1).
    Used for fine-tuning.
    """
    def __init__(self, config: Config):
        super().__init__(config)
        self.alpha_t_fn = config.ALPHA_T_FN
        self.beta_t_fn = config.BETA_T_FN

    def _kappa_t(self, t: float) -> float:
        # kappa_t = dot(alpha_t) / alpha_t
        # Assuming alpha_t = t, dot(alpha_t) = 1
        # Add a small epsilon to avoid division by zero at t=0
        return 1.0 / (t + self.config.H) 

    def _eta_t(self, t: float) -> float:
        # eta_t = beta_t * (dot(alpha_t)/alpha_t * beta_t - dot(beta_t))
        # Assuming alpha_t = t, beta_t = 1-t
        # dot(alpha_t) = 1, dot(beta_t) = -1
        # eta_t = (1-t) * (1/t * (1-t) - (-1)) = (1-t) * ((1-t)/t + 1) = (1-t) * (1/t)
        # Add a small epsilon to avoid division by zero at t=0
        return (1 - t) / (t + self.config.H)

    def sigma(self, t: float) -> torch.Tensor:
        """
        Memoryless noise schedule: sigma(t) = sqrt(2 * eta_t).
        As per G.1, with offset for stability: sigma(t) = sqrt(2 * (1 - t + h) / (t + h))
        """
        # Ensure t is a tensor for calculations
        t_tensor = torch.tensor(t, device=self.config.DEVICE, dtype=self.config.DTYPE)
        
        # This matches the simplified expression in G.1 when alpha_t=t and beta_t=1-t
        # sigma(t) = sqrt(2 * (1 - t + h) / (t + h))
        return torch.sqrt(2 * (self.beta_t_fn(t) + self.config.H) / (self.alpha_t_fn(t) + self.config.H))

    def b(self, x: torch.Tensor, t: float, score_fn: Callable) -> torch.Tensor:
        """
        Drift coefficient for Memoryless Flow Matching SDE (Eq. 11).
        b(x, t) = kappa_t * x + (sigma(t)^2 / 2 + eta_t) * s(x,t)
        """
        kappa_t = self._kappa_t(t)
        eta_t = self._eta_t(t)
        sigma_t_sq = self.sigma(t)**2
        
        # score_fn should return the score s(x, t)
        score = score_fn(x, t) 
        
        drift = kappa_t * x + (sigma_t_sq / 2 + eta_t) * score
        return drift

class ODE(SDE):
    """
    Base class for Ordinary Differential Equations.
    dX_t = v(X_t, t)dt
    """
    def __init__(self, config: Config):
        super().__init__(config)
        
    def v(self, x: torch.Tensor, t: float, vector_field_fn: Callable) -> torch.Tensor:
        """Velocity field."""
        raise NotImplementedError

    def sde_step(self, x: torch.Tensor, t: float, dt: float, vector_field_fn: Callable) -> torch.Tensor:
        """
        Euler integration step for ODEs (no diffusion term).
        X_{t+dt} = X_t + v(X_t, t)dt
        """
        velocity = self.v(x, t, vector_field_fn) * dt
        return x + velocity

class FlowMatchingODE(ODE):
    """
    Standard Flow Matching ODE (Eq. 3).
    Used for sampling after fine-tuning (with sigma(t)=0).
    """
    def __init__(self, config: Config):
        super().__init__(config)

    def v(self, x: torch.Tensor, t: float, vector_field_fn: Callable) -> torch.Tensor:
        """
        Velocity field is directly provided by the vector_field_fn (e.g., fine-tuned U-Net).
        """
        return vector_field_fn(x, t) # This is v^finetune(x,t)

    def sigma(self, t: float) -> torch.Tensor:
        """
        For ODEs, diffusion coefficient is zero.
        """
        return torch.tensor(0.0, device=self.config.DEVICE, dtype=self.config.DTYPE)

def solve_sde_fwd(
    x_0: torch.Tensor,
    sde_model: SDE,
    score_fn: Callable, # This is the base/finetuned model (e.g., U-Net outputting epsilon or v)
    t_span: Tuple[float, float],
    dt: float,
    num_steps: int,
    return_trajectory: bool = False
) -> torch.Tensor:
    """
    Solve SDE forward in time using Euler-Maruyama.
    """
    t_start, t_end = t_span
    x = x_0.to(sde_model.config.DEVICE, sde_model.config.DTYPE)
    trajectory = [x] if return_trajectory else None

    times = torch.linspace(t_start, t_end, num_steps + 1, device=x.device, dtype=x.dtype)

    for i in range(num_steps):
        t = times[i].item()
        x = sde_model.sde_step(x, t, dt, score_fn)
        if return_trajectory:
            trajectory.append(x)
            
    return (x, trajectory) if return_trajectory else x

def solve_adjoint_ode_bwd(
    x_trajectory: list[torch.Tensor],
    sde_model: SDE, # The SDE model used for forward pass
    base_score_fn: Callable, # Base model's score function
    reward_fn: Callable, # Reward function r(x)
    dt: float,
    num_steps: int,
    lambda_reward: float
) -> list[torch.Tensor]:
    """
    Solve the lean adjoint ODE backwards in time (Eq. 38, 39).
    d(tilde_a)/dt = -[tilde_a^T * nabla_x b(X_t, t) + nabla_x f(X_t, t)]
    tilde_a(1) = nabla_x g(X_1)
    
    Here, g(X_1) = -r(X_1) and f(X_t, t) = 0.
    So, tilde_a(1) = -nabla_x r(X_1).
    And d(tilde_a)/dt = -[tilde_a^T * nabla_x b(X_t, t)]
    
    We need to approximate nabla_x b(X_t, t).
    b(x, t) = kappa_t * x + (sigma(t)^2 / 2 + eta_t) * s(x,t)
    nabla_x b(x,t) approx kappa_t * I + (sigma(t)^2 / 2 + eta_t) * nabla_x s(x,t)
    nabla_x s(x,t) is the Jacobian of the score function.
    
    The paper uses a simplified form:
    tilde_a_t-h = tilde_a_t + h * tilde_a_t^T * nabla_X_t (2 * v^base(X_t, t) - dot(alpha_t)/alpha_t * X_t) (Eq. 41)
    tilde_a_1 = -nabla_X_1 r(X_1) (Eq. 41)
    
    This specific form is for Flow Matching where b(x,t) + sigma(t)u(x,t) = 2v_finetune(x,t) - kappa_t * x
    And the lean adjoint is for the *base* drift.
    
    Let's derive `nabla_x b_base` for the adjoint ODE (Eq. 38/41).
    The paper simplifies `nabla_x b(X_t, t)` in the lean adjoint update for Flow Matching.
    From Eq. 27 (Memoryless Flow Matching case):
    b(x, t) + sigma(t)u(x, t) = 2 * v_finetune(x, t) - kappa_t * x
    This is the controlled drift.
    The lean adjoint uses the *base* drift for backward pass.
    What is the base drift for FM? It's v_base(x,t) when sigma(t)=0 and u(x,t)=0.
    However, the lean adjoint in Eq. 41 is for the *memoryless* FM.
    
    From Eq. 11, the base drift is b(x,t) = kappa_t * x + (sigma(t)^2 / 2 + eta_t) * s(x,t)
    In Algorithm 1, the lean adjoint is given as:
    tilde_a_t-h = tilde_a_t + h * tilde_a_t^T * nabla_X_t (2 * v^base(X_t, t) - dot(alpha_t)/alpha_t * X_t) (Eq. 41)
    This implies the drift in the adjoint ODE is `2 * v^base(X_t, t) - dot(alpha_t)/alpha_t * X_t`
    Let's reconcile this with the general form d(tilde_a)/dt = -[tilde_a^T * nabla_x b(X_t, t)]
    
    The term `nabla_X_t (2 * v^base(X_t, t) - dot(alpha_t)/alpha_t * X_t)` is the Jacobian of the drift function
    for the *forward* SDE used in sampling.
    In Memoryless Flow Matching, the effective drift is `b(x,t) + sigma(t)u(x,t)`.
    From the first line of Algorithm 1, the forward sampling equation is:
    X_{t+h} = X_t + h * (2 * v_theta_finetune(X_t, t) - dot(alpha_t)/alpha_t * X_t) + sqrt(h) * sigma(t) * epsilon_t
    So the effective drift here is `(2 * v_theta_finetune(X_t, t) - dot(alpha_t)/alpha_t * X_t)`.
    And for the lean adjoint, we use the `v_base` version of this drift.
    Let `drift_fn(x, t) = 2 * v_base(x, t) - (self.alpha_t_fn(t) / self.alpha_t_fn(t)) * x`
    
    """
    
    adjoint_trajectory = []
    
    # Initialize tilde_a_1 = -nabla_X_1 r(X_1) (Eq. 41).
    # X_1 is obtained by a final noiseless update from X_1-h (G.1)
    # X_1_minus_h is the last element of x_trajectory (i.e. x_trajectory[-1])
    # The paper says X_1 := X_1-h + h * v^base(X_1-h, 1-h) (G.1)
    
    # Get X_1_minus_h from the trajectory.
    # The trajectory is collected from t=0 to t=1-h, so x_trajectory[-1] is X_{1-h}
    x_1_minus_h = x_trajectory[-1].clone().detach().requires_grad_(True)
    
    # Compute X_1 with noiseless update using base_vector_field_fn
    # Note: the paper mentions v^base, but the implementation typically uses the U-Net output directly
    # So we'll assume base_score_fn is actually base_vector_field_fn here for FM.
    
    # Need to be careful with indexing if t_span was [0,1] and dt was such that t_end=1.0
    # Let's assume x_trajectory contains states for t = [0, h, ..., 1-h]
    # The last recorded time for x_trajectory[-1] is effectively `times[num_steps-1]` which is `1-h`
    
    # Let's re-construct the actual t for x_trajectory
    times = torch.linspace(sde_model.config.H, 1.0, num_steps, device=x_trajectory[0].device, dtype=x_trajectory[0].dtype)
    t_1_minus_h = times[-1] # This should be 1.0, or close to it if last step is full 'h' from 1-h
    
    # This part is a bit ambiguous in the paper.
    # "X_1 := X_{1-h} + h * v^base(X_{1-h}, 1-h)" (G.1)
    # The `base_score_fn` here should be the `v^base` if we are in Flow Matching context.
    # We need to compute v^base(x_1_minus_h, t_1_minus_h)
    
    # Create a dummy FlowMatchingODE to get the v field for base_score_fn (which acts as v^base)
    fm_ode = FlowMatchingODE(sde_model.config)
    v_base_at_1_minus_h = fm_ode.v(x_1_minus_h, t_1_minus_h, base_score_fn)
    
    x_1 = x_1_minus_h + sde_model.config.H * v_base_at_1_minus_h # x_1 is detached here.
    
    # Compute gradient of reward function wrt x_1
    reward_at_x_1 = reward_fn(x_1) * lambda_reward # Apply lambda scaling (Eq. 7)
    
    # nabla_X_1 r(X_1)
    # It's actually -nabla_X_1 r(X_1) based on Eq. 41 and g=-r
    grad_r_x_1 = torch.autograd.grad(reward_at_x_1.sum(), x_1, retain_graph=True)[0]
    
    tilde_a = -grad_r_x_1
    adjoint_trajectory.append(tilde_a.detach()) # Store detached for consistency

    # Solve backwards from t=1-h to t=0
    # Times are decreasing from 1-h down to 0, or k from K-1 down to 0
    
    # The indexing of x_trajectory is forward, so reverse it for backward pass
    x_trajectory_reversed = list(reversed(x_trajectory[:-1])) # Exclude X_1-h, handled for X_1 init
    
    # Also reverse times, but correctly manage t_current and t_prev_in_trajectory
    # If x_trajectory is [X_0, X_h, ..., X_{1-h}]
    # We iterate t from 1-h down to 0
    # Current t for tilde_a_t is from time_points_for_backward, which are [1-h, 1-2h, ..., h]
    
    # Algorithm 1: tilde_a_t-h = tilde_a_t + h * ...
    # This means we compute tilde_a for time k*h using tilde_a for time (k+1)*h
    
    # The actual times for x_trajectory are 0, h, 2h, ..., (K-1)h
    # So x_trajectory[i] corresponds to time i*h
    
    # We need to compute tilde_a for t = (K-1)h, (K-2)h, ..., 0
    
    # The loop should go from K-1 down to 0 (inclusive for t=0)
    # The previous point in forward pass was x_k. The next was x_{k+1}.
    # The `tilde_a_t` comes from previous step, `tilde_a_t_plus_h`.
    
    # Loop for k from (K-1) down to 0. (t = k*h)
    for i in range(num_steps - 1, -1, -1): # K-1 down to 0
        t_current = times[i].item() # current time for X_t
        x_current = x_trajectory[i].clone().detach().requires_grad_(True)
        
        # d_t(adjoint) = - (adjoint^T * nabla_x_drift)
        # So adjoint_t-h = adjoint_t + h * (adjoint_t^T * nabla_x_drift_t)
        # Or tilde_a_k = tilde_a_k+1 + h * tilde_a_k+1^T * nabla_x_drift(X_k, k)
        
        # In Algo 1, tilde_a_t-h (tilde_a_k) is calculated from tilde_a_t (tilde_a_k+1)
        # tilde_a_t refers to the value of adjoint at time t+h in forward pass (index k+1)
        # and tilde_a_t-h refers to value of adjoint at time t in forward pass (index k)
        
        # For the formula tilde_a_t-h = tilde_a_t + h * tilde_a_t^T * nabla_X_t(...)
        # `tilde_a_t` corresponds to the current `tilde_a` in our loop (from previous iteration, i.e., `tilde_a_k+1`)
        # `X_t` corresponds to `x_current` in our loop (i.e., `X_k`)
        
        # We need the Jacobian of the drift function `(2 * v^base(X_t, t) - dot(alpha_t)/alpha_t * X_t)`
        # w.r.t X_t.
        # This is `nabla_X_t (2 * v^base(X_t, t) - kappa_t * X_t)`
        
        kappa_t_val = sde_model._kappa_t(t_current) # This is dot(alpha_t)/alpha_t
        
        # drift_fn_for_adjoint = 2 * v^base(X_t, t) - kappa_t * X_t
        
        # To get the Jacobian, we can compute the vector-Jacobian product (VJP)
        # tilde_a_t.T @ Jacobian_of_drift
        # This is equivalent to (jacobian_of_drift.T @ tilde_a_t).T
        
        # Compute the drift function using base_score_fn (which is v^base in FM context)
        fm_ode = FlowMatchingODE(sde_model.config)
        v_base_at_x_current = fm_ode.v(x_current, t_current, base_score_fn)
        
        drift_for_adjoint_calc = 2 * v_base_at_x_current - kappa_t_val * x_current
        
        # Compute VJP: tilde_a.T @ (nabla_X_t drift_for_adjoint_calc)
        # In PyTorch, torch.autograd.grad(outputs, inputs, grad_outputs=...) computes VJP
        # grad_outputs = tilde_a (which is from the previous step, i.e., tilde_a_t)
        
        grad_drift_wrt_x_current = torch.autograd.grad(
            drift_for_adjoint_calc, x_current, grad_outputs=tilde_a, retain_graph=True
        )[0]
        
        # tilde_a_t-h = tilde_a_t + h * (grad_drift_wrt_x_current)
        tilde_a = tilde_a + sde_model.config.H * grad_drift_wrt_x_current
        adjoint_trajectory.append(tilde_a.detach())
        
    # The trajectory is built by appending. So, tilde_a for t=0 is the last element.
    # Reverse it to match forward trajectory order (t=0, h, ..., 1-h)
    adjoint_trajectory.reverse()
    
    return adjoint_trajectory


