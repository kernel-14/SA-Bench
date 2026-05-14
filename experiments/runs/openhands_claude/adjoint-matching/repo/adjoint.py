"""
Adjoint state computation for the Adjoint Matching algorithm.

Implements:
1. Lean Adjoint ODE (Eq. 38-39) for Flow Matching
2. Lean Adjoint ODE (Eq. 221-222) for DDPM
3. Full Adjoint ODE (Eq. 30-31) for the Continuous Adjoint baseline
4. Discrete Adjoint (differentiate-then-discretize) for the Discrete Adjoint baseline

The lean adjoint removes terms that have expectation zero at the optimum,
reducing variance compared to the full adjoint (Section 5.2).

Lean adjoint ODE (Eq. 38-39):
  d/dt ã(t; X) = -(ã(t; X)^T * nabla_x b(X_t, t) + nabla_x f(X_t, t))
  ã(1; X) = nabla_x g(X_1)

For reward fine-tuning with f=0, g=-r:
  d/dt ã(t; X) = -ã(t; X)^T * nabla_x b(X_t, t)
  ã(1; X) = -nabla_x r(X_1)

For Flow Matching with memoryless schedule, b(x,t) = 2*v_base(x,t) - kappa_t*x:
  nabla_x b(x,t) = 2*nabla_x v_base(x,t) - kappa_t*I
"""

import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple

from noise_schedules import FlowMatchingSchedule, DDIMSchedule


# ---------------------------------------------------------------------------
# Lean Adjoint for Flow Matching (Algorithm 1)
# ---------------------------------------------------------------------------

def compute_lean_adjoint_fm(
    xs: List[torch.Tensor],
    reward_grad: torch.Tensor,
    v_base_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    schedule: FlowMatchingSchedule,
    text_embeddings: Optional[torch.Tensor] = None,
    f_running_cost: Optional[Callable] = None,
) -> List[torch.Tensor]:
    """
    Solve the lean adjoint ODE backwards in time (Algorithm 1, Eq. 41).

    The lean adjoint ODE for FM with memoryless schedule:
      ã_{t-h} = ã_t + h * ã_t^T * nabla_{X_t}(2*v_base(X_t,t) - kappa_t*X_t)
      ã_1 = -nabla_{X_1} r(X_1)

    Args:
        xs: List of trajectory states [X_0, X_h, ..., X_1], length K+1
        reward_grad: nabla_{X_1} r(X_1), shape [B, ...]
        v_base_fn: Base velocity field v_base(x, t, text_emb)
        schedule: FlowMatchingSchedule
        text_embeddings: Text conditioning
        f_running_cost: Optional running cost gradient (None for reward fine-tuning)

    Returns:
        List of adjoint states [ã_0, ã_h, ..., ã_1], length K+1
    """
    K = schedule.K
    h = schedule.h
    device = reward_grad.device

    # Initialize: ã_1 = -nabla r(X_1)
    # Use the noiseless final state X_hat_1 for initialization (Appendix G.1)
    a_tilde = [-reward_grad]  # ã at t=1

    # Solve backwards: from t=1-h down to t=0
    for i in range(K - 1, -1, -1):
        t_val = schedule.timesteps[i]
        t = torch.full((xs[i].shape[0],), t_val.item(), device=device, dtype=xs[i].dtype)

        x_t = xs[i].detach()
        a_t = a_tilde[-1].detach()

        # Compute nabla_{X_t} b_base(X_t, t) where b_base = 2*v_base - kappa_t*X_t
        # We need: a_t^T * nabla_{X_t} b_base(X_t, t)
        # = a_t^T * (2*nabla_{X_t} v_base(X_t, t) - kappa_t * I)
        # = 2 * a_t^T * nabla_{X_t} v_base(X_t, t) - kappa_t * a_t

        kappa = schedule.kappa(t).view(-1, *([1] * (x_t.dim() - 1)))

        # Compute vector-Jacobian product: a_t^T * nabla_{X_t} v_base
        x_t_req = x_t.requires_grad_(True)
        v_base = v_base_fn(x_t_req, t, text_embeddings)
        # vjp: a_t^T * J_v where J_v = nabla_{x_t} v_base
        vjp = torch.autograd.grad(
            outputs=v_base,
            inputs=x_t_req,
            grad_outputs=a_t,
            create_graph=False,
            retain_graph=False,
        )[0]

        # Lean adjoint update (Eq. 41):
        # ã_{t-h} = ã_t + h * (2 * vjp - kappa_t * ã_t)
        a_prev = a_t + h * (2.0 * vjp - kappa * a_t)

        # Add running cost gradient if provided
        if f_running_cost is not None:
            x_t_req2 = x_t.requires_grad_(True)
            f_val = f_running_cost(x_t_req2, t)
            f_grad = torch.autograd.grad(f_val.sum(), x_t_req2)[0]
            a_prev = a_prev - h * f_grad

        a_tilde.append(a_prev)

    # Reverse to get [ã_0, ã_h, ..., ã_1]
    a_tilde.reverse()
    return a_tilde


def compute_lean_adjoint_fm_noiseless_init(
    xs: List[torch.Tensor],
    x1_hat: torch.Tensor,
    reward_fn: Callable[[torch.Tensor], torch.Tensor],
    v_base_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    schedule: FlowMatchingSchedule,
    text_embeddings: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """
    Lean adjoint with noiseless final state initialization (Appendix G.1).

    X_hat_1 = X_{1-h} + h * v_base(X_{1-h}, 1-h)  (noiseless final step)
    ã_1 = -nabla_{X_hat_1} r(X_hat_1)

    This removes bias from the noise added in the final step.
    """
    x1_req = x1_hat.requires_grad_(True)
    r_val = reward_fn(x1_req)
    reward_grad = torch.autograd.grad(r_val.sum(), x1_req)[0]

    return compute_lean_adjoint_fm(
        xs=xs,
        reward_grad=reward_grad,
        v_base_fn=v_base_fn,
        schedule=schedule,
        text_embeddings=text_embeddings,
    )


# ---------------------------------------------------------------------------
# Lean Adjoint for DDPM (Algorithm 2)
# ---------------------------------------------------------------------------

def compute_lean_adjoint_ddpm(
    xs: List[torch.Tensor],
    reward_grad: torch.Tensor,
    eps_base_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    ddim_schedule: DDIMSchedule,
    text_embeddings: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """
    Lean adjoint ODE for DDPM (Algorithm 2, Eq. 221-222).

    The base drift for DDPM with memoryless schedule:
      b_base(x, k) = (alpha_bar_{k+1} - alpha_bar_k)/(2*alpha_bar_k) * x
                   - (alpha_bar_{k+1} - alpha_bar_k)/(alpha_bar_k * sqrt(1-alpha_bar_k)) * eps_base(x,k)

    Lean adjoint update:
      ã_k = ã_{k+1} + ã_{k+1}^T * nabla_{X_k}(DDPM_base_drift(X_k, k))
      ã_K = -nabla_{X_K} r(X_K)
    """
    K = ddim_schedule.K
    device = reward_grad.device

    a_tilde = [-reward_grad]

    for k in range(K - 1, -1, -1):
        x_k = xs[k].detach()
        a_k1 = a_tilde[-1].detach()

        ab_k = ddim_schedule.alpha_bar[k].to(device)
        ab_k1 = ddim_schedule.alpha_bar[k + 1].to(device)
        d_ab = ab_k1 - ab_k

        t = torch.full((x_k.shape[0],), k / K, device=device, dtype=x_k.dtype)

        # Compute VJP through DDPM base drift
        x_k_req = x_k.requires_grad_(True)
        eps_base = eps_base_fn(x_k_req, t, text_embeddings)

        # b_base(x_k, k) = (d_ab/(2*ab_k))*x_k - (d_ab/(ab_k*sqrt(1-ab_k)))*eps_base
        b_base = (d_ab / (2.0 * ab_k)) * x_k_req - \
                 (d_ab / (ab_k * torch.sqrt(1.0 - ab_k))) * eps_base

        vjp = torch.autograd.grad(
            outputs=b_base,
            inputs=x_k_req,
            grad_outputs=a_k1,
            create_graph=False,
            retain_graph=False,
        )[0]

        a_k = a_k1 + vjp
        a_tilde.append(a_k)

    a_tilde.reverse()
    return a_tilde


# ---------------------------------------------------------------------------
# Full Adjoint ODE (Continuous Adjoint baseline, Section 5.1.1)
# ---------------------------------------------------------------------------

def compute_full_adjoint_fm(
    xs: List[torch.Tensor],
    reward_grad: torch.Tensor,
    v_finetune_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    v_base_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    schedule: FlowMatchingSchedule,
    text_embeddings: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """
    Full adjoint ODE (Eq. 30-31) for the Continuous Adjoint method.

    d/dt a(t; X, u) = -[a(t; X, u)^T * nabla_{X_t}(b(X_t,t) + sigma(t)*u(X_t,t))
                        + nabla_{X_t}(f(X_t,t) + 1/2*||u(X_t,t)||^2)]
    a(1; X, u) = nabla g(X_1) = -nabla r(X_1)

    For FM with memoryless schedule and f=0:
      b + sigma*u = 2*v_finetune - kappa_t*x
      1/2*||u||^2 = (2/sigma(t))^2/2 * ||v_finetune - v_base||^2 = 2/eta_t * ||v_finetune - v_base||^2

    Full adjoint includes extra terms compared to lean adjoint:
      nabla_{X_t}(1/2*||u(X_t,t)||^2) and nabla_{X_t}(u(X_t,t))^T * a(t)
    """
    K = schedule.K
    h = schedule.h
    device = reward_grad.device

    a = [-reward_grad]

    for i in range(K - 1, -1, -1):
        t_val = schedule.timesteps[i]
        t = torch.full((xs[i].shape[0],), t_val.item(), device=device, dtype=xs[i].dtype)

        x_t = xs[i].detach()
        a_t = a[-1].detach()

        kappa = schedule.kappa(t).view(-1, *([1] * (x_t.dim() - 1)))
        sigma = schedule.sigma_memoryless(t).view(-1, *([1] * (x_t.dim() - 1)))
        eta = schedule.eta(t).view(-1, *([1] * (x_t.dim() - 1)))

        x_t_req = x_t.requires_grad_(True)
        v_ft = v_finetune_fn(x_t_req, t, text_embeddings)
        v_bs = v_base_fn(x_t_req, t, text_embeddings)

        # Full drift: b + sigma*u = 2*v_finetune - kappa*x
        full_drift = 2.0 * v_ft - kappa * x_t_req

        # Control cost: 1/2 * ||u||^2 = 2/eta_t * ||v_ft - v_bs||^2
        u = torch.sqrt(2.0 / eta) * (v_ft - v_bs)
        control_cost = 0.5 * (u ** 2).sum(dim=list(range(1, u.dim())))

        # VJP through full drift
        vjp_drift = torch.autograd.grad(
            outputs=full_drift,
            inputs=x_t_req,
            grad_outputs=a_t,
            create_graph=False,
            retain_graph=True,
        )[0]

        # Gradient of control cost
        grad_cost = torch.autograd.grad(
            outputs=control_cost.sum(),
            inputs=x_t_req,
            create_graph=False,
            retain_graph=False,
        )[0]

        a_prev = a_t + h * (-vjp_drift - grad_cost)
        a.append(a_prev)

    a.reverse()
    return a


# ---------------------------------------------------------------------------
# Discrete Adjoint (differentiate-then-discretize, Section 5.1.1)
# ---------------------------------------------------------------------------

def compute_discrete_adjoint_fm(
    xs: List[torch.Tensor],
    reward_fn: Callable[[torch.Tensor], torch.Tensor],
    v_finetune_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    v_base_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    schedule: FlowMatchingSchedule,
    text_embeddings: Optional[torch.Tensor] = None,
    reward_lambda: float = 1.0,
) -> torch.Tensor:
    """
    Discrete Adjoint: differentiate through the entire trajectory.
    Uses gradient checkpointing to reduce memory.

    Computes gradient of:
      L = -lambda * r(X_1) + 1/2 * sum_t ||u(X_t, t)||^2 * h
    with respect to model parameters, by backpropagating through the SDE.

    This is the "discretize-then-differentiate" approach.
    """
    K = schedule.K
    h = schedule.h
    device = xs[0].device

    # Compute control cost along trajectory
    total_cost = torch.tensor(0.0, device=device)

    for i in range(K):
        t_val = schedule.timesteps[i]
        t = torch.full((xs[i].shape[0],), t_val.item(), device=device, dtype=xs[i].dtype)
        eta = schedule.eta(t).view(-1, *([1] * (xs[i].dim() - 1)))

        v_ft = v_finetune_fn(xs[i], t, text_embeddings)
        v_bs = v_base_fn(xs[i].detach(), t, text_embeddings)

        # Control cost: 2/eta_t * ||v_ft - v_bs||^2 * h
        u_sq = (2.0 / eta) * ((v_ft - v_bs) ** 2).sum(dim=list(range(1, v_ft.dim())))
        total_cost = total_cost + h * u_sq.mean()

    # Terminal reward
    x1 = xs[-1]
    reward = reward_fn(x1)
    total_cost = total_cost - reward_lambda * reward.mean()

    return total_cost


# ---------------------------------------------------------------------------
# Adjoint state for control computation
# ---------------------------------------------------------------------------

def adjoint_to_control_fm(
    a_tilde: torch.Tensor,
    t: torch.Tensor,
    schedule: FlowMatchingSchedule,
) -> torch.Tensor:
    """
    Convert lean adjoint state to target control (Eq. 37):
      target = -sigma(t)^T * ã(t; X)
             = -sigma(t) * ã(t; X)   [scalar sigma]

    The Adjoint Matching loss minimizes:
      ||u(X_t, t) + sigma(t) * ã(t; X)||^2
    """
    sigma = schedule.sigma_memoryless(t).view(-1, *([1] * (a_tilde.dim() - 1)))
    return -sigma * a_tilde
