"""
SDE/ODE simulation utilities for Flow Matching and diffusion models.

Implements:
- Euler-Maruyama discretization of the controlled SDE (Eq. 13)
- Flow Matching ODE sampling (Eq. 3)
- Memoryless Flow Matching SDE sampling (Algorithm 1, Eq. 40)
- DDIM/DDPM sampling (Eq. 5, 6, 7)
- Trajectory sampling for fine-tuning
"""

import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple

from noise_schedules import FlowMatchingSchedule, DDIMSchedule


# ---------------------------------------------------------------------------
# Flow Matching trajectory sampling
# ---------------------------------------------------------------------------

def sample_fm_ode(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    x0: torch.Tensor,
    schedule: FlowMatchingSchedule,
    text_embeddings: Optional[torch.Tensor] = None,
    return_trajectory: bool = False,
) -> torch.Tensor:
    """
    Sample from Flow Matching ODE (Eq. 3):
      dX_t = v(X_t, t) dt,  X_0 ~ N(0, I)

    Euler discretization:
      X_{t+h} = X_t + h * v(X_t, t)
    """
    x = x0.clone()
    trajectory = [x.clone()] if return_trajectory else None

    for i in range(schedule.K):
        t_val = schedule.timesteps[i]
        t = torch.full((x.shape[0],), t_val.item(), device=x.device, dtype=x.dtype)
        with torch.no_grad():
            v = velocity_fn(x, t, text_embeddings)
        x = x + schedule.h * v
        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return x, torch.stack(trajectory, dim=1)  # [B, K+1, ...]
    return x


def sample_fm_sde_memoryless(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    x0: torch.Tensor,
    schedule: FlowMatchingSchedule,
    text_embeddings: Optional[torch.Tensor] = None,
    return_trajectory: bool = False,
    no_grad: bool = True,
) -> torch.Tensor:
    """
    Sample from Memoryless Flow Matching SDE (Algorithm 1, Eq. 40):
      X_{t+h} = X_t + h*(2*v_finetune(X_t,t) - kappa_t*X_t) + sqrt(h)*sigma(t)*eps

    This uses the memoryless noise schedule sigma(t) = sqrt(2*(1-t+h)/(t+h)).
    The full drift is: b(x,t) + sigma(t)*u(x,t) = 2*v_finetune(x,t) - kappa_t*x
    """
    x = x0.clone()
    trajectory = [x.clone()] if return_trajectory else None
    h = schedule.h

    context = torch.no_grad() if no_grad else torch.enable_grad()
    with context:
        for i in range(schedule.K):
            t_val = schedule.timesteps[i]
            t = torch.full((x.shape[0],), t_val.item(), device=x.device, dtype=x.dtype)

            v = velocity_fn(x, t, text_embeddings)
            kappa = schedule.kappa(t).view(-1, *([1] * (x.dim() - 1)))
            sigma = schedule.sigma_memoryless(t).view(-1, *([1] * (x.dim() - 1)))

            drift = 2.0 * v - kappa * x
            noise = torch.randn_like(x)
            x = x + h * drift + (h ** 0.5) * sigma * noise

            if return_trajectory:
                trajectory.append(x.clone())

    if return_trajectory:
        return x, torch.stack(trajectory, dim=1)
    return x


def sample_fm_sde_memoryless_with_grad(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    x0: torch.Tensor,
    schedule: FlowMatchingSchedule,
    text_embeddings: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Sample trajectory with gradient tracking for fine-tuning.
    Returns: (x_final, list of x_t, list of noise_t)
    """
    x = x0.clone()
    xs = [x]
    noises = []
    h = schedule.h

    for i in range(schedule.K):
        t_val = schedule.timesteps[i]
        t = torch.full((x.shape[0],), t_val.item(), device=x.device, dtype=x.dtype)

        v = velocity_fn(x, t, text_embeddings)
        kappa = schedule.kappa(t).view(-1, *([1] * (x.dim() - 1)))
        sigma = schedule.sigma_memoryless(t).view(-1, *([1] * (x.dim() - 1)))

        drift = 2.0 * v - kappa * x
        noise = torch.randn_like(x)
        noises.append(noise)
        x = x + h * drift + (h ** 0.5) * sigma * noise
        xs.append(x)

    return x, xs, noises


# ---------------------------------------------------------------------------
# DDPM trajectory sampling
# ---------------------------------------------------------------------------

def sample_ddpm(
    denoiser_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    x0: torch.Tensor,
    ddim_schedule: DDIMSchedule,
    text_embeddings: Optional[torch.Tensor] = None,
    return_trajectory: bool = False,
    no_grad: bool = True,
) -> torch.Tensor:
    """
    Sample from DDPM (Algorithm 2, Eq. 219).
    x0 is the initial noise X_0 ~ N(0, I).
    """
    x = x0.clone()
    trajectory = [x.clone()] if return_trajectory else None

    context = torch.no_grad() if no_grad else torch.enable_grad()
    with context:
        for k in range(ddim_schedule.K):
            t = torch.full((x.shape[0],), k / ddim_schedule.K,
                           device=x.device, dtype=x.dtype)
            eps = denoiser_fn(x, t, text_embeddings)
            noise = torch.randn_like(x)
            x = ddim_schedule.ddpm_step(x, eps, k, noise)
            if return_trajectory:
                trajectory.append(x.clone())

    if return_trajectory:
        return x, torch.stack(trajectory, dim=1)
    return x


def sample_ddim(
    denoiser_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    x0: torch.Tensor,
    ddim_schedule: DDIMSchedule,
    text_embeddings: Optional[torch.Tensor] = None,
    return_trajectory: bool = False,
    no_grad: bool = True,
) -> torch.Tensor:
    """
    Sample from DDIM (deterministic ODE, sigma_k = 0).
    """
    x = x0.clone()
    trajectory = [x.clone()] if return_trajectory else None

    context = torch.no_grad() if no_grad else torch.enable_grad()
    with context:
        for k in range(ddim_schedule.K):
            t = torch.full((x.shape[0],), k / ddim_schedule.K,
                           device=x.device, dtype=x.dtype)
            eps = denoiser_fn(x, t, text_embeddings)
            ab_k = ddim_schedule.alpha_bar[k]
            ab_k1 = ddim_schedule.alpha_bar[k + 1]
            # DDIM deterministic update
            x0_pred = (x - torch.sqrt(1.0 - ab_k) * eps) / torch.sqrt(ab_k)
            x = torch.sqrt(ab_k1) * x0_pred + torch.sqrt(1.0 - ab_k1) * eps
            if return_trajectory:
                trajectory.append(x.clone())

    if return_trajectory:
        return x, torch.stack(trajectory, dim=1)
    return x


# ---------------------------------------------------------------------------
# Classifier-free guidance sampling
# ---------------------------------------------------------------------------

def sample_fm_cfg(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    x0: torch.Tensor,
    schedule: FlowMatchingSchedule,
    text_embeddings: torch.Tensor,
    null_embeddings: torch.Tensor,
    cfg_scale: float = 1.0,
    sigma_type: str = "zero",
    no_grad: bool = True,
) -> torch.Tensor:
    """
    Classifier-free guidance sampling (Section 7):
      v_guided(x, t | y) = (1+w)*v(x,t|y) - w*v(x,t)
    where w = cfg_scale.
    """
    def guided_velocity(x, t, _):
        v_cond = velocity_fn(x, t, text_embeddings)
        if cfg_scale == 0.0:
            return v_cond
        v_uncond = velocity_fn(x, t, null_embeddings)
        return (1.0 + cfg_scale) * v_cond - cfg_scale * v_uncond

    if sigma_type == "zero":
        return sample_fm_ode(guided_velocity, x0, schedule, no_grad=no_grad)
    elif sigma_type == "memoryless":
        return sample_fm_sde_memoryless(guided_velocity, x0, schedule, no_grad=no_grad)
    else:
        raise ValueError(f"Unknown sigma_type: {sigma_type}")


# ---------------------------------------------------------------------------
# Interpolation (reference flow)
# ---------------------------------------------------------------------------

def interpolate_reference_flow(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Reference flow interpolation (Eq. 2):
      X_t = beta_t * X_0 + alpha_t * X_1
    For alpha_t = t, beta_t = 1-t:
      X_t = (1-t)*X_0 + t*X_1
    """
    t_view = t.view(-1, *([1] * (x0.dim() - 1)))
    return (1.0 - t_view) * x0 + t_view * x1


def sample_reference_flow_conditional(
    x1: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample X_t ~ p(X_t | X_1) from the reference flow.
    X_t = (1-t)*eps + t*X_1,  eps ~ N(0, I)
    Returns (X_t, eps).
    """
    eps = torch.randn_like(x1)
    x_t = interpolate_reference_flow(eps, x1, t)
    return x_t, eps
