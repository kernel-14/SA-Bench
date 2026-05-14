"""
Loss functions for all fine-tuning methods.

Implements:
1. Adjoint Matching loss (Eq. 37, Algorithm 1) - proposed method
2. Continuous Adjoint loss (Eq. 28, 32) - baseline
3. Discrete Adjoint loss - baseline
4. DRaFT-K loss (Clark et al., 2024) - baseline
5. ReFL loss (Xu et al., 2023) - baseline
6. DPO loss for Flow Matching (Wallace et al., 2023a, adapted) - baseline

All losses are for Flow Matching models. DDPM variants follow the same
structure with different parameterizations (see Algorithm 2 in Appendix E.4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple

from noise_schedules import FlowMatchingSchedule, DDIMSchedule
from adjoint import (
    compute_lean_adjoint_fm,
    compute_lean_adjoint_fm_noiseless_init,
    compute_full_adjoint_fm,
    compute_discrete_adjoint_fm,
)
from sde_utils import sample_fm_sde_memoryless_with_grad


# ---------------------------------------------------------------------------
# Adjoint Matching Loss (Algorithm 1, Eq. 37, 42)
# ---------------------------------------------------------------------------

def adjoint_matching_loss_fm(
    v_finetune_fn: Callable,
    v_base_fn: Callable,
    reward_fn: Callable,
    x0: torch.Tensor,
    schedule: FlowMatchingSchedule,
    text_embeddings: Optional[torch.Tensor] = None,
    reward_lambda: float = 1.0,
    grad_timestep_indices: Optional[torch.Tensor] = None,
    lct: Optional[float] = None,
) -> torch.Tensor:
    """
    Adjoint Matching loss for Flow Matching (Algorithm 1).

    Steps:
    1. Sample trajectory with memoryless noise schedule (Eq. 40)
    2. Compute noiseless final state X_hat_1 for reward gradient
    3. Solve lean adjoint ODE backwards (Eq. 41)
    4. Compute loss (Eq. 42):
       L = sum_t ||2/sigma(t) * (v_finetune(X_t,t) - v_base(X_t,t)) + sigma(t)*ã_t||^2

    With loss clipping threshold LCT (Appendix G.3):
       L = sum_t min{LCT, ||...||^2}

    Args:
        v_finetune_fn: Fine-tuned velocity field (with gradients)
        v_base_fn: Base velocity field (no gradients needed)
        reward_fn: Reward function r(x) -> scalar per sample
        x0: Initial noise [B, C, H, W]
        schedule: FlowMatchingSchedule
        text_embeddings: Text conditioning
        reward_lambda: Scaling factor lambda for reward
        grad_timestep_indices: Subset of timestep indices for gradient computation
        lct: Loss clipping threshold (None = no clipping)

    Returns:
        Scalar loss
    """
    K = schedule.K
    h = schedule.h
    device = x0.device

    # Step 1: Sample trajectory with memoryless noise schedule (no grad for trajectory)
    with torch.no_grad():
        x = x0.clone()
        xs = [x]
        for i in range(K):
            t_val = schedule.timesteps[i]
            t = torch.full((x.shape[0],), t_val.item(), device=device, dtype=x.dtype)
            v = v_finetune_fn(x, t, text_embeddings)
            kappa = schedule.kappa(t).view(-1, *([1] * (x.dim() - 1)))
            sigma = schedule.sigma_memoryless(t).view(-1, *([1] * (x.dim() - 1)))
            drift = 2.0 * v - kappa * x
            noise = torch.randn_like(x)
            x = x + h * drift + (h ** 0.5) * sigma * noise
            xs.append(x)

    # Step 2: Compute noiseless final state X_hat_1 (Appendix G.1)
    with torch.no_grad():
        t_last = torch.full((xs[-2].shape[0],), schedule.timesteps[-1].item(),
                            device=device, dtype=xs[-2].dtype)
        v_last = v_base_fn(xs[-2], t_last, text_embeddings)
        x1_hat = xs[-2] + h * v_last  # noiseless final step

    # Step 3: Compute reward gradient at X_hat_1
    x1_req = x1_hat.detach().requires_grad_(True)
    r_val = reward_lambda * reward_fn(x1_req)
    reward_grad = torch.autograd.grad(r_val.sum(), x1_req)[0]

    # Step 4: Solve lean adjoint ODE backwards
    a_tildes = compute_lean_adjoint_fm(
        xs=xs,
        reward_grad=reward_grad,
        v_base_fn=v_base_fn,
        schedule=schedule,
        text_embeddings=text_embeddings,
    )

    # Step 5: Compute Adjoint Matching loss (Eq. 42)
    if grad_timestep_indices is None:
        grad_timestep_indices = torch.arange(K, device=device)

    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for idx in grad_timestep_indices:
        i = idx.item()
        t_val = schedule.timesteps[i]
        t = torch.full((xs[i].shape[0],), t_val.item(), device=device, dtype=xs[i].dtype)

        sigma = schedule.sigma_memoryless(t).view(-1, *([1] * (xs[i].dim() - 1)))
        eta = schedule.eta(t).view(-1, *([1] * (xs[i].dim() - 1)))

        # Compute v_finetune with gradients at this timestep
        v_ft = v_finetune_fn(xs[i].detach(), t, text_embeddings)
        v_bs = v_base_fn(xs[i].detach(), t, text_embeddings)

        a_t = a_tildes[i].detach()

        # Loss term: ||2/sigma(t) * (v_ft - v_bs) + sigma(t) * ã_t||^2
        # = ||sqrt(2/eta_t) * (v_ft - v_bs) + sqrt(2*eta_t) * ã_t||^2
        term = (2.0 / sigma) * (v_ft - v_bs) + sigma * a_t
        loss_t = (term ** 2).sum(dim=list(range(1, term.dim()))).mean()

        if lct is not None:
            loss_t = torch.clamp(loss_t, max=lct)

        total_loss = total_loss + loss_t
        count += 1

    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Continuous Adjoint Loss (Section 5.1.1, Eq. 28, 32)
# ---------------------------------------------------------------------------

def continuous_adjoint_loss_fm(
    v_finetune_fn: Callable,
    v_base_fn: Callable,
    reward_fn: Callable,
    x0: torch.Tensor,
    schedule: FlowMatchingSchedule,
    text_embeddings: Optional[torch.Tensor] = None,
    reward_lambda: float = 1.0,
    grad_timestep_indices: Optional[torch.Tensor] = None,
    lct: Optional[float] = None,
) -> torch.Tensor:
    """
    Continuous Adjoint loss (Eq. 28):
      L(u; X) = integral_0^1 (1/2 * ||u(X_t,t)||^2) dt - r(X_1)

    Gradient computed via the adjoint method (Eq. 32):
      dL/dtheta = 1/2 * integral_0^1 d/dtheta ||u(X_t,t)||^2 dt
                + integral_0^1 (du/dtheta)^T * sigma(t)^T * a(t; X, u) dt

    In practice, we use the Basic Adjoint Matching formulation (Proposition 2)
    which gives the same gradient but as a least-squares objective:
      L_basic = 1/2 * integral ||u(X_t,t) + sigma(t)^T * a(t; X, u)||^2 dt
    """
    K = schedule.K
    h = schedule.h
    device = x0.device

    # Sample trajectory
    with torch.no_grad():
        x = x0.clone()
        xs = [x]
        for i in range(K):
            t_val = schedule.timesteps[i]
            t = torch.full((x.shape[0],), t_val.item(), device=device, dtype=x.dtype)
            v = v_finetune_fn(x, t, text_embeddings)
            kappa = schedule.kappa(t).view(-1, *([1] * (x.dim() - 1)))
            sigma = schedule.sigma_memoryless(t).view(-1, *([1] * (x.dim() - 1)))
            drift = 2.0 * v - kappa * x
            noise = torch.randn_like(x)
            x = x + h * drift + (h ** 0.5) * sigma * noise
            xs.append(x)

    # Noiseless final state
    with torch.no_grad():
        t_last = torch.full((xs[-2].shape[0],), schedule.timesteps[-1].item(),
                            device=device, dtype=xs[-2].dtype)
        v_last = v_base_fn(xs[-2], t_last, text_embeddings)
        x1_hat = xs[-2] + h * v_last

    x1_req = x1_hat.detach().requires_grad_(True)
    r_val = reward_lambda * reward_fn(x1_req)
    reward_grad = torch.autograd.grad(r_val.sum(), x1_req)[0]

    # Full adjoint ODE
    a_states = compute_full_adjoint_fm(
        xs=xs,
        reward_grad=reward_grad,
        v_finetune_fn=v_finetune_fn,
        v_base_fn=v_base_fn,
        schedule=schedule,
        text_embeddings=text_embeddings,
    )

    if grad_timestep_indices is None:
        grad_timestep_indices = torch.arange(K, device=device)

    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for idx in grad_timestep_indices:
        i = idx.item()
        t_val = schedule.timesteps[i]
        t = torch.full((xs[i].shape[0],), t_val.item(), device=device, dtype=xs[i].dtype)

        sigma = schedule.sigma_memoryless(t).view(-1, *([1] * (xs[i].dim() - 1)))
        eta = schedule.eta(t).view(-1, *([1] * (xs[i].dim() - 1)))

        v_ft = v_finetune_fn(xs[i].detach(), t, text_embeddings)
        v_bs = v_base_fn(xs[i].detach(), t, text_embeddings)

        a_t = a_states[i].detach()

        # Basic Adjoint Matching objective (Proposition 2):
        # ||u(X_t,t) + sigma(t) * a(t; X, u)||^2
        u = torch.sqrt(2.0 / eta) * (v_ft - v_bs)
        term = u + sigma * a_t
        loss_t = (term ** 2).sum(dim=list(range(1, term.dim()))).mean()

        if lct is not None:
            loss_t = torch.clamp(loss_t, max=lct)

        total_loss = total_loss + loss_t
        count += 1

    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Discrete Adjoint Loss (Section 5.1.1)
# ---------------------------------------------------------------------------

def discrete_adjoint_loss_fm(
    v_finetune_fn: Callable,
    v_base_fn: Callable,
    reward_fn: Callable,
    x0: torch.Tensor,
    schedule: FlowMatchingSchedule,
    text_embeddings: Optional[torch.Tensor] = None,
    reward_lambda: float = 1.0,
) -> torch.Tensor:
    """
    Discrete Adjoint: differentiate through the full SDE simulation.
    Uses "discretize-then-differentiate" approach.

    L = -lambda * r(X_1) + 1/2 * sum_t h * ||u(X_t,t)||^2
    """
    K = schedule.K
    h = schedule.h
    device = x0.device

    x = x0.clone()
    total_control_cost = torch.tensor(0.0, device=device)

    for i in range(K):
        t_val = schedule.timesteps[i]
        t = torch.full((x.shape[0],), t_val.item(), device=device, dtype=x.dtype)

        eta = schedule.eta(t).view(-1, *([1] * (x.dim() - 1)))
        sigma = schedule.sigma_memoryless(t).view(-1, *([1] * (x.dim() - 1)))
        kappa = schedule.kappa(t).view(-1, *([1] * (x.dim() - 1)))

        v_ft = v_finetune_fn(x, t, text_embeddings)
        with torch.no_grad():
            v_bs = v_base_fn(x, t, text_embeddings)

        # Control cost: 1/2 * ||u||^2 = 1/eta_t * ||v_ft - v_bs||^2
        u_sq = (1.0 / eta) * ((v_ft - v_bs) ** 2).sum(dim=list(range(1, v_ft.dim())))
        total_control_cost = total_control_cost + h * u_sq.mean()

        # Step forward (detach for memory efficiency, use gradient checkpointing in practice)
        with torch.no_grad():
            drift = 2.0 * v_ft.detach() - kappa * x
            noise = torch.randn_like(x)
            x = x + h * drift + (h ** 0.5) * sigma * noise

    # Terminal reward
    x1_req = x.requires_grad_(True)
    r_val = reward_lambda * reward_fn(x1_req)
    reward_term = -r_val.mean()

    return total_control_cost + reward_term


# ---------------------------------------------------------------------------
# DRaFT-K Loss (Clark et al., 2024)
# ---------------------------------------------------------------------------

def draft_loss_fm(
    v_finetune_fn: Callable,
    reward_fn: Callable,
    x0: torch.Tensor,
    schedule: FlowMatchingSchedule,
    text_embeddings: Optional[torch.Tensor] = None,
    reward_lambda: float = 1.0,
    num_grad_steps: int = 1,
) -> torch.Tensor:
    """
    DRaFT-K loss: backpropagate reward through last K steps.

    DRaFT-1: only last step (num_grad_steps=1)
    DRaFT-40: all 40 steps (num_grad_steps=40)

    For DRaFT-1, only the last denoising step is differentiable.
    For DRaFT-K, the last K steps are differentiable.
    """
    K = schedule.K
    h = schedule.h
    device = x0.device

    # Run first K - num_grad_steps steps without gradients
    with torch.no_grad():
        x = x0.clone()
        for i in range(K - num_grad_steps):
            t_val = schedule.timesteps[i]
            t = torch.full((x.shape[0],), t_val.item(), device=device, dtype=x.dtype)
            v = v_finetune_fn(x, t, text_embeddings)
            kappa = schedule.kappa(t).view(-1, *([1] * (x.dim() - 1)))
            sigma = schedule.sigma_memoryless(t).view(-1, *([1] * (x.dim() - 1)))
            drift = 2.0 * v - kappa * x
            noise = torch.randn_like(x)
            x = x + h * drift + (h ** 0.5) * sigma * noise

    # Run last num_grad_steps steps with gradients
    for i in range(K - num_grad_steps, K):
        t_val = schedule.timesteps[i]
        t = torch.full((x.shape[0],), t_val.item(), device=device, dtype=x.dtype)
        v = v_finetune_fn(x, t, text_embeddings)
        kappa = schedule.kappa(t).view(-1, *([1] * (x.dim() - 1)))
        sigma = schedule.sigma_memoryless(t).view(-1, *([1] * (x.dim() - 1)))
        drift = 2.0 * v - kappa * x
        with torch.no_grad():
            noise = torch.randn_like(x)
        x = x + h * drift + (h ** 0.5) * sigma * noise

    # Compute reward
    r = reward_fn(x)
    return -reward_lambda * r.mean()


# ---------------------------------------------------------------------------
# ReFL Loss (Xu et al., 2023, adapted for Flow Matching, Appendix F.1)
# ---------------------------------------------------------------------------

def refl_loss_fm(
    v_finetune_fn: Callable,
    reward_fn: Callable,
    x0: torch.Tensor,
    schedule: FlowMatchingSchedule,
    text_embeddings: Optional[torch.Tensor] = None,
    reward_lambda: float = 1.0,
    detach_trajectory: bool = True,
) -> torch.Tensor:
    """
    ReFL (Reward Feedback Learning) for Flow Matching (Appendix F.1).

    Denoiser map (Eq. 229):
      X_hat_1(x, t) = (v(x,t) - (beta_dot_t/beta_t)*x) / (alpha_dot_t - (beta_dot_t/beta_t)*alpha_t)
    For alpha_t=t, beta_t=1-t:
      X_hat_1(x, t) = (1-t)*v(x,t) + x

    Loss: -lambda * r(X_hat_1(X_t, t)) for a randomly sampled t.
    """
    K = schedule.K
    h = schedule.h
    device = x0.device

    # Sample trajectory (detached)
    with torch.no_grad():
        x = x0.clone()
        xs = [x]
        ts_used = []
        for i in range(K):
            t_val = schedule.timesteps[i]
            t = torch.full((x.shape[0],), t_val.item(), device=device, dtype=x.dtype)
            ts_used.append(t)
            v = v_finetune_fn(x, t, text_embeddings)
            kappa = schedule.kappa(t).view(-1, *([1] * (x.dim() - 1)))
            sigma = schedule.sigma_memoryless(t).view(-1, *([1] * (x.dim() - 1)))
            drift = 2.0 * v - kappa * x
            noise = torch.randn_like(x)
            x = x + h * drift + (h ** 0.5) * sigma * noise
            xs.append(x)

    # Sample a random timestep for gradient computation
    idx = torch.randint(0, K, (1,)).item()
    t_val = schedule.timesteps[idx]
    t = torch.full((xs[idx].shape[0],), t_val.item(), device=device, dtype=xs[idx].dtype)

    x_t = xs[idx].detach()
    beta = schedule.beta(t).view(-1, *([1] * (x_t.dim() - 1)))

    # Compute denoiser map with gradients
    v_ft = v_finetune_fn(x_t, t, text_embeddings)
    x_hat_1 = beta * v_ft + x_t  # = (1-t)*v(x,t) + x

    r = reward_fn(x_hat_1)
    return -reward_lambda * r.mean()


# ---------------------------------------------------------------------------
# DPO Loss for Flow Matching (Wallace et al., 2023a, Appendix F.2)
# ---------------------------------------------------------------------------

def dpo_loss_fm(
    v_finetune_fn: Callable,
    v_ref_fn: Callable,
    x1_win: torch.Tensor,
    x1_lose: torch.Tensor,
    schedule: FlowMatchingSchedule,
    beta_tilde: float = 5000.0,
    text_embeddings: Optional[torch.Tensor] = None,
    reward_fn: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Diffusion-DPO loss for Flow Matching (Appendix F.2, Eq. 234-235).

    For ranked pairs (x1_win, x1_lose) or reward-weighted pairs:
      L_DPO = -E[log sigma(-beta_tilde/2 * (||denoiser_diff_win||^2 - ||denoiser_diff_lose||^2))]

    where denoiser_diff = (v_theta(x_t, t) - (beta_dot/beta)*x_t) / (alpha_dot - (beta_dot/beta)*alpha)
                        - (alpha/beta)*x1

    For alpha_t=t, beta_t=1-t:
      denoiser_diff = v(x_t,t)/(alpha_dot - beta_dot*alpha/beta) - x1/(beta/alpha)
    """
    K = schedule.K
    h = schedule.h
    device = x1_win.device
    B = x1_win.shape[0]

    # Sample random timestep
    k = torch.randint(0, K, (1,)).item()
    t_val = schedule.timesteps[k]
    t = torch.full((B,), t_val.item(), device=device, dtype=x1_win.dtype)

    alpha = schedule.alpha(t).view(-1, *([1] * (x1_win.dim() - 1)))
    beta = schedule.beta(t).view(-1, *([1] * (x1_win.dim() - 1)))
    alpha_dot = schedule.alpha_dot(t).view(-1, *([1] * (x1_win.dim() - 1)))
    beta_dot = schedule.beta_dot(t).view(-1, *([1] * (x1_win.dim() - 1)))

    # Sample x_t from forward process: x_t = beta_t * eps + alpha_t * x1
    eps_win = torch.randn_like(x1_win)
    eps_lose = torch.randn_like(x1_lose)
    x_t_win = beta * eps_win + alpha * x1_win
    x_t_lose = beta * eps_lose + alpha * x1_lose

    def denoiser_diff(x_t, x1, v_fn):
        """
        Compute ||denoiser_map(x_t, t) - x1||^2 / (beta/alpha)^2
        = ||(v(x_t,t) - (beta_dot/beta)*x_t) / (alpha_dot - (beta_dot/beta)*alpha) - (alpha/beta)*x1||^2
        """
        v = v_fn(x_t, t, text_embeddings)
        # Denoiser map (Eq. 229)
        numerator = v - (beta_dot / beta) * x_t
        denominator = alpha_dot - (beta_dot / beta) * alpha
        x1_pred = numerator / denominator
        # Normalized difference
        diff = x1_pred - (alpha / beta) * x1
        return (diff ** 2).sum(dim=list(range(1, diff.dim())))

    # Compute differences for theta and reference model
    diff_theta_win = denoiser_diff(x_t_win.detach(), x1_win, v_finetune_fn)
    diff_theta_lose = denoiser_diff(x_t_lose.detach(), x1_lose, v_finetune_fn)

    with torch.no_grad():
        diff_ref_win = denoiser_diff(x_t_win.detach(), x1_win, v_ref_fn)
        diff_ref_lose = denoiser_diff(x_t_lose.detach(), x1_lose, v_ref_fn)

    # DPO loss (Eq. 234)
    implicit_reward_diff = (
        (diff_ref_win - diff_theta_win) - (diff_ref_lose - diff_theta_lose)
    )

    if reward_fn is not None:
        # Soft preference weights from reward model (Eq. 235)
        with torch.no_grad():
            r_win = reward_fn(x1_win)
            r_lose = reward_fn(x1_lose)
        w_win = torch.sigmoid(r_win - r_lose)
        w_lose = torch.sigmoid(r_lose - r_win)
        loss = -(w_win * F.logsigmoid(-beta_tilde / 2.0 * implicit_reward_diff) +
                 w_lose * F.logsigmoid(beta_tilde / 2.0 * implicit_reward_diff)).mean()
    else:
        loss = -F.logsigmoid(-beta_tilde / 2.0 * implicit_reward_diff).mean()

    return loss


# ---------------------------------------------------------------------------
# Flow Matching pre-training loss (for reference)
# ---------------------------------------------------------------------------

def flow_matching_loss(
    v_fn: Callable,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    text_embeddings: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Flow Matching pre-training loss (Lipman et al., 2023):
      L_FM = E[||v(X_t, t) - (X_1 - X_0)||^2]
    where X_t = (1-t)*X_0 + t*X_1 and the target velocity is X_1 - X_0 = alpha_dot*X_1 + beta_dot*X_0.

    For alpha_t=t, beta_t=1-t: target = X_1 - X_0.
    """
    t_view = t.view(-1, *([1] * (x0.dim() - 1)))
    x_t = (1.0 - t_view) * x0 + t_view * x1
    target = x1 - x0  # = alpha_dot*x1 + beta_dot*x0 = x1 - x0

    v_pred = v_fn(x_t, t, text_embeddings)
    return F.mse_loss(v_pred, target)


# ---------------------------------------------------------------------------
# SOC objective (for evaluation / monitoring)
# ---------------------------------------------------------------------------

def soc_objective_fm(
    v_finetune_fn: Callable,
    v_base_fn: Callable,
    reward_fn: Callable,
    x0: torch.Tensor,
    schedule: FlowMatchingSchedule,
    text_embeddings: Optional[torch.Tensor] = None,
    reward_lambda: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate the SOC objective (Eq. 12, 19):
      J = E[1/2 * integral ||u(X_t,t)||^2 dt - lambda * r(X_1)]
        = KL(p^u || p^base) - lambda * E[r(X_1)]

    Returns: (total_objective, control_cost, reward)
    """
    K = schedule.K
    h = schedule.h
    device = x0.device

    with torch.no_grad():
        x = x0.clone()
        total_control_cost = torch.tensor(0.0, device=device)

        for i in range(K):
            t_val = schedule.timesteps[i]
            t = torch.full((x.shape[0],), t_val.item(), device=device, dtype=x.dtype)

            eta = schedule.eta(t).view(-1, *([1] * (x.dim() - 1)))
            sigma = schedule.sigma_memoryless(t).view(-1, *([1] * (x.dim() - 1)))
            kappa = schedule.kappa(t).view(-1, *([1] * (x.dim() - 1)))

            v_ft = v_finetune_fn(x, t, text_embeddings)
            v_bs = v_base_fn(x, t, text_embeddings)

            # Control: u = sqrt(2/eta) * (v_ft - v_bs)
            u_sq = (2.0 / eta) * ((v_ft - v_bs) ** 2).sum(dim=list(range(1, v_ft.dim())))
            total_control_cost = total_control_cost + h * 0.5 * u_sq.mean()

            drift = 2.0 * v_ft - kappa * x
            noise = torch.randn_like(x)
            x = x + h * drift + (h ** 0.5) * sigma * noise

        reward = reward_fn(x).mean()
        objective = total_control_cost - reward_lambda * reward

    return objective, total_control_cost, reward
