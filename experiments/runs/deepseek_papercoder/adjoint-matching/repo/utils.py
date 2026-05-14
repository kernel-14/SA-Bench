## utils.py

import math
from typing import Callable, List

import torch


def get_sigma(t: float, h: float) -> float:
    """
    Compute the memoryless noise schedule coefficient.

    The schedule follows equation (237) from the paper:
        sigma(t) = sqrt( 2 * (1 - t + h) / (t + h) )

    Args:
        t: Continuous time (0 ≤ t ≤ 1).
        h: Offset to avoid division by zero (usually dt = 1/K).

    Returns:
        Scalar sigma(t).
    """
    return math.sqrt(2.0 * (1.0 - t + h) / (t + h))


def memoryless_drift(
    v_base: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: float,
    h: float,
) -> torch.Tensor:
    """
    Evaluate the drift of the memoryless base SDE.

    Using the linear interpolation α_t = t, the drift is:
        b_m(x, t) = 2 * v_base(x, t) - x / (t + h)

    Args:
        v_base: Base velocity field callable.
                Expects (latent_batch, time_tensor) and returns velocity of same shape.
        x: Batch of latent states at time t. Shape (B, C, H, W).
        t: Scalar time.
        h: Offset constant (usually dt).

    Returns:
        Drift tensor of same shape as x.
    """
    batch = x.shape[0]
    time_tensor = torch.full((batch,), t, device=x.device, dtype=x.dtype)
    v = v_base(x, time_tensor)          # shape (B, C, H, W)
    return 2.0 * v - x / (t + h)


def sample_memoryless_sde(
    v_fine: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    v_base: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    dt: float,
    K: int,
    h: float,
) -> List[torch.Tensor]:
    """
    Simulate the fine-tuned controlled SDE with memoryless noise schedule.

    Uses Euler–Maruyama discretisation of:
        dX_t = (2 * v_fine(X_t, t) - X_t / (t + h)) dt + sqrt(dt) * sigma(t) dW_t

    Args:
        v_fine: Fine-tuned velocity field callable.
        v_base: Not used in the forward simulation; kept for interface symmetry.
        x0: Initial latent noise. Shape (B, C, H, W).
        dt: Step size.
        K: Number of steps.
        h: Offset for sigma and drift denominator.

    Returns:
        List of length K+1 containing detached states at times 0, dt, 2dt, …, 1.
    """
    traj = [x0.detach().clone()]
    x = x0.detach().clone()

    for k in range(K):
        t = k * dt
        sigma_t = get_sigma(t, h)
        batch = x.shape[0]
        time_tensor = torch.full((batch,), t, device=x.device, dtype=x.dtype)

        with torch.no_grad():
            # fine-tuned velocity (evaluated without gradient tracking)
            v_f = v_fine(x, time_tensor)
            drift = 2.0 * v_f - x / (t + h)
            eps = torch.randn_like(x)
            x = x + dt * drift + math.sqrt(dt) * sigma_t * eps
            traj.append(x.detach().clone())

    return traj


def solve_adjoint_ode_backward(
    v_base: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    trajectory: List[torch.Tensor],
    dt: float,
    reward_grad_fn: Callable[[torch.Tensor], torch.Tensor],
) -> List[torch.Tensor]:
    """
    Solve the lean adjoint ODE backward in time.

    The adjoint states satisfy:
        d\tilde{a}_t / dt = - ( \tilde{a}_t^T ∇_x b_m(x, t) )
        \tilde{a}_1 = -∇_x r( \hat{X}_1 )
    where \hat{X}_1 is a noiseless endpoint estimate.

    The integration uses forward Euler steps from t=1 down to 0, utilising the
    relationship a_{t-dt} = a_t + dt * a_t^T ∇_x b_m(x_t, t).

    Args:
        v_base: Base velocity field callable.
        trajectory: List of K+1 detached latent states (times 0 to 1).
        dt: Step size.
        reward_grad_fn: A callable that computes ∇_x r( \hat{X}_1 );
                        it receives the batched noiseless endpoint estimate and
                        returns a gradient tensor of the same shape (detached).

    Returns:
        List of K adjoint states corresponding to times 0, dt, 2dt, …, 1-dt.
    """
    K = len(trajectory) - 1
    h_offset = dt   # offset used in drift denominator

    # ---- 1. Noiseless endpoint estimate ----
    X_last = trajectory[-2]  # state at time 1 - dt
    t_last = 1.0 - dt
    batch = X_last.shape[0]
    time_tensor_last = torch.full((batch,), t_last, device=X_last.device, dtype=X_last.dtype)

    with torch.no_grad():
        v_last = v_base(X_last, time_tensor_last)
    X_hat_1 = X_last + dt * v_last          # noiseless endpoint

    # ---- 2. Reward gradient ----
    grad_r = reward_grad_fn(X_hat_1)        # should be detached externally
    if grad_r.requires_grad:
        grad_r = grad_r.detach()

    # ---- 3. Initialise adjoint at t = 1 ----
    a = -grad_r.clone().detach()

    adjoints_reverse: List[torch.Tensor] = []

    # ---- 4. Backward integration loop ----
    for i in range(K, 0, -1):               # i from K down to 1
        t = i * dt                          # current time (1, 1-dt, …, dt)
        x_t = trajectory[i]                 # detached state

        # Ensure gradient is computed only w.r.t. this local tensor
        x_t.requires_grad_(True)
        batch_i = x_t.shape[0]
        time_tensor = torch.full((batch_i,), t, device=x_t.device, dtype=x_t.dtype)

        # drift at time t
        b_m = 2.0 * v_base(x_t, time_tensor) - x_t / (t + h_offset)

        # scalar product a·b_m, then gradient ↔ a^T ∇_x b_m
        scalar = (a.detach() * b_m).sum()
        vjp = torch.autograd.grad(scalar, x_t, create_graph=False)[0]

        x_t.requires_grad_(False)

        # Euler step backwards: a_{t-dt} = a_t + dt * vjp
        a = a + dt * vjp.detach()

        # Record adjoint at time (i-1)*dt
        adjoints_reverse.append(a.clone().detach())

    # Reverse to time‑order: [a_0, a_dt, …, a_{1-dt}]
    adjoints = adjoints_reverse[::-1]

    # Sanity check: length should equal K (times 0, dt, …, 1-dt)
    assert len(adjoints) == K, f"Expected {K} adjoint states, got {len(adjoints)}"

    return adjoints
