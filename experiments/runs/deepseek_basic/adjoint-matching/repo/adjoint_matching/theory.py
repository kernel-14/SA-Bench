"""
Theoretical foundations of Adjoint Matching and memoryless SOC.

This module provides reference implementations and documentation for the
key mathematical results in the paper:

1. Proposition 1: Memoryless noise schedule condition
   σ(t)² = 2η_t + χ(t) where χ satisfies certain limit conditions

2. Theorem 1: Fine-tuning must use memoryless noise schedule

3. Proposition 2: Basic Adjoint Matching has correct critical points

4. Proposition 7: Adjoint Matching loss has correct critical points (the
   lean adjoint removes terms with zero expectation at optimum)

These serve as reference implementations to verify the theoretical claims.
"""

import torch
import math
from typing import Callable


def compute_eta_flow_matching(t: torch.Tensor) -> torch.Tensor:
    """
    Compute η_t for Flow Matching with α_t = t, β_t = 1 - t.

    η_t = β_t (α̇_t/α_t · β_t - β̇_t)
        = (1-t)((1-t)/t + 1)
        = (1-t)/t

    This is the key coefficient that determines the memoryless noise schedule.
    """
    return (1.0 - t) / t


def compute_eta_ddim(alpha_bar: torch.Tensor, alpha_dot_bar: torch.Tensor) -> torch.Tensor:
    """
    Compute η_t for DDIM.

    η_t = ᾱ̇_t / (2ᾱ_t)

    where ᾱ_t is the cumulative product of α_t in the DDIM framework.
    """
    return alpha_dot_bar / (2.0 * alpha_bar)


def memoryless_noise_schedule_fm(t: torch.Tensor, h: float = 0.025) -> torch.Tensor:
    """
    Memoryless noise schedule for Flow Matching.

    σ(t) = √(2η_t) = √(2(1-t)/t)

    With numerical offset (Appendix G.1):
    σ(t) = √(2(1-t+h)/(t+h))

    Args:
        t: Time values in (0, 1].
        h: Step size for numerical stability.

    Returns:
        σ(t) values.
    """
    t_safe = t + h
    one_minus_t_safe = 1.0 - t + h
    return torch.sqrt(2.0 * one_minus_t_safe / t_safe)


def memoryless_noise_schedule_ddim(
    alpha_bar: torch.Tensor, alpha_dot_bar: torch.Tensor
) -> torch.Tensor:
    """
    Memoryless noise schedule for DDIM/DDPM.

    σ(t) = √(2η_t) = √(ᾱ̇_t / ᾱ_t)

    This recovers the continuous-time limit of the DDPM generative process.
    """
    return torch.sqrt(alpha_dot_bar / alpha_bar)


def compute_control_flow_matching(
    v_finetune: torch.Tensor,
    v_base: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the control u from fine-tuned and base velocity fields.

    Equation (27) from the paper:
    u(x, t) = √(2/(β_t(α̇_t/α_t · β_t - β̇_t))) · (v^finetune(x,t) - v^base(x,t))

    With α_t = t, β_t = 1 - t:
    η_t = β_t(α̇_t/α_t · β_t - β̇_t) = (1-t)/t

    So:
    u(x, t) = √(2t/(1-t)) · (v^finetune(x,t) - v^base(x,t))
    """
    eta = compute_eta_flow_matching(t)
    # Add small epsilon for numerical stability
    return torch.sqrt(2.0 / (eta + 1e-8)) * (v_finetune - v_base)


def compute_control_ddim(
    epsilon_finetune: torch.Tensor,
    epsilon_base: torch.Tensor,
    alpha_bar: torch.Tensor,
    alpha_dot_bar: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the control u from fine-tuned and base epsilon predictors.

    Equation (26) from the paper:
    u(x, t) = -√(α̇̄_t / (ᾱ_t (1 - ᾱ_t))) · (ε^finetune - ε^base)

    This uses the memoryless noise schedule for DDIM.
    """
    coeff = torch.sqrt(alpha_dot_bar / (alpha_bar * (1.0 - alpha_bar) + 1e-8))
    return -coeff * (epsilon_finetune - epsilon_base)


def compute_value_function_bias(
    base_model_log_prob: Callable,
    reward_fn: Callable,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
) -> torch.Tensor:
    """
    Demonstrate the initial value function bias (Section 4.2).

    The naive SOC formulation gives:
    p*(X₀, X₁) = p^base(X₀, X₁) · exp(r(X₁) + V(X₀, 0))

    rather than the desired:
    p*(X₁) ∝ p^base(X₁) · exp(r(X₁))

    This function computes V(X₀, 0) - the bias term.
    """
    # V(x, 0) = -log E[exp(r(X₁)) | X₀ = x]
    # This is the negative log of the normalization constant
    reward = reward_fn(x_1)
    expected_exp_reward = torch.exp(reward).mean()
    V = -torch.log(expected_exp_reward + 1e-8)
    return V


def verify_memoryless_property(
    X_0: torch.Tensor,
    X_1: torch.Tensor,
    sigma_fn: Callable,
    num_samples: int = 10000,
) -> float:
    """
    Empirically verify that the memoryless noise schedule produces
    independent X₀ and X₁.

    Uses Hilbert-Schmidt Independence Criterion (HSIC) as a measure
    of dependence. Lower values indicate more independence.

    For a true memoryless process, HSIC → 0 as num_samples → ∞.
    """
    # Simple correlation-based independence test
    # Center the data
    X_0_c = X_0 - X_0.mean(dim=0, keepdim=True)
    X_1_c = X_1 - X_1.mean(dim=0, keepdim=True)

    # Compute cross-covariance Frobenius norm
    cross_cov = torch.mm(X_0_c.T, X_1_c) / (num_samples - 1)
    hsic = torch.sum(cross_cov ** 2)

    return hsic.item()


def lean_adjoint_ode_rhs(
    a_t: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    base_drift_fn: Callable,
    state_cost_fn: Callable,
) -> torch.Tensor:
    """
    Right-hand side of the lean adjoint ODE (Eqs. 38-39).

    d/dt ã(t; X) = -(ã(t; X)ᵀ ∇_x b(X_t, t) + ∇_x f(X_t, t))

    This is the key equation for the Adjoint Matching algorithm.
    The lean adjoint removes terms involving ∇_x||u||² that have
    zero expectation at the optimal control.

    Args:
        a_t: Current adjoint state ã(t).
        x_t: Current state X_t.
        t: Current time.
        base_drift_fn: Base drift function b(x, t).
        state_cost_fn: Running state cost f(x, t).

    Returns:
        dã/dt.
    """
    # For reward fine-tuning: f = 0, so ∇_x f = 0
    # The lean adjoint only depends on the base drift, not the control
    b = base_drift_fn(x_t, t)

    # Compute VJP: a_tᵀ ∇_x b(x_t, t)
    # This is the computationally expensive part
    vjp = torch.autograd.grad(
        b, x_t, grad_outputs=a_t, retain_graph=True, create_graph=True
    )[0]

    # Add ∇_x f (zero for reward fine-tuning)
    grad_f = state_cost_fn(x_t, t) if state_cost_fn is not None else 0.0

    return -(vjp + grad_f)


def full_adjoint_ode_rhs(
    a_t: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    base_drift_fn: Callable,
    control_fn: Callable,
    state_cost_fn: Callable,
    sigma_t: torch.Tensor,
) -> torch.Tensor:
    """
    Right-hand side of the full continuous adjoint ODE (Eq. 30-31).

    da/dt = -[aᵀ ∇_x(b + σu) + ∇_x(f + ½||u||²)]
    a(1) = ∇g(X₁)

    This is more expensive than the lean adjoint because it requires
    computing ∇_x u and ∇_x||u||².

    Args:
        a_t: Current adjoint state a(t).
        x_t: Current state X_t.
        t: Current time.
        base_drift_fn: Base drift function b(x, t).
        control_fn: Control function u(x, t).
        state_cost_fn: Running state cost f(x, t).
        sigma_t: Diffusion coefficient σ(t).

    Returns:
        da/dt.
    """
    u = control_fn(x_t, t)
    drift = base_drift_fn(x_t, t) + sigma_t * u

    # VJP: a_tᵀ ∇_x(b + σu)
    vjp_drift = torch.autograd.grad(
        drift, x_t, grad_outputs=a_t, retain_graph=True, create_graph=True
    )[0]

    # ∇_x(f + ½||u||²)
    cost = state_cost_fn(x_t, t) if state_cost_fn is not None else 0.0
    cost = cost + 0.5 * torch.sum(u ** 2, dim=-1)
    grad_cost = torch.autograd.grad(
        cost.sum(), x_t, retain_graph=True, create_graph=True
    )[0]

    return -(vjp_drift + grad_cost)
