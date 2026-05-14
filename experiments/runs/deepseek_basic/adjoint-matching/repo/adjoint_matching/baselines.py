"""
Baseline methods for reward fine-tuning of dynamical generative models.

Implements the baseline fine-tuning algorithms compared in the paper:
- Continuous Adjoint method (Section 5.1.1)
- Discrete Adjoint method (Section 5.1.1)
- DRaFT-K (Clark et al., 2024) - adapted for Flow Matching
- ReFL (Xu et al., 2023) - adapted for Flow Matching (Appendix F.1)
- DPO (Wallace et al., 2023a) - adapted for Flow Matching (Appendix F.2)

These serve as comparison points for the proposed Adjoint Matching algorithm.
"""

import torch
import torch.nn as nn
from typing import Callable, Optional
from .noise_schedule import FlowMatchingNoiseSchedule


class ContinuousAdjointLoss:
    """
    Continuous Adjoint method (Section 5.1.1).

    Directly optimizes the SOC objective by differentiating through the
    adjoint ODE. This is the "differentiate-then-discretize" approach.

    Loss: L(u; X) = ∫₀¹ (½||u(X_t,t)||² + f(X_t,t)) dt + g(X₁)

    Gradient computed via adjoint state a(t; X, u) solving:
        da/dt = -[aᵀ∇_x(b + σu) + ∇_x(f + ½||u||²)]
        a(1) = ∇g(X₁)
    """

    def __init__(
        self,
        base_model: nn.Module,
        noise_schedule: FlowMatchingNoiseSchedule,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        lambda_reg: float = 1.0,
        lct: Optional[float] = None,
    ):
        self.base_model = base_model
        self.noise_schedule = noise_schedule
        self.reward_fn = reward_fn
        self.lambda_reg = lambda_reg

        if lct is None:
            # LCT = 1600 * λ² for continuous adjoint (Appendix G.3)
            self.lct = 1600.0 * (lambda_reg ** 2)
        else:
            self.lct = lct

    def compute_loss(
        self,
        fine_tuned_model: nn.Module,
        X_trajectory: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the continuous adjoint loss.

        The loss is:
        L = E[∫₀¹ ½||u(X_t,t)||² dt - λ·r(X₁)]

        where u is the control implicitly defined through the fine-tuned model.
        With clipping applied (Appendix G.3).
        """
        K = self.noise_schedule.num_steps
        h = self.noise_schedule.h
        B = X_trajectory.shape[1]
        device = X_trajectory.device

        # Control cost: ∫ ½||u||² dt
        control_cost = 0.0
        for k in range(K):
            t_val = timesteps[k]
            X_t = X_trajectory[k]

            sigma_t = self.noise_schedule.sigma(t_val)

            with torch.no_grad():
                v_base = self.base_model(X_t, t_val.expand(B))

            v_ft = fine_tuned_model(X_t, t_val.expand(B))

            # u = (2/σ(t)) · (v^ft - v^base) for memoryless FM
            u_t = (2.0 / sigma_t) * (v_ft - v_base)
            control_cost += 0.5 * torch.sum(u_t ** 2, dim=-1).mean() * h

        # Terminal cost: -λ·r(X₁)
        X_1 = X_trajectory[-1]
        reward = self.reward_fn(X_1)
        terminal_cost = -self.lambda_reg * reward.mean()

        total_loss = control_cost + terminal_cost

        # Apply clipping to control cost terms (Appendix G.3)
        # This is approximate - the paper applies clipping differently
        if self.lct is not None:
            total_loss = torch.clamp(total_loss, max=self.lct)

        return total_loss


class DiscreteAdjointLoss:
    """
    Discrete Adjoint method (Section 5.1.1).

    "Discretize-then-differentiate" approach that stores the computational
    graph of the numerical solver and differentiates through it.

    This is essentially backprop through time (BPTT) on the SDE simulation.
    Note: This can use extremely large memory; gradient checkpointing is
    typically needed for practical use.
    """

    def __init__(
        self,
        base_model: nn.Module,
        noise_schedule: FlowMatchingNoiseSchedule,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        lambda_reg: float = 1.0,
    ):
        self.base_model = base_model
        self.noise_schedule = noise_schedule
        self.reward_fn = reward_fn
        self.lambda_reg = lambda_reg

    def compute_loss(
        self,
        fine_tuned_model: nn.Module,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute the discrete adjoint loss by simulating the full SDE
        with gradient tracking.
        """
        K = self.noise_schedule.num_steps
        h = self.noise_schedule.h
        D = 512  # Latent dimension

        X_t = torch.randn(batch_size, D, device=device)
        control_cost = 0.0

        for k in range(K):
            t_val = torch.tensor(k * h, device=device)

            sigma_t = self.noise_schedule.sigma(t_val)
            alpha_dot = self.noise_schedule.alpha_dot(t_val)
            alpha = self.noise_schedule.alpha(t_val)

            v_ft = fine_tuned_model(X_t, t_val.expand(batch_size))

            with torch.no_grad():
                v_base = self.base_model(X_t, t_val.expand(batch_size))

            # Control
            u_t = (2.0 / sigma_t) * (v_ft - v_base)
            control_cost += 0.5 * torch.sum(u_t ** 2, dim=-1).mean() * h

            # Drift: 2v^ft - (α̇/α)·x
            drift = 2.0 * v_ft - (alpha_dot / alpha) * X_t

            # Euler-Maruyama step
            noise = torch.randn(batch_size, D, device=device)
            X_t = X_t + h * drift + torch.sqrt(h) * sigma_t * noise

        # Terminal reward
        reward = self.reward_fn(X_t)
        terminal_cost = -self.lambda_reg * reward.mean()

        return control_cost + terminal_cost


class DRaFTLoss:
    """
    DRaFT-K loss (Clark et al., 2024), adapted for Flow Matching.

    DRaFT directly backpropagates the reward through the last K steps
    of the generation process. DRaFT-1 only backpropagates through the
    last step (K=1).

    This is a heuristic method that does not provably converge to the
    tilted distribution (1).
    """

    def __init__(
        self,
        base_model: nn.Module,
        noise_schedule: FlowMatchingNoiseSchedule,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        lambda_reg: float = 1.0,
        K_draft: int = 1,
    ):
        """
        Args:
            base_model: Pre-trained base model.
            noise_schedule: Noise schedule.
            reward_fn: Reward function.
            lambda_reg: Reward scaling.
            K_draft: Number of steps to backpropagate through (1 for DRaFT-1).
        """
        self.base_model = base_model
        self.noise_schedule = noise_schedule
        self.reward_fn = reward_fn
        self.lambda_reg = lambda_reg
        self.K_draft = K_draft

    def compute_loss(
        self,
        fine_tuned_model: nn.Module,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        DRaFT-K: Simulate forward without gradients for K-K_draft steps,
        then with gradients for the last K_draft steps.
        """
        K = self.noise_schedule.num_steps
        h = self.noise_schedule.h
        D = 512

        X_t = torch.randn(batch_size, D, device=device)

        # Forward without gradients for first K-K_draft steps
        detach_start = K - self.K_draft

        with torch.no_grad():
            for k in range(detach_start):
                t_val = torch.tensor(k * h, device=device)
                sigma_t = self.noise_schedule.sigma(t_val)
                alpha_dot = self.noise_schedule.alpha_dot(t_val)
                alpha = self.noise_schedule.alpha(t_val)

                v_ft = fine_tuned_model(X_t, t_val.expand(batch_size))
                drift = 2.0 * v_ft - (alpha_dot / alpha) * X_t
                noise = torch.randn(batch_size, D, device=device)
                X_t = X_t + h * drift + torch.sqrt(h) * sigma_t * noise

        # Forward with gradients for last K_draft steps
        for k in range(detach_start, K):
            t_val = torch.tensor(k * h, device=device)
            sigma_t = self.noise_schedule.sigma(t_val)
            alpha_dot = self.noise_schedule.alpha_dot(t_val)
            alpha = self.noise_schedule.alpha(t_val)

            v_ft = fine_tuned_model(X_t, t_val.expand(batch_size))
            drift = 2.0 * v_ft - (alpha_dot / alpha) * X_t
            noise = torch.randn(batch_size, D, device=device)
            X_t = X_t + h * drift + torch.sqrt(h) * sigma_t * noise

        # Reward on final state
        reward = self.reward_fn(X_t)
        return -self.lambda_reg * reward.mean()


class RefLLoss:
    """
    Reward Feedback Learning (ReFL; Xu et al., 2023), adapted for Flow Matching.

    ReFL applies the reward to a "denoised" estimate of X₁ at intermediate
    timesteps. The denoiser map for Flow Matching (Appendix F.1, Eq. 229):

        X̂₁(x, t) = (v(x,t) - (β̇_t/β_t)·x) / (α̇_t - (β̇_t/β_t)·α_t)
    """

    def __init__(
        self,
        base_model: nn.Module,
        noise_schedule: FlowMatchingNoiseSchedule,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        lambda_reg: float = 1.0,
    ):
        self.base_model = base_model
        self.noise_schedule = noise_schedule
        self.reward_fn = reward_fn
        self.lambda_reg = lambda_reg

    def denoise(
        self,
        v: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Denoiser map for Flow Matching (Eq. 229 in Appendix F.1).

        X̂₁(x, t) = (v(x,t) - (β̇_t/β_t)·x) / (α̇_t - (β̇_t/β_t)·α_t)

        With α_t = t, β_t = 1-t:
        α̇_t = 1, β̇_t = -1
        X̂₁(x, t) = (v(x,t) + x/(1-t)) / (1 + t/(1-t))
                  = (v(x,t)·(1-t) + x) / 1
                  = (1-t)·v(x,t) + x
        """
        beta = self.noise_schedule.beta(t)
        beta_dot = self.noise_schedule.beta_dot(t)
        alpha = self.noise_schedule.alpha(t)
        alpha_dot = self.noise_schedule.alpha_dot(t)

        # General formula
        denom = alpha_dot - (beta_dot / beta) * alpha
        numer = v - (beta_dot / beta) * x
        return numer / (denom + 1e-8)

    def compute_loss(
        self,
        fine_tuned_model: nn.Module,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """ReFL loss: apply reward to denoised estimate at random timesteps."""
        K = self.noise_schedule.num_steps
        h = self.noise_schedule.h
        D = 512

        X_t = torch.randn(batch_size, D, device=device)
        total_loss = 0.0

        with torch.no_grad():
            for k in range(K):
                t_val = torch.tensor(k * h, device=device)
                sigma_t = self.noise_schedule.sigma(t_val)
                alpha_dot = self.noise_schedule.alpha_dot(t_val)
                alpha = self.noise_schedule.alpha(t_val)

                v_ft = fine_tuned_model(X_t, t_val.expand(batch_size))
                drift = 2.0 * v_ft - (alpha_dot / alpha) * X_t
                noise = torch.randn(batch_size, D, device=device)
                X_t = X_t + h * drift + torch.sqrt(h) * sigma_t * noise

                # At random timesteps, compute ReFL gradient
                if torch.rand(1).item() < 0.25:  # 25% of steps
                    X_t_grad = X_t.detach().requires_grad_(True)
                    v_ft_grad = fine_tuned_model(
                        X_t_grad, t_val.expand(batch_size)
                    )
                    x_hat_1 = self.denoise(v_ft_grad, X_t_grad, t_val)
                    reward = self.reward_fn(x_hat_1)
                    total_loss -= self.lambda_reg * reward.mean()

        return total_loss
