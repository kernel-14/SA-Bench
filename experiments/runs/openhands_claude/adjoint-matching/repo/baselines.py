"""
Baseline fine-tuning methods for comparison with Adjoint Matching.

Implements:
1. DRaFT-1 and DRaFT-40 (Clark et al., 2024) - directly fine-tune with reward gradients
2. ReFL (Xu et al., 2023) - reward feedback learning via denoiser map
3. Diffusion-DPO (Wallace et al., 2023a) - direct preference optimization
4. Continuous Adjoint - SOC baseline using full adjoint ODE
5. Discrete Adjoint - SOC baseline using backprop through simulation

All baselines are adapted to the Flow Matching framework as described in
Appendix F and Section 7 of the paper.
"""

import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple

from noise_schedules import FlowMatchingSchedule
from losses import (
    draft_loss_fm,
    refl_loss_fm,
    dpo_loss_fm,
    continuous_adjoint_loss_fm,
    discrete_adjoint_loss_fm,
    adjoint_matching_loss_fm,
)


# ---------------------------------------------------------------------------
# Base fine-tuner class
# ---------------------------------------------------------------------------

class BaseFineTuner:
    """Base class for all fine-tuning methods."""

    def __init__(
        self,
        model: nn.Module,
        base_model: nn.Module,
        reward_fn: Callable,
        schedule: FlowMatchingSchedule,
        optimizer: torch.optim.Optimizer,
        reward_lambda: float = 1.0,
        device: torch.device = None,
    ):
        self.model = model
        self.base_model = base_model
        self.reward_fn = reward_fn
        self.schedule = schedule
        self.optimizer = optimizer
        self.reward_lambda = reward_lambda
        self.device = device or next(model.parameters()).device

        # Freeze base model
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        self.base_model.eval()

    def velocity_fn(self, x, t, text_emb):
        return self.model(x, t, text_emb)

    def base_velocity_fn(self, x, t, text_emb):
        with torch.no_grad():
            return self.base_model(x, t, text_emb)

    def compute_loss(self, x0, text_embeddings) -> torch.Tensor:
        raise NotImplementedError

    def step(self, x0: torch.Tensor, text_embeddings: Optional[torch.Tensor] = None) -> Dict:
        self.optimizer.zero_grad()
        loss = self.compute_loss(x0, text_embeddings)
        loss.backward()
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {"loss": loss.item()}


# ---------------------------------------------------------------------------
# DRaFT-K (Clark et al., 2024)
# ---------------------------------------------------------------------------

class DRaFTFineTuner(BaseFineTuner):
    """
    DRaFT-K: Directly fine-tune diffusion models on differentiable rewards.

    Backpropagates reward gradient through the last K denoising steps.
    DRaFT-1: K=1 (only last step), DRaFT-40: K=40 (all steps).

    From the paper (Section 7): DRaFT-1 performs best among baselines.
    Uses heuristic gradient stopping to stay close to base model.
    """

    def __init__(self, *args, num_grad_steps: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_grad_steps = num_grad_steps

    def compute_loss(self, x0, text_embeddings=None):
        return draft_loss_fm(
            v_finetune_fn=self.velocity_fn,
            reward_fn=self.reward_fn,
            x0=x0,
            schedule=self.schedule,
            text_embeddings=text_embeddings,
            reward_lambda=self.reward_lambda,
            num_grad_steps=self.num_grad_steps,
        )


# ---------------------------------------------------------------------------
# ReFL (Xu et al., 2023)
# ---------------------------------------------------------------------------

class ReFLFineTuner(BaseFineTuner):
    """
    ReFL: Reward Feedback Learning.

    Uses the denoiser map to compute a clean image prediction at each step,
    then optimizes the reward on this prediction.

    Adapted to Flow Matching via the denoiser map (Appendix F.1):
      X_hat_1(x, t) = (1-t)*v(x,t) + x
    """

    def compute_loss(self, x0, text_embeddings=None):
        return refl_loss_fm(
            v_finetune_fn=self.velocity_fn,
            reward_fn=self.reward_fn,
            x0=x0,
            schedule=self.schedule,
            text_embeddings=text_embeddings,
            reward_lambda=self.reward_lambda,
        )


# ---------------------------------------------------------------------------
# Diffusion-DPO (Wallace et al., 2023a)
# ---------------------------------------------------------------------------

class DPOFineTuner(BaseFineTuner):
    """
    Diffusion-DPO adapted for Flow Matching (Appendix F.2).

    Uses ranked pairs of generated samples to fine-tune the model.
    When a reward model is available, uses soft preference weights.

    Note: As discussed in Appendix F.2, DPO with on-policy samples
    (generated from current model) performs similarly to the base model.
    """

    def __init__(self, *args, beta_tilde: float = 5000.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_tilde = beta_tilde

    def compute_loss(self, x0, text_embeddings=None):
        # Generate two sets of samples for preference pairs
        B = x0.shape[0] // 2
        x0_a, x0_b = x0[:B], x0[B:]

        # Generate samples from current model
        with torch.no_grad():
            x = x0_a.clone()
            for i in range(self.schedule.K):
                t_val = self.schedule.timesteps[i]
                t = torch.full((x.shape[0],), t_val.item(),
                               device=x.device, dtype=x.dtype)
                v = self.velocity_fn(x, t, text_embeddings[:B] if text_embeddings is not None else None)
                kappa = self.schedule.kappa(t).view(-1, *([1] * (x.dim() - 1)))
                sigma = self.schedule.sigma_memoryless(t).view(-1, *([1] * (x.dim() - 1)))
                drift = 2.0 * v - kappa * x
                noise = torch.randn_like(x)
                x = x + self.schedule.h * drift + (self.schedule.h ** 0.5) * sigma * noise
            x1_a = x

            x = x0_b.clone()
            for i in range(self.schedule.K):
                t_val = self.schedule.timesteps[i]
                t = torch.full((x.shape[0],), t_val.item(),
                               device=x.device, dtype=x.dtype)
                v = self.velocity_fn(x, t, text_embeddings[B:] if text_embeddings is not None else None)
                kappa = self.schedule.kappa(t).view(-1, *([1] * (x.dim() - 1)))
                sigma = self.schedule.sigma_memoryless(t).view(-1, *([1] * (x.dim() - 1)))
                drift = 2.0 * v - kappa * x
                noise = torch.randn_like(x)
                x = x + self.schedule.h * drift + (self.schedule.h ** 0.5) * sigma * noise
            x1_b = x

        return dpo_loss_fm(
            v_finetune_fn=self.velocity_fn,
            v_ref_fn=self.base_velocity_fn,
            x1_win=x1_a,
            x1_lose=x1_b,
            schedule=self.schedule,
            beta_tilde=self.beta_tilde,
            text_embeddings=text_embeddings[:B] if text_embeddings is not None else None,
            reward_fn=self.reward_fn,
        )


# ---------------------------------------------------------------------------
# Continuous Adjoint (Section 5.1.1)
# ---------------------------------------------------------------------------

class ContinuousAdjointFineTuner(BaseFineTuner):
    """
    Continuous Adjoint method for SOC.

    Uses the full adjoint ODE (Eq. 30-31) to compute gradients.
    Includes extra terms compared to Adjoint Matching (lean adjoint).
    """

    def __init__(self, *args, lct_factor: float = 1600.0,
                 grad_timestep_selector=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lct_factor = lct_factor
        self.grad_timestep_selector = grad_timestep_selector

    def compute_loss(self, x0, text_embeddings=None):
        lct = self.lct_factor * (self.reward_lambda ** 2)
        grad_indices = None
        if self.grad_timestep_selector is not None:
            grad_indices = self.grad_timestep_selector(x0.device)

        return continuous_adjoint_loss_fm(
            v_finetune_fn=self.velocity_fn,
            v_base_fn=self.base_velocity_fn,
            reward_fn=self.reward_fn,
            x0=x0,
            schedule=self.schedule,
            text_embeddings=text_embeddings,
            reward_lambda=self.reward_lambda,
            grad_timestep_indices=grad_indices,
            lct=lct,
        )


# ---------------------------------------------------------------------------
# Discrete Adjoint (Section 5.1.1)
# ---------------------------------------------------------------------------

class DiscreteAdjointFineTuner(BaseFineTuner):
    """
    Discrete Adjoint method: differentiate through the SDE simulation.

    Uses "discretize-then-differentiate" approach.
    Requires gradient checkpointing for memory efficiency.
    Note: Paper reports instability with default hyperparameters;
    uses lower learning rate (1e-5) for stable training (Table 6).
    """

    def compute_loss(self, x0, text_embeddings=None):
        return discrete_adjoint_loss_fm(
            v_finetune_fn=self.velocity_fn,
            v_base_fn=self.base_velocity_fn,
            reward_fn=self.reward_fn,
            x0=x0,
            schedule=self.schedule,
            text_embeddings=text_embeddings,
            reward_lambda=self.reward_lambda,
        )


# ---------------------------------------------------------------------------
# Adjoint Matching (proposed method, Section 5.2)
# ---------------------------------------------------------------------------

class AdjointMatchingFineTuner(BaseFineTuner):
    """
    Adjoint Matching: proposed method for SOC-based reward fine-tuning.

    Key features:
    1. Uses memoryless noise schedule (Theorem 1) for unbiased fine-tuning
    2. Lean adjoint ODE (Eq. 38-39) for lower variance gradients
    3. Loss clipping threshold LCT (Appendix G.3)
    4. Selective gradient timesteps (Appendix G.2)
    """

    def __init__(self, *args, lct_factor: float = 1.6,
                 grad_timestep_selector=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lct_factor = lct_factor
        self.grad_timestep_selector = grad_timestep_selector

    def compute_loss(self, x0, text_embeddings=None):
        lct = self.lct_factor * (self.reward_lambda ** 2)
        grad_indices = None
        if self.grad_timestep_selector is not None:
            grad_indices = self.grad_timestep_selector(x0.device)

        return adjoint_matching_loss_fm(
            v_finetune_fn=self.velocity_fn,
            v_base_fn=self.base_velocity_fn,
            reward_fn=self.reward_fn,
            x0=x0,
            schedule=self.schedule,
            text_embeddings=text_embeddings,
            reward_lambda=self.reward_lambda,
            grad_timestep_indices=grad_indices,
            lct=lct,
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_finetuner(
    method: str,
    model: nn.Module,
    base_model: nn.Module,
    reward_fn: Callable,
    schedule: FlowMatchingSchedule,
    optimizer: torch.optim.Optimizer,
    reward_lambda: float = 1.0,
    device: torch.device = None,
    **kwargs,
) -> BaseFineTuner:
    """
    Build a fine-tuner for the specified method.

    Args:
        method: One of "adjoint_matching", "cont_adjoint", "disc_adjoint",
                "draft_1", "draft_40", "refl", "dpo"
        model: Fine-tuned model (with gradients)
        base_model: Frozen base model
        reward_fn: Reward function r(x) -> [B]
        schedule: FlowMatchingSchedule
        optimizer: Optimizer for model parameters
        reward_lambda: Reward scaling factor lambda
        device: Target device
        **kwargs: Method-specific arguments

    Returns:
        Fine-tuner instance
    """
    common_args = dict(
        model=model,
        base_model=base_model,
        reward_fn=reward_fn,
        schedule=schedule,
        optimizer=optimizer,
        reward_lambda=reward_lambda,
        device=device,
    )

    if method == "adjoint_matching":
        return AdjointMatchingFineTuner(
            **common_args,
            lct_factor=kwargs.get("lct_factor", 1.6),
            grad_timestep_selector=kwargs.get("grad_timestep_selector",
                                              schedule.select_grad_timesteps),
        )
    elif method == "cont_adjoint":
        return ContinuousAdjointFineTuner(
            **common_args,
            lct_factor=kwargs.get("lct_factor", 1600.0),
            grad_timestep_selector=kwargs.get("grad_timestep_selector",
                                              schedule.select_grad_timesteps),
        )
    elif method == "disc_adjoint":
        return DiscreteAdjointFineTuner(**common_args)
    elif method == "draft_1":
        return DRaFTFineTuner(**common_args, num_grad_steps=1)
    elif method == "draft_40":
        return DRaFTFineTuner(**common_args, num_grad_steps=40)
    elif method == "refl":
        return ReFLFineTuner(**common_args)
    elif method == "dpo":
        return DPOFineTuner(
            **common_args,
            beta_tilde=kwargs.get("beta_tilde", 5000.0),
        )
    else:
        raise ValueError(f"Unknown fine-tuning method: {method}")
