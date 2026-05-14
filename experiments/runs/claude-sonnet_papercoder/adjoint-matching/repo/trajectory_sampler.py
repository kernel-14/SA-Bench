## trajectory_sampler.py
"""Trajectory sampling for Adjoint Matching fine-tuning experiments.

This module implements the TrajectorySampler class, which handles three
operations from the paper:

1. **Forward trajectory sampling** using the memoryless Flow Matching SDE
   (Algorithm 1, equation 40):
       X_{t+h} = X_t + h*(2*v_finetune(X_t,t) - κ_t*X_t) + sqrt(h)*σ(t)*ε

2. **Noiseless terminal state computation** for the lean adjoint terminal
   condition (Appendix G.1):
       X̂_1 = X_{1-h} + h*v_base(X_{1-h}, 1-h)  [no noise]

3. **Timestep subset selection** for gradient evaluation (Appendix G.2):
   - 10 random early timesteps from t ≤ 0.725
   - All late timesteps t ≥ 0.75 (last 10 steps for K=40)

Configuration alignment (config.yaml):
    sampling.K: 40                    → K timesteps
    sampling.h: 0.025                 → step size
    noise_schedule.offset: 0.025      → sigma offset
    timestep_subset.num_early_samples: 10
    timestep_subset.early_t_max: 0.725
    timestep_subset.late_t_min: 0.75
    model.num_train_timesteps: 1000   → UNet integer timestep range

Dependencies:
    - noise_schedule.py: NoiseSchedule (sigma_memoryless, kappa, h)
    - utils.py: get_unet_timestep (continuous t → integer UNet timestep)
    - torch, math, random, typing (standard)

No other project file dependencies.
"""

from __future__ import annotations

import logging
import math
import random
from typing import List, Optional

import torch
import torch.nn as nn

from noise_schedule import NoiseSchedule
from utils import get_unet_timestep

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (from config.yaml)
# ---------------------------------------------------------------------------

# Number of UNet training timesteps (config.yaml model.num_train_timesteps: 1000)
_NUM_TRAIN_TIMESTEPS: int = 1000

# Timestep subset boundaries (config.yaml timestep_subset section)
_EARLY_T_MAX: float = 0.725   # config.yaml timestep_subset.early_t_max
_LATE_T_MIN: float = 0.75     # config.yaml timestep_subset.late_t_min
_NUM_EARLY_SAMPLES: int = 10  # config.yaml timestep_subset.num_early_samples


class TrajectorySampler:
    """Samples trajectories using the memoryless Flow Matching SDE.

    Implements Algorithm 1 from the paper for forward trajectory sampling,
    noiseless terminal state computation, and timestep subset selection.

    The memoryless SDE (equation 40) is:
        X_{t+h} = X_t + h*(2*v_θ(X_t,t) - κ_t*X_t) + sqrt(h)*σ(t)*ε

    where:
        κ_t = 1/t  (for α_t = t)
        σ(t) = sqrt(2*(1-t+h)/(t+h))  (memoryless schedule with offset)
        ε ~ N(0, I)

    All trajectory tensors are detached (stop-gradient) after each step.
    Only v_θ(X_t, t) calls inside losses.py retain gradients.

    Attributes:
        noise_schedule: NoiseSchedule instance providing σ(t) and κ_t.
        device: PyTorch device string for tensor allocation.

    Example:
        >>> ns = NoiseSchedule(h=0.025)
        >>> sampler = TrajectorySampler(ns, device="cuda")
        >>> X0 = torch.randn(4, 4, 64, 64, device="cuda")
        >>> text_emb = torch.randn(4, 77, 768, device="cuda")
        >>> timesteps = ns.get_timesteps(K=40)
        >>> traj = sampler.sample_trajectory(v_theta, v_base, X0, timesteps, text_emb)
        >>> len(traj)  # K+1 = 41 states
        41
    """

    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        device: str = "cuda",
    ) -> None:
        """Initialize the trajectory sampler.

        Args:
            noise_schedule: NoiseSchedule instance providing σ(t), κ_t, and h.
                Sourced from config.yaml noise_schedule and sampling sections.
                Must have h = 1/K = 0.025 for K=40 (config.yaml sampling.h).
            device: PyTorch device string for tensor allocation.
                From config.yaml: training.device (inferred from num_gpus).
                Examples: "cuda", "cpu", "cuda:0", "cuda:1".
        """
        self.noise_schedule: NoiseSchedule = noise_schedule
        self.device: str = device

        logger.info(
            "TrajectorySampler initialized: device='%s', h=%.4f",
            device,
            noise_schedule.h,
        )

    # ------------------------------------------------------------------
    # Forward trajectory sampling (Algorithm 1, equation 40)
    # ------------------------------------------------------------------

    def sample_trajectory(
        self,
        v_theta: nn.Module,
        v_base: nn.Module,
        X0: torch.Tensor,
        timesteps: List[float],
        text_emb: torch.Tensor,
        uncond_emb: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Sample a trajectory using the memoryless Flow Matching SDE.

        Implements Algorithm 1, equation (40) from the paper:
            X_{t+h} = X_t + h*(2*v_θ(X_t,t) - κ_t*X_t) + sqrt(h)*σ(t)*ε

        The trajectory is sampled using v_theta (the fine-tuned model).
        v_base is accepted for API consistency but not used here — it is
        used in get_noiseless_terminal().

        All trajectory tensors are detached after each step (stop-gradient
        discipline). The gradient only flows through v_θ(X_t, t) calls
        inside losses.py, not through the trajectory itself.

        Args:
            v_theta: Fine-tuned velocity field (trainable UNet). Called as
                v_theta(latent, timestep_tensor, encoder_hidden_states=text_emb).
                Returns UNet2DConditionOutput with .sample attribute.
            v_base: Frozen base velocity field. Accepted for API consistency
                but not used in this method. Used in get_noiseless_terminal().
            X0: Initial noise tensor of shape (batch_size, C, H, W).
                Sampled from N(0, I) in trainer.py.
                For SD1.5: shape (B, 4, 64, 64).
            timesteps: List of K float timestep values [h, 2h, ..., 1.0].
                Generated by noise_schedule.get_timesteps(K=40).
                From config.yaml: sampling.K=40, sampling.h=0.025.
            text_emb: CLIP text embeddings of shape (batch_size, seq_len, hidden_dim).
                For SD1.5: shape (B, 77, 768).
                Passed as encoder_hidden_states to the UNet.
            uncond_emb: Optional unconditional text embeddings for CFG.
                Not used during trajectory sampling (CFG is applied at inference).
                Accepted for API completeness.

        Returns:
            List of K+1 detached tensors [X_0, X_h, X_{2h}, ..., X_{Kh=1}].
            Each tensor has the same shape as X0: (batch_size, C, H, W).
            All tensors are on self.device and detached from the autograd graph.

        Note:
            The UNet timestep conversion uses the diffusers reverse-time
            convention: timestep_int = int((1-t) * 1000).
            At t=0.025 (most noisy): timestep_int = 975.
            At t=0.975 (nearly clean): timestep_int = 25.
        """
        batch_size: int = X0.shape[0]
        h: float = self.noise_schedule.h

        # Initialize trajectory with the detached initial state
        X_t: torch.Tensor = X0.detach().to(self.device)
        trajectory: List[torch.Tensor] = [X_t]

        # All UNet calls are wrapped in no_grad — trajectory is stop-grad
        with torch.no_grad():
            for t in timesteps:
                # ----------------------------------------------------------
                # Step 1: Convert continuous t to UNet integer timestep.
                # Diffusers uses reverse-time convention:
                #   t=0 (noise) → timestep=1000 (high noise)
                #   t=1 (clean) → timestep=0 (clean)
                # get_unet_timestep handles clamping to [0, 999].
                # ----------------------------------------------------------
                timestep_int: int = get_unet_timestep(
                    t_continuous=t,
                    num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
                )
                # Create batch of identical timesteps: shape (batch_size,)
                timestep_tensor: torch.Tensor = torch.tensor(
                    [timestep_int] * batch_size,
                    dtype=torch.long,
                    device=self.device,
                )

                # ----------------------------------------------------------
                # Step 2: Get velocity prediction from fine-tuned model.
                # UNet call: (latent, timestep, encoder_hidden_states) → output
                # output.sample is the predicted velocity/noise field.
                # ----------------------------------------------------------
                unet_output = v_theta(
                    X_t,
                    timestep_tensor,
                    encoder_hidden_states=text_emb,
                    return_dict=True,
                )
                v_pred: torch.Tensor = unet_output.sample  # (B, C, H, W)

                # ----------------------------------------------------------
                # Step 3: Compute drift = 2*v_pred - κ_t * X_t
                # From eq. (27) and Algorithm 1 eq. (40):
                #   full_drift = 2*v_finetune(X_t, t) - (α̇_t/α_t)*X_t
                # For α_t = t: κ_t = α̇_t/α_t = 1/t
                # ----------------------------------------------------------
                kappa_t: float = self.noise_schedule.kappa(t)
                drift: torch.Tensor = 2.0 * v_pred - kappa_t * X_t

                # ----------------------------------------------------------
                # Step 4: Compute noise term = sqrt(h) * σ(t) * ε
                # σ(t) = sqrt(2*(1-t+h)/(t+h)) from Appendix G.1.
                # ε ~ N(0, I), same shape as X_t.
                # ----------------------------------------------------------
                sigma_t: float = self.noise_schedule.sigma_memoryless(
                    t=t,
                    h=h,
                )
                noise: torch.Tensor = torch.randn_like(X_t)
                noise_term: torch.Tensor = math.sqrt(h) * sigma_t * noise

                # ----------------------------------------------------------
                # Step 5: Euler-Maruyama update (eq. 40).
                # X_{t+h} = X_t + h*drift + sqrt(h)*σ(t)*ε
                # ----------------------------------------------------------
                X_next: torch.Tensor = X_t + h * drift + noise_term

                # ----------------------------------------------------------
                # Step 6: CRITICAL — detach to enforce stop-gradient.
                # The trajectory tensors must not carry gradient history.
                # Gradients only flow through v_θ(X_t, t) in losses.py.
                # ----------------------------------------------------------
                X_next = X_next.detach()
                trajectory.append(X_next)
                X_t = X_next

        # Verify trajectory length: should be K+1 states
        assert len(trajectory) == len(timesteps) + 1, (
            f"Expected {len(timesteps) + 1} trajectory states, "
            f"got {len(trajectory)}."
        )

        return trajectory

    # ------------------------------------------------------------------
    # Noiseless terminal state (Appendix G.1)
    # ------------------------------------------------------------------

    def get_noiseless_terminal(
        self,
        trajectory: List[torch.Tensor],
        v_base: nn.Module,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the noiseless terminal state X̂_1 for the lean adjoint.

        Implements the noiseless final step from Appendix G.1:
            X̂_1 = X_{1-h} + h * v_base(X_{1-h}, 1-h)

        This avoids gradient distortion from the noise injected in the final
        Euler-Maruyama step. In the continuous-time limit h → 0, X̂_1 = X_1,
        so this is consistent with the theory.

        The terminal condition for the lean adjoint ODE is:
            ã(1; X) = ∇_{X̂_1} g(X̂_1) = -∇_{X̂_1} r(X̂_1)

        Uses v_base (frozen) rather than v_theta, as the terminal condition
        is based on the base model's prediction of the clean image.

        Args:
            trajectory: List of K+1 tensors from sample_trajectory().
                trajectory[-2] = X_{1-h} (second-to-last state).
                trajectory[-1] = X_1 (noisy terminal, not used here).
            v_base: Frozen base velocity field (UNet with frozen parameters).
                Called as v_base(latent, timestep_tensor, encoder_hidden_states).
            text_emb: CLIP text embeddings of shape (batch_size, seq_len, hidden_dim).
                Same embeddings used in sample_trajectory().

        Returns:
            Noiseless terminal latent X̂_1 of shape (batch_size, C, H, W).
            Detached from the autograd graph. On self.device.

        Note:
            For K=40, h=0.025:
                t_last = 1.0 - 0.025 = 0.975
                timestep_int = int((1 - 0.975) * 1000) = int(25) = 25
        """
        h: float = self.noise_schedule.h

        # X_{1-h}: the second-to-last trajectory state (index -2)
        # trajectory[-1] is X_1 (noisy), trajectory[-2] is X_{1-h} (cleaner)
        X_last: torch.Tensor = trajectory[-2].detach()
        batch_size: int = X_last.shape[0]

        # Compute the timestep for t = 1 - h = 0.975 (for K=40)
        t_last: float = 1.0 - h
        timestep_int: int = get_unet_timestep(
            t_continuous=t_last,
            num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
        )
        timestep_tensor: torch.Tensor = torch.tensor(
            [timestep_int] * batch_size,
            dtype=torch.long,
            device=self.device,
        )

        # Get base model velocity at X_{1-h} (no gradient, frozen model)
        with torch.no_grad():
            unet_output = v_base(
                X_last,
                timestep_tensor,
                encoder_hidden_states=text_emb,
                return_dict=True,
            )
            v_pred_base: torch.Tensor = unet_output.sample  # (B, C, H, W)

        # Noiseless Euler step: X̂_1 = X_{1-h} + h * v_base(X_{1-h}, 1-h)
        # No noise term — this is the key difference from sample_trajectory
        X_hat_1: torch.Tensor = X_last + h * v_pred_base

        # Detach: this is a constant for the lean adjoint terminal condition
        return X_hat_1.detach()

    # ------------------------------------------------------------------
    # Timestep subset selection (Appendix G.2)
    # ------------------------------------------------------------------

    def select_timestep_subset(
        self,
        timesteps: List[float],
    ) -> List[float]:
        """Select the timestep subset for gradient evaluation.

        Implements Appendix G.2's strategy for selecting which timesteps
        to evaluate the Adjoint Matching loss at:

        1. Sample ``num_early_samples`` timesteps uniformly WITHOUT replacement
           from {t ∈ timesteps : t ≤ early_t_max} (early region).
        2. Always include ALL timesteps {t ∈ timesteps : t ≥ late_t_min}
           (late region — critical for image quality).
        3. Return the sorted union.

        Configuration (config.yaml timestep_subset section):
            num_early_samples: 10    → _NUM_EARLY_SAMPLES
            early_t_max: 0.725       → _EARLY_T_MAX
            late_t_min: 0.75         → _LATE_T_MIN

        For K=40 with h=0.025:
            - Early steps (t ≤ 0.725): [0.025, 0.050, ..., 0.725] = 29 steps
            - Late steps (t ≥ 0.75):   [0.750, 0.775, ..., 0.975] = 10 steps
              (t=1.0 is the terminal state, not included in loss computation)
            - Selected: 10 random early + 10 late = ~20 steps total

        The last 25% of timesteps (t ∈ [0.75, 1.0]) are always included
        because fine-tuning the final denoising steps is critical for
        image quality (Appendix G.2).

        Randomness: Uses Python's built-in random.sample(), which is seeded
        by utils.set_seed() for reproducibility. Each call during training
        produces a different subset, providing stochastic gradient estimation.

        Args:
            timesteps: Full list of K timestep values [h, 2h, ..., 1-h].
                Typically the same list passed to sample_trajectory(),
                excluding the terminal t=1.0 (which is not a loss evaluation
                point). Generated by noise_schedule.get_timesteps(K=40).

        Returns:
            Sorted list of selected timestep values. Length is approximately
            num_early_samples + len(late_steps), but may be less if there
            are fewer than num_early_samples early timesteps available.

        Note:
            The terminal timestep t=1.0 is excluded from the loss computation
            subset because the loss is evaluated at X_t for t < 1.0 (the
            trajectory states, not the terminal state itself).
        """
        # Partition timesteps into early and late regions
        # Exclude t=1.0 (terminal state) from both regions
        early_steps: List[float] = [
            t for t in timesteps
            if t <= _EARLY_T_MAX and t < 1.0
        ]
        late_steps: List[float] = [
            t for t in timesteps
            if t >= _LATE_T_MIN and t < 1.0
        ]

        # Sample from early steps without replacement
        # If fewer early steps than requested, use all of them
        k_early: int = min(_NUM_EARLY_SAMPLES, len(early_steps))
        if k_early > 0:
            selected_early: List[float] = random.sample(early_steps, k_early)
        else:
            selected_early = []
            logger.warning(
                "No early timesteps found with t <= %.3f. "
                "Timestep subset will only contain late steps.",
                _EARLY_T_MAX,
            )

        # Combine early (random subset) and late (all) steps
        combined: List[float] = selected_early + late_steps

        # Remove duplicates (in case early and late regions overlap)
        combined_unique: List[float] = list(set(combined))

        # Sort in ascending order for consistent processing in losses.py
        subset: List[float] = sorted(combined_unique)

        if len(subset) == 0:
            logger.warning(
                "Empty timestep subset selected. "
                "Check that timesteps list is non-empty and "
                "early_t_max/late_t_min boundaries are correct."
            )

        return subset

    def __repr__(self) -> str:
        """Human-readable representation of the trajectory sampler."""
        return (
            f"TrajectorySampler("
            f"device='{self.device}', "
            f"h={self.noise_schedule.h:.4f}, "
            f"early_t_max={_EARLY_T_MAX}, "
            f"late_t_min={_LATE_T_MIN}, "
            f"num_early_samples={_NUM_EARLY_SAMPLES}"
            f")"
        )
