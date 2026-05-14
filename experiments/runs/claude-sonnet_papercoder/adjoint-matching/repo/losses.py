```python
## losses.py
"""Loss functions for Adjoint Matching and Continuous Adjoint fine-tuning.

This module implements the core loss functions from the paper:

1. **AdjointMatchingLoss.compute()** — the Adjoint Matching objective from
   Algorithm 1, equation (42) and the clipped variant from Appendix G.3:

       L_AdjMatch(θ) = Σ_{t∈κ} min{ LCT, ||(2/σ(t))*(v_θ-v_base) + σ(t)*ã_t||² }

   where LCT = 1.6 * λ² (config.yaml loss.lct_constant: 1.6).

2. **continuous_adjoint_loss()** — the Continuous Adjoint baseline from
   equation (32), using the full adjoint state with LCT = 1600 * λ²
   (config.yaml loss.lct_constant_cont_adjoint: 1600.0).

Mathematical background:
    The Adjoint Matching loss casts SOC as a regression problem (Section 5.2).
    The unique critical point of E[L_AdjMatch] is the optimal control u*
    (Proposition 7). The lean adjoint ã removes control-dependent terms from
    the full adjoint ODE, reducing variance while preserving the critical point.

Configuration alignment (config.yaml):
    loss.lct_constant: 1.6              → LCT = 1.6 * λ² for AdjointMatching
    loss.lct_constant_cont_adjoint: 1600.0 → LCT = 1600 * λ² for ContAdjoint
    sampling.K: 40                      → K timesteps
    sampling.h: 0.025                   → step size (sigma offset)
    model.num_train_timesteps: 1000     → UNet integer timestep range

Dependencies:
    - noise_schedule.py: NoiseSchedule (sigma_memoryless, h)
    - utils.py: get_unet_timestep (continuous t → diffusers int timestep)
    - torch, typing (standard)

No other project file dependencies.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

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

# Minimum sigma value to prevent division by zero in (2/sigma_t)
_SIGMA_MIN: float = 1e-6


class AdjointMatchingLoss:
    """Computes the Adjoint Matching loss for reward fine-tuning.

    Implements Algorithm 1, equation (42) from the paper with the loss
    clipping scheme from Appendix G.3:

        L_AdjMatch(θ) = Σ_{t∈κ} min{ LCT, ||(2/σ(t))*(v_θ-v_base) + σ(t)*ã_t||² }

    where:
        σ(t) = sqrt(2*(1-t+h)/(t+h))  [memoryless schedule, Appendix G.1]
        ã_t = lean adjoint state at time t [equations 38-39]
        LCT = 1.6 * λ²  [loss clipping threshold, Appendix G.3]
        κ = timestep subset [Appendix G.2: 10 early + 10 late steps]

    The gradient flows ONLY through v_θ(X_t, t). All other quantities
    (X_t, ã_t, v_base) are detached constants.

    Gradient flow:
        v_theta.parameters()
            → v_theta(X_t.detach(), t_int, text_emb)  [gradient here]
            → (2/σ)*(v_ft - v_bs.detach()) + σ*ã_t.detach()
            → term.pow(2).flatten(1).sum(1)
            → clamp(max=lct)
            → mean()  →  scalar loss  →  backward()

    Attributes:
        noise_schedule: NoiseSchedule instance providing σ(t) and h.

    Example:
        >>> ns = NoiseSchedule(h=0.025)
        >>> loss_fn = AdjointMatchingLoss(ns)
        >>> loss = loss_fn.compute(
        ...     v_theta, v_base, trajectory, adjoints,
        ...     timestep_subset, text_emb, lct=1.6e8
        ... )
        >>> loss.backward()
    """

    def __init__(self, noise_schedule: NoiseSchedule) -> None:
        """Initialize the Adjoint Matching loss.

        Args:
            noise_schedule: NoiseSchedule instance providing sigma_memoryless()
                and h. Sourced from config.yaml noise_schedule and sampling
                sections. Must have h = 1/K = 0.025 for K=40.
        """
        self.noise_schedule: NoiseSchedule = noise_schedule

        logger.info(
            "AdjointMatchingLoss initialized: h=%.4f",
            noise_schedule.h,
        )

    def compute(
        self,
        v_theta: nn.Module,
        v_base: nn.Module,
        trajectory: List[torch.Tensor],
        adjoints: Dict[float, torch.Tensor],
        timestep_subset: List[float],
        text_emb: torch.Tensor,
        lct: float = 1.6e8,
    ) -> torch.Tensor:
        """Compute the clipped Adjoint Matching loss.

        Implements Algorithm 1, equation (42) with clipping (Appendix G.3):

            L̂_AdjMatch(θ) = Σ_{t∈κ} min{ LCT, ||(2/σ(t))*(v_θ-v_base) + σ(t)*ã_t||² }

        The loss is averaged over both the batch dimension and the timestep
        subset, giving a stable scalar regardless of batch size or subset size.

        Args:
            v_theta: Fine-tuned velocity field (trainable UNet). Called as
                v_theta(latent, timestep_tensor, encoder_hidden_states=text_emb).
                Returns UNet2DConditionOutput with .sample attribute of shape
                (batch_size, 4, 64, 64) for SD1.5.
                MUST have requires_grad=True for its parameters.
            v_base: Frozen base velocity field (UNet with frozen parameters).
                Called with the same signature as v_theta.
                Parameters must have requires_grad=False.
            trajectory: List of K+1 detached tensors from
                TrajectorySampler.sample_trajectory(). trajectory[i]
                corresponds to the state at timesteps[i] for i in [0, K-1].
                All tensors have shape (batch_size, C, H, W) = (B, 4, 64, 64).
                All tensors are already detached (stop-gradient).
            adjoints: Dict[float, Tensor] from LeanAdjointSolver.solve().
                Maps each timestep t to the lean adjoint state ã_t of shape
                (batch_size, C, H, W). All tensors are already detached.
            timestep_subset: List of float timestep values selected by
                TrajectorySampler.select_timestep_subset(). Typically ~20
                values for K=40 (10 early + 10 late, Appendix G.2).
            text_emb: CLIP text embeddings of shape (batch_size, seq_len, hidden_dim).
                For SD1.5: shape (B, 77, 768). Passed as encoder_hidden_states.
                Should be detached (frozen text encoder output).
            lct: Loss Clipping Threshold from config.yaml.
                For Adjoint Matching: lct = lct_constant * λ² = 1.6 * λ².
                For λ=12500: lct = 1.6 * 12500² = 2.5e8.
                For λ=1000:  lct = 1.6 * 1000² = 1.6e6.
                From config.py: config.lct = config.lct_constant * config.lambda_reward².

        Returns:
            Scalar tensor with gradient graph through v_theta.parameters() only.
            Value is the mean clipped squared norm over the timestep subset
            and batch. Returns zero tensor (no gradient) if timestep_subset
            is empty or no valid timesteps are found in adjoints.

        Note:
            The UNet timestep conversion uses diffusers reverse-time convention:
                timestep_int = int((1 - t) * 1000)
            At t=0.025 (most noisy): timestep_int = 975
            At t=0.975 (nearly clean): timestep_int = 25

            The trajectory index for timestep t is determined by finding t
            in the full timestep list. Since trajectory[0] = X_0 (initial
            noise) and trajectory[i] = X_{t_i} for i ≥ 1, the index for
            timestep t_i is i (0-indexed in the trajectory list, where
            trajectory[0] is X_0 and trajectory[1] is X_{t_1=h}).
        """
        if len(timestep_subset) == 0:
            logger.warning(
                "AdjointMatchingLoss.compute: empty timestep_subset. "
                "Returning zero loss."
            )
            # Return a zero tensor that participates in the computation graph
            # so that .backward() doesn't fail
            dummy_param = next(v_theta.parameters())
            return dummy_param.sum() * 0.0

        h: float = self.noise_schedule.h
        device: torch.device = text_emb.device
        batch_size: int = trajectory[0].shape[0]

        # Build a mapping from timestep value to trajectory index.
        # trajectory[0] = X_0 (initial noise, not a loss evaluation point)
        # trajectory[i] = X_{t_{i-1}} for i in [1, K]
        # where t_{i-1} is the (i-1)-th element of the full timestep list.
        # However, TrajectorySampler stores trajectory as:
        #   trajectory[0] = X_0
        #   trajectory[1] = X_{h}   (after first step at t=h)
        #   trajectory[i] = X_{t_i} where t_i = i * h
        # So trajectory[i] corresponds to timestep t_i = i * h for i >= 1,
        # and trajectory[0] = X_0 corresponds to t=0 (not in timestep list).
        # The timestep list is [h, 2h, ..., Kh=1.0] with K elements.
        # trajectory[i] = X_{timesteps[i-1]} for i in [1, K+1].
        # Therefore: for timestep t = timesteps[j], trajectory index = j + 1.
        # We build this mapping dynamically from the trajectory length.
        K: int = len(trajectory) - 1  # Number of steps = K
        # Reconstruct the full timestep list to build the index mapping
        # (we need to find the index of each t in timestep_subset)
        full_timesteps: List[float] = self.noise_schedule.get_timesteps(K=K)
        # Build t -> trajectory_index mapping
        # trajectory[0] = X_0, trajectory[i] = X_{full_timesteps[i-1]}
        t_to_traj_idx: Dict[float, int] = {}
        for j, t_val in enumerate(full_timesteps):
            t_to_traj_idx[t_val] = j + 1  # trajectory index = j + 1

        # Accumulate loss over timestep subset
        total_loss: torch.Tensor = torch.zeros(
            1, dtype=torch.float32, device=device
        )
        count: int = 0

        for t in timestep_subset:
            # ------------------------------------------------------------------
            # Step 1: Retrieve trajectory state X_t (stop-gradient).
            # Find the trajectory index for this timestep.
            # ------------------------------------------------------------------
            traj_idx: Optional[int] = _find_trajectory_index(
                t, t_to_traj_idx, full_timesteps
            )
            if traj_idx is None:
                logger.warning(
                    "AdjointMatchingLoss.compute: timestep t=%.4f not found "
                    "in trajectory index map. Skipping.",
                    t,
                )
                continue

            if traj_idx >= len(trajectory):
                logger.warning(
                    "AdjointMatchingLoss.compute: trajectory index %d out of "
                    "bounds (trajectory length %d) for t=%.4f. Skipping.",
                    traj_idx,
                    len(trajectory),
                    t,
                )
                continue

            # X_t: (B, C, H, W), detached (stop-gradient)
            X_t: torch.Tensor = trajectory[traj_idx].detach().to(device)

            # ------------------------------------------------------------------
            # Step 2: Retrieve lean adjoint state ã_t (stop-gradient).
            # ------------------------------------------------------------------
            if t not in adjoints:
                logger.warning(
                    "AdjointMatchingLoss.compute: adjoint for t=%.4f not found "
                    "in adjoints dict. Skipping.",
                    t,
                )
                continue

            a_tilde: torch.Tensor = adjoints[t].detach().to(device)

            # ------------------------------------------------------------------
            # Step 3: Compute sigma_t = sqrt(2*(1-t+h)/(t+h)).
            # Uses the practical offset from Appendix G.1.
            # ------------------------------------------------------------------
            sigma_t_float: float = self.noise_schedule.sigma_memoryless(
                t=t, h=h
            )
            # Clamp to prevent division by zero in (2/sigma_t)
            sigma_t_float = max(sigma_t_float, _SIGMA_MIN)

            # ------------------------------------------------------------------
            # Step 4: Convert continuous t to UNet integer timestep.
            # Diffusers convention: t=0 (noise) → 1000, t=1 (clean) → 0.
            # ------------------------------------------------------------------
            timestep_int: int = get_unet_timestep(
                t_continuous=t,
                num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
            )
            timestep_tensor: torch.Tensor = torch.tensor(
                [timestep_int] * batch_size,
                dtype=torch.long,
                device=device,
            )

            # ------------------------------------------------------------------
            # Step 5: Compute fine-tuned velocity WITH gradients.
            # This is the ONLY operation that retains the gradient graph.
            # v_theta.parameters() will receive gradients through this call.
            # ------------------------------------------------------------------
            # Cast X_t to match v_theta's expected dtype (bfloat16 in training)
            X_t_input: torch.Tensor = X_t.to(dtype=_get_model_dtype(v_theta))
            text_emb_input: torch.Tensor = text_emb.to(
                device=device, dtype=_get_model_dtype(v_theta)
            )

            unet_output_ft = v_theta(
                X_t_input,
                timestep_tensor,
                encoder_hidden_states=text_emb_input,
                return_dict=True,
            )
            v_ft: torch.Tensor = unet_output_ft.sample  # (B, C, H, W)

            # ------------------------------------------------------------------
            # Step 6: Compute base velocity WITHOUT gradients.
            # v_base is frozen; detach for belt-and-suspenders safety.
            # ------------------------------------------------------------------
            with torch.no_grad():
                unet_output_bs = v_base(
                    X_t_input,
                    timestep_tensor,
                    encoder_hidden_states=text_emb_input,
                    return_dict=True,
                )
                v_bs: torch.Tensor = unet_output_bs.sample.detach()  # (B, C, H, W)

            # ------------------------------------------------------------------
            # Step 7: Compute the residual term (equation 42).
            # term = (2/σ(t)) * (v_θ - v_base) + σ(t) * ã_t
            # Cast sigma and a_tilde to match v_ft dtype for consistency.
            # ------------------------------------------------------------------
            target_dtype: torch.dtype = v_ft.dtype
            sigma_t: torch.Tensor = torch.tensor(
                sigma_t_float, dtype=target_dtype, device=device
            )
            a_tilde_cast: torch.Tensor = a_tilde.to(dtype=target_dtype)

            # Velocity difference: (B, C, H, W)
            v_diff: torch.Tensor = v_ft - v_bs.to(dtype=target_dtype)

            # Residual: (2/σ) * (v_θ - v_base) + σ * ã_t
            term: torch.Tensor = (2.0 / sigma_t) * v_diff + sigma_t * a_tilde_cast
            # term shape: (B, C, H, W)

            # ------------------------------------------------------------------
            # Step 8: Compute per-sample squared Frobenius norm.
            # Flatten all non-batch dimensions, then sum.
            # Result shape: (B,)
            # ------------------------------------------------------------------
            # Cast to float32 for the norm computation to avoid bfloat16 overflow
            term_f32: torch.Tensor = term.float()
            loss_t: torch.Tensor = term_f32.pow(2).flatten(1).sum(dim=1)
            # loss_t shape: (B,)

            # ------------------------------------------------------------------
            # Step 9: Apply loss clipping (Appendix G.3).
            # LCT = 1.6 * λ² for Adjoint Matching.
            # Prevents high-magnitude early-timestep terms from dominating.
            # ------------------------------------------------------------------
            loss_t = torch.clamp(loss_t, max=float(lct))

            # ------------------------------------------------------------------
            # Step 10: Accumulate mean over batch.
            # ------------------------------------------------------------------
            # Cast back to the computation dtype for accumulation
            total_loss = total_loss + loss_t.mean().to(dtype=torch.float32)
            count += 1

        # ------------------------------------------------------------------
        # Average over the timestep subset.
        # ------------------------------------------------------------------
        if count == 0:
            logger.warning(
                "AdjointMatchingLoss.compute: no valid timesteps processed. "
                "Returning zero loss."
            )
            dummy_param = next(v_theta.parameters())
            return dummy_param.sum() * 0.0

        final_loss: torch.Tensor = total_loss / float(count)

        logger.debug(
            "AdjointMatchingLoss.compute: loss=%.6f, timesteps_processed=%d",
            final_loss.item(),
            count,
        )

        return final_loss

    def __repr__(self) -> str:
        """Human-readable representation of the loss function."""
        return (
            f"AdjointMatchingLoss("
            f"h={self.noise_schedule.h:.4f}"
            f")"
        )


# ---------------------------------------------------------------------------
# Continuous Adjoint baseline loss (standalone function)
# ---------------------------------------------------------------------------


def continuous_adjoint_loss(
    v_theta: nn.Module,
    v_base: nn.Module,
    trajectory: List[torch.Tensor],
    full_adjoints: Dict[float, torch.Tensor],
    timestep_subset: List[float],
    text_emb: torch.Tensor,
    noise_schedule: NoiseSchedule,
    lct: float = 1.6e11,
) -> torch.Tensor:
    """Compute the Continuous Adjoint baseline loss.

    Implements the basic Adjoint Matching form (equation 34) but using the
    FULL adjoint state a(t; X, u) rather than the lean adjoint ã(t; X).

    The full adjoint satisfies equations (30)-(31):
        da/dt = -[a(t)ᵀ ∇_x(b + σ*u) + ∇_x(f + ½||u||²)]
        a(1) = ∇g(X_1) = -∇r(X_1)

    The loss form (analogous to equation 34 with full adjoint):
        L_ContAdj(θ) = Σ_{t∈κ} min{ LCT, ||u(X_t,t) + σ(t)ᵀ a(t; X, u)||² }

    where u = (2/σ(t)) * (v_θ - v_base) is the control parameterization
    from equation (27).

    Key difference from AdjointMatchingLoss.compute():
        - Uses full adjoint a(t) instead of lean adjoint ã(t)
        - LCT = 1600 * λ² (much larger, Appendix G.3) because full adjoint
          states have much larger magnitude than lean adjoint states
        - The full adjoint is computed in lean_adjoint.py with the controlled
          drift Jacobian (more expensive, higher variance)

    Args:
        v_theta: Fine-tuned velocity field (trainable UNet).
            Parameters must have requires_grad=True.
        v_base: Frozen base velocity field.
            Parameters must have requires_grad=False.
        trajectory: List of K+1 detached tensors from
            TrajectorySampler.sample_trajectory().
        full_adjoints: Dict[float, Tensor] mapping timestep t to the full
            adjoint state a(t; X, u) of shape (batch_size, C, H, W).
            Computed by LeanAdjointSolver with the controlled drift Jacobian.
            All tensors are detached.
        timestep_subset: List of float timestep values for gradient evaluation.
            Same subset as used in AdjointMatchingLoss.compute().
        text_emb: CLIP text embeddings of shape (batch_size, seq_len, hidden_dim).
        noise_schedule: NoiseSchedule instance providing sigma_memoryless() and h.
        lct: Loss Clipping Threshold for Continuous Adjoint.
            From config.yaml: loss.lct_constant_cont_adjoint * λ² = 1600 * λ².
            For λ=12500: lct = 1600 * 12500² = 2.5e11.
            Default 1.6e11 corresponds to λ ≈ 10000.

    Returns:
        Scalar tensor with gradient graph through v_theta.parameters() only.
        Returns zero tensor if timestep_subset is empty.

    Note:
        The LCT for Continuous Adjoint is 1000× larger than for Adjoint
        Matching (1600 vs 1.6 constant) because the full adjoint states
        have much larger magnitude. This is documented in Appendix G.3:
        "the magnitude of the regular adjoint states is significantly larger
        than the magnitude of the lean adjoint states."
    """
    if len(timestep_subset) == 0:
        logger.warning(
            "continuous_adjoint_loss: empty timestep_subset. "
            "Returning zero loss."
        )
        dummy_param = next(v_theta.parameters())
        return dummy_param.sum() * 0.0

    h: float = noise_schedule.h
    device: torch.device = text_emb.device
    batch_size: int = trajectory[0].shape[0]

    # Build trajectory index mapping (same logic as AdjointMatchingLoss.compute)
    K: int = len(trajectory) - 1
    full_timesteps: List[float] = noise_schedule.get_timesteps(K=K)
    t_to_traj_idx: Dict[float, int] = {}
    for j, t_val in enumerate(full_timesteps):
        t_to_traj_idx[t_val] = j + 1

    total_loss: torch.Tensor = torch.zeros(
        1, dtype=torch.float32, device=device
    )
    count: int = 0

    for t in timestep_subset:
        # ------------------------------------------------------------------
        # Step 1: Retrieve trajectory state X_t (stop-gradient).
        # ------------------------------------------------------------------
        traj_idx: Optional[int] = _find_trajectory_index(
            t, t_to_traj_idx, full_timesteps
        )
        if traj_idx is None or traj_idx >= len(trajectory):
            logger.warning(
                "continuous_adjoint_loss: timestep t=%.4f not found or "
                "out of bounds. Skipping.",
                t,
            )
            continue

        X_t: torch.Tensor = trajectory[traj_idx].detach().to(device)

        # ------------------------------------------------------------------
        # Step 2: Retrieve full adjoint state a(t; X, u) (stop-gradient).
        # ------------------------------------------------------------------
        if t not in full_adjoints:
            logger.warning(
                "continuous_adjoint_loss: full adjoint for t=%.4f not found. "
                "Skipping.",
                t,
            )
            continue

        a_full: torch.Tensor = full_adjoints[t].detach().to(device)

        # ------------------------------------------------------------------
        # Step 3: Compute sigma_t.
        # ------------------------------------------------------------------
        sigma_t_float: float = noise_schedule.sigma_memoryless(t=t, h=h)
        sigma_t_float = max(sigma_t_float, _SIGMA_MIN)

        # ------------------------------------------------------------------
        # Step 4: Convert t to UNet integer timestep.
        # ------------------------------------------------------------------
        timestep_int: int = get_unet_timestep(
            t_continuous=t,
            num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
        )
        timestep_tensor: torch.Tensor = torch.tensor(
            [timestep_int] * batch_size,
            dtype=torch.long,
            device=device,
        )

        # ------------------------------------------------------------------
        # Step 5: Compute fine-tuned velocity WITH gradients.
        # ------------------------------------------------------------------
        model_dtype: torch.dtype = _get_model_dtype(v_theta)
        X_t_input: torch.Tensor = X_t.to(dtype=model_dtype)
        text_emb_input: torch.Tensor = text_emb.to(
            device=device, dtype=model_dtype
        )

        unet_output_ft = v_theta(
            X_t_input,
            timestep_tensor,
            encoder_hidden_states=text_emb_input,
            return_dict=True,
        )
        v_ft: torch.Tensor = unet_output_ft.sample  # (B, C, H, W)

        # ------------------------------------------------------------------
        # Step 6: Compute base velocity WITHOUT gradients.
        # ------------------------------------------------------------------
        with torch.no_grad():
            unet_output_bs = v_base(
                X_t_input,
                timestep_tensor,
                encoder_hidden_states=text_emb_input,
                return_dict=True,
            )
            v_bs: torch.Tensor = unet_output_bs.sample.detach()

        # ------------------------------------------------------------------
        # Step 7: Compute control u = (2/σ) * (v_θ - v_base).
        # From equation (27): u(x,t) = (2/σ(t)) * (v_finetune - v_base)
        # ------------------------------------------------------------------
        target_dtype: torch.dtype = v_ft.dtype
        sigma_t: torch.Tensor = torch.tensor(
            sigma_t_float, dtype=target_dtype, device=device
        )
        a_full_cast: torch.Tensor = a_full.to(dtype=target_dtype)

        v_diff: torch.Tensor = v_ft - v_bs.to(dtype=target_dtype)
        u: torch.Tensor = (2.0 / sigma_t) * v_diff  # (B, C, H, W)

        # ------------------------------------------------------------------
        # Step 8: Compute the residual term.
        # From equation (34) with full adjoint:
        #   term = u(X_t, t) + σ(t)ᵀ a(t; X, u)
        # ------------------------------------------------------------------
        term: torch.Tensor = u + sigma_t * a_full_cast
        # term shape: (B, C, H, W)

        # ------------------------------------------------------------------
        # Step 9: Compute per-sample squared Frobenius norm.
        # ------------------------------------------------------------------
        term_f32: torch.Tensor = term.float()
        loss_t: torch.Tensor = term_f32.pow(2).flatten(1).sum(dim=1)
        # loss_t shape: (B,)

        # ------------------------------------------------------------------
        # Step 10: Apply loss clipping with the larger LCT for full adjoint.
        # LCT = 1600 * λ² (Appendix G.3).
        # ------------------------------------------------------------------
        loss_t = torch.clamp(loss_t, max=float(lct))

        # ------------------------------------------------------------------
        # Step 11: Accumulate mean over batch.
        # ------------------------------------------------------------------
        total_loss = total_loss + loss_t.mean().to(dtype=torch.float32)
        count += 1

    # Average over timestep subset
    if count == 0:
        logger.warning(
            "continuous_adjoint_loss: no valid timesteps processed. "
            "Returning zero loss."
        )
        dummy_param = next(v_theta.parameters())
        return dummy_param.sum() * 0.0

    final_loss: torch.Tensor = total_loss / float(count)

    logger.debug(
        "continuous_adjoint_loss: loss=%.6f, timesteps_processed=%d",
        final_loss.item(),
        count,
    )

    return final_loss


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------


def _find_trajectory_index(
    t: float,
    t_to_traj_idx: Dict[float, int],
    full_timesteps: List[float],
    tolerance: float = 1e-6,
) -> Optional[int]:
    """Find the trajectory index for a given timestep value.

    Handles floating-point comparison issues by first trying exact lookup,
    then falling back to nearest-neighbor search within tolerance.

    Args:
        t: Target timestep value to look up.
        t_to_traj_idx: Dict mapping timestep float → trajectory index.
            Built from noise_schedule.get_timesteps(K).
        full_timesteps: Full list of timestep values for nearest-neighbor
            fallback search.
        tolerance: Maximum absolute difference for nearest-neighbor match.
            Default 1e-6 handles typical floating-point rounding errors.

    Returns:
        Trajectory index (int) if found, None if no match within tolerance.
    """
    # Try exact lookup first (handles most cases)
    if t in t_to_traj_idx:
        return t_to_traj_idx[t]

    # Fallback: nearest-neighbor search within tolerance
    # This handles cases where floating-point rounding causes slight mismatches
    # (e.g., 0.025 * 3 = 0.07500000000000001 vs 0.075)
    best_idx: Optional[int] = None
    best_diff: float = float("inf")

    for t_key, idx in t_to_traj_idx.items():
        diff: float = abs(t_key - t)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx

    if best_diff <= tolerance:
        return best_idx

    # No match found within tolerance
    return None


def _get_model_dtype(model: nn.Module) -> torch.dtype:
    """Get the dtype of the first parameter of