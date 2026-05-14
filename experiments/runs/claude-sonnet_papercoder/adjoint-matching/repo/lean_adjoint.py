## lean_adjoint.py
"""Lean adjoint ODE solver for Adjoint Matching fine-tuning experiments.

This module implements the backward-in-time integration of the lean adjoint
ODE (equations 38-39 from the paper), which is the core computational
primitive enabling Adjoint Matching to cast SOC as a regression problem
without importance weighting.

The lean adjoint ODE (equations 38-39):
    d/dt ã(t; X) = -(ã(t; X)ᵀ ∇_x b(X_t, t) + ∇_x f(X_t, t))
    ã(1; X) = ∇_x g(X_1)

For reward fine-tuning with f=0 and g=-r:
    d/dt ã(t; X) = -ã(t; X)ᵀ ∇_x b(X_t, t)
    ã(1; X) = -∇_x r(X_1)

The base drift under the memoryless schedule (σ²/2 = η_t):
    b(x, t) = κ_t * x + 2*η_t * s(x, t) = 2*v_base(x, t) - κ_t * x

Euler backward step (Algorithm 1, equation 41):
    ã_{t-h} = ã_t + h * ã_tᵀ ∇_{X_t}(2*v_base(X_t, t) - κ_t * X_t)
    ã_1 = -∇_{X_1} r(X_1)

Configuration alignment (config.yaml):
    sampling.K: 40                    → K timesteps
    sampling.h: 0.025                 → step size
    noise_schedule.offset: 0.025      → sigma offset
    model.num_train_timesteps: 1000   → UNet integer timestep range

Dependencies:
    - noise_schedule.py: NoiseSchedule (kappa, h)
    - utils.py: get_unet_timestep (continuous t → integer UNet timestep)
    - torch, typing, logging (standard)

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


class LeanAdjointSolver:
    """Solves the lean adjoint ODE backwards in time for Adjoint Matching.

    Implements the backward integration of equations (38)-(39) from the paper,
    which defines the lean adjoint state ã(t; X) used in the Adjoint Matching
    loss (equation 37/42).

    The lean adjoint removes the control-dependent terms from the full adjoint
    ODE (equation 30), retaining only the base drift Jacobian:
        d/dt ã(t; X) = -ã(t; X)ᵀ ∇_x b(X_t, t)

    For the memoryless Flow Matching base drift:
        b(x, t) = 2*v_base(x, t) - κ_t * x

    The Euler backward step (Algorithm 1, equation 41) is:
        ã_{t-h} = ã_t + h * ã_tᵀ ∇_{X_t}(2*v_base(X_t, t) - κ_t * X_t)

    All computations run under torch.no_grad() except the VJP computation
    inside _vjp_base_drift(), which uses torch.enable_grad() locally.
    All returned adjoint states are detached from the autograd graph.

    Attributes:
        noise_schedule: NoiseSchedule instance providing kappa(t) and h.
        device: PyTorch device string for tensor allocation.

    Example:
        >>> ns = NoiseSchedule(h=0.025)
        >>> solver = LeanAdjointSolver(ns, device="cuda")
        >>> terminal_grad = torch.randn(4, 4, 64, 64, device="cuda")
        >>> timesteps = ns.get_timesteps(K=40)
        >>> adjoints = solver.solve(trajectory, v_base, terminal_grad, timesteps, text_emb)
        >>> adjoints[0.025].shape
        torch.Size([4, 4, 64, 64])
    """

    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        device: str = "cuda",
    ) -> None:
        """Initialize the lean adjoint solver.

        Args:
            noise_schedule: NoiseSchedule instance providing kappa(t) and h.
                Sourced from config.yaml noise_schedule and sampling sections.
                Must have h = 1/K = 0.025 for K=40 (config.yaml sampling.h).
            device: PyTorch device string for tensor allocation.
                From config.yaml: training.device (inferred from num_gpus).
                Examples: "cuda", "cpu", "cuda:0", "cuda:1".
        """
        self.noise_schedule: NoiseSchedule = noise_schedule
        self.device: str = device

        logger.info(
            "LeanAdjointSolver initialized: device='%s', h=%.4f",
            device,
            noise_schedule.h,
        )

    # ------------------------------------------------------------------
    # Public API: reward gradient terminal condition
    # ------------------------------------------------------------------

    def compute_reward_gradient(
        self,
        X_hat_1: torch.Tensor,
        prompts: List[str],
        reward_fn: "RewardModel",  # type: ignore[name-defined]  # noqa: F821
        lambda_r: float = 12500.0,
    ) -> torch.Tensor:
        """Compute the terminal condition gradient for the lean adjoint ODE.

        Computes ã(1; X) = ∇_x g(X̂_1) = -λ * ∇_{X̂_1} r(X̂_1) by delegating
        to the reward model's gradient() method.

        This is a convenience wrapper that calls reward_fn.gradient() with
        the correct arguments. The actual gradient computation (including
        the differentiable path through the VAE decoder) is implemented in
        reward_models.py.

        Args:
            X_hat_1: Noiseless terminal latent of shape (batch_size, C, H, W).
                Computed by TrajectorySampler.get_noiseless_terminal().
                Represents X̂_1 = X_{1-h} + h * v_base(X_{1-h}, 1-h).
            prompts: List of text prompt strings, length = batch_size.
                Used by the reward model for text-image alignment scoring.
            reward_fn: RewardModel instance (e.g., ImageRewardModel).
                Must implement gradient(X_latent, prompts, vae, lambda_r).
                Note: This signature requires vae; callers should use
                reward_fn.gradient() directly when vae is available.
                This method is provided for API completeness.
            lambda_r: Reward scaling factor λ from config.yaml
                reward.lambda_reward (default 12500).

        Returns:
            Terminal gradient tensor of shape (batch_size, C, H, W)
            representing -lambda_r * d(r)/dX_latent. Detached, float32.

        Note:
            In practice, trainer.py calls reward_fn.gradient() directly
            (passing the vae argument), then passes the result to solve().
            This method is provided for cases where the vae is embedded
            in the reward_fn or not needed.
        """
        # Delegate to the reward model's gradient computation.
        # The reward model handles the VAE decode path internally.
        # We pass vae=None here; callers should use reward_fn.gradient()
        # directly when a VAE is needed.
        logger.debug(
            "compute_reward_gradient: batch_size=%d, lambda_r=%.1f",
            X_hat_1.shape[0],
            lambda_r,
        )
        # This method is a convenience wrapper; the actual implementation
        # is in reward_models.ImageRewardModel.gradient().
        # Callers with a VAE should call reward_fn.gradient(X_hat_1, prompts, vae, lambda_r).
        raise NotImplementedError(
            "compute_reward_gradient() requires a VAE for the decode path. "
            "Call reward_fn.gradient(X_hat_1, prompts, vae, lambda_r) directly "
            "from trainer.py, then pass the result to solve() as terminal_grad."
        )

    # ------------------------------------------------------------------
    # Core VJP computation
    # ------------------------------------------------------------------

    def _vjp_base_drift(
        self,
        v_base: nn.Module,
        X_t: torch.Tensor,
        t: float,
        a_tilde: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the VJP of the base drift with respect to X_t.

        Computes the vector-Jacobian product:
            VJP = ã_tᵀ * ∂/∂X_t [2*v_base(X_t, t) - κ_t * X_t]

        This is the key operation in the lean adjoint Euler step (eq. 41):
            ã_{t-h} = ã_t + h * VJP(ã_t, X_t, t)

        The computation uses torch.autograd.grad() to compute the VJP
        efficiently without materializing the full Jacobian matrix.

        Memory efficiency: Since create_graph=False, the temporary computation
        graph for the VJP is freed immediately after the .grad call.

        Args:
            v_base: Frozen base velocity field (UNet with frozen parameters).
                Parameters must have requires_grad=False.
                Called as v_base(X_t_grad, timestep_tensor, encoder_hidden_states).
                Returns UNet2DConditionOutput with .sample attribute.
            X_t: Trajectory state at time t, shape (batch_size, C, H, W).
                Must be detached (stop-grad) — a fresh requires_grad=True
                copy is created internally.
            t: Continuous time value in (0, 1]. Used to compute κ_t = 1/t
                and the UNet integer timestep via get_unet_timestep().
            a_tilde: Current lean adjoint state, shape (batch_size, C, H, W).
                Used as grad_outputs in the VJP computation.
                Must be detached.
            text_emb: CLIP text embeddings, shape (batch_size, seq_len, hidden_dim).
                Passed as encoder_hidden_states to the UNet.
                Must be detached (frozen text encoder output).

        Returns:
            VJP tensor of shape (batch_size, C, H, W) representing
            ã_tᵀ * J_{base_drift}. Detached from the autograd graph.
            NaN/Inf values are replaced with zeros for numerical safety.

        Note:
            The UNet timestep conversion uses the diffusers reverse-time
            convention: timestep_int = int((1-t) * 1000).
            At t=0.025 (most noisy): timestep_int = 975.
            At t=0.975 (nearly clean): timestep_int = 25.

            The batch dimension is handled correctly: torch.autograd.grad
            computes the sum over the batch of ã_i^T * J_i, which is the
            correct batched VJP for independent per-sample computations.
        """
        batch_size: int = X_t.shape[0]

        # ------------------------------------------------------------------
        # Step 1: Create a fresh leaf tensor with requires_grad=True.
        # X_t arrives detached (stop-grad from TrajectorySampler).
        # We need a new leaf node to compute the Jacobian w.r.t. X_t.
        # ------------------------------------------------------------------
        X_t_grad: torch.Tensor = X_t.detach().to(self.device).requires_grad_(True)

        # ------------------------------------------------------------------
        # Step 2: Compute the base drift inside torch.enable_grad().
        # This is necessary because solve() runs under torch.no_grad(),
        # but we need gradient computation for the VJP.
        # ------------------------------------------------------------------
        with torch.enable_grad():
            # Convert continuous t to UNet integer timestep
            # Diffusers convention: t=0 (noise) → 1000, t=1 (clean) → 0
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

            # UNet forward pass: get base velocity prediction
            # v_base parameters have requires_grad=False (frozen),
            # so gradient only flows through X_t_grad.
            unet_output = v_base(
                X_t_grad,
                timestep_tensor,
                encoder_hidden_states=text_emb,
                return_dict=True,
            )
            v_pred: torch.Tensor = unet_output.sample  # (B, C, H, W)

            # Compute base drift: b(x, t) = 2*v_base(x, t) - κ_t * x
            # For α_t = t: κ_t = α̇_t / α_t = 1/t
            kappa_t: float = self.noise_schedule.kappa(t)
            base_drift: torch.Tensor = 2.0 * v_pred - kappa_t * X_t_grad
            # base_drift shape: (B, C, H, W)

            # ------------------------------------------------------------------
            # Step 3: Compute VJP via torch.autograd.grad.
            # grad_outputs=a_tilde means we compute:
            #   VJP = sum_i a_tilde_i * (∂base_drift_i / ∂X_t_grad)
            # where the sum is over all elements of the output tensor.
            # This is equivalent to ã_tᵀ * J_{base_drift} in matrix notation.
            #
            # create_graph=False: we don't need higher-order gradients.
            # retain_graph=False: free the computation graph immediately.
            # allow_unused=False: X_t_grad must appear in the computation graph.
            # ------------------------------------------------------------------
            grad_tuple = torch.autograd.grad(
                outputs=base_drift,
                inputs=X_t_grad,
                grad_outputs=a_tilde.to(dtype=base_drift.dtype, device=self.device),
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )
            vjp: torch.Tensor = grad_tuple[0]  # shape: (B, C, H, W)

        # ------------------------------------------------------------------
        # Step 4: Sanitize and detach the VJP result.
        # Replace NaN/Inf with zeros to prevent gradient explosion.
        # Detach to enforce stop-grad discipline.
        # ------------------------------------------------------------------
        vjp = torch.nan_to_num(vjp, nan=0.0, posinf=0.0, neginf=0.0)
        return vjp.detach()

    # ------------------------------------------------------------------
    # Main backward integration
    # ------------------------------------------------------------------

    def solve(
        self,
        trajectory: List[torch.Tensor],
        v_base: nn.Module,
        terminal_grad: torch.Tensor,
        timesteps: List[float],
        text_emb: torch.Tensor,
    ) -> Dict[float, torch.Tensor]:
        """Solve the lean adjoint ODE backwards from t=1 to t=h.

        Implements the backward Euler integration of equations (38)-(39):
            ã_{t-h} = ã_t + h * ã_tᵀ ∇_{X_t}(2*v_base(X_t, t) - κ_t * X_t)
            ã_1 = terminal_grad  (= -λ * ∇_{X̂_1} r(X̂_1))

        The entire method runs under torch.no_grad() as the outer context.
        The _vjp_base_drift() method uses torch.enable_grad() locally for
        the VJP computation, then returns a detached result.

        All adjoint states in the returned dict are detached from the
        autograd graph and ready for use in AdjointMatchingLoss.compute().

        Args:
            trajectory: List of K+1 detached tensors from
                TrajectorySampler.sample_trajectory(). trajectory[i]
                corresponds to the state at timesteps[i] for i < K,
                and trajectory[K] is the terminal state X_1.
                All tensors have shape (batch_size, C, H, W).
            v_base: Frozen base velocity field (UNet with frozen parameters).
                Parameters must have requires_grad=False.
                Used in _vjp_base_drift() to compute the base drift Jacobian.
            terminal_grad: Terminal condition tensor of shape (batch_size, C, H, W).
                Represents ã(1; X) = ∇_x g(X̂_1) = -λ * ∇_{X̂_1} r(X̂_1).
                Computed by reward_models.ImageRewardModel.gradient() and
                passed from trainer.py. Already detached.
            timesteps: List of K float timestep values [h, 2h, ..., 1.0].
                Generated by noise_schedule.get_timesteps(K=40).
                Length must equal len(trajectory) - 1.
                From config.yaml: sampling.K=40, sampling.h=0.025.
            text_emb: CLIP text embeddings of shape (batch_size, seq_len, hidden_dim).
                Same embeddings used in TrajectorySampler.sample_trajectory().
                Passed as encoder_hidden_states to v_base.

        Returns:
            Dict[float, Tensor] mapping each timestep t to its lean adjoint
            state ã_t of shape (batch_size, C, H, W). Keys include all
            timesteps from h to 1.0 (inclusive). All tensors are detached,
            float32 (or the dtype of terminal_grad), on self.device.

            The AdjointMatchingLoss.compute() will look up adjoints[t] for
            each t in the timestep subset selected by
            TrajectorySampler.select_timestep_subset().

        Raises:
            ValueError: If len(trajectory) != len(timesteps) + 1.
            ValueError: If timesteps is empty.

        Note:
            Iteration boundaries for K=40, h=0.025:
                timesteps = [0.025, 0.050, ..., 0.975, 1.000]  (40 elements)
                trajectory has 41 elements: [X_0, X_h, ..., X_{1-h}, X_1]
                trajectory[i] = X_{timesteps[i]} for i in [0, K-1]
                trajectory[K] = X_1 (terminal, not used in backward pass)

            The adjoint at t=1.0 is initialized from terminal_grad.
            We then compute adjoints at t=0.975, 0.950, ..., 0.025.
            The adjoint at t=0.025 (first timestep) is the last computed.

            Sign convention for Euler backward step:
                d/dt ã = -ã^T ∇_x b
                Integrating backward from t to t-h:
                ã(t-h) ≈ ã(t) - h * (d/dt ã)|_t
                        = ã(t) - h * (-ã^T ∇_x b)
                        = ã(t) + h * ã^T ∇_x b
                        = ã(t) + h * VJP
                Hence the sign is +h * vjp (not -h * vjp).
        """
        # ------------------------------------------------------------------
        # Input validation
        # ------------------------------------------------------------------
        if len(timesteps) == 0:
            raise ValueError(
                "timesteps list is empty. "
                "Provide at least one timestep for the lean adjoint solve."
            )

        expected_traj_len: int = len(timesteps) + 1
        if len(trajectory) != expected_traj_len:
            raise ValueError(
                f"trajectory length ({len(trajectory)}) must equal "
                f"len(timesteps) + 1 = {expected_traj_len}. "
                f"Ensure trajectory was generated with the same timesteps list."
            )

        h: float = self.noise_schedule.h
        batch_size: int = terminal_grad.shape[0]

        logger.debug(
            "LeanAdjointSolver.solve: K=%d, h=%.4f, batch_size=%d",
            len(timesteps),
            h,
            batch_size,
        )

        # ------------------------------------------------------------------
        # Build timestep → trajectory index mapping for O(1) lookup.
        # timesteps[i] corresponds to trajectory[i] for i in [0, K-1].
        # trajectory[K] is the terminal state X_1 (not used in backward pass).
        # ------------------------------------------------------------------
        t_to_idx: Dict[float, int] = {
            t: i for i, t in enumerate(timesteps)
        }

        # ------------------------------------------------------------------
        # Initialize adjoint at terminal time t=1.0.
        # terminal_grad = -λ * ∇_{X̂_1} r(X̂_1) = ∇_x g(X̂_1) since g = -r.
        # This is already the correct terminal condition ã(1; X) = ∇_x g(X_1).
        # ------------------------------------------------------------------
        a_tilde: torch.Tensor = terminal_grad.detach().to(
            device=self.device, dtype=torch.float32
        )

        # Initialize the output dict with the terminal adjoint
        # Use the last timestep value (should be 1.0) as the key
        t_terminal: float = timesteps[-1]  # Should be 1.0
        adjoints: Dict[float, torch.Tensor] = {t_terminal: a_tilde.clone()}

        # ------------------------------------------------------------------
        # Backward Euler integration.
        # Iterate from the second-to-last timestep down to the first.
        # For K=40: from index K-2 (t=0.975) down to index 0 (t=0.025).
        #
        # At each step:
        #   1. Get X_t from trajectory at index i
        #   2. Compute VJP = ã_tᵀ * ∂/∂X_t [2*v_base(X_t,t) - κ_t*X_t]
        #   3. Update: ã_{t-h} = ã_t + h * VJP
        #   4. Store ã_{t-h} in adjoints dict
        #
        # The outer torch.no_grad() context prevents gradient accumulation
        # through the adjoint states themselves. _vjp_base_drift() uses
        # torch.enable_grad() locally for the VJP computation only.
        # ------------------------------------------------------------------
        with torch.no_grad():
            # Iterate backwards: from index K-2 down to 0
            # (skipping the last timestep t=1.0 which is already initialized)
            for i in range(len(timesteps) - 2, -1, -1):
                t: float = timesteps[i]

                # Get trajectory state at time t (detached by TrajectorySampler)
                X_t: torch.Tensor = trajectory[i].detach().to(
                    device=self.device, dtype=torch.float32
                )

                # Ensure text_emb is on the correct device
                text_emb_device: torch.Tensor = text_emb.to(
                    device=self.device, dtype=torch.float32
                )

                # Compute VJP: ã_tᵀ * ∂/∂X_t [2*v_base(X_t,t) - κ_t*X_t]
                # _vjp_base_drift uses torch.enable_grad() internally
                vjp: torch.Tensor = self._vjp_base_drift(
                    v_base=v_base,
                    X_t=X_t,
                    t=t,
                    a_tilde=a_tilde,
                    text_emb=text_emb_device,
                )

                # Euler backward step:
                # ã(t-h) = ã(t) + h * ã(t)^T ∇_x b(X_t, t)
                # Sign is +h because we integrate d/dt ã = -ã^T ∇_x b
                # backward in time: ã(t-h) = ã(t) - h*(d/dt ã) = ã(t) + h*ã^T ∇_x b
                a_tilde = (a_tilde + h * vjp).detach()

                # Sanitize to prevent NaN/Inf propagation through time
                a_tilde = torch.nan_to_num(
                    a_tilde, nan=0.0, posinf=0.0, neginf=0.0
                )

                # Store adjoint at current timestep t
                # Note: a_tilde now represents ã at time t (after the update,
                # it represents ã at t-h, but we store it as the adjoint
                # for use at timestep t in the loss computation).
                # The loss uses adjoints[t] where t is the timestep at which
                # v_theta(X_t, t) is evaluated, so we store the adjoint
                # that was computed using X_t (before the backward step).
                # Correction: we store the updated a_tilde as adjoints[t]
                # because the Adjoint Matching loss at timestep t uses
                # the adjoint ã_t (not ã_{t-h}).
                # The update computes ã_{t-h} from ã_t, so we need to
                # store ã_t BEFORE the update.
                # We fix this by storing before the update below.
                adjoints[t] = a_tilde.clone()

        # ------------------------------------------------------------------
        # Post-processing: re-store adjoints with correct timing.
        # The above loop stores ã_{t-h} at key t, but the loss needs ã_t
        # at key t. We need to re-run with correct storage order.
        # ------------------------------------------------------------------
        # Reset and redo with correct storage order
        a_tilde = terminal_grad.detach().to(
            device=self.device, dtype=torch.float32
        )
        adjoints = {t_terminal: a_tilde.clone()}

        with torch.no_grad():
            for i in range(len(timesteps) - 2, -1, -1):
                t = timesteps[i]

                X_t = trajectory[i].detach().to(
                    device=self.device, dtype=torch.float32
                )
                text_emb_device = text_emb.to(
                    device=self.device, dtype=torch.float32
                )

                # Store ã_t BEFORE the backward Euler update
                # This is the adjoint at time t, used in the loss at timestep t
                adjoints[t] = a_tilde.clone()

                # Compute VJP for the backward step
                vjp = self._vjp_base_drift(
                    v_base=v_base,
                    X_t=X_t,
                    t=t,
                    a_tilde=a_tilde,
                    text_emb=text_emb_device,
                )

                # Euler backward step: ã_{t-h} = ã_t + h * VJP
                a_tilde = (a_tilde + h * vjp).detach()
                a_tilde = torch.nan_to_num(
                    a_tilde, nan=0.0, posinf=0.0, neginf=0.0
                )

        logger.debug(
            "LeanAdjointSolver.solve: computed %d adjoint states.",
            len(adjoints),
        )

        return adjoints

    def __repr__(self) -> str:
        """Human-readable representation of the lean adjoint solver."""
        return (
            f"LeanAdjointSolver("
            f"device='{self.device}', "
            f"h={self.noise_schedule.h:.4f}"
            f")"
        )
