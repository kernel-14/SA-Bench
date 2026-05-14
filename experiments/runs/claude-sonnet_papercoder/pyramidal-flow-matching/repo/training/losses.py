## training/losses.py
"""Flow matching and VAE loss functions for Pyramidal Flow Matching.

Implements two distinct training objectives:

1. **Unified Flow Matching Loss** (Paper Section 3.2.1, Eq. 6):
   The core pyramidal flow matching objective used to train the MM-DiT
   across all K=3 pyramid stages with a single unified model:
       E_{k, t, (x_hat_ek, x_hat_sk)} || v_t(x_hat_t) - (x_hat_ek - x_hat_sk) ||^2

2. **VAE Loss** (Paper Section 4.1, Appendix B):
   Reconstruction + KL divergence loss for pretraining the 3D causal VAE:
       L_VAE = L_recon + kl_weight * L_KL

This module is stateless with respect to training step — it provides pure
loss computation utilities consumed by Trainer.train_step(). No learnable
parameters are defined here.

Stage indexing convention (from Shared Knowledge):
    - stage_id=0: full resolution, stage_bounds[2] = [0.667, 1.0]
    - stage_id=1: half resolution, stage_bounds[1] = [0.333, 0.667]
    - stage_id=2: quarter resolution, stage_bounds[0] = [0.0, 0.333]

Config references (configs/default.yaml):
    - pyramid.num_stages: 3
    - pyramid.stage_bounds: [[0.0, 0.333], [0.333, 0.667], [0.667, 1.0]]
    - vae.kl_weight: 1.0e-6

Usage:
    from training.losses import FlowMatchingLoss

    loss_fn = FlowMatchingLoss(config)

    # Sample stage and timestep for this training step
    stage_id = loss_fn.sample_stage_id(K=3)
    s_k = float(config.pyramid.stage_bounds[stage_id][0])
    e_k = float(config.pyramid.stage_bounds[stage_id][1])
    t = loss_fn.sample_t_in_stage(s_k, e_k, batch_size=8)

    # Compute flow matching loss (velocities from PyramidFlowModel.forward())
    loss = loss_fn.compute(pred_velocity, target_velocity, mask=attn_mask)

    # VAE pretraining loss
    vae_loss = loss_fn.vae_loss(recon, x, mu, logvar, kl_weight=1e-6)
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from utils.logging import get_logger

## ---------------------------------------------------------------------------
## Module-level logger
## ---------------------------------------------------------------------------
logger = get_logger(__name__)

## ---------------------------------------------------------------------------
## Numerical stability constants
## ---------------------------------------------------------------------------
_EPS: float = 1e-8          # Denominator epsilon for masked loss
_LOGVAR_MIN: float = -30.0  # Clamp logvar before exp() to prevent overflow
_LOGVAR_MAX: float = 20.0   # Clamp logvar before exp() to prevent overflow


class FlowMatchingLoss:
    """Unified flow matching and VAE loss functions for pyramidal training.

    Provides the core loss computation utilities for the three-stage training
    procedure described in the paper (Appendix B, Table 4). This class has
    no learnable parameters and no internal mutable state — it is a pure
    collection of loss computation methods parameterized by the project config.

    All methods are safe to call from multiple threads/processes in distributed
    training, as they operate only on the input tensors and stored config values.

    Attributes:
        K: Number of pyramid stages (3 from config.pyramid.num_stages).
        stage_bounds: List of [s_k, e_k] time window pairs per stage_id.
            stage_bounds[0] = [0.0, 0.333] (lowest resolution, stage_id=2 in paper)
            stage_bounds[1] = [0.333, 0.667] (mid resolution, stage_id=1 in paper)
            stage_bounds[2] = [0.667, 1.0] (full resolution, stage_id=0 in paper)
            Indexed by stage_id (0=full res, K-1=lowest res) per Shared Knowledge.
        kl_weight: KL divergence loss weight for VAE training (1e-6 from config).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes FlowMatchingLoss from the project config.

        Reads all required values from configs/default.yaml via the
        omegaconf DictConfig (or plain dict) passed as ``config``.

        Args:
            config: Project configuration dictionary. Expected keys:
                - config['pyramid']['num_stages'] (int): 3
                - config['pyramid']['stage_bounds'] (list): [[0.0, 0.333],
                  [0.333, 0.667], [0.667, 1.0]]
                - config['vae']['kl_weight'] (float): 1.0e-6

        Raises:
            ValueError: If len(stage_bounds) != num_stages, or if any
                stage bound has s_k >= e_k (invalid time window).
        """
        # ----------------------------------------------------------------
        # Parse pyramid configuration
        # ----------------------------------------------------------------
        pyramid_cfg: Dict[str, Any] = config.get("pyramid", {})
        vae_cfg: Dict[str, Any] = config.get("vae", {})

        self.K: int = int(pyramid_cfg.get("num_stages", 3))

        # Parse stage_bounds: list of [s_k, e_k] pairs
        # Config ordering: index 0 = lowest res (k=K-1), index K-1 = full res (k=0)
        # This matches the stage_id convention from Shared Knowledge.
        raw_bounds: List[Any] = list(
            pyramid_cfg.get(
                "stage_bounds",
                [[0.0, 0.333], [0.333, 0.667], [0.667, 1.0]],
            )
        )

        # Convert to list of [float, float] pairs, handling omegaconf ListConfig
        self.stage_bounds: List[List[float]] = [
            [float(b[0]), float(b[1])] for b in raw_bounds
        ]

        # Validate stage_bounds length
        if len(self.stage_bounds) != self.K:
            raise ValueError(
                f"len(stage_bounds)={len(self.stage_bounds)} must equal "
                f"num_stages={self.K}. "
                f"Got stage_bounds={self.stage_bounds}. "
                f"Check configs/default.yaml pyramid.stage_bounds."
            )

        # Validate each stage window: s_k < e_k
        for stage_id, (s_k, e_k) in enumerate(self.stage_bounds):
            if s_k >= e_k:
                raise ValueError(
                    f"Invalid stage bounds at stage_id={stage_id}: "
                    f"s_k={s_k} must be strictly less than e_k={e_k}. "
                    f"Check configs/default.yaml pyramid.stage_bounds."
                )

        # ----------------------------------------------------------------
        # Parse VAE configuration
        # ----------------------------------------------------------------
        self.kl_weight: float = float(vae_cfg.get("kl_weight", 1.0e-6))

        logger.info(
            "FlowMatchingLoss initialized: K=%d, stage_bounds=%s, "
            "kl_weight=%.2e",
            self.K,
            self.stage_bounds,
            self.kl_weight,
        )

    # -----------------------------------------------------------------------
    # Core flow matching loss
    # -----------------------------------------------------------------------

    def compute(
        self,
        pred_velocity: Tensor,
        target_velocity: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Computes the unified flow matching MSE loss.

        Implements the pyramidal flow matching training objective from
        Section 3.2.1 of the paper (Eq. 6):
            E_{k,t,(x_hat_ek, x_hat_sk)} || v_t(x_hat_t) - (x_hat_ek - x_hat_sk) ||^2

        The target velocity ``(x_hat_ek - x_hat_sk)`` is precomputed by
        ``PyramidFlowModel.compute_velocity_target()`` and passed directly.
        This method only computes the MSE between predicted and target velocities.

        Supports optional masking for Patch n' Pack batches (Section 3.4,
        following Dehghani et al., 2023) where padding tokens should not
        contribute to the loss gradient.

        Args:
            pred_velocity: Predicted velocity field from the MM-DiT transformer.
                Shape can be any of:
                - [B, seq_len, patch_dim]: Patch-flattened transformer output
                - [B, C, H, W]: Spatial latent format (4D)
                - [B, C, T, H, W]: Spatiotemporal latent format (5D)
                The shape must exactly match ``target_velocity``.
            target_velocity: Target velocity tensor, precomputed as
                ``x_hat_ek - x_hat_sk`` by PyramidFlowModel.
                Same shape as ``pred_velocity``.
            mask: Optional attention/validity mask for packed sequences.
                Shape should be broadcastable to ``pred_velocity``:
                - [B, seq_len]: Boolean or float mask for sequence format.
                  True/1.0 = valid token, False/0.0 = padding token.
                - [B, C, H, W]: Spatial mask for 4D latent format.
                - [B, C, T, H, W]: Spatiotemporal mask for 5D format.
                If None, standard mean reduction is applied over all elements.

        Returns:
            Scalar loss tensor (0-dimensional) containing the MSE flow
            matching loss. Dtype matches ``pred_velocity``.

        Raises:
            ValueError: If ``pred_velocity`` and ``target_velocity`` have
                different shapes.

        Example:
            >>> loss_fn = FlowMatchingLoss(config)
            >>> pred_v = torch.randn(2, 576, 64)   # [B, seq_len, patch_dim]
            >>> target_v = torch.randn(2, 576, 64)
            >>> loss = loss_fn.compute(pred_v, target_v)
            >>> loss.shape
            torch.Size([])  # scalar

            >>> # With Patch n' Pack mask
            >>> mask = torch.ones(2, 576, dtype=torch.bool)
            >>> mask[0, 400:] = False  # Last 176 tokens are padding
            >>> loss_masked = loss_fn.compute(pred_v, target_v, mask=mask)
        """
        # ----------------------------------------------------------------
        # Input validation
        # ----------------------------------------------------------------
        if pred_velocity.shape != target_velocity.shape:
            raise ValueError(
                f"pred_velocity and target_velocity must have the same shape. "
                f"Got pred_velocity.shape={tuple(pred_velocity.shape)}, "
                f"target_velocity.shape={tuple(target_velocity.shape)}."
            )

        # ----------------------------------------------------------------
        # Compute element-wise squared difference
        # ----------------------------------------------------------------
        # diff: same shape as pred_velocity and target_velocity
        diff: Tensor = (pred_velocity - target_velocity) ** 2

        # ----------------------------------------------------------------
        # Apply mask if provided (Patch n' Pack support)
        # ----------------------------------------------------------------
        if mask is None:
            # Standard mean reduction over all elements
            loss: Tensor = diff.mean()
        else:
            # ----------------------------------------------------------------
            # Masked mean reduction: only valid (non-padding) tokens contribute
            # ----------------------------------------------------------------
            # Convert mask to float for multiplication
            # Handles both bool masks (True=valid) and float masks (1.0=valid)
            mask_float: Tensor = mask.to(dtype=diff.dtype, device=diff.device)

            # Expand mask to match diff shape for broadcasting
            # diff shape: [B, seq_len, patch_dim] or [B, C, H, W] or [B, C, T, H, W]
            # mask shape: [B, seq_len] or [B, C, H, W] or [B, C, T, H, W]
            #
            # For sequence format [B, seq_len]: expand to [B, seq_len, 1]
            # so it broadcasts over the patch_dim dimension.
            # For spatial/spatiotemporal formats: mask should already match.
            if mask_float.dim() < diff.dim():
                # Add trailing singleton dimensions for broadcasting
                num_extra_dims: int = diff.dim() - mask_float.dim()
                for _ in range(num_extra_dims):
                    mask_float = mask_float.unsqueeze(-1)
                # mask_float: [B, seq_len, 1, ...] — broadcasts over feature dims

            # Apply mask: zero out padding token contributions
            masked_diff: Tensor = diff * mask_float

            # Compute mean over valid tokens only
            # Sum of valid elements = mask_float.sum() * (feature_dims product)
            # We use the total sum of the mask (counting each feature dim once)
            # and multiply by the feature dimension size for correct normalization.
            #
            # More precisely: we want mean over all valid (token, feature) pairs.
            # valid_count = sum of mask_float broadcast to diff shape
            # = mask_float.expand_as(diff).sum()
            # This correctly counts each (token, feature) pair.
            valid_count: Tensor = mask_float.expand_as(diff).sum()

            # Normalize: sum of masked squared differences / number of valid elements
            # Add eps to prevent division by zero if all tokens are masked
            loss = masked_diff.sum() / (valid_count + _EPS)

        return loss

    # -----------------------------------------------------------------------
    # Timestep and stage sampling utilities
    # -----------------------------------------------------------------------

    def sample_t_in_stage(
        self,
        s_k: float,
        e_k: float,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Samples timesteps uniformly within a pyramid stage's time window.

        Implements the timestep sampling for the flow matching objective.
        The paper uses uniform sampling within each stage's time window
        [s_k, e_k] (Section 3.4: "different pyramidal stages are uniformly
        sampled in each update iteration").

        The returned absolute timestep t ∈ [s_k, e_k] is used to compute
        the rescaled timestep t' = (t - s_k) / (e_k - s_k) ∈ [0, 1] in
        PyramidFlowModel.interpolate_latent().

        Args:
            s_k: Stage start time (lower bound of time window).
                Example: 0.0 for lowest-resolution stage (stage_id=2).
                Example: 0.667 for full-resolution stage (stage_id=0).
            e_k: Stage end time (upper bound of time window).
                Example: 0.333 for lowest-resolution stage (stage_id=2).
                Example: 1.0 for full-resolution stage (stage_id=0).
            batch_size: Number of timestep samples to draw. Typically equals
                the per-GPU batch size for the current training step.
            device: Target device for the output tensor. If None, defaults
                to CPU. Should be set to the training device (e.g., cuda:0).

        Returns:
            Float32 tensor of shape [batch_size] with values uniformly
            sampled from [s_k, e_k]. Each element is an absolute timestep
            for one sample in the batch.

        Raises:
            ValueError: If s_k >= e_k (invalid time window) or batch_size <= 0.

        Example:
            >>> loss_fn = FlowMatchingLoss(config)
            >>> # Sample timesteps for the full-resolution stage
            >>> t = loss_fn.sample_t_in_stage(s_k=0.667, e_k=1.0, batch_size=8)
            >>> t.shape
            torch.Size([8])
            >>> (t >= 0.667).all() and (t <= 1.0).all()
            True
        """
        if s_k >= e_k:
            raise ValueError(
                f"s_k={s_k} must be strictly less than e_k={e_k}. "
                f"Invalid time window for stage sampling."
            )
        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got batch_size={batch_size}."
            )

        # Sample uniform random values in [0, 1) and scale to [s_k, e_k)
        # torch.rand gives values in [0, 1), so the result is in [s_k, e_k)
        # This is a half-open interval, which is fine for continuous distributions.
        t: Tensor = torch.rand(
            batch_size,
            dtype=torch.float32,
            device=device if device is not None else torch.device("cpu"),
        )

        # Scale from [0, 1) to [s_k, e_k)
        t = t * (e_k - s_k) + s_k

        return t

    def sample_stage_id(self, K: int) -> int:
        """Uniformly samples a pyramid stage index.

        Implements the stage sampling described in Section 3.4:
        "During training, different pyramidal stages are uniformly sampled
        in each update iteration."

        With K=3 stages, each stage has probability 1/3 of being selected.

        Args:
            K: Total number of pyramid stages. From config.pyramid.num_stages.
                Defaults to 3 in all experiments (Paper: Section 4.1).
                Must be >= 1.

        Returns:
            Python int in [0, K-1] representing the sampled stage index.
            The stage_id convention (from Shared Knowledge):
                - 0: full resolution (final stage, time window [0.667, 1.0])
                - 1: half resolution (mid stage, time window [0.333, 0.667])
                - 2: quarter resolution (first stage, time window [0.0, 0.333])

        Raises:
            ValueError: If K < 1.

        Example:
            >>> loss_fn = FlowMatchingLoss(config)
            >>> stage_id = loss_fn.sample_stage_id(K=3)
            >>> stage_id in [0, 1, 2]
            True
            >>> # Get corresponding time bounds
            >>> s_k = float(config['pyramid']['stage_bounds'][stage_id][0])
            >>> e_k = float(config['pyramid']['stage_bounds'][stage_id][1])
        """
        if K < 1:
            raise ValueError(
                f"K must be >= 1, got K={K}. "
                f"Number of pyramid stages must be at least 1."
            )

        # torch.randint(low, high, size) samples from {low, low+1, ..., high-1}
        # We want {0, 1, ..., K-1}, so low=0, high=K
        stage_id: int = int(torch.randint(0, K, (1,)).item())

        return stage_id

    # -----------------------------------------------------------------------
    # VAE loss
    # -----------------------------------------------------------------------

    def vae_loss(
        self,
        recon: Tensor,
        x: Tensor,
        mu: Tensor,
        logvar: Tensor,
        kl_weight: float = 1.0e-6,
    ) -> Tensor:
        """Computes the combined VAE training loss.

        Implements the standard VAE objective for pretraining the 3D causal
        VAE described in the paper (Section 4.1, Appendix B):
            L_VAE = L_recon + kl_weight * L_KL

        where:
            L_recon = MSE(recon, x)  [pixel/latent reconstruction]
            L_KL = -0.5 * E[1 + logvar - mu^2 - exp(logvar)]  [KL divergence]

        The KL term regularizes the latent space toward N(0, I), preventing
        posterior collapse while maintaining high reconstruction quality.
        The very small kl_weight=1e-6 (from config.vae.kl_weight) ensures
        reconstruction quality dominates, consistent with MAGVIT-v2 style
        training (Paper: Appendix B).

        Numerical stability: logvar is clamped to [-30, 20] before
        exponentiation to prevent overflow in bfloat16 training.

        Args:
            recon: Reconstructed video tensor from the VAE decoder.
                Shape: [B, C, T, H, W] with values in [-1, 1] (Tanh output).
                Must have the same shape as ``x``.
            x: Original input video tensor (ground truth).
                Shape: [B, C, T, H, W] with values in [-1, 1] (normalized).
                Must have the same shape as ``recon``.
            mu: Encoder mean output from the VAE encoder.
                Shape: [B, latent_channels, T//8, H//8, W//8].
                Represents the mean of the approximate posterior q(z|x).
            logvar: Encoder log-variance output from the VAE encoder.
                Shape: [B, latent_channels, T//8, H//8, W//8].
                Represents log(sigma^2) of the approximate posterior q(z|x).
                Clamped to [-30, 20] internally for numerical stability.
            kl_weight: Weight for the KL divergence term. Defaults to 1e-6
                as specified in config.vae.kl_weight. This very small value
                ensures reconstruction quality dominates during VAE training.

        Returns:
            Scalar loss tensor (0-dimensional) containing the combined
            VAE loss L_recon + kl_weight * L_KL. Dtype matches ``recon``.

        Raises:
            ValueError: If ``recon`` and ``x`` have different shapes, or if
                ``mu`` and ``logvar`` have different shapes.

        Example:
            >>> loss_fn = FlowMatchingLoss(config)
            >>> B, C, T, H, W = 2, 3, 8, 64, 64
            >>> recon = torch.randn(B, C, T, H, W)
            >>> x = torch.randn(B, C, T, H, W)
            >>> mu = torch.randn(B, 16, 1, 8, 8)
            >>> logvar = torch.randn(B, 16, 1, 8, 8)
            >>> loss = loss_fn.vae_loss(recon, x, mu, logvar)
            >>> loss.shape
            torch.Size([])  # scalar
        """
        # ----------------------------------------------------------------
        # Input validation
        # ----------------------------------------------------------------
        if recon.shape != x.shape:
            raise ValueError(
                f"recon and x must have the same shape. "
                f"Got recon.shape={tuple(recon.shape)}, "
                f"x.shape={tuple(x.shape)}."
            )

        if mu.shape != logvar.shape:
            raise ValueError(
                f"mu and logvar must have the same shape. "
                f"Got mu.shape={tuple(mu.shape)}, "
                f"logvar.shape={tuple(logvar.shape)}."
            )

        # ----------------------------------------------------------------
        # Reconstruction loss: MSE between reconstructed and original video
        # ----------------------------------------------------------------
        # F.mse_loss with reduction='mean' computes:
        #   L_recon = (1 / N) * sum((recon_i - x_i)^2)
        # where N = total number of elements (B * C * T * H * W)
        l_recon: Tensor = F.mse_loss(recon, x, reduction="mean")

        # ----------------------------------------------------------------
        # KL divergence loss: KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # ----------------------------------------------------------------
        # Closed-form KL divergence for diagonal Gaussian:
        #   KL(N(mu, sigma^2) || N(0, I))
        #   = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        #   = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        #
        # Clamp logvar for numerical stability in bfloat16:
        # exp(20) ≈ 4.85e8 (safe), exp(30) ≈ 1.07e13 (may overflow in bf16)
        logvar_clamped: Tensor = logvar.clamp(
            min=_LOGVAR_MIN, max=_LOGVAR_MAX
        )

        # Per-element KL: -0.5 * (1 + logvar - mu^2 - exp(logvar))
        # Mean reduction over all latent dimensions (B, latent_channels, T//8, H//8, W//8)
        l_kl: Tensor = -0.5 * (
            1.0 + logvar_clamped - mu.pow(2) - logvar_clamped.exp()
        ).mean()

        # ----------------------------------------------------------------
        # Combined VAE loss
        # ----------------------------------------------------------------
        # L_total = L_recon + kl_weight * L_KL
        # kl_weight=1e-6 ensures reconstruction quality dominates
        l_total: Tensor = l_recon + float(kl_weight) * l_kl

        logger.debug(
            "VAE loss: total=%.6f, recon=%.6f, kl=%.6f (kl_weight=%.2e)",
            l_total.item(),
            l_recon.item(),
            l_kl.item(),
            kl_weight,
        )

        return l_total

    # -----------------------------------------------------------------------
    # Convenience accessors
    # -----------------------------------------------------------------------

    def get_stage_bounds(self, stage_id: int) -> List[float]:
        """Returns the [s_k, e_k] time bounds for a given stage_id.

        Convenience accessor that handles omegaconf ListConfig conversion
        and validates the stage_id range.

        Args:
            stage_id: Pyramid stage index in [0, K-1].
                0 = full resolution (final stage).
                K-1 = lowest resolution (first stage).

        Returns:
            List [s_k, e_k] of float time bounds for the stage.

        Raises:
            ValueError: If stage_id is outside [0, K-1].

        Example:
            >>> loss_fn = FlowMatchingLoss(config)
            >>> s_k, e_k = loss_fn.get_stage_bounds(stage_id=0)
            >>> s_k, e_k
            (0.667, 1.0)
        """
        if stage_id < 0 or stage_id >= self.K:
            raise ValueError(
                f"stage_id={stage_id} is out of range [0, {self.K - 1}]. "
                f"Must be in [0, K-1] where K={self.K}."
            )

        bounds: List[float] = self.stage_bounds[stage_id]
        return [float(bounds[0]), float(bounds[1])]

    def __repr__(self) -> str:
        """Returns a string representation of the loss configuration.

        Returns:
            String describing key hyperparameters for debugging.
        """
        return (
            f"FlowMatchingLoss("
            f"K={self.K}, "
            f"stage_bounds={self.stage_bounds}, "
            f"kl_weight={self.kl_weight:.2e}"
            f")"
        )
