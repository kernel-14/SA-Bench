```python
## models/pyramid_flow.py
"""Pyramidal Flow Matching model for efficient video generative modeling.

Implements the core pyramidal flow matching algorithm from the paper
"Pyramidal Flow Matching for Efficient Video Generative Modeling".

This module orchestrates the spatial pyramid (Section 3.2) and temporal
pyramid (Section 3.3) designs into a unified training and inference interface.

Key algorithms implemented:
- Coupled endpoint sampling (Eqs. 9-10): shared noise for both stage endpoints
- Unified flow matching objective (Section 3.2.1): single DiT for all stages
- Renoising at jump points (Eq. 15): maintains probability path continuity
- Temporal pyramid history conditioning (Section 3.3): compressed history

Tensor convention (from Shared Knowledge):
- Videos: [B, C, T, H, W] in pixel space
- Latents: [B, latent_dim, T//8, H//8, W//8] in latent space
- Stage indexing: stage_id=0 = full resolution (final), stage_id=K-1 = lowest res

Usage:
    from models.pyramid_flow import PyramidFlowModel

    model = PyramidFlowModel(vae, transformer, text_encoders, pos_enc, config)

    # Training forward pass
    outputs = model.forward(batch)
    loss = loss_fn.compute(outputs['pred_velocity'], outputs['target_velocity'])

    # Inference: see inference/sampler.py
    x_hat_ek, x_hat_sk = model.sample_coupled_endpoints(x1, stage_id=0)
    x_hat_sk_next = model.renoise_at_jump(x_hat_ek, s_k=0.667)
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.mmdit import MMDiT
from models.positional_encoding import PositionalEncoding
from models.text_encoders import TextEncoders
from models.vae_3d import VAE3D
from utils.logging import get_logger

## ---------------------------------------------------------------------------
## Module-level logger
## ---------------------------------------------------------------------------
logger = get_logger(__name__)


class PyramidFlowModel(nn.Module):
    """Unified pyramidal flow matching model for image and video generation.

    Combines the 3D VAE, MM-DiT transformer, text encoders, and positional
    encoding into a single end-to-end trainable module. Implements the
    pyramidal flow matching algorithm from the paper.

    The spatial pyramid divides the flow trajectory into K=3 stages, each
    operating at a different spatial resolution. Only the final stage (stage_id=0)
    operates at full resolution. The temporal pyramid compresses history frames
    for autoregressive video generation.

    Stage indexing convention (from Shared Knowledge):
        - stage_id=0: full resolution, stage_bounds[2] = [0.667, 1.0]
        - stage_id=1: half resolution, stage_bounds[1] = [0.333, 0.667]
        - stage_id=2: quarter resolution, stage_bounds[0] = [0.0, 0.333]

    Downsampling factors: downsample_factors[stage_id] = 2^stage_id
        - stage_id=0: factor=1 (no downsampling)
        - stage_id=1: factor=2 (2x downsampling)
        - stage_id=2: factor=4 (4x downsampling)

    Attributes:
        vae: 3D causal VAE for pixel↔latent compression.
        transformer: MM-DiT backbone predicting the velocity field v_t.
        text_encoders: Frozen T5 + CLIP text encoders.
        pos_enc: Positional encoding (sinusoidal + RoPE).
        K: Number of pyramid stages (3 from config).
        stage_bounds: List of [s_k, e_k] per stage_id. Ordered [k=2, k=1, k=0].
        gamma: Blockwise covariance parameter for renoising (-1/3 from config).
        upsample_mode: Upsampling mode for noisy latents ("nearest" from config).
        downsample_mode: Downsampling mode for clean latents ("bilinear" from config).
        use_coupled_noise: Whether to use coupled noise sampling (True from config).
        history_noise_min: Min corruptive noise strength for history (0.0 from config).
        history_noise_max: Max corruptive noise strength for history (1/3 from config).
        num_history_frames: Number of history frames to condition on (2 from config).
        downsample_factors: Spatial downsampling factor per stage_id ([1, 2, 4]).
        latent_channels: VAE latent channel count (16 from config).
    """

    def __init__(
        self,
        vae: VAE3D,
        transformer: MMDiT,
        text_encoders: TextEncoders,
        pos_enc: PositionalEncoding,
        config: Dict[str, Any],
    ) -> None:
        """Initializes PyramidFlowModel.

        Args:
            vae: Pretrained or randomly initialized 3D causal VAE.
                Used to encode pixel videos to latents and decode back.
            transformer: MM-DiT backbone that predicts the velocity field.
                Initialized from SD3 Medium weights (Paper: Appendix B).
            text_encoders: Frozen T5-XXL + CLIP ViT-L/14 text encoders.
                Provides conditioning embeddings for the transformer.
            pos_enc: Positional encoding module for spatial and temporal dims.
                Handles pyramid extrapolation and interpolation.
            config: Project configuration dictionary from configs/default.yaml.
                Expected keys under config['pyramid']:
                    - num_stages (int): 3
                    - stage_bounds (list): [[0.0, 0.333], [0.333, 0.667], [0.667, 1.0]]
                    - gamma (float): -0.3333...
                    - upsample_mode (str): "nearest"
                    - downsample_mode (str): "bilinear"
                    - use_coupled_noise (bool): True
                    - downsample_factors (list): [1, 2, 4]
                    - temporal.history_noise_min (float): 0.0
                    - temporal.history_noise_max (float): 0.333
                    - temporal.num_history_frames (int): 2
                Also reads config['vae']['latent_channels'] (int): 16.
        """
        super().__init__()

        # ----------------------------------------------------------------
        # Store sub-modules
        # ----------------------------------------------------------------
        self.vae: VAE3D = vae
        self.transformer: MMDiT = transformer
        self.text_encoders: TextEncoders = text_encoders
        self.pos_enc: PositionalEncoding = pos_enc

        # ----------------------------------------------------------------
        # Parse pyramid configuration
        # ----------------------------------------------------------------
        pyramid_cfg: Dict[str, Any] = config.get("pyramid", {})
        temporal_cfg: Dict[str, Any] = pyramid_cfg.get("temporal", {})
        vae_cfg: Dict[str, Any] = config.get("vae", {})

        self.K: int = int(pyramid_cfg.get("num_stages", 3))

        # stage_bounds[i] = [s_k, e_k] for stage_id=i
        # Config ordering: [k=2 (lowest res), k=1 (mid), k=0 (full res)]
        # So stage_bounds[stage_id] directly gives [s_k, e_k] for that stage_id
        raw_bounds: List[List[float]] = list(
            pyramid_cfg.get(
                "stage_bounds",
                [[0.0, 0.333], [0.333, 0.667], [0.667, 1.0]],
            )
        )
        self.stage_bounds: List[List[float]] = [
            [float(b[0]), float(b[1])] for b in raw_bounds
        ]

        if len(self.stage_bounds) != self.K:
            raise ValueError(
                f"len(stage_bounds)={len(self.stage_bounds)} must equal "
                f"num_stages={self.K}. Got stage_bounds={self.stage_bounds}."
            )

        self.gamma: float = float(pyramid_cfg.get("gamma", -1.0 / 3.0))
        self.upsample_mode: str = str(pyramid_cfg.get("upsample_mode", "nearest"))
        self.downsample_mode: str = str(pyramid_cfg.get("downsample_mode", "bilinear"))
        self.use_coupled_noise: bool = bool(
            pyramid_cfg.get("use_coupled_noise", True)
        )

        # Downsampling factors: downsample_factors[stage_id] = 2^stage_id
        raw_factors: List[int] = list(
            pyramid_cfg.get("downsample_factors", [1, 2, 4])
        )
        self.downsample_factors: List[int] = [int(f) for f in raw_factors]

        if len(self.downsample_factors) != self.K:
            raise ValueError(
                f"len(downsample_factors)={len(self.downsample_factors)} must "
                f"equal num_stages={self.K}. "
                f"Got downsample_factors={self.downsample_factors}."
            )

        # Temporal pyramid parameters
        self.history_noise_min: float = float(
            temporal_cfg.get("history_noise_min", 0.0)
        )
        self.history_noise_max: float = float(
            temporal_cfg.get("history_noise_max", 1.0 / 3.0)
        )
        self.num_history_frames: int = int(
            temporal_cfg.get("num_history_frames", 2)
        )

        # VAE latent channels
        self.latent_channels: int = int(vae_cfg.get("latent_channels", 16))

        logger.info(
            "PyramidFlowModel initialized: K=%d, stage_bounds=%s, "
            "gamma=%.4f, upsample_mode=%s, downsample_mode=%s, "
            "use_coupled_noise=%s, downsample_factors=%s, "
            "history_noise=[%.3f, %.3f], num_history_frames=%d, "
            "latent_channels=%d",
            self.K,
            self.stage_bounds,
            self.gamma,
            self.upsample_mode,
            self.downsample_mode,
            self.use_coupled_noise,
            self.downsample_factors,
            self.history_noise_min,
            self.history_noise_max,
            self.num_history_frames,
            self.latent_channels,
        )

    # -----------------------------------------------------------------------
    # Spatial resampling utilities
    # -----------------------------------------------------------------------

    def downsample(
        self,
        x: Tensor,
        factor: int,
        mode: str = "bilinear",
    ) -> Tensor:
        """Spatially downsamples a latent tensor by an integer factor.

        Handles both 4D [B, C, H, W] and 5D [B, C, T, H, W] tensors.
        For 5D video tensors, downsampling is applied spatially (H and W only),
        not temporally. The temporal dimension is preserved.

        Args:
            x: Input latent tensor of shape [B, C, H, W] or [B, C, T, H, W].
            factor: Integer downsampling factor. Must be >= 1.
                factor=1 returns x unchanged (no-op for full-resolution stage).
                factor=2 halves H and W. factor=4 quarters H and W.
            mode: Interpolation mode for F.interpolate. Defaults to "bilinear"
                for clean x_1 latents (Paper: bilinear downsampling for data).
                Use "nearest" for noisy latents at jump points.

        Returns:
            Downsampled tensor of shape [B, C, H//factor, W//factor] or
            [B, C, T, H//factor, W//factor].

        Raises:
            ValueError: If factor < 1 or if H/W are not divisible by factor.

        Example:
            >>> x = torch.randn(2, 16, 16, 96, 96)  # [B, C, T, H, W]
            >>> x_down = model.downsample(x, factor=4)
            >>> x_down.shape
            torch.Size([2, 16, 16, 24, 24])
        """
        if factor < 1:
            raise ValueError(
                f"Downsampling factor must be >= 1, got factor={factor}."
            )

        # No-op for factor=1 (full resolution stage)
        if factor == 1:
            return x

        is_5d: bool = x.dim() == 5

        if is_5d:
            # [B, C, T, H, W] → reshape to [B*T, C, H, W] for 2D interpolation
            B, C, T, H, W = x.shape
            x_2d: Tensor = x.reshape(B * T, C, H, W)
        elif x.dim() == 4:
            x_2d = x
            B, C, H, W = x.shape
            T = 1
        else:
            raise ValueError(
                f"Input tensor must be 4D [B, C, H, W] or 5D [B, C, T, H, W], "
                f"got shape {tuple(x.shape)}."
            )

        # Validate divisibility
        if H % factor != 0 or W % factor != 0:
            raise ValueError(
                f"H={H} and W={W} must be divisible by factor={factor}. "
                f"Input shape: {tuple(x.shape)}."
            )

        # Apply 2D spatial interpolation
        scale: float = 1.0 / float(factor)
        align_corners: Optional[bool] = (
            False if mode in ("bilinear", "bicubic") else None
        )

        if align_corners is not None:
            x_down_2d: Tensor = F.interpolate(
                x_2d,
                scale_factor=scale,
                mode=mode,
                align_corners=align_corners,
                recompute_scale_factor=False,
            )
        else:
            x_down_2d = F.interpolate(
                x_2d,
                scale_factor=scale,
                mode=mode,
                recompute_scale_factor=False,
            )

        if is_5d:
            # Reshape back to [B, C, T, H//factor, W//factor]
            _, _, H_out, W_out = x_down_2d.shape
            return x_down_2d.reshape(B, C, T, H_out, W_out)

        return x_down_2d

    def upsample(
        self,
        x: Tensor,
        factor: int = 2,
        mode: str = "nearest",
    ) -> Tensor:
        """Spatially upsamples a latent tensor by an integer factor.

        Handles both 4D [B, C, H, W] and 5D [B, C, T, H, W] tensors.
        For 5D video tensors, upsampling is applied spatially (H and W only).

        The default mode is "nearest" because the paper's renoising derivation
        (Appendix A) uses nearest-neighbor upsampling, which produces the
        blockwise covariance structure Sigma_block needed for the corrective
        noise calculation.

        Args:
            x: Input latent tensor of shape [B, C, H, W] or [B, C, T, H, W].
            factor: Integer upsampling factor. Must be >= 1.
                factor=1 returns x unchanged. factor=2 doubles H and W.
            mode: Interpolation mode. Defaults to "nearest" for noisy latents
                at jump points (Paper: nearest-neighbor for renoising derivation).

        Returns:
            Upsampled tensor of shape [B, C, H*factor, W*factor] or
            [B, C, T, H*factor, W*factor].

        Raises:
            ValueError: If factor < 1.

        Example:
            >>> x = torch.randn(2, 16, 16, 24, 24)  # [B, C, T, H, W]
            >>> x_up = model.upsample(x, factor=2)
            >>> x_up.shape
            torch.Size([2, 16, 16, 48, 48])
        """
        if factor < 1:
            raise ValueError(
                f"Upsampling factor must be >= 1, got factor={factor}."
            )

        # No-op for factor=1
        if factor == 1:
            return x

        is_5d: bool = x.dim() == 5

        if is_5d:
            B, C, T, H, W = x.shape
            x_2d: Tensor = x.reshape(B * T, C, H, W)
        elif x.dim() == 4:
            x_2d = x
            B, C, H, W = x.shape
            T = 1
        else:
            raise ValueError(
                f"Input tensor must be 4D [B, C, H, W] or 5D [B, C, T, H, W], "
                f"got shape {tuple(x.shape)}."
            )

        # Apply 2D spatial upsampling
        scale: float = float(factor)
        align_corners: Optional[bool] = (
            False if mode in ("bilinear", "bicubic") else None
        )

        if align_corners is not None:
            x_up_2d: Tensor = F.interpolate(
                x_2d,
                scale_factor=scale,
                mode=mode,
                align_corners=align_corners,
            )
        else:
            x_up_2d = F.interpolate(
                x_2d,
                scale_factor=scale,
                mode=mode,
            )

        if is_5d:
            _, _, H_out, W_out = x_up_2d.shape
            return x_up_2d.reshape(B, C, T, H_out, W_out)

        return x_up_2d

    # -----------------------------------------------------------------------
    # Core pyramidal flow matching algorithms
    # -----------------------------------------------------------------------

    def sample_coupled_endpoints(
        self,
        x1: Tensor,
        stage_id: int,
    ) -> Tuple[Tensor, Tensor]:
        """Samples coupled start and end points for a pyramid stage.

        Implements Equations 9-10 from Section 3.2.1 of the paper:
            x_hat_{e_k} = e_k * Down(x_1, 2^k) + (1 - e_k) * n
            x_hat_{s_k} = s_k * Up(Down(x_1, 2^(k+1))) + (1 - s_k) * n

        The key insight is that the same noise tensor n is shared between
        both endpoints ("coupled noise"), which improves flow trajectory
        straightness by reducing intersections (Appendix C.4, Fig. 13).

        Args:
            x1: Clean data latent at full resolution, shape
                [B, latent_channels, T//8, H//8, W//8] or [B, C, H, W].
                This is the VAE-encoded video/image.
            stage_id: Pyramid stage index. 0=full resolution (final stage),
                K-1=lowest resolution (first stage).
                Must be in [0, K-1].

        Returns:
            Tuple (x_hat_ek, x_hat_sk) where:
                - x_hat_ek: End point of stage k, shape at stage_id resolution.
                  Noisy version of Down(x_1, 2^stage_id).
                - x_hat_sk: Start point of stage k, shape at stage_id resolution.
                  Noisy version of Up(Down(x_1, 2^(stage_id+1))).
                Both tensors have the same shape (stage_id resolution).

        Raises:
            ValueError: If stage_id is outside [0, K-1].

        Example:
            >>> x1 = torch.randn(2, 16, 16, 96, 96)  # Full-res latent
            >>> x_end, x_start = model.sample_coupled_endpoints(x1, stage_id=2)
            >>> x_end.shape  # Quarter resolution
            torch.Size([2, 16, 16, 24, 24])
            >>> x_start.shape  # Also quarter resolution
            torch.Size([2, 16, 16, 24, 24])
        """
        if stage_id < 0 or stage_id >= self.K:
            raise ValueError(
                f"stage_id={stage_id} is out of range [0, {self.K - 1}]. "
                f"Must be in [0, K-1] where K={self.K}."
            )

        # Retrieve stage time bounds: [s_k, e_k]
        s_k: float = self.stage_bounds[stage_id][0]
        e_k: float = self.stage_bounds[stage_id][1]

        # Downsampling factors for current and next coarser stage
        # factor_k = 2^stage_id (current stage resolution factor)
        # factor_k1 = 2^(stage_id+1) (next coarser stage resolution factor)
        factor_k: int = self.downsample_factors[stage_id]
        factor_k1: int = 2 ** (stage_id + 1)

        # ----------------------------------------------------------------
        # Step 1: Compute Down(x_1, 2^k) — clean latent at current resolution
        # ----------------------------------------------------------------
        x1_down_k: Tensor = self.downsample(
            x1, factor=factor_k, mode=self.downsample_mode
        )
        # Shape: [B, C, T//8, H//(8*factor_k), W//(8*factor_k)]
        # For stage_id=0 (full res): same as x1
        # For stage_id=2 (quarter res): H//32, W//32

        # ----------------------------------------------------------------
        # Step 2: Compute Up(Down(x_1, 2^(k+1))) — pixelated version at
        # current resolution (upsampled from next coarser level)
        # ----------------------------------------------------------------
        x1_down_k1: Tensor = self.downsample(
            x1, factor=factor_k1, mode=self.downsample_mode
        )
        # Shape: [B, C, T//8, H//(8*factor_k1), W//(8*factor_k1)]

        # Upsample by 2 to bring back to current stage resolution
        x1_up_down: Tensor = self.upsample(
            x1_down_k1, factor=2, mode=self.upsample_mode
        )
        # Shape: [B, C, T//8, H//(8*factor_k), W//(8*factor_k)]
        # Same spatial shape as x1_down_k

        # ----------------------------------------------------------------
        # Step 3: Sample shared noise at current stage resolution
        # ----------------------------------------------------------------
        if self.use_coupled_noise:
            # Coupled noise: same n for both endpoints (Eqs. 9-10)
            # Improves flow straightness by reducing trajectory intersections
            n: Tensor = torch.randn_like(x1_down_k)
        else:
            # Uncoupled noise: independent n for each endpoint (ablation)
            n = torch.randn_like(x1_down_k)

        # ----------------------------------------------------------------
        # Step 4: Compute endpoints
        # x_hat_{e_k} = e_k * Down(x_1, 2^k) + (1 - e_k) * n  [Eq. 9]
        # x_hat_{s_k} = s_k * Up(Down(x_1, 2^(k+1))) + (1 - s_k) * n  [Eq. 10]
        # ----------------------------------------------------------------
        x_hat_ek: Tensor = e_k * x1_down_k + (1.0 - e_k) * n
        x_hat_sk: Tensor = s_k * x1_up_down + (1.0 - s_k) * n

        return x_hat_ek, x_hat_sk

    def interpolate_latent(
        self,
        x_start: Tensor,
        x_end: Tensor,
        t_prime: Tensor,
    ) -> Tensor:
        """Linearly interpolates between stage endpoints at rescaled timestep t'.

        Implements the piecewise flow within a pyramid stage (Section 3.2):
            x_hat_t = t' * x_end + (1 - t') * x_start

        where t' = (t - s_k) / (e_k - s_k) is the rescaled timestep in [0, 1].

        At t'=0: returns x_start (noisy, pixelated start of stage)
        At t'=1: returns x_end (cleaner, higher-res end of stage)

        Args:
            x_start: Start point of the stage (x_hat_sk), shape [B, C, ...].
                Noisy and pixelated (upsampled from lower resolution).
            x_end: End point of the stage (x_hat_ek), shape [B, C, ...].
                Cleaner and at current stage resolution.
            t_prime: Rescaled timestep tensor of shape [B] with values in [0, 1].
                Computed as (t - s_k) / (e_k - s_k) from the absolute timestep.

        Returns:
            Interpolated latent x_hat_t of the same shape as x_start and x_end.

        Raises:
            ValueError: If x_start and x_end have different shapes.

        Example:
            >>> x_start = torch.randn(2, 16, 16, 24, 24)
            >>> x_end = torch.randn(2, 16, 16, 24, 24)
            >>> t_prime = torch.tensor([0.3, 0.7])
            >>> x_t = model.interpolate_latent(x_start, x_end, t_prime)
            >>> x_t.shape
            torch.Size([2, 16, 16, 24, 24])
        """
        if x_start.shape != x_end.shape:
            raise ValueError(
                f"x_start and x_end must have the same shape. "
                f"Got x_start.shape={tuple(x_start.shape)}, "
                f"x_end.shape={tuple(x_end.shape)}."
            )

        # Reshape t_prime for broadcasting over spatial/temporal dimensions
        # t_prime: [B] → [B, 1, 1, 1] (4D) or [B, 1, 1, 1, 1] (5D)
        ndim: int = x_start.dim()
        t_view: Tensor = t_prime.to(
            dtype=x_start.dtype, device=x_start.device
        )

        # Add singleton dimensions for broadcasting
        for _ in range(ndim - 1):
            t_view = t_view.unsqueeze(-1)
        # t_view: [B, 1, ...] with ndim-1 trailing singleton dims

        # Linear interpolation: x_hat_t = t' * x_end + (1 - t') * x_start
        x_hat_t: Tensor = t_view * x_end + (1.0 - t_view) * x_start

        return x_hat_t

    def compute_velocity_target(
        self,
        x_end: Tensor,
        x_start: Tensor,
    ) -> Tensor:
        """Computes the target velocity for the flow matching objective.

        Implements the conditional vector field from Section 3.2.1:
            u_t(x_hat_t | x_1) = x_hat_{e_k} - x_hat_{s_k}

        This is the constant velocity field within each pyramid stage,
        analogous to u(x_t | x_1) = x_1 - x_0 in standard flow matching.

        The unified training objective (Section 3.2.1) minimizes:
            E_{k,t,(x_hat_ek, x_hat_sk)} || v_t(x_hat_t) - (x_hat_ek - x_hat_sk) ||^2

        Args:
            x_end: End point of the stage (x_hat_ek), shape [B, C, ...].
            x_start: Start point of the stage (x_hat_sk), shape [B, C, ...].

        Returns:
            Target velocity tensor of the same shape as x_end and x_start.
            This is the regression target for the transformer's velocity output.

        Example:
            >>> x_end = torch.randn(2, 16, 16, 24, 24)
            >>> x_start = torch.randn(2, 16, 16, 24, 24)
            >>> target = model.compute_velocity_target(x_end, x_start)
            >>> target.shape
            torch.Size([2, 16, 16, 24, 24])
        """
        return x_end - x_start

    def generate_correlated_noise(
        self,
        shape: Tuple[int, ...],
        gamma: float = -1.0 / 3.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Generates noise with blockwise negative correlation structure.

        Implements the corrective noise n' from Section 3.2.2 and Appendix A.
        The noise has covariance matrix Sigma'_block with off-diagonal elements
        gamma = -1/3, which decorrelates the spatially correlated pixels
        introduced by nearest-neighbor upsampling at jump points.

        The blockwise structure: each 2×2 spatial block (corresponding to
        4 pixels upsampled from a single lower-resolution pixel) has the
        covariance:
            Sigma'_block = [[1, γ, γ, γ],
                            [γ, 1, γ, γ],
                            [γ, γ, 1, γ],
                            [γ, γ, γ, 1]]
        where γ = -1/3.

        Generation procedure (derived from Appendix A):
        For each 2×2 block with 4 pixels z_1, z_2, z_3, z_4 ~ N(0,1):
            z_mean = (z_1 + z_2 + z_3 + z_4) / 4
            n'_i = sqrt(4/3) * (z_i - z_mean)

        Verification:
            Var(n'_i) = (4/3) * Var(z_i - z_mean) = (4/3) * (3/4) = 1 ✓
            Cov(n'_i, n'_j) = (4/3) * (-1/4) = -1/3 = γ ✓

        Args:
            shape: Shape of the output noise tensor. Must be 4D [B, C, H, W]
                or 5D [B, C, T, H, W]. H and W must be even (divisible by 2)
                since the blockwise structure operates on 2×2 spatial blocks.
            gamma: Off-diagonal covariance value. Must be in [-1/3, 0].
                Defaults to -1/3 (minimum, maximally decorrelating).
                