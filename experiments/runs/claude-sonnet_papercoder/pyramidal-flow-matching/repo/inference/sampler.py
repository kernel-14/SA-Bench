## Code: inference/sampler.py

```python
## inference/sampler.py
"""Pyramidal flow matching inference sampler.

Implements the full pyramidal inference pipeline described in Algorithm 1
of the paper "Pyramidal Flow Matching for Efficient Video Generative Modeling".

The sampler orchestrates:
1. Multi-stage Euler ODE integration within each pyramid stage
2. Renoising at jump points between stages (Eq. 15, Appendix A)
3. Classifier-free guidance (CFG) at each denoising step
4. Autoregressive long-video generation with temporal pyramid history

Stage indexing convention (from Shared Knowledge):
    - stage_id=0: full resolution (final stage), stage_bounds[2] = [0.667, 1.0]
    - stage_id=1: half resolution (mid stage), stage_bounds[1] = [0.333, 0.667]
    - stage_id=2: quarter resolution (first stage), stage_bounds[0] = [0.0, 0.333]

Config references (configs/default.yaml):
    inference.num_steps_per_stage: 20
    inference.cfg_scale: 7.5
    inference.solver: "euler"
    inference.apply_cfg_all_stages: true
    inference.segment_latent_frames: 16
    pyramid.num_stages: 3
    pyramid.stage_bounds: [[0.0, 0.333], [0.333, 0.667], [0.667, 1.0]]
    vae.temporal_compression: 8
    vae.spatial_compression: 8
    vae.latent_channels: 16
    model.patch_size: 2
    model.dtype: "bfloat16"

Usage:
    from inference.sampler import InferenceSampler

    sampler = InferenceSampler(model=pyramid_flow_model, config=config)

    # Single-clip text-to-video generation
    video = sampler.sample_video(
        prompt="A beautiful sunset over the ocean",
        num_frames=121,
        resolution=(768, 768),
        guidance_scale=7.5,
    )
    # video: Tensor [1, 3, 121, 768, 768] in pixel space, values in [-1, 1]

    # Long-video autoregressive generation
    long_video = sampler.autoregressive_generate(
        prompt="A person walking through a forest",
        total_frames=241,
        segment_frames=16,
        resolution=(768, 768),
    )
    # long_video: Tensor [1, 3, 241, 768, 768]
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from models.pyramid_flow import PyramidFlowModel
from utils.logging import get_logger

## ---------------------------------------------------------------------------
## Module-level logger
## ---------------------------------------------------------------------------
logger = get_logger(__name__)


class InferenceSampler:
    """Full pyramidal inference pipeline for image and video generation.

    Implements Algorithm 1 from the paper: multi-stage Euler ODE integration
    with renoising at stage transitions, classifier-free guidance, and
    autoregressive long-video generation.

    The inference process follows the spatial pyramid in reverse order:
    starting from pure noise at the lowest resolution (stage k=K-1), running
    Euler steps within each stage, then renoising and upsampling to transition
    to the next higher-resolution stage, until the final full-resolution
    output is produced at stage k=0.

    Attributes:
        model: PyramidFlowModel providing transformer, VAE, and text encoders.
        num_steps_per_stage: Number of Euler steps per pyramid stage (20).
        cfg_scale: Classifier-free guidance scale (7.5).
        solver: ODE solver type ("euler").
        apply_cfg_all_stages: Whether to apply CFG at all stages (True).
        segment_latent_frames: Latent frames per autoregressive segment (16).
        K: Number of pyramid stages (3).
        stage_bounds: List of [s_k, e_k] per config index.
            stage_bounds[0] = [0.0, 0.333] (k=2, lowest res)
            stage_bounds[1] = [0.333, 0.667] (k=1, mid res)
            stage_bounds[2] = [0.667, 1.0] (k=0, full res)
        temporal_compression: VAE temporal compression factor (8).
        spatial_compression: VAE spatial compression factor (8).
        latent_channels: VAE latent channel count (16).
        patch_size: Spatial patch size for patchification (2).
        upsample_mode: Upsampling mode for renoising ("nearest").
        gamma: Blockwise covariance parameter for renoising (-1/3).
        device: Torch device for inference tensors.
        dtype: Torch dtype for inference tensors (bfloat16).
        use_noisy_history_at_inference: Whether to add noise to history (False).
    """

    def __init__(
        self,
        model: PyramidFlowModel,
        config: Dict[str, Any],
    ) -> None:
        """Initializes InferenceSampler from the project config.

        Reads all inference hyperparameters from configs/default.yaml via
        the omegaconf DictConfig (or plain dict) passed as ``config``.

        Args:
            model: Trained PyramidFlowModel. Must have:
                - model.transformer: MMDiT for velocity prediction
                - model.vae: VAE3D for encode/decode
                - model.text_encoders: TextEncoders for T5+CLIP encoding
                - model.renoise_at_jump(): renoising at stage transitions
            config: Project configuration dictionary. Expected keys:
                - config['inference']['num_steps_per_stage'] (int): 20
                - config['inference']['cfg_scale'] (float): 7.5
                - config['inference']['solver'] (str): "euler"
                - config['inference']['apply_cfg_all_stages'] (bool): True
                - config['inference']['segment_latent_frames'] (int): 16
                - config['pyramid']['num_stages'] (int): 3
                - config['pyramid']['stage_bounds'] (list): [[0,0.333],...]
                - config['pyramid']['upsample_mode'] (str): "nearest"
                - config['pyramid']['gamma'] (float): -0.333
                - config['pyramid']['temporal']['use_noisy_history_at_inference'] (bool): False
                - config['vae']['temporal_compression'] (int): 8
                - config['vae']['spatial_compression'] (int): 8
                - config['vae']['latent_channels'] (int): 16
                - config['model']['patch_size'] (int): 2
                - config['model']['dtype'] (str): "bfloat16"
        """
        self.model: PyramidFlowModel = model

        # ----------------------------------------------------------------
        # Parse inference configuration
        # ----------------------------------------------------------------
        inference_cfg: Dict[str, Any] = config.get("inference", {})

        self.num_steps_per_stage: int = int(
            inference_cfg.get("num_steps_per_stage", 20)
        )
        self.cfg_scale: float = float(inference_cfg.get("cfg_scale", 7.5))
        self.solver: str = str(inference_cfg.get("solver", "euler"))
        self.apply_cfg_all_stages: bool = bool(
            inference_cfg.get("apply_cfg_all_stages", True)
        )
        self.segment_latent_frames: int = int(
            inference_cfg.get("segment_latent_frames", 16)
        )

        # ----------------------------------------------------------------
        # Parse pyramid configuration
        # ----------------------------------------------------------------
        pyramid_cfg: Dict[str, Any] = config.get("pyramid", {})
        temporal_cfg: Dict[str, Any] = pyramid_cfg.get("temporal", {})

        self.K: int = int(pyramid_cfg.get("num_stages", 3))

        # stage_bounds[i] = [s_k, e_k] for config index i
        # Config ordering: index 0 = k=K-1 (lowest res), index K-1 = k=0 (full res)
        raw_bounds: List[Any] = list(
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
                f"num_stages={self.K}. "
                f"Got stage_bounds={self.stage_bounds}."
            )

        self.upsample_mode: str = str(pyramid_cfg.get("upsample_mode", "nearest"))
        self.gamma: float = float(pyramid_cfg.get("gamma", -1.0 / 3.0))

        self.use_noisy_history_at_inference: bool = bool(
            temporal_cfg.get("use_noisy_history_at_inference", False)
        )

        # ----------------------------------------------------------------
        # Parse VAE configuration
        # ----------------------------------------------------------------
        vae_cfg: Dict[str, Any] = config.get("vae", {})

        self.temporal_compression: int = int(
            vae_cfg.get("temporal_compression", 8)
        )
        self.spatial_compression: int = int(
            vae_cfg.get("spatial_compression", 8)
        )
        self.latent_channels: int = int(vae_cfg.get("latent_channels", 16))

        # ----------------------------------------------------------------
        # Parse model configuration
        # ----------------------------------------------------------------
        model_cfg: Dict[str, Any] = config.get("model", {})

        self.patch_size: int = int(model_cfg.get("patch_size", 2))

        # Determine dtype from config
        dtype_str: str = str(model_cfg.get("dtype", "bfloat16"))
        if dtype_str == "bfloat16":
            self.dtype: torch.dtype = torch.bfloat16
        elif dtype_str == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        # ----------------------------------------------------------------
        # Determine device from model parameters
        # ----------------------------------------------------------------
        try:
            self.device: torch.device = next(
                self.model.transformer.parameters()
            ).device
        except StopIteration:
            # Fallback if model has no parameters
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        logger.info(
            "InferenceSampler initialized: K=%d, stage_bounds=%s, "
            "num_steps_per_stage=%d, cfg_scale=%.1f, solver=%s, "
            "apply_cfg_all_stages=%s, segment_latent_frames=%d, "
            "temporal_compression=%d, spatial_compression=%d, "
            "latent_channels=%d, patch_size=%d, dtype=%s, device=%s",
            self.K,
            self.stage_bounds,
            self.num_steps_per_stage,
            self.cfg_scale,
            self.solver,
            self.apply_cfg_all_stages,
            self.segment_latent_frames,
            self.temporal_compression,
            self.spatial_compression,
            self.latent_channels,
            self.patch_size,
            dtype_str,
            self.device,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _get_stage_bounds_for_k(self, k: int) -> Tuple[float, float]:
        """Returns (s_k, e_k) time bounds for pyramid stage k.

        Maps from stage k (0=full-res, K-1=lowest-res) to the config
        index (0=lowest-res, K-1=full-res).

        Config storage order: stage_bounds[0] = k=K-1 (lowest res),
        stage_bounds[K-1] = k=0 (full res).

        Mapping: config_index = K - 1 - k

        Args:
            k: Pyramid stage index. 0=full resolution (final stage),
               K-1=lowest resolution (first stage).

        Returns:
            Tuple (s_k, e_k) of float time bounds for stage k.

        Raises:
            ValueError: If k is outside [0, K-1].
        """
        if k < 0 or k >= self.K:
            raise ValueError(
                f"k={k} is out of range [0, {self.K - 1}]. "
                f"Must be in [0, K-1] where K={self.K}."
            )
        # Config index: 0 = k=K-1 (lowest res), K-1 = k=0 (full res)
        config_idx: int = self.K - 1 - k
        bounds: List[float] = self.stage_bounds[config_idx]
        return float(bounds[0]), float(bounds[1])

    def _compute_frame_token_counts(
        self,
        latent: Tensor,
    ) -> List[int]:
        """Computes the number of tokens per latent frame for causal masking.

        After patchification (2×2 patches), each latent frame of spatial
        size (H_latent, W_latent) contributes
        (H_latent // patch_size) * (W_latent // patch_size) tokens.

        Args:
            latent: Latent tensor of shape [B, C, T, H, W] (5D video) or
                [B, C, H, W] (4D image/single-frame).

        Returns:
            List of length T (or 1 for images) where each entry is the
            number of tokens for that latent frame.
        """
        if latent.dim() == 5:
            # [B, C, T, H, W]
            T: int = latent.shape[2]
            H: int = latent.shape[3]
            W: int = latent.shape[4]
        elif latent.dim() == 4:
            # [B, C, H, W] — treat as single frame
            T = 1
            H = latent.shape[2]
            W = latent.shape[3]
        else:
            raise ValueError(
                f"latent must be 4D [B, C, H, W] or 5D [B, C, T, H, W], "
                f"got shape {tuple(latent.shape)}."
            )

        # Tokens per frame after patchification
        tokens_h: int = max(1, H // self.patch_size)
        tokens_w: int = max(1, W // self.patch_size)
        tokens_per_frame: int = tokens_h * tokens_w

        return [tokens_per_frame] * T

    def _concat_text_cond(
        self,
        cond: Dict[str, Tensor],
        uncond: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Concatenates conditional and unconditional text embeddings for CFG.

        Doubles the batch dimension by concatenating [uncond, cond] along
        dim=0. The split order is [uncond | cond] so that after the forward
        pass, `v_uncond, v_cond = output.chunk(2, dim=0)`.

        Args:
            cond: Conditional text embeddings dict with keys:
                't5_embeds', 'clip_embeds', 'attention_mask'.
            uncond: Unconditional (null) text embeddings dict with same keys.

        Returns:
            Combined dict with doubled batch dimension [uncond | cond].
        """
        combined: Dict[str, Tensor] = {}
        for key in cond:
            if key in uncond:
                # Concatenate uncond first, then cond: [uncond | cond]
                combined[key] = torch.cat(
                    [uncond[key], cond[key]], dim=0
                )
            else:
                combined[key] = cond[key]
        return combined

    def _build_history_for_stage(
        self,
        generated_latents: List[Tensor],
        k: int,
    ) -> Optional[Tensor]:
        """Builds the temporal pyramid history condition for stage k.

        Implements Section 3.3 of the paper: history frames are compressed
        at different factors depending on their distance from the current frame.

        At stage k:
        - Most recent history frame: Down(x, 2^k) — same compression as current stage
        - Older history frames: Down(x, 2^(k+1)) — more aggressive compression

        At inference, clean generated latents are used (no noise added),
        consistent with config.pyramid.temporal.use_noisy_history_at_inference=false.

        Args:
            generated_latents: List of previously generated latent tensors,
                each of shape [1, C, T_seg, H_latent, W_latent] at full
                latent resolution (before pyramid downsampling).
            k: Current pyramid stage index. 0=full res, K-1=lowest res.

        Returns:
            History tensor of shape [1, C, T_history, H_k, W_k] where
            T_history is the total number of history latent frames and
            H_k, W_k are the spatial dimensions at stage k.
            Returns None if no history is available (first segment).
        """
        if not generated_latents:
            return None

        # Downsampling factors for history frames
        # factor_k = 2^k (current stage spatial compression)
        # factor_k1 = 2^(k+1) (older frames, more compressed)
        factor_k: int = 2 ** k
        factor_k1: int = 2 ** (k + 1)

        history_frames: List[Tensor] = []

        # Process history latents: most recent last
        # Per Section 3.3: most recent frame at factor_k, older at factor_k1
        num_history: int = len(generated_latents)

        for hist_idx, hist_latent in enumerate(generated_latents):
            # hist_latent: [1, C, T_seg, H_latent, W_latent]
            is_most_recent: bool = (hist_idx == num_history - 1)
            compression_factor: int = factor_k if is_most_recent else factor_k1

            if compression_factor > 1:
                # Spatially downsample the history latent
                B, C, T_seg, H_lat, W_lat = hist_latent.shape
                # Reshape to [B*T_seg, C, H_lat, W_lat] for 2D interpolation
                hist_2d: Tensor = hist_latent.reshape(B * T_seg, C, H_lat, W_lat)
                scale: float = 1.0 / float(compression_factor)
                hist_down_2d: Tensor = F.interpolate(
                    hist_2d.float(),
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=False,
                ).to(dtype=self.dtype)
                _, _, H_down, W_down = hist_down_2d.shape
                hist_down: Tensor = hist_down_2d.reshape(B, C, T_seg, H_down, W_down)
            else:
                hist_down = hist_latent.to(dtype=self.dtype)

            history_frames.append(hist_down)

        if not history_frames:
            return None

        # Concatenate all history frames along the temporal dimension
        # Each history_frames[i]: [1, C, T_seg, H_k, W_k]
        history_tensor: Tensor = torch.cat(history_frames, dim=2)
        # Shape: [1, C, T_total_history, H_k, W_k]

        return history_tensor

    # -----------------------------------------------------------------------
    # Core inference primitives
    # -----------------------------------------------------------------------

    def euler_step(
        self,
        x: Tensor,
        velocity: Tensor,
        dt: float,
    ) -> Tensor:
        """Performs a single Euler ODE integration step.

        Implements the first-order Euler method for integrating the flow ODE:
            dx/dt = v_t(x_t)
        Discretized as:
            x_{t+dt} = x_t + dt * v_t(x_t)

        Since we integrate from s_k (noisy) toward e_k (clean), dt is
        positive: dt = (e_k - s_k) / num_steps > 0.

        Args:
            x: Current noisy latent tensor at timestep t.
                Shape: [B, C, H, W] (4D) or [B, C, T, H, W] (5D).
            velocity: Predicted velocity field v_t(x_t) from the transformer.
                Same shape as x.
            dt: Timestep increment. Positive for forward integration
                (noise → data direction in flow matching).

        Returns:
            Updated latent tensor x_{t+dt} of the same shape as x.

        Example:
            >>> x = torch.randn(1, 16, 16, 24, 24)
            >>> v = torch.randn_like(x)
            >>> x_next = sampler.euler_step(x, v, dt=0.333/20)
            >>> x_next.shape
            torch.Size([1, 16, 16, 24, 24])
        """
        return x + dt * velocity

    def sample_within_stage(
        self,
        x_start: Tensor,
        stage_id: int,
        text_cond: Dict[str, Tensor],
        num_steps: int,
        cfg_scale: float,
        history_latent: Optional[Tensor] = None,
    ) -> Tensor:
        """Runs Euler ODE integration within a single pyramid stage.

        Integrates the velocity field from s_k to e_k using ``num_steps``
        uniform Euler steps, applying classifier-free guidance at each step.

        The transformer is called with the absolute timestep t ∈ [s_k, e_k]
        and the stage_id, which together determine the pyramid stage context.

        For efficiency, conditional and unconditional passes are batched
        together by doubling the batch dimension (CFG batching), halving
        the number of transformer forward passes.

        Args:
            x_start: Starting noisy latent at timestep s_k.
                Shape: [B, C, T, H_k, W_k] (5D video) or [B, C, H_k, W_k] (4D image).
                H_k = H_latent // 2^stage_id, W_k = W_latent // 2^stage_id.
            stage_id: Pyramid stage index. 0=full resolution, K-1=lowest resolution.
                Determines the time window [s_k, e_k] and spatial resolution.
            text_cond: Conditional text embeddings dict from TextEncoders.encode().
                Keys: 't5_embeds' [B, seq_len, 4096], 'clip_embeds' [B, 768],
                'attention_mask' [B, seq_len].
            num_steps: Number of Euler integration steps within this stage.
                From config.inference.num_steps_per_stage (20).
            cfg_scale: Classifier-free guidance scale. Values > 1.0 enable CFG.
                From config.inference.cfg_scale (7.5).
            history_latent: Optional compressed history tensor for autoregressive
                generation. Shape: [B, C, T_history, H_k, W_k].
                None for single-clip generation (no history).

        Returns:
            Final latent x_hat_{e_k} at the end of this stage.
            Same shape as x_start.

        Example:
            >>> x_start = torch.randn(1, 16, 16, 24, 24)  # Stage k=2 start
            >>> x_end = sampler.sample_within_stage(
            ...     x_start, stage_id=2, text_cond=text_cond,
            ...     num_steps=20, cfg_scale=7.5
            ... )
            >>> x_end.shape
            torch.Size([1, 16, 16, 24, 24])
        """
        # ----------------------------------------------------------------
        # Retrieve stage time bounds
        # ----------------------------------------------------------------
        s_k: float
        e_k: float
        s_k, e_k = self._get_stage_bounds_for_k(stage_id)

        # Uniform step size within the stage
        dt: float = (e_k - s_k) / float(max(1, num_steps))

        # ----------------------------------------------------------------
        # Prepare batch size and device
        # ----------------------------------------------------------------
        batch_size: int = x_start.shape[0]
        device: torch.device = x_start.device

        # ----------------------------------------------------------------
        # Compute frame token counts for causal attention mask
        # ----------------------------------------------------------------
        frame_token_counts: List[int] = self._compute_frame_token_counts(x_start)

        # ----------------------------------------------------------------
        # Prepare unconditional text conditioning for CFG
        # ----------------------------------------------------------------
        use_cfg: bool = cfg_scale > 1.0 and self.apply_cfg_all_stages
        text_cond_uncond: Optional[Dict[str, Tensor]] = None

        if use_cfg:
            text_cond_uncond = self.model.text_encoders.null_embed(
                batch_size=batch_size
            )
            # Move to correct device and dtype
            text_cond_uncond = {
                k: v.to(device=device, dtype=self.dtype)
                for k, v in text_cond_uncond.items()
            }

        # Move conditional text cond to correct device and dtype
        text_cond_device: Dict[str, Tensor] = {
            k: v.to(device=device, dtype=self.dtype)
            for k, v in text_cond.items()
        }

        # ----------------------------------------------------------------
        # Euler integration loop
        # ----------------------------------------------------------------
        x: Tensor = x_start.to(device=device, dtype=self.dtype)
        t: float = s_k

        for step_idx in range(num_steps):
            # Current absolute timestep tensor [B]
            t_tensor: Tensor = torch.full(
                (batch_size,),
                fill_value=t,
                dtype=self.dtype,
                device=device,
            )

            if use_cfg and text_cond_uncond is not None:
                # --------------------------------------------------------
                # CFG batching: double batch for single forward pass
                # Concatenate [uncond | cond] along batch dimension
                # --------------------------------------------------------
                x_double: Tensor = torch.cat([x, x], dim=0)
                # [2B, C, ...]

                t_double: Tensor = torch.cat([t_tensor, t_tensor], dim=0)
                # [2B]

                text_cond_double: Dict[str, Tensor] = self._concat_text_cond(
                    cond=text_cond_device,
                    uncond=text_cond_uncond,
                )
                # Each value: [2B, ...]

                # History latent doubled for CFG batch
                history_double: Optional[Tensor] = None
                if history_latent is not None:
                    history_double = torch.cat(
                        [history_latent, history_latent], dim=0
                    )

                # Frame token counts for doubled batch (same per sample)
                frame_token_counts_double: List[int] = frame_token_counts * 2

                # Single forward pass for both uncond and cond
                v_double: Tensor = self.model.transformer.forward(
                    latent=x_double,
                    timesteps=t_double,
                    text_cond=text_cond_double,
                    frame_token_counts=frame_token_counts_double,
                    stage_id=stage_id,
                    history_latent=history_double,
                )
                # v_double: [2B, C, ...] — [uncond | cond]

                # Split: first half is uncond, second half is cond
                v_uncond: Tensor
                v_cond: Tensor
                v_uncond, v_cond = v_double.chunk(2, dim=0)
                # Each: [B, C, ...]

                # CFG combination: v = v_uncond + scale * (v_cond - v_uncond)
                v: Tensor = v_uncond + cfg_scale * (v_cond - v_uncond)

            else:
                # --------------------------------------------------------
                # No CFG: single conditional forward pass
                # --------------------------------------------------------
                v = self.model.transformer.forward(
                    latent=x,
                    timesteps=t_tensor,
                    text_cond=text_cond_device,
                    frame_token_counts=frame_token_counts,
                    stage_id=stage_id,
                    history_latent=history_latent,
                )
                # v: [B, C, ...]

            # ----------------------------------------------------------------
            # Euler step: x_{t+dt} = x_t + dt * v_t(x_t)
            # ----------------------------------------------------------------
            x = self.euler_step(x, v, dt)

            # Advance timestep
            t = t + dt

        # x is now x_hat_{e_k}
        return x

    def apply_renoise(
        self,
        x_end: Tensor,
        s_k: float,
    ) -> Tensor:
        """Applies renoising at a pyramid stage jump point.

        Implements Equation 15 from Section 3.2.2 of the paper:
            x_hat_{s_k} = ((1 + s_k) / 2) * Up(x_hat_{e_{k+1}})
                        + (sqrt(3) * (1 - s_k) / 2) * n'

        where n' has blockwise correlated structure with gamma = -1/3.

        This maintains continuity of the probability path between stages
        by matching the Gaussian distributions at the jump point. The
        upsampling doubles the spatial resolution (H and W), while the
        correlated noise decorrelates the spatially adjacent pixels that
        were upsampled from the same lower-resolution pixel.

        Delegates to model.renoise_at_jump() which implements the full
        correlated noise generation and renoising formula.

        Args:
            x_end: Output of the previous (lower-resolution) stage,
                x_hat_{e_{k+1}}. Shape: [B, C, T, H_low, W_low] or
                [B, C, H_low, W_low].
                After renoising, spatial dims are doubled: H_high = H_low * 2.
            s_k: Start timestep of the next (higher-resolution) stage k.
                This is stage_bounds[K-1-k][0] for the destination stage k.
                Used to compute the rescaling coefficient (1+s_k)/2 and
                noise weight sqrt(3)*(1-s_k)/2.

        Returns:
            Renoised latent x_hat_{s_k} at the start of stage k.
            Shape: [B, C, T, H_high, W_high] or [B, C, H_high, W_high]
            where H_high = H_low * 2, W_high = W_low * 2.

        Example:
            >>> x_end = torch.randn(1, 16, 16, 24, 24)  # Stage k=2 output
            >>> s_k = 0.333  # Start of stage k=1
            >>> x_start_k1 = sampler.apply_renoise(x_end, s_k=s_k)
            >>> x_