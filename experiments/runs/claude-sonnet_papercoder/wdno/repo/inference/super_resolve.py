## Code: inference/super_resolve.py

```python
## inference/super_resolve.py
"""Zero-shot super-resolution inference module for WDNO.

This module implements the SuperResolver class that iteratively applies the
Super-Resolution Model (SRM) to progressively double the resolution of BRM
outputs beyond the training resolution.

The core principle is approximate scale invariance (paper Section 3.2):
the pattern of change between resolution levels is consistent, so the SRM
trained on downsampled pairs can generalize to unseen finer resolutions.

Inference protocol (paper Section 3.2):
    "we utilize the Super-Resolution Model to generate the data based on
    both the wavelet coefficients of lower-resolution results with size N×M
    and the wavelet coefficients of a_h at the post-super-resolution
    resolution 2N×2M. This process is iterated."

Evaluation protocol (paper Section 4.6, Table 16, Table 17):
    "we interpolate the outcomes of each super-resolution step to the
    highest resolution level" for fair MSE comparison.

Paper sources:
    - SR inference: Section 3.2 ("Inference" paragraph)
    - SR evaluation: Section 4.6, Figure 4a, Figure 4b
    - 1D SR resolutions: Section 4.6 (80×120 → 160×240 → 320×480 → 640×960)
    - 2D SR resolutions: Section 4.6 (32×64×64 → 32×128×128)
    - Odd dimension handling: Appendix F.3
    - Boundary duplication: Appendix H.2

Config references:
    - super_resolution.num_levels: 0-3 for 1D, 0-1 for 2D
    - super_resolution.eval_interp_modes: [linear, nearest]
    - data.burgers.sr_resolutions: [[80,120],[160,240],[320,480],[640,960]]
    - data.fluid_2d.sr_resolutions: [[32,64,64],[32,128,128]]
    - wavelet.burgers.wavelet_type: bior2.4
    - wavelet.fluid_2d.wavelet_type: bior1.3
    - inference.<experiment>.ddim_steps
    - inference.<experiment>.ddim_eta
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.wdno_pipeline import WDNOPipeline
from utils.helpers import linear_interpolate, nearest_interpolate
from wavelet.wavelet_transform import WaveletTransform

logger = logging.getLogger(__name__)


class SuperResolver:
    """Zero-shot super-resolution inference wrapper for WDNO.

    Iteratively applies the SRM to double the resolution of a BRM output,
    enabling zero-shot generalization to resolutions not seen during training.

    The SRM is accessed via ``pipeline.srm`` (a Diffusion instance). At each
    SR level, the SuperResolver:
        1. Applies the wavelet transform to the current (low-res) output.
        2. Duplicates the low-res wavelet coefficients to match the 2x shape,
           with special boundary handling for odd dimensions.
        3. Extracts and wavelet-transforms the conditioning at 2x resolution.
        4. Runs SRM DDIM sampling conditioned on the concatenated inputs.
        5. Applies the inverse wavelet transform to get the 2x output.

    Attributes:
        pipeline: WDNOPipeline holding the trained SRM (``pipeline.srm``).
            The SRM is a Diffusion instance trained on multi-resolution pairs.
        wavelet_transform: WaveletTransform configured for the current
            experiment. Must match the instance used during SRM training.
        device: Compute device string ('cuda', 'cpu', 'cuda:0', etc.).
        spatial_dim: Spatial dimensionality of the PDE (1 or 2). Derived
            from wavelet_transform.spatial_dim. Drives which dimensions
            are upsampled in _handle_odd_dimensions.
    """

    def __init__(
        self,
        pipeline: WDNOPipeline,
        wavelet_transform: WaveletTransform,
        device: str = "cuda",
    ) -> None:
        """Initialize the SuperResolver.

        Args:
            pipeline: Fully initialized WDNOPipeline with a loaded SRM
                checkpoint (``pipeline.srm`` must not be None when
                ``resolve()`` is called with ``num_levels > 0``).
                The SRM model is moved to ``device`` during initialization.
            wavelet_transform: WaveletTransform instance configured for the
                current experiment. Must match the wavelet type, mode, level,
                and spatial_dim used during SRM training.
                Config: wavelet.burgers.wavelet_type='bior2.4' for 1D,
                wavelet.fluid_2d.wavelet_type='bior1.3' for 2D.
            device: Compute device string. All tensors are moved to this
                device during inference. Config: experiment.device.
        """
        self.pipeline: WDNOPipeline = pipeline
        self.wavelet_transform: WaveletTransform = wavelet_transform
        self.device: str = device
        self.spatial_dim: int = wavelet_transform.spatial_dim

        # Move pipeline models to device
        target_device = torch.device(device)
        self.pipeline.brm = self.pipeline.brm.to(target_device)
        if self.pipeline.srm is not None:
            self.pipeline.srm = self.pipeline.srm.to(target_device)

        logger.info(
            "SuperResolver initialized: spatial_dim=%d, device=%s, "
            "srm=%s",
            self.spatial_dim,
            device,
            "enabled" if pipeline.srm is not None else "disabled (num_levels must be 0)",
        )

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def resolve(
        self,
        base_output_raw: torch.Tensor,
        cond_hr_raw: torch.Tensor,
        num_levels: int = 1,
    ) -> torch.Tensor:
        """Iteratively apply SRM to produce a super-resolved output.

        Starting from the BRM output at base resolution, applies the SRM
        ``num_levels`` times to progressively double the resolution. At each
        level, the conditioning is extracted at the appropriate 2x resolution
        from ``cond_hr_raw`` (which is provided at the finest target resolution).

        Paper Section 3.2 inference protocol:
            1. BRM generates base-resolution wavelet coefficients (done externally).
            2. SRM generates 2x output from (W_low_dup, W_cond_hr_2x).
            3. Repeat step 2 for each additional SR level.

        Args:
            base_output_raw: BRM output in physical space at base resolution.
                Shape [B, T, X] for 1D PDEs (e.g., [B, 81, 120] for Burgers')
                or [B, T, H, W] for 2D PDEs (e.g., [B, 32, 64, 64] for fluid).
                dtype=float32. May be on any device (moved to self.device).
            cond_hr_raw: Equation parameters (conditioning) at the finest
                target resolution. Used to extract conditioning at each
                intermediate 2x resolution during the SR loop.
                Shape [B, T_hr, X_hr] for 1D (e.g., [B, 81, 960] for 3x SR)
                or [B, T_hr, H_hr, W_hr] for 2D (e.g., [B, 32, 128, 128]).
                For 1D experiments, this is typically the force f at the
                finest resolution. For 2D, it is the full state trajectory.
                dtype=float32. May be on any device.
            num_levels: Number of SR levels to apply. 0 returns
                ``base_output_raw`` unchanged. Config:
                super_resolution.num_levels.
                - 1D Burgers': 0–3 (paper Section 4.6)
                - 2D fluid: 0–1 (paper Section 4.6)

        Returns:
            Super-resolved output in physical space. Shape:
            - 0 levels: same as base_output_raw
            - 1 level: [B, T*2, X*2] or [B, T, H*2, W*2]
            - 2 levels: [B, T*4, X*4] or [B, T, H*4, W*4]
            - 3 levels: [B, T*8, X*8] (1D only)
            dtype=float32.

        Raises:
            ValueError: If num_levels > 0 and pipeline.srm is None.
            ValueError: If num_levels < 0.
        """
        if num_levels < 0:
            raise ValueError(
                f"num_levels must be >= 0, got {num_levels}."
            )

        if num_levels == 0:
            logger.debug("resolve: num_levels=0, returning base_output_raw unchanged.")
            return base_output_raw.float()

        if self.pipeline.srm is None:
            raise ValueError(
                f"num_levels={num_levels} requires a trained SRM, but "
                "pipeline.srm is None. Either set num_levels=0 or provide "
                "a trained SRM in the pipeline."
            )

        device = torch.device(self.device)

        # Move inputs to device
        current_output: torch.Tensor = base_output_raw.to(
            device=device, dtype=torch.float32
        )
        cond_hr: torch.Tensor = cond_hr_raw.to(
            device=device, dtype=torch.float32
        )

        logger.info(
            "resolve: num_levels=%d, base_shape=%s, cond_hr_shape=%s",
            num_levels,
            tuple(current_output.shape),
            tuple(cond_hr.shape),
        )

        # Iterative SR loop
        for level in range(num_levels):
            logger.info(
                "resolve: applying SR level %d/%d, current_shape=%s",
                level + 1,
                num_levels,
                tuple(current_output.shape),
            )

            current_output = self._apply_one_sr_level(
                current_output=current_output,
                cond_hr_raw=cond_hr,
                level=level,
                total_levels=num_levels,
            )

            logger.info(
                "resolve: SR level %d complete, output_shape=%s",
                level + 1,
                tuple(current_output.shape),
            )

        return current_output.float()

    def interpolate_to_finest(
        self,
        output: torch.Tensor,
        target_shape: Tuple[int, ...],
        mode: str = "linear",
    ) -> torch.Tensor:
        """Interpolate an output tensor to the finest resolution for evaluation.

        Used to normalize all SR-level outputs to the same resolution before
        computing MSE, enabling fair comparison across SR levels (paper
        Section 4.6, Table 16, Table 17).

        Paper Section 4.6: "we interpolate the outcomes of each super-resolution
        step to the highest resolution level. This allows us to assess whether
        the model can accurately generate data on finer grid points beyond the
        resolutions encountered during training."

        Handles both 1D PDE data (3D tensors [B, T, X]) and 2D PDE data
        (4D tensors [B, T, H, W]) by adding a channel dimension for
        F.interpolate compatibility.

        Args:
            output: Output tensor at any resolution. Shape [B, T, X] for
                1D PDEs or [B, T, H, W] for 2D PDEs. dtype=float32.
            target_shape: Target shape for the non-batch dimensions.
                For 1D: (T_finest, X_finest) e.g. (641, 960) for 3x SR.
                For 2D: (T_finest, H_finest, W_finest) e.g. (32, 128, 128).
                Note: includes the temporal dimension.
            mode: Interpolation mode. 'linear' uses bilinear/trilinear
                interpolation; 'nearest' uses nearest-neighbor.
                Config: super_resolution.eval_interp_modes=[linear, nearest].
                Paper Table 16/17 reports both modes.

        Returns:
            Interpolated tensor of shape [B, *target_shape]. dtype=float32.

        Raises:
            ValueError: If mode is not 'linear' or 'nearest'.
            ValueError: If output has unexpected number of dimensions.
        """
        if mode not in ("linear", "nearest"):
            raise ValueError(
                f"mode must be 'linear' or 'nearest', got '{mode}'. "
                "Config: super_resolution.eval_interp_modes=[linear, nearest]."
            )

        output = output.float()
        B: int = output.shape[0]

        if output.dim() == 3:
            # 1D PDE data: [B, T, X]
            if len(target_shape) != 2:
                raise ValueError(
                    f"3D output [B, T, X] requires target_shape of length 2 "
                    f"(T_finest, X_finest), got length {len(target_shape)}: "
                    f"{target_shape}."
                )
            T_target, X_target = target_shape

            # Check if already at target shape
            if output.shape[1] == T_target and output.shape[2] == X_target:
                return output

            # Add channel dim for F.interpolate: [B, T, X] → [B, 1, T, X]
            output_4d: torch.Tensor = output.unsqueeze(1)

            if mode == "linear":
                result_4d: torch.Tensor = F.interpolate(
                    output_4d,
                    size=(T_target, X_target),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                result_4d = F.interpolate(
                    output_4d,
                    size=(T_target, X_target),
                    mode="nearest",
                )

            # Remove channel dim: [B, 1, T_target, X_target] → [B, T_target, X_target]
            return result_4d.squeeze(1)

        elif output.dim() == 4:
            # 2D PDE data: [B, T, H, W]
            if len(target_shape) != 3:
                raise ValueError(
                    f"4D output [B, T, H, W] requires target_shape of length 3 "
                    f"(T_finest, H_finest, W_finest), got length {len(target_shape)}: "
                    f"{target_shape}."
                )
            T_target, H_target, W_target = target_shape

            # Check if already at target shape
            if (output.shape[1] == T_target
                    and output.shape[2] == H_target
                    and output.shape[3] == W_target):
                return output

            # For 2D PDE data, interpolate spatial dims per timestep
            # Reshape: [B, T, H, W] → [B*T, 1, H, W] for F.interpolate
            T_cur: int = output.shape[1]
            H_cur: int = output.shape[2]
            W_cur: int = output.shape[3]

            output_4d_spatial: torch.Tensor = output.reshape(
                B * T_cur, 1, H_cur, W_cur
            )

            if mode == "linear":
                result_spatial: torch.Tensor = F.interpolate(
                    output_4d_spatial,
                    size=(H_target, W_target),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                result_spatial = F.interpolate(
                    output_4d_spatial,
                    size=(H_target, W_target),
                    mode="nearest",
                )

            # Reshape back: [B*T_cur, 1, H_target, W_target] → [B, T_cur, H_target, W_target]
            result_spatial = result_spatial.squeeze(1).reshape(
                B, T_cur, H_target, W_target
            )

            # Handle temporal interpolation if T_target != T_cur
            if T_cur != T_target:
                # [B, T_cur, H_target, W_target] → [B, 1, T_cur, H_target, W_target]
                result_5d: torch.Tensor = result_spatial.unsqueeze(1)
                if mode == "linear":
                    result_5d = F.interpolate(
                        result_5d,
                        size=(T_target, H_target, W_target),
                        mode="trilinear",
                        align_corners=False,
                    )
                else:
                    result_5d = F.interpolate(
                        result_5d,
                        size=(T_target, H_target, W_target),
                        mode="nearest",
                    )
                result_spatial = result_5d.squeeze(1)

            return result_spatial

        else:
            raise ValueError(
                f"interpolate_to_finest expects 3D [B, T, X] or 4D [B, T, H, W] "
                f"input, got {output.dim()}D tensor with shape {tuple(output.shape)}."
            )

    # -----------------------------------------------------------------------
    # Internal SR level application
    # -----------------------------------------------------------------------

    def _apply_one_sr_level(
        self,
        current_output: torch.Tensor,
        cond_hr_raw: torch.Tensor,
        level: int,
        total_levels: int,
    ) -> torch.Tensor:
        """Apply one SR level: double the resolution of current_output.

        Implements one iteration of the SR loop from paper Section 3.2:
            1. Wavelet-transform current (low-res) output → W_low
            2. Duplicate W_low to match 2x wavelet coefficient shape → W_low_dup
            3. Extract conditioning at 2x resolution from cond_hr_raw
            4. Wavelet-transform 2x conditioning → W_cond_hr
            5. Run SRM DDIM sampling conditioned on cat([W_low_dup, W_cond_hr])
            6. Inverse wavelet transform → 2x output

        Args:
            current_output: Current output at the input resolution.
                Shape [B, T, X] (1D) or [B, T, H, W] (2D). float32.
            cond_hr_raw: Conditioning at the finest target resolution.
                Shape [B, T_hr, X_hr] (1D) or [B, T_hr, H_hr, W_hr] (2D).
            level: Current SR level index (0-indexed). Used to compute
                the downsampling factor for conditioning extraction.
            total_levels: Total number of SR levels. Used with level to
                compute the downsampling factor.

        Returns:
            Output at 2x resolution. Shape [B, T*2, X*2] (1D) or
            [B, T, H*2, W*2] (2D). float32.
        """
        device = torch.device(self.device)

        # --- Step 1: Wavelet-transform current (low-res) output ---
        with torch.no_grad():
            W_low: torch.Tensor = self.wavelet_transform.forward(current_output)
        # W_low: [B, C_coeff, T_c_low, X_c_low] (1D)
        #        [B, C_coeff, T_c_low, H_c_low, W_c_low] (2D)

        logger.debug(
            "_apply_one_sr_level level=%d: W_low.shape=%s",
            level,
            tuple(W_low.shape),
        )

        # --- Step 2: Compute target 2x raw shape ---
        target_2x_raw_shape: Tuple[int, ...] = self._compute_2x_raw_shape(
            current_shape=tuple(current_output.shape),
        )
        # target_2x_raw_shape: (B, T*2, X*2) or (B, T, H*2, W*2)

        logger.debug(
            "_apply_one_sr_level level=%d: target_2x_raw_shape=%s",
            level,
            target_2x_raw_shape,
        )

        # --- Step 3: Compute target 2x wavelet coefficient shape ---
        target_2x_coeff_shape: Tuple[int, ...] = self.wavelet_transform.get_output_shape(
            target_2x_raw_shape
        )
        # target_2x_coeff_shape: (B, C_coeff, T_c_high, X_c_high) or
        #                         (B, C_coeff, T_c_high, H_c_high, W_c_high)

        logger.debug(
            "_apply_one_sr_level level=%d: target_2x_coeff_shape=%s",
            level,
            target_2x_coeff_shape,
        )

        # --- Step 4: Duplicate W_low to match 2x coefficient shape ---
        W_low_dup: torch.Tensor = self._handle_odd_dimensions(
            low_res_coeffs=W_low,
            high_res_shape=target_2x_coeff_shape,
        )
        # W_low_dup: same shape as target_2x_coeff_shape

        logger.debug(
            "_apply_one_sr_level level=%d: W_low_dup.shape=%s",
            level,
            tuple(W_low_dup.shape),
        )

        # --- Step 5: Extract conditioning at 2x resolution ---
        # cond_hr_raw is at finest resolution (total_levels doublings from base)
        # We need conditioning at (level+1) doublings from base
        # = finest resolution / 2^(total_levels - level - 1)
        cond_at_2x: torch.Tensor = self._extract_cond_at_2x_resolution(
            cond_hr_raw=cond_hr_raw,
            target_raw_shape=target_2x_raw_shape,
        )
        # cond_at_2x: [B, T*2, X*2] (1D) or [B, T, H*2, W*2] (2D)

        logger.debug(
            "_apply_one_sr_level level=%d: cond_at_2x.shape=%s",
            level,
            tuple(cond_at_2x.shape),
        )

        # --- Step 6: Wavelet-transform 2x conditioning ---
        with torch.no_grad():
            W_cond_hr: torch.Tensor = self.wavelet_transform.forward(cond_at_2x)
        # W_cond_hr: [B, C_coeff, T_c_high, X_c_high] (1D)
        #             [B, C_coeff, T_c_high, H_c_high, W_c_high] (2D)

        logger.debug(
            "_apply_one_sr_level level=%d: W_cond_hr.shape=%s",
            level,
            tuple(W_cond_hr.shape),
        )

        # --- Step 7: Concatenate W_low_dup and W_cond_hr for SRM conditioning ---
        # SRM conditioning = cat([W_low_dup, W_cond_hr], dim=1)
        # This matches the SRM training format from MultiResolutionDataset:
        # cond = torch.cat([W_low_dup, W_cond_high], dim=1)
        W_srm_cond: torch.Tensor = torch.cat([W_low_dup, W_cond_hr], dim=1)
        # W_srm_cond: [B, 2*C_coeff, T_c_high, X_c_high] (1D)

        logger.debug(
            "_apply_one_sr_level level=%d: W_srm_cond.shape=%s",
            level,
            tuple(W_srm_cond.shape),
        )

        # --- Step 8: Run SRM DDIM sampling ---
        assert self.pipeline.srm is not None, (
            "pipeline.srm is None but num_levels > 0. "
            "This should have been caught in resolve()."
        )

        # Determine output shape for SRM sampling
        # SRM output has same spatial/temporal coeff dims as W_low_dup,
        # but with out_channels from the SRM model
        srm_out_channels: int = self.pipeline.srm.model.out_channels
        srm_output_shape: Tuple[int, ...] = (
            (W_low_dup.shape[0], srm_out_channels) + tuple(W_low_dup.shape[2:])
        )

        logger.debug(
            "_apply_one_sr_level level=%d: srm_output_shape=%s",
            level,
            srm_output_shape,
        )

        with torch.no_grad():
            W_high: torch.Tensor = self.pipeline.srm.ddim_sample(
                shape=srm_output_shape,
                cond=W_srm_cond,
                ddim_steps=self.pipeline.ddim_steps,
                eta=self.pipeline.ddim_eta,
                cfg_weight=self.pipeline.cfg_weight,
                guidance_fn=None,
                guidance_lambda_schedule=None,
            )
        # W_high: [B, C_coeff, T_c_high, X_c_high] (1D)
        #          [B, C_coeff, T_c_high, H_c_high, W_c_high] (2D)

        logger.debug(
            "_apply_one_sr_level level=%d: W_high.shape=%s",
            level,
            tuple(W_high.shape),
        )

        # --- Step 9: Inverse wavelet transform to get 2x physical-space output ---
        with torch.no_grad():
            output_2x: torch.Tensor = self.wavelet_transform.inverse(
                W_high,
                original_shape=target_2x_raw_shape,
            )
        # output_2x: [B, T*2, X*2] (1D) or [B, T, H*2, W*2] (2D)

        logger.debug(
            "_apply_one_sr_level level=%d: output_2x.shape=%s",
            level,
            tuple(output_2x.shape),
        )

        return output_2x.float()

    # -----------------------------------------------------------------------
    # Shape computation helpers
    # -----------------------------------------------------------------------

    def _compute_2x_raw_shape(
        self,
        current_shape: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        """Compute the 2x upsampled raw data shape.

        For 1D experiments (spatial_dim=1): doubles both temporal and spatial
        dimensions. Shape [B, T, X] → [B, T*2, X*2].

        For 2D experiments (spatial_dim=2): doubles only spatial dimensions
        (H and W), preserving the temporal dimension. Shape [B, T, H, W] →
        [B, T, H*2, W*2]. This matches the paper's 2D SR: 32×64×64 → 32×128×128.

        Args:
            current_shape: Shape tuple of the current output tensor.
                (B, T, X) for 1D or (B, T, H, W) for 2D.

        Returns:
            Shape tuple for the 2x upsampled output.
            (B, T*2, X*2) for 1D or (B, T, H*2, W*2) for 2D.

        Raises:
            ValueError: If current_shape has wrong number of dimensions
                for the configured spatial_dim.
        """
        if self.spatial_dim == 1:
            if len(current_shape) != 3:
                raise ValueError(
                    f"spatial_dim=1 expects 3D shape (B, T, X), "
                    f"got {len(current_shape)}D: {current_shape}."
                )
            B, T, X = current_shape
            return (B, T * 2, X * 2)

        else:  # spatial_dim == 2
            if len(current_shape) != 4:
                raise ValueError(
                    f"spatial_dim=2 expects 4D shape (B, T, H, W), "
                    f"got {len(current_shape)}D: {current_shape}."
                )
            B, T, H, W = current_shape
            # Temporal dimension preserved; only H and W are doubled
            # Paper Section 4.6: 32×64×64 → 32×128×128
            return (B, T, H * 2, W * 2)

    def _extract_cond_at_2x_resolution(
        self,
        cond_hr_raw: torch.Tensor,
        target_raw_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Extract conditioning at the target 2x resolution from finest-res conditioning.

        Downsamples ``cond_hr_raw`` (at finest resolution) to match
        ``target_raw_shape`` (the 2x resolution for the current SR level).

        Uses average pooling for downsampling (consistent with
        MultiResolutionDataset._downsample_by_2) or returns cond_hr_raw
        directly if it already matches the target shape.

        Args:
            cond_hr_raw: Conditioning at the finest target resolution.
                Shape [B, T_hr, X_hr] (1D) or [B, T_hr, H_hr, W_hr] (2D).
                float32, on self.device.
            target_raw_shape: Target shape for the 2x conditioning.
                (B, T_2x, X_2x) for 1D or (B, T_2x, H_2x, W_2x) for 2D.

        Returns:
            Conditioning tensor at the target 2x resolution.
            Shape matches target_raw_shape. float32.
        """
        # Check if already at target shape
        if tuple(cond_hr_raw.shape) == tuple(target_raw_shape):
            return cond_hr_raw

        if self.spatial_dim == 1:
            # cond_hr_raw: [B, T_hr, X_hr]
            # target: (B, T_2x, X_2x)
            B_target, T_target, X_target = target_raw_shape
            B_cur, T_cur, X_cur = cond_hr_raw.shape

            if T_cur == T_target and X_cur == X_target:
                return cond_hr_raw

            # Use F.interpolate for flexible downsampling/upsampling
            # [B, T_hr, X_hr] → [B, 1, T_hr, X_hr] → interpolate → [B, 1, T_2x, X_2x]
            cond_4d: torch.Tensor = cond_hr_raw.unsqueeze(1)  # [B, 1, T_hr, X_hr]
            cond_resized: torch.Tensor = F.interpolate(
                cond_4d,
                size=(T_target, X_target),
                mode="bilinear",
                align_corners=False,
            )  # [B, 1, T_2x, X_2x]
            return cond_resized.squeeze(1)  # [B, T_2x, X_2x]

        else:  # spatial_dim == 2
            