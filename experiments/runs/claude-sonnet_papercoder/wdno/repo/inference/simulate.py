## Code: inference/simulate.py

```python
## inference/simulate.py
"""Simulation inference module for WDNO (Wavelet Diffusion Neural Operator).

This module implements the Simulator class that bridges raw PDE conditioning
data and the WDNOPipeline for simulation inference. It handles:
    1. Transforming raw conditioning inputs into wavelet coefficient space
    2. Invoking the pipeline's DDIM sampling
    3. Reconstructing physical-space predictions via inverse wavelet transform
    4. Excluding the initial condition (t=0) from output per the MSE evaluation
       protocol (paper Section 4: "MSE measured on entire state sequences
       excluding initial conditions")

Conditioning preparation is experiment-specific:
    - 1D PDE experiments (Burgers', advection, compressible NS):
      Apply 2D DWT to force f, 1D DWT to u_0, tile u_0 coefficients along
      time dimension, concatenate channel-wise.
      Paper Appendix F.3: "we take the 1D wavelet transform, repeat the
      coefficients, and then concatenate them with the 2D coefficients."
    - 2D PDE experiments (fluid_2d, ERA5):
      Apply 3D DWT to control/state, 2D DWT to initial density, 1D DWT to
      smoke percentage, tile and concatenate.
      Paper Appendix H.2: "we take the 2D and 1D wavelet transform and
      repeat the coefficients to concatenate them."

Paper sources:
    - Simulation inference algorithm: Section 3.1, Algorithm 1
    - MSE evaluation protocol: Section 4 (exclude initial condition)
    - 1D conditioning: Appendix F.3
    - 2D conditioning: Appendix H.2
    - DDIM sampling: Section 3.1 (DDIM for fast inference)

Config references:
    - experiment.spatial_dim: 1 or 2 (drives conditioning logic)
    - experiment.name: experiment identifier
    - inference.<experiment>.ddim_steps: DDIM sampling steps
    - inference.<experiment>.ddim_eta: DDIM stochasticity
    - super_resolution.num_levels: SR levels at inference
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from config import Config
from models.wdno_pipeline import WDNOPipeline
from wavelet.wavelet_transform import WaveletTransform

logger = logging.getLogger(__name__)


class Simulator:
    """Simulation inference wrapper for WDNO.

    Transforms raw PDE conditioning data into wavelet space, invokes the
    WDNOPipeline's DDIM sampling, and reconstructs physical-space state
    trajectory predictions.

    The output excludes the initial condition (t=0) to match the MSE
    evaluation protocol from the paper.

    Attributes:
        pipeline: Fully initialized WDNOPipeline with loaded BRM (and
            optionally SRM) checkpoints. Provides the simulate() method
            that runs DDIM sampling in wavelet space.
        wavelet_transform: WaveletTransform instance configured for the
            current experiment. Must match the instance used during training:
            bior2.4/periodization/2D DWT for 1D PDE experiments;
            bior1.3/zero/3D DWT for 2D PDE experiments.
        device: Compute device string ('cuda', 'cpu', 'cuda:0', etc.).
        spatial_dim: Spatial dimensionality of the PDE (1 or 2). Drives
            the conditioning preparation logic. Derived from
            wavelet_transform.spatial_dim.
        experiment: Experiment name string. Used for experiment-specific
            conditioning logic (e.g., compressible NS has 3 output variables).
    """

    def __init__(
        self,
        pipeline: WDNOPipeline,
        wavelet_transform: WaveletTransform,
        device: str = "cuda",
        config: Optional[Config] = None,
    ) -> None:
        """Initialize the Simulator.

        Args:
            pipeline: Fully initialized WDNOPipeline with loaded BRM and
                optionally SRM checkpoints. The pipeline's BRM and SRM
                models are moved to device if not already there.
            wavelet_transform: WaveletTransform instance configured for the
                current experiment. Must match the wavelet type, mode, level,
                and spatial_dim used during training.
                Config: wavelet.<experiment>.wavelet_type,
                wavelet.<experiment>.padding_mode.
            device: Compute device string. All tensors are moved to this
                device during inference. Config: experiment.device.
            config: Optional Config instance. If provided, used to read
                experiment name and spatial_dim. If None, spatial_dim is
                inferred from wavelet_transform.spatial_dim and experiment
                defaults to 'burgers'.
        """
        self.pipeline: WDNOPipeline = pipeline
        self.wavelet_transform: WaveletTransform = wavelet_transform
        self.device: str = device
        self.spatial_dim: int = wavelet_transform.spatial_dim

        # Experiment name for experiment-specific conditioning logic
        if config is not None:
            self.experiment: str = config.experiment
            self._config: Optional[Config] = config
        else:
            self.experiment = "burgers"
            self._config = None

        # Move pipeline models to device
        target_device = torch.device(device)
        self.pipeline.brm = self.pipeline.brm.to(target_device)
        if self.pipeline.srm is not None:
            self.pipeline.srm = self.pipeline.srm.to(target_device)

        logger.info(
            "Simulator initialized: experiment=%s, spatial_dim=%d, device=%s",
            self.experiment,
            self.spatial_dim,
            device,
        )

    # -----------------------------------------------------------------------
    # Public inference methods
    # -----------------------------------------------------------------------

    def run(
        self,
        cond_raw: torch.Tensor,
        num_sr_levels: int = 0,
    ) -> torch.Tensor:
        """Run simulation inference for a single sample or small batch.

        Processes conditioning data through wavelet transform, runs DDIM
        sampling via the pipeline, and reconstructs the physical-space
        state trajectory. The initial condition (t=0) is excluded from
        the output per the paper's MSE evaluation protocol.

        Args:
            cond_raw: Raw conditioning tensor in physical space. Format
                depends on experiment:
                - 1D Burgers'/advection/compressible NS:
                  Dict or stacked tensor containing u_0 [B, X] and f [B, T, X].
                  If a single tensor, expected shape [B, T+1, X] where the
                  first time slice is u_0 and remaining slices are f.
                  Alternatively, a tuple (u0, f) where u0: [B, X] and
                  f: [B, T, X].
                - 2D fluid/ERA5:
                  Tuple (initial_density, control, smoke_pct) where
                  initial_density: [B, H, W], control: [B, T, H_ctrl, W_ctrl],
                  smoke_pct: [B, T].
                For single-sample inputs (no batch dim), a batch dimension
                is added automatically.
            num_sr_levels: Number of super-resolution levels to apply after
                BRM sampling. 0 = base resolution only. Config:
                super_resolution.num_levels. Requires pipeline.srm to be
                non-None when > 0.

        Returns:
            Predicted state trajectory in physical space, excluding the
            initial condition (t=0). Shape:
            - 1D experiments: [B, T, X] e.g. [B, 80, 120] for Burgers'
            - 2D experiments: [B, T, H, W] e.g. [B, 31, 64, 64] for fluid_2d
            dtype=float32.

        Note:
            This method wraps run_batch() with automatic batch dimension
            handling. For large batches, use run_batch() directly.
        """
        # Handle single-sample inputs by adding batch dimension
        was_unbatched: bool = False

        if isinstance(cond_raw, (tuple, list)):
            # Check if first element lacks batch dimension
            first_elem = cond_raw[0]
            if isinstance(first_elem, torch.Tensor):
                if self.spatial_dim == 1 and first_elem.dim() == 1:
                    # u_0 is [X] without batch dim → add batch dim to all
                    cond_raw = tuple(
                        t.unsqueeze(0) if isinstance(t, torch.Tensor) else t
                        for t in cond_raw
                    )
                    was_unbatched = True
                elif self.spatial_dim == 2 and first_elem.dim() == 2:
                    # initial_density is [H, W] without batch dim
                    cond_raw = tuple(
                        t.unsqueeze(0) if isinstance(t, torch.Tensor) else t
                        for t in cond_raw
                    )
                    was_unbatched = True
        elif isinstance(cond_raw, torch.Tensor):
            if self.spatial_dim == 1 and cond_raw.dim() == 2:
                # [T+1, X] without batch dim
                cond_raw = cond_raw.unsqueeze(0)
                was_unbatched = True
            elif self.spatial_dim == 2 and cond_raw.dim() == 3:
                # [T, H, W] without batch dim
                cond_raw = cond_raw.unsqueeze(0)
                was_unbatched = True

        # Run batch inference
        result: torch.Tensor = self.run_batch(cond_raw, num_sr_levels=num_sr_levels)

        # Remove batch dimension if input was unbatched
        if was_unbatched:
            result = result.squeeze(0)

        return result

    def run_batch(
        self,
        cond_batch: object,
        num_sr_levels: int = 0,
    ) -> torch.Tensor:
        """Run simulation inference for a batch of samples.

        The primary inference method. Processes a batch of conditioning
        data through the full WDNO simulation pipeline:
            1. Move data to device
            2. Apply wavelet transform to conditioning data
            3. Run BRM DDIM sampling in wavelet space
            4. Optionally apply SRM for super-resolution
            5. Apply inverse wavelet transform to get physical-space output
            6. Exclude initial condition (t=0) from output

        All operations are performed under torch.no_grad() for memory
        efficiency during inference.

        Args:
            cond_batch: Batch of conditioning data. Format depends on
                experiment (see run() docstring). Can be:
                - Tuple (u0, f) for 1D experiments: u0 [B, X], f [B, T, X]
                - Tuple (initial_density, control, smoke_pct) for 2D
                - Single tensor [B, T+1, X] for 1D (first slice = u0)
            num_sr_levels: Number of super-resolution levels. Config:
                super_resolution.num_levels.

        Returns:
            Predicted state trajectories in physical space, excluding t=0.
            Shape: [B, T, X] for 1D or [B, T, H, W] for 2D. dtype=float32.
        """
        with torch.no_grad():
            # --- Step 1: Prepare conditioning in wavelet space ---
            W_cond: torch.Tensor = self._prepare_conditioning(cond_batch)
            # W_cond: [B, C_cond, T_c, X_c] (1D) or [B, C_cond, T_c, H_c, W_c] (2D)

            batch_size: int = W_cond.shape[0]

            logger.debug(
                "run_batch: batch_size=%d, W_cond.shape=%s, num_sr_levels=%d",
                batch_size,
                tuple(W_cond.shape),
                num_sr_levels,
            )

            # --- Step 2: Run BRM DDIM sampling (+ optional SRM) ---
            W_u_pred: torch.Tensor = self.pipeline.simulate(
                W_cond=W_cond,
                num_sr_levels=num_sr_levels,
            )
            # W_u_pred: [B, C_state, T_c, X_c] or [B, C_state, T_c, H_c, W_c]

            logger.debug(
                "run_batch: W_u_pred.shape=%s",
                tuple(W_u_pred.shape),
            )

            # --- Step 3: Determine original physical shape for inverse DWT ---
            original_physical_shape: Tuple[int, ...] = self._get_original_physical_shape(
                batch_size=batch_size,
                num_sr_levels=num_sr_levels,
            )

            # --- Step 4: Apply inverse wavelet transform ---
            u_pred: torch.Tensor = self.wavelet_transform.inverse(
                W_u_pred,
                original_shape=original_physical_shape,
            )
            # u_pred: [B, T+1, X] (1D) or [B, T+1, H, W] (2D)
            # Includes t=0 (initial condition)

            logger.debug(
                "run_batch: u_pred.shape=%s (before IC exclusion)",
                tuple(u_pred.shape),
            )

            # --- Step 5: Exclude initial condition (t=0) ---
            # Paper Section 4: "MSE measured on entire state sequences
            # excluding initial conditions"
            # u_pred[:, 0, ...] is the t=0 slice (initial condition)
            # u_pred[:, 1:, ...] is the prediction target (t=1 to T)
            u_pred_no_ic: torch.Tensor = u_pred[:, 1:, ...]
            # Shape: [B, T, X] (1D) or [B, T, H, W] (2D)

            logger.debug(
                "run_batch: output shape=%s (IC excluded)",
                tuple(u_pred_no_ic.shape),
            )

            return u_pred_no_ic.float()

    # -----------------------------------------------------------------------
    # Conditioning preparation
    # -----------------------------------------------------------------------

    def _prepare_conditioning(self, cond_raw: object) -> torch.Tensor:
        """Transform raw conditioning data into packed wavelet coefficients.

        Dispatches to experiment-specific conditioning logic based on
        self.spatial_dim:
            - spatial_dim=1: 1D PDE experiments (Burgers', advection,
              compressible NS). Applies 2D DWT to force f and 1D DWT to
              initial condition u_0, tiles u_0 coefficients along time,
              concatenates channel-wise.
            - spatial_dim=2: 2D PDE experiments (fluid_2d, ERA5). Applies
              3D DWT to control/state, 2D DWT to initial density, 1D DWT
              to smoke percentage, tiles and concatenates.

        Args:
            cond_raw: Raw conditioning data. Format:
                1D experiments: tuple (u0, f) where u0: [B, X], f: [B, T, X]
                    OR single tensor [B, T+1, X] (first slice = u0)
                2D experiments: tuple (initial_density, control, smoke_pct)
                    where initial_density: [B, H, W],
                    control: [B, T, H_ctrl, W_ctrl] or [B, T, N_ctrl],
                    smoke_pct: [B, T]

        Returns:
            Packed wavelet conditioning tensor. Shape:
            - 1D: [B, C_cond, T_c, X_c] e.g. [B, 6, 41, 60] for Burgers'
              (4 channels from f DWT + 2 channels from u0 1D DWT tiled)
            - 2D: [B, C_cond, T_c, H_c, W_c] e.g. [B, C, 18, 34, 34]
            dtype=float32, on self.device.

        Raises:
            ValueError: If cond_raw format is unrecognized.
        """
        if self.spatial_dim == 1:
            return self._prepare_conditioning_1d(cond_raw)
        else:
            return self._prepare_conditioning_2d(cond_raw)

    def _prepare_conditioning_1d(self, cond_raw: object) -> torch.Tensor:
        """Prepare conditioning for 1D PDE experiments.

        Implements the conditioning strategy from paper Appendix F.3:
        "we take the 1D wavelet transform, repeat the coefficients, and
        then concatenate them with the 2D coefficients."

        Steps:
            1. Extract u_0 [B, X] and f [B, T, X] from cond_raw
            2. Apply 2D DWT to f: [B, T, X] → [B, 4, T_c, X_c]
               (4 coefficient sets: cA, cH, cV, cD)
            3. Apply 1D DWT to u_0: [B, X] → [B, 2, X_c]
               (2 coefficient sets: cA, cD at finest level)
            4. Tile u_0 coefficients along time: [B, 2, X_c] → [B, 2, T_c, X_c]
            5. Concatenate: [B, 4, T_c, X_c] + [B, 2, T_c, X_c] → [B, 6, T_c, X_c]

        Args:
            cond_raw: Either:
                - Tuple (u0, f): u0 [B, X], f [B, T, X]
                - Single tensor [B, T+1, X]: first slice is u0, rest is f
                - Dict with keys 'u0' and 'f'

        Returns:
            Packed conditioning wavelet coefficients [B, 6, T_c, X_c].
            dtype=float32, on self.device.

        Raises:
            ValueError: If cond_raw format cannot be parsed.
        """
        device = torch.device(self.device)

        # --- Extract u0 and f from cond_raw ---
        u0: torch.Tensor
        f: torch.Tensor

        if isinstance(cond_raw, (tuple, list)) and len(cond_raw) >= 2:
            u0 = cond_raw[0].to(device=device, dtype=torch.float32)
            f = cond_raw[1].to(device=device, dtype=torch.float32)
        elif isinstance(cond_raw, dict):
            if "u0" in cond_raw and "f" in cond_raw:
                u0 = cond_raw["u0"].to(device=device, dtype=torch.float32)
                f = cond_raw["f"].to(device=device, dtype=torch.float32)
            elif "u" in cond_raw and "f" in cond_raw:
                # u contains full trajectory; extract t=0 as u0
                u_full = cond_raw["u"].to(device=device, dtype=torch.float32)
                u0 = u_full[:, 0, :]  # [B, X]
                f = cond_raw["f"].to(device=device, dtype=torch.float32)
            else:
                raise ValueError(
                    f"Dict cond_raw must have keys ('u0', 'f') or ('u', 'f'), "
                    f"got keys: {list(cond_raw.keys())}."
                )
        elif isinstance(cond_raw, torch.Tensor):
            # Single tensor [B, T+1, X]: first slice is u0, rest is f
            cond_tensor = cond_raw.to(device=device, dtype=torch.float32)
            if cond_tensor.dim() != 3:
                raise ValueError(
                    f"Single-tensor cond_raw for 1D experiments must be 3D "
                    f"[B, T+1, X], got shape {tuple(cond_tensor.shape)}."
                )
            u0 = cond_tensor[:, 0, :]   # [B, X]
            f = cond_tensor[:, 1:, :]   # [B, T, X]
        else:
            raise ValueError(
                f"Unrecognized cond_raw format for 1D experiment: "
                f"type={type(cond_raw).__name__}. Expected tuple (u0, f), "
                "dict with keys 'u0'/'f', or tensor [B, T+1, X]."
            )

        # Validate shapes
        if u0.dim() != 2:
            raise ValueError(
                f"u0 must be 2D [B, X], got shape {tuple(u0.shape)}."
            )
        if f.dim() != 3:
            raise ValueError(
                f"f must be 3D [B, T, X], got shape {tuple(f.shape)}."
            )

        B: int = f.shape[0]
        T: int = f.shape[1]
        X: int = f.shape[2]

        logger.debug(
            "_prepare_conditioning_1d: B=%d, T=%d, X=%d, u0.shape=%s, f.shape=%s",
            B, T, X, tuple(u0.shape), tuple(f.shape),
        )

        # --- Step 2: Apply 2D DWT to force f ---
        # f: [B, T, X] → W_f: [B, 4, T_c, X_c]
        # WaveletTransform.forward() expects [B, T, X] for spatial_dim=1
        W_f: torch.Tensor = self.wavelet_transform.forward(f)
        # W_f: [B, 4, T_c, X_c] e.g. [B, 4, 41, 60] for T=80, X=120

        T_c: int = W_f.shape[2]
        X_c: int = W_f.shape[3]

        logger.debug(
            "_prepare_conditioning_1d: W_f.shape=%s (T_c=%d, X_c=%d)",
            tuple(W_f.shape), T_c, X_c,
        )

        # --- Step 3: Apply 1D DWT to u_0 ---
        # u0: [B, X] → W_u0: [B, 2, X_c]
        # The 1D DWT at level=1 yields 2 coefficient sets (cA, cD)
        W_u0: torch.Tensor = self._apply_1d_wavelet(u0)
        # W_u0: [B, 2, X_c] e.g. [B, 2, 60] for X=120

        logger.debug(
            "_prepare_conditioning_1d: W_u0.shape=%s",
            tuple(W_u0.shape),
        )

        # --- Step 4: Tile u_0 coefficients along time dimension ---
        # W_u0: [B, 2, X_c] → [B, 2, T_c, X_c]
        # Use repeat_1d_to_nd from WaveletTransform
        # target_shape includes batch and channel dims: (B, C_nd, T_c, X_c)
        target_shape_for_tiling: Tuple[int, ...] = (B, W_f.shape[1], T_c, X_c)
        W_u0_tiled: torch.Tensor = self.wavelet_transform.repeat_1d_to_nd(
            coeffs_lower=W_u0,
            target_shape=target_shape_for_tiling,
        )
        # W_u0_tiled: [B, 2, T_c, X_c]

        logger.debug(
            "_prepare_conditioning_1d: W_u0_tiled.shape=%s",
            tuple(W_u0_tiled.shape),
        )

        # --- Step 5: Concatenate along channel dimension ---
        # [B, 4, T_c, X_c] + [B, 2, T_c, X_c] → [B, 6, T_c, X_c]
        W_cond: torch.Tensor = torch.cat([W_f, W_u0_tiled], dim=1)

        logger.debug(
            "_prepare_conditioning_1d: W_cond.shape=%s (final)",
            tuple(W_cond.shape),
        )

        return W_cond.float()

    def _prepare_conditioning_2d(self, cond_raw: object) -> torch.Tensor:
        """Prepare conditioning for 2D PDE experiments.

        Implements the conditioning strategy from paper Appendix H.2:
        "we take the 2D and 1D wavelet transform and repeat the coefficients
        to concatenate them."

        Steps:
            1. Extract initial_density [B, H, W], control [B, T, H_ctrl, W_ctrl]
               or [B, T, N_ctrl], and smoke_pct [B, T] from cond_raw
            2. Apply 3D DWT to control (if spatial): [B, T, H, W] → [B, 8, T_c, H_c, W_c]
               OR handle flattened control [B, T, N_ctrl] by reshaping
            3. Apply 2D DWT to initial_density: [B, H, W] → [B, 4, H_c, W_c]
               Tile along time: [B, 4, H_c, W_c] → [B, 4, T_c, H_c, W_c]
            4. Apply 1D DWT to smoke_pct: [B, T] → [B, 2, T_c]
               Tile along spatial dims: [B, 2, T_c] → [B, 2, T_c, H_c, W_c]
            5. Concatenate all along channel dim

        Args:
            cond_raw: Either:
                - Tuple (initial_density, control, smoke_pct)
                - Dict with keys 'initial_density', 'control', 'smoke_pct'

        Returns:
            Packed conditioning wavelet coefficients [B, C_cond, T_c, H_c, W_c].
            dtype=float32, on self.device.

        Raises:
            ValueError: If cond_raw format cannot be parsed.
        """
        device = torch.device(self.device)

        # --- Extract components from cond_raw ---
        initial_density: torch.Tensor
        control: torch.Tensor
        smoke_pct: torch.Tensor

        if isinstance(cond_raw, (tuple, list)) and len(cond_raw) >= 3:
            initial_density = cond_raw[0].to(device=device, dtype=torch.float32)
            control = cond_raw[1].to(device=device, dtype=torch.float32)
            smoke_pct = cond_raw[2].to(device=device, dtype=torch.float32)
        elif isinstance(cond_raw, (tuple, list)) and len(cond_raw) == 2:
            # Fallback: (initial_density, control) without smoke_pct
            initial_density = cond_raw[0].to(device=device, dtype=torch.float32)
            control = cond_raw[1].to(device=device, dtype=torch.float32)
            # Create zero smoke_pct
            B_tmp = initial_density.shape[0]
            T_tmp = control.shape[1] if control.dim() >= 2 else 32
            smoke_pct = torch.zeros(B_tmp, T_tmp, device=device, dtype=torch.float32)
        elif isinstance(cond_raw, dict):
            initial_density = cond_raw.get(
                "initial_density",
                cond_raw.get("density", cond_raw.get("smoke", None))
            )
            control = cond_raw.get("control", cond_raw.get("force", None))
            smoke_pct = cond_raw.get(
                "smoke_pct",
                cond_raw.get("smoke_percentage", None)
            )

            if initial_density is None:
                raise ValueError(
                    f"Dict cond_raw missing 'initial_density' key. "
                    f"Available keys: {list(cond_raw.keys())}."
                )
            if control is None:
                raise ValueError(
                    f"Dict cond_raw missing 'control' key. "
                    f"Available keys: {list(cond_raw.keys())}."
                )

            initial_density = initial_density.to(device=device, dtype=torch.float32)
            control = control.to(device=device, dtype=torch.float32)

            if smoke_pct is None:
                B_tmp = initial_density.shape[0]
                T_tmp = control.shape[1] if control.dim() >= 2 else 32
                smoke_pct = torch.zeros(B_tmp, T_tmp, device=device, dtype=torch.float32)
            else:
                smoke_pct = smoke_pct.to(device=device, dtype=torch.float32)
        else:
            raise ValueError(
                f"Unrecognized cond_raw format for 2D experiment: "
                f"type={type(cond_raw).__name__}. Expected tuple "
                "(initial_density, control, smoke_pct) or dict."
            )

        B: int = initial_density.shape[0]

        logger.debug(
            "_prepare_conditioning_2d: B=%d, initial_density.shape=%s, "
            "control.shape=%s, smoke_pct.shape=%s",
            B,
            tuple(initial_density.shape),
            tuple(control.shape),
            tuple(smoke_pct.shape),
        )

        # --- Step 2: Apply 3D DWT to control ---
        # Handle both spatial control [B, T, H, W] and flattened [B, T, N_ctrl]
        W_ctrl: torch.Tensor
        T_c: int
        H_c: int
        W_c: int

        if control.dim() == 4:
            # Spatial control: [B, T, H_ctrl, W_ctrl]
            # Apply 3D DWT: [B, T, H, W] → [B, 8, T_c, H_c, W_c]
            W_ctrl = self.wavelet_transform.forward(control)
            T_c = W_ctrl.shape[2]
            H_c = W_ctrl.shape[3]
            W_c = W_ctrl.shape[4]
        elif control.dim() == 3:
            # Flattened control: [B, T, N_ctrl]
            # Reshape to spatial: try to infer H, W from N_ctrl
            # For fluid_2d: N_ctrl = 3584 (peripheral cells)
            # We treat the flattened control as a 1D spatial signal
            # and apply a 2D DWT treating (T, N_ctrl) as (time, space)
            # This is consistent with the 1D PDE treatment
            W_ctrl = self.wavelet_transform.forward(control)
            # W_ctrl: [B, 4, T_c, N_c] (treating N_ctrl as spatial dim)
            T_c = W_ctrl.shape[2]
            # For 2D experiments, we need 3D coefficient shape
            # Expand to [B, 4, T_c, N_c, 1] and treat as degenerate 3D
            # This is a fallback; ideally control should be spatial
            H_c = W_ctrl.shape[3]
            W_c = 1
            W_ctrl = W_ctrl.unsqueeze(-