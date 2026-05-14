## Code: models/wdno_pipeline.py

```python
## models/wdno_pipeline.py
"""WDNO Pipeline: orchestrates BRM, SRM, and WaveletTransform for inference.

This module implements WDNOPipeline, the central class that ties together the
Base-Resolution Model (BRM), Super-Resolution Model (SRM), and WaveletTransform
to implement the full WDNO inference algorithm described in the paper.

Three inference modes are supported:
    1. Simulation: BRM DDIM sampling conditioned on equation parameters.
    2. Control: BRM DDIM sampling with combined CFG + classifier guidance
       to steer generation toward lower control objective I.
    3. Zero-shot super-resolution: iterative SRM application to progressively
       double resolution beyond the training resolution.

All operations are performed on wavelet coefficient tensors. Raw PDE data
never enters this class directly — callers (Simulator, Controller,
SuperResolver) apply WaveletTransform.forward() before calling pipeline
methods and WaveletTransform.inverse() on the returned tensors.

Exception: _control_guidance_step internally calls wavelet_transform.inverse()
to convert W_f_hat to physical-space f_hat before evaluating objective_fn,
because the objective function operates in physical space.

Paper sources:
    - Simulation inference: Section 3.1, Algorithm 1 (Sampling, simulation branch)
    - Control inference: Section 3.1, Eq. 4, Algorithm 1 (Sampling, control branch)
    - Super-resolution inference: Section 3.2 ("Inference" paragraph)
    - Guidance gradient: Section 3.1 (W_f_hat^(k) computation)
    - IC guidance: Appendix F.4, H.3
    - Cosine guidance schedule: Table 18 ("Scheduler of guidance: cosine")

Config references:
    - inference.burgers.ddim_steps: 50
    - inference.burgers.ddim_eta: 1.0
    - inference.burgers.guidance_lambda: 120000
    - inference.burgers.guidance_schedule: cosine
    - inference.compressible_ns.ddim_steps: 850
    - inference.fluid_2d.ddim_steps: 100
    - inference.fluid_2d.guidance_lambda: 100
    - diffusion.cfg_weight: 1.0
"""

from __future__ import annotations

import logging
import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from config import Config
from models.diffusion import Diffusion
from wavelet.wavelet_transform import WaveletTransform

logger = logging.getLogger(__name__)


class WDNOPipeline:
    """Orchestrates BRM, SRM, and WaveletTransform for WDNO inference.

    This class implements the three inference modes of WDNO:
        1. simulate(): BRM DDIM sampling for PDE state prediction.
        2. control(): BRM DDIM sampling with classifier guidance for
           optimal control force generation.
        3. _super_resolve(): Iterative SRM application for zero-shot
           super-resolution beyond training resolution.

    The pipeline operates exclusively on wavelet coefficient tensors.
    Callers are responsible for applying WaveletTransform.forward() to
    raw PDE data before calling pipeline methods, and applying
    WaveletTransform.inverse() to the returned wavelet coefficients.

    Attributes:
        brm: Base-Resolution Model (Diffusion instance). Trained on
            full-resolution wavelet coefficients. Used for both simulation
            and control inference.
        srm: Super-Resolution Model (Diffusion instance or None). Trained
            on multi-resolution data pairs. Used for zero-shot SR.
            None for experiments without SR (e.g., compressible NS).
        wavelet_transform: Shared WaveletTransform instance configured
            for the current experiment. Used in _control_guidance_step
            to convert W_f_hat to physical-space f_hat.
        config: Experiment configuration. Drives ddim_steps, ddim_eta,
            guidance_lambda, guidance_schedule, cfg_weight, and device.
        ddim_steps: DDIM sampling steps from config.inference[experiment].
        ddim_eta: DDIM stochasticity parameter from config.
        cfg_weight: Classifier-free guidance weight from config.
        guidance_lambda: Maximum guidance weight lambda_max from config.
        guidance_schedule: Guidance schedule type ('cosine') from config.
        device: Compute device string from config.
    """

    def __init__(
        self,
        brm: Diffusion,
        srm: Optional[Diffusion],
        wavelet_transform: WaveletTransform,
        config: Config,
    ) -> None:
        """Initialize the WDNOPipeline.

        Extracts inference hyperparameters from config for the current
        experiment. No model initialization occurs here — models are
        passed in already built and (optionally) trained.

        Args:
            brm: Base-Resolution Model. Must be a trained Diffusion instance
                with a U-Net configured for the current experiment's wavelet
                coefficient shape. Used for both simulation and control.
            srm: Super-Resolution Model. Trained Diffusion instance for
                multi-resolution pairs. Pass None if super-resolution is
                not needed (e.g., compressible NS experiment, or when
                config.super_resolution.num_levels=0).
            wavelet_transform: Shared WaveletTransform instance. Must be
                configured with the correct wavelet type, mode, level, and
                spatial_dim for the current experiment. Used internally in
                _control_guidance_step to convert wavelet coefficients to
                physical-space force for objective evaluation.
            config: Experiment configuration. The pipeline reads inference
                hyperparameters from config.inference[config.experiment].
                Config: inference.burgers.ddim_steps=50,
                inference.burgers.guidance_lambda=120000, etc.
        """
        self.brm: Diffusion = brm
        self.srm: Optional[Diffusion] = srm
        self.wavelet_transform: WaveletTransform = wavelet_transform
        self.config: Config = config

        # Extract inference hyperparameters from config
        self.ddim_steps: int = config.ddim_steps
        self.ddim_eta: float = config.ddim_eta
        self.cfg_weight: float = config.cfg_weight
        self.guidance_lambda: float = config.guidance_lambda
        self.guidance_schedule: str = config.guidance_schedule
        self.device: str = config.device

        logger.info(
            "WDNOPipeline initialized: experiment=%s, ddim_steps=%d, "
            "ddim_eta=%.2f, cfg_weight=%.2f, guidance_lambda=%.1f, "
            "guidance_schedule=%s, srm=%s",
            config.experiment,
            self.ddim_steps,
            self.ddim_eta,
            self.cfg_weight,
            self.guidance_lambda,
            self.guidance_schedule,
            "enabled" if srm is not None else "disabled",
        )

    # -----------------------------------------------------------------------
    # Simulation inference
    # -----------------------------------------------------------------------

    def simulate(
        self,
        W_cond: torch.Tensor,
        num_sr_levels: int = 0,
        W_cond_hr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run BRM DDIM sampling to generate state trajectory wavelet coefficients.

        Implements the simulation branch of Algorithm 1 (paper Section 3.1):
            1. Initialize W_u^(K) ~ N(0, I)
            2. For k = K, ..., 1: apply DDIM step conditioned on W_cond
            3. Return W_u^(0) (wavelet coefficients of predicted state)

        Optionally applies SRM for zero-shot super-resolution after BRM.

        Args:
            W_cond: Conditioning wavelet coefficient tensor. Contains
                wavelet-transformed equation parameters (e.g., initial
                condition u_0 and force f for Burgers'). Shape:
                [B, C_cond, T_c, X_c] for 1D PDEs or
                [B, C_cond, T_c, H_c, W_c] for 2D PDEs.
                dtype=float32, on self.device.
            num_sr_levels: Number of super-resolution levels to apply after
                BRM. 0 = base resolution only. Config:
                super_resolution.num_levels. Requires srm to be non-None
                and W_cond_hr to be provided when > 0.
            W_cond_hr: High-resolution conditioning for SRM. Required when
                num_sr_levels > 0. Contains wavelet-transformed equation
                parameters at the target high resolution. Shape:
                [B, C_cond, T_c_hr, X_c_hr] or
                [B, C_cond, T_c_hr, H_c_hr, W_c_hr].
                If None and num_sr_levels > 0, W_cond is used as HR cond
                (appropriate when W_cond is already at target resolution).

        Returns:
            Wavelet coefficient tensor of the predicted state trajectory.
            Shape matches W_cond's spatial/temporal dimensions (base res)
            or the super-resolved dimensions (if num_sr_levels > 0).
            dtype=float32. Apply WaveletTransform.inverse() to get u[0,T].

        Raises:
            ValueError: If num_sr_levels > 0 and srm is None.
        """
        if num_sr_levels > 0 and self.srm is None:
            raise ValueError(
                f"num_sr_levels={num_sr_levels} requires a trained SRM, "
                "but srm=None. Either set num_sr_levels=0 or provide an SRM."
            )

        # Determine output shape for BRM sampling.
        # The output has the same spatial/temporal coefficient dimensions as W_cond,
        # but with the number of channels corresponding to the state u (not the
        # conditioning). We infer this from the BRM model's out_channels.
        batch_size: int = W_cond.shape[0]
        out_channels: int = self.brm.model.out_channels
        spatial_coeff_dims: Tuple[int, ...] = tuple(W_cond.shape[2:])
        output_shape: Tuple[int, ...] = (batch_size, out_channels) + spatial_coeff_dims

        logger.debug(
            "simulate: batch=%d, output_shape=%s, ddim_steps=%d",
            batch_size,
            output_shape,
            self.ddim_steps,
        )

        # Run BRM DDIM sampling (no guidance function for simulation)
        W_u_base: torch.Tensor = self.brm.ddim_sample(
            shape=output_shape,
            cond=W_cond,
            ddim_steps=self.ddim_steps,
            eta=self.ddim_eta,
            cfg_weight=self.cfg_weight,
            guidance_fn=None,
            guidance_lambda_schedule=None,
        )

        # Apply super-resolution if requested
        if num_sr_levels > 0:
            # Use W_cond_hr if provided, otherwise fall back to W_cond
            cond_hr: torch.Tensor = W_cond_hr if W_cond_hr is not None else W_cond
            W_u_final: torch.Tensor = self._super_resolve(
                W_base=W_u_base,
                W_cond_hr=cond_hr,
                num_levels=num_sr_levels,
            )
        else:
            W_u_final = W_u_base

        return W_u_final

    # -----------------------------------------------------------------------
    # Control inference
    # -----------------------------------------------------------------------

    def control(
        self,
        W_cond: torch.Tensor,
        objective_fn: Callable[[torch.Tensor], torch.Tensor],
        num_sr_levels: int = 0,
        W_cond_hr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run control inference with combined CFG + classifier guidance.

        Implements the control branch of Algorithm 1 (paper Section 3.1,
        Eq. 4):
            1. Initialize W_f^(K) ~ N(0, I)
            2. For k = K, ..., 1:
               a. Estimate W_f_hat^(k) from W_f^(k) and eps_theta
               b. Compute gradient of I(W_f_hat^(k)) w.r.t. W_f^(k)
               c. Apply DDIM step with combined eps + lambda * grad
            3. Return W_f^(0) (wavelet coefficients of optimal control force)

        The guidance weight lambda follows a cosine schedule (paper Table 18:
        "Scheduler of guidance: cosine"), ramping up from 0 to lambda_max
        as denoising progresses toward cleaner estimates.

        Args:
            W_cond: Conditioning wavelet coefficient tensor. Contains
                wavelet-transformed equation parameters (e.g., u_0 and
                u_star for Burgers' control). Shape:
                [B, C_cond, T_c, X_c] for 1D or
                [B, C_cond, T_c, H_c, W_c] for 2D. dtype=float32.
            objective_fn: Differentiable control objective function.
                Signature: (f_hat: Tensor) -> Tensor (scalar).
                Input f_hat is the physical-space force estimate obtained
                by applying inverse wavelet transform to W_f_hat.
                Must be differentiable w.r.t. its input for autograd.grad.
                For 1D Burgers': I = ||u(T) - u*||^2 + alpha*||f||^2.
                For 2D fluid: I = percentage of smoke NOT reaching bucket.
            num_sr_levels: Number of super-resolution levels after BRM.
                Config: super_resolution.num_levels.
            W_cond_hr: High-resolution conditioning for SRM (optional).
                Required when num_sr_levels > 0.

        Returns:
            Wavelet coefficient tensor of the generated control force.
            Shape: [B, C_f, T_c, X_c] for 1D or [B, C_f, T_c, H_c, W_c]
            for 2D. Apply WaveletTransform.inverse() to get f[0,T].
            dtype=float32.

        Raises:
            ValueError: If num_sr_levels > 0 and srm is None.
        """
        if num_sr_levels > 0 and self.srm is None:
            raise ValueError(
                f"num_sr_levels={num_sr_levels} requires a trained SRM, "
                "but srm=None."
            )

        batch_size: int = W_cond.shape[0]
        device: torch.device = torch.device(self.device)

        # Determine output shape for W_f (force wavelet coefficients).
        # Same spatial/temporal coefficient dims as W_cond, but with
        # out_channels from the BRM model.
        out_channels: int = self.brm.model.out_channels
        spatial_coeff_dims: Tuple[int, ...] = tuple(W_cond.shape[2:])
        output_shape: Tuple[int, ...] = (batch_size, out_channels) + spatial_coeff_dims

        # Store original shape for inverse wavelet transform in guidance step
        # Physical-space shape: [B, T, X] for 1D or [B, T, H, W] for 2D
        # We reconstruct this from the wavelet coefficient shape
        original_physical_shape: Tuple[int, ...] = self._infer_physical_shape(
            output_shape
        )

        logger.debug(
            "control: batch=%d, output_shape=%s, physical_shape=%s, "
            "ddim_steps=%d, guidance_lambda=%.1f",
            batch_size,
            output_shape,
            original_physical_shape,
            self.ddim_steps,
            self.guidance_lambda,
        )

        # Initialize W_f from Gaussian noise: W_f^(K) ~ N(0, I)
        W_f: torch.Tensor = torch.randn(output_shape, device=device)

        # Build DDIM timestep sequence (same as in Diffusion.ddim_sample)
        time_seq: torch.Tensor = torch.linspace(
            0,
            self.brm.timesteps - 1,
            self.ddim_steps + 1,
            dtype=torch.long,
        )
        time_seq = time_seq.flip(0)  # descending: [T-1, ..., 0]

        # Create (t, t_prev) pairs
        time_pairs: List[Tuple[int, int]] = [
            (int(time_seq[i].item()), int(time_seq[i + 1].item()))
            for i in range(self.ddim_steps)
        ]
        total_steps: int = len(time_pairs)

        # Iterative denoising with guidance
        for step_idx, (t_val, t_prev_val) in enumerate(time_pairs):
            # Compute cosine-scheduled guidance weight
            lambda_t: float = self._cosine_guidance_schedule(
                step=step_idx,
                total_steps=total_steps,
                lambda_max=self.guidance_lambda,
            )

            # Single DDIM step with combined CFG + classifier guidance
            W_f = self._control_guidance_step(
                W_f_noisy=W_f,
                W_cond=W_cond,
                t=t_val,
                t_prev=t_prev_val,
                lambda_t=lambda_t,
                objective_fn=objective_fn,
                eta=self.ddim_eta,
                cfg_weight=self.cfg_weight,
                original_physical_shape=original_physical_shape,
            )

        # Apply super-resolution if requested
        if num_sr_levels > 0:
            cond_hr: torch.Tensor = W_cond_hr if W_cond_hr is not None else W_cond
            W_f = self._super_resolve(
                W_base=W_f,
                W_cond_hr=cond_hr,
                num_levels=num_sr_levels,
            )

        return W_f

    # -----------------------------------------------------------------------
    # Control guidance step
    # -----------------------------------------------------------------------

    def _control_guidance_step(
        self,
        W_f_noisy: torch.Tensor,
        W_cond: torch.Tensor,
        t: int,
        t_prev: int,
        lambda_t: float,
        objective_fn: Callable[[torch.Tensor], torch.Tensor],
        eta: float = 1.0,
        cfg_weight: float = 1.0,
        original_physical_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Single DDIM step with combined CFG + classifier guidance for control.

        Implements Eq. 4 from paper Section 3.1:
            W_f_hat^(k) = (W_f^(k) - sqrt(1-alpha_bar_k)*eps_theta) / sqrt(alpha_bar_k)
            W_f^(k-1) = W_f^(k) - eta*(eps_theta + lambda*nabla_I(W_f_hat^(k))) + xi

        The gradient nabla_I is computed on the denoised estimate W_f_hat
        (not the noisy W_f^(k)) to avoid noise contaminating the gradient.
        The gradient flows through: W_f_noisy → W_f_hat → f_hat → I(f_hat).

        The actual update uses the DDIM formula with the guided noise prediction:
            eps_guided = eps_pred + lambda_t * grad
            x0_hat = predict_x0_from_eps(W_f_noisy, t, eps_pred)  [detached eps]
            sigma_t = eta * sqrt((1-alpha_bar_{t-1})/(1-alpha_bar_t) * (1-alpha_bar_t/alpha_bar_{t-1}))
            W_f_prev = sqrt(alpha_bar_{t-1})*x0_hat
                     + sqrt(1-alpha_bar_{t-1}-sigma_t^2)*eps_guided
                     + sigma_t*noise

        Args:
            W_f_noisy: Current noisy force wavelet coefficients at timestep t.
                Shape [B, C, *spatial_coeff_dims]. dtype=float32.
            W_cond: Conditioning wavelet coefficients. Shape [B, C_cond, *dims].
            t: Current timestep index (integer, 0-indexed). Higher = noisier.
            t_prev: Previous (cleaner) timestep index. In range [0, t-1].
            lambda_t: Guidance weight at current step. From cosine schedule.
                Config: inference.burgers.guidance_lambda=120000.
            objective_fn: Differentiable control objective. Takes physical-space
                force f_hat and returns scalar loss tensor.
            eta: DDIM stochasticity parameter. Config: inference.*.ddim_eta=1.0.
            cfg_weight: CFG guidance weight. Config: diffusion.cfg_weight=1.0.
            original_physical_shape: Physical-space shape for inverse wavelet
                transform. If None, inferred from W_f_noisy shape.

        Returns:
            Updated W_f at timestep t_prev. Same shape as W_f_noisy.
            dtype=float32.
        """
        B: int = W_f_noisy.shape[0]
        device: torch.device = W_f_noisy.device

        t_batch: torch.Tensor = torch.full(
            (B,), t, device=device, dtype=torch.long
        )

        # -----------------------------------------------------------------------
        # Step 1: Compute eps_pred via CFG (no gradient tracking needed here)
        # -----------------------------------------------------------------------
        with torch.no_grad():
            eps_pred: torch.Tensor = self._get_eps_pred_cfg(
                W_f_noisy, W_cond, t_batch, cfg_weight
            )

        # -----------------------------------------------------------------------
        # Step 2: Estimate W_f_hat (denoised estimate) — used for gradient path
        # -----------------------------------------------------------------------
        # W_f_hat = (W_f_noisy - sqrt(1-alpha_bar_t)*eps_pred) / sqrt(alpha_bar_t)
        # eps_pred is detached so gradient flows only through W_f_noisy
        alpha_bar_t: float = float(self.brm.alphas_cumprod[t].item())
        sqrt_alpha_bar_t: float = math.sqrt(max(alpha_bar_t, 1e-20))
        sqrt_one_minus_alpha_bar_t: float = math.sqrt(max(1.0 - alpha_bar_t, 0.0))

        # -----------------------------------------------------------------------
        # Step 3 & 4: Compute gradient of objective through W_f_hat → f_hat → I
        # -----------------------------------------------------------------------
        grad: Optional[torch.Tensor] = None

        if lambda_t > 0.0:
            # Create a fresh computation graph for gradient computation
            W_f_for_grad: torch.Tensor = W_f_noisy.detach().requires_grad_(True)

            # Recompute W_f_hat with gradient tracking
            # eps_pred is detached — gradient flows only through W_f_for_grad
            n_trailing: int = W_f_for_grad.ndim - 1
            view_shape: Tuple[int, ...] = (-1,) + (1,) * n_trailing

            W_f_hat: torch.Tensor = (
                W_f_for_grad
                - sqrt_one_minus_alpha_bar_t * eps_pred.detach()
            ) / sqrt_alpha_bar_t

            # Clamp for numerical stability (standard practice)
            W_f_hat = W_f_hat.clamp(-10.0, 10.0)

            # Apply inverse wavelet transform to get physical-space f_hat
            if original_physical_shape is None:
                original_physical_shape = self._infer_physical_shape(
                    tuple(W_f_noisy.shape)
                )

            try:
                f_hat: torch.Tensor = self.wavelet_transform.inverse(
                    W_f_hat, original_physical_shape
                )
            except Exception as exc:
                logger.warning(
                    "Inverse wavelet transform failed in guidance step: %s. "
                    "Skipping guidance for this step.",
                    exc,
                )
                grad = torch.zeros_like(W_f_noisy)
            else:
                # Compute objective and gradient
                try:
                    obj_val: torch.Tensor = objective_fn(f_hat)

                    # Ensure obj_val is a scalar
                    if obj_val.numel() > 1:
                        obj_val = obj_val.mean()

                    # Compute gradient w.r.t. W_f_for_grad
                    grad_tuple = torch.autograd.grad(
                        outputs=obj_val,
                        inputs=W_f_for_grad,
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=True,
                    )
                    grad = grad_tuple[0]

                    if grad is None:
                        logger.warning(
                            "autograd.grad returned None for objective_fn. "
                            "Ensure objective_fn is differentiable w.r.t. its input."
                        )
                        grad = torch.zeros_like(W_f_noisy)

                except Exception as exc:
                    logger.warning(
                        "Gradient computation failed in guidance step: %s. "
                        "Skipping guidance for this step.",
                        exc,
                    )
                    grad = torch.zeros_like(W_f_noisy)

        # -----------------------------------------------------------------------
        # Step 5: DDIM update with guided noise prediction
        # -----------------------------------------------------------------------
        # Get alpha_bar values for current and previous timesteps
        alpha_bar_t_prev: float
        if t_prev >= 0 and t_prev < self.brm.timesteps:
            alpha_bar_t_prev = float(self.brm.alphas_cumprod[t_prev].item())
        else:
            # t_prev = -1 or 0 edge case: alpha_bar_{-1} = 1.0 by convention
            alpha_bar_t_prev = 1.0

        sqrt_alpha_bar_t_prev: float = math.sqrt(max(alpha_bar_t_prev, 1e-20))

        # Compute DDIM sigma_t (stochasticity)
        # sigma_t = eta * sqrt((1-alpha_bar_{t-1})/(1-alpha_bar_t) * (1 - alpha_bar_t/alpha_bar_{t-1}))
        one_minus_alpha_bar_t: float = max(1.0 - alpha_bar_t, 1e-20)
        one_minus_alpha_bar_t_prev: float = max(1.0 - alpha_bar_t_prev, 0.0)

        # Ratio for sigma computation
        ratio: float = (
            one_minus_alpha_bar_t_prev / one_minus_alpha_bar_t
            * max(1.0 - alpha_bar_t / max(alpha_bar_t_prev, 1e-20), 0.0)
        )
        sigma_t: float = eta * math.sqrt(max(ratio, 0.0))

        # Direction coefficient: sqrt(1 - alpha_bar_{t-1} - sigma_t^2)
        dir_coeff_sq: float = max(
            one_minus_alpha_bar_t_prev - sigma_t ** 2, 0.0
        )
        dir_coeff: float = math.sqrt(dir_coeff_sq)

        # Compute x0_hat from detached eps_pred (for DDIM update)
        with torch.no_grad():
            x0_hat: torch.Tensor = (
                W_f_noisy - sqrt_one_minus_alpha_bar_t * eps_pred
            ) / sqrt_alpha_bar_t
            x0_hat = x0_hat.clamp(-10.0, 10.0)

        # Combine eps_pred with guidance gradient
        if grad is not None and lambda_t > 0.0:
            eps_guided: torch.Tensor = eps_pred + lambda_t * grad.detach()
        else:
            eps_guided = eps_pred

        # DDIM update:
        # W_f_prev = sqrt(alpha_bar_{t-1}) * x0_hat
        #          + sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * eps_guided
        #          + sigma_t * noise
        with torch.no_grad():
            W_f_prev: torch.Tensor = (
                sqrt_alpha_bar_t_prev * x0_hat
                + dir_coeff * eps_guided
            )

            # Add stochastic noise if sigma_t > 0 and not at final step
            if sigma_t > 0.0 and t_prev > 0:
                noise: torch.Tensor = torch.randn_like(W_f_noisy)
                W_f_prev = W_f_prev + sigma_t * noise

        return W_f_prev.detach()

    # -----------------------------------------------------------------------
    # Denoised estimate
    # -----------------------------------------------------------------------

    def _estimate_x0(
        self,
        W_noisy: torch.Tensor,
        W_cond: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """Estimate the clean x0 from noisy x_t at timestep t.

        Convenience wrapper used internally by _control_guidance_step and
        _apply_initial_condition_guidance. Uses the BRM model with CFG.

        Formula (paper Section 3.1):
            W_f_hat^(k) = (W_f^(k) - sqrt(1-alpha_bar_k)*eps_theta) / sqrt(alpha_bar_k)

        Args:
            W_noisy: Noisy wavelet coefficients at timestep t.
                Shape [B, C, *spatial_coeff_dims]. dtype=float32.
            W_cond: Conditioning wavelet coefficients.
                Shape [B, C_cond, *spatial_coeff_dims]. dtype=float32.
            t: Current timestep index (integer, 0-indexed).

        Returns:
            Estimated clean wavelet coefficients W_x0_hat.
            Same shape as W_noisy. dtype=float32.
        """
        B: int = W_noisy.shape[0]
        device: torch.device = W_noisy.device

        t_batch: torch.Tensor = torch.full(
            (B,), t, device=device, dtype=torch.long
        )

        with torch.no_grad():
            eps_pred: torch.Tensor = self._get_eps_pred_cfg(
                W_noisy, W_cond, t_batch, self.cfg_weight
            )
            W_x0_hat: torch.Tensor = self.brm.predict_x0_from_eps(
                W_noisy, t_batch, eps_pred
            )

        return W_x0_hat

    # -----------------------------------------------------------------------
    # Cosine guidance schedule
    # -----------------------------------------------------------------------

    def _cosine_guidance_schedule(
        self,
        step: int,
        total_steps: int,
        lambda_max: float,
        lambda_min: float = 0.0,
    ) -> float:
        """Compute cosine-scheduled guidance weight at a given denoising step.

        The schedule ramps UP from lambda_min at step=0 (most noisy, start
        of denoising) to lambda_max at step=total_steps (cleanest, end of
        denoising). This ensures guidance is strongest when the denoised
        estimate W_f_hat is most