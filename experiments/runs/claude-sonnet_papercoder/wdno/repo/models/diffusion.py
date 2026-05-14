## Code: models/diffusion.py

```python
## models/diffusion.py
"""DDPM training and DDIM sampling for WDNO (Wavelet Diffusion Neural Operator).

This module implements the Diffusion class that wraps a U-Net denoising model
with the full DDPM forward process (noise addition) and DDIM reverse process
(fast sampling). All operations are performed on wavelet coefficient tensors —
raw PDE data never enters this class directly.

The class supports three use cases:
    1. BRM simulation: pure DDIM sampling conditioned on equation parameters
    2. BRM control: DDIM sampling with classifier guidance gradient injection
    3. SRM super-resolution: DDIM sampling conditioned on low-res + high-res params

Mathematical foundations (paper Section 2.2):
    Forward process:
        q(x_{k+1} | x_k) = N(x_{k+1}; sqrt(alpha_k)*x_k, (1-alpha_k)*I)
        x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps

    Training loss (simplified ELBO):
        L = E_{k,x0,eps}[||eps - eps_theta(sqrt(alpha_bar_k)*x0 + sqrt(1-alpha_bar_k)*eps, k)||^2]

    DDIM reverse step (Song et al. 2020):
        x0_hat = (x_t - sqrt(1-alpha_bar_t)*eps_theta) / sqrt(alpha_bar_t)
        x_{t-1} = sqrt(alpha_bar_{t-1})*x0_hat
                + sqrt(1-alpha_bar_{t-1}-sigma_t^2)*eps_theta
                + sigma_t*noise

    Control guidance (paper Section 3.1, Eq. 4):
        W_f_hat^(k) = (W_f^(k) - sqrt(1-alpha_bar_k)*eps_theta) / sqrt(alpha_bar_k)
        W_f^(k-1) = W_f^(k) - eta*(eps_theta + lambda*nabla_I(W_f_hat^(k))) + xi

Paper sources:
    - DDPM: Section 2.2, Ho et al. 2020
    - DDIM: Section 3.1 (inference), Song et al. 2020
    - CFG: Section 2.2, Ho & Salimans 2022
    - Control guidance: Section 3.1, Eq. 4
    - Cosine schedule: Nichol & Dhariwal 2021

Config references:
    - diffusion.num_timesteps: 1000
    - diffusion.beta_schedule: cosine
    - diffusion.cfg_dropout_prob: 0.1
    - diffusion.cfg_weight: 1.0
    - inference.burgers.ddim_steps: 50
    - inference.burgers.ddim_eta: 1.0
    - inference.compressible_ns.ddim_steps: 850
    - inference.fluid_2d.ddim_steps: 100
"""

from __future__ import annotations

import logging
import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import UNet

logger = logging.getLogger(__name__)

# Cosine schedule offset (Nichol & Dhariwal 2021)
_COSINE_SCHEDULE_S: float = 0.008

# Clipping bounds for x0 estimate (numerical stability)
_X0_CLIP_MIN: float = -1.0
_X0_CLIP_MAX: float = 1.0

# Beta clipping bounds (prevent numerical issues)
_BETA_CLIP_MIN: float = 0.0
_BETA_CLIP_MAX: float = 0.999


class Diffusion(nn.Module):
    """DDPM training and DDIM sampling wrapper for WDNO.

    Wraps a U-Net denoising model with the complete diffusion machinery:
    forward process (noise addition), training loss, and DDIM reverse process.
    All schedule tensors are registered as non-parameter buffers so they
    move with .to(device) and are saved in checkpoints.

    This class is instantiated twice in the WDNO pipeline:
        - BRM (Base-Resolution Model): operates at training resolution
        - SRM (Super-Resolution Model): operates on multi-resolution pairs

    Both instances share the same class but differ in their UNet's
    in_channels and cond_channels.

    Attributes:
        model: U-Net denoising network (eps_theta). Predicts noise from
            noisy wavelet coefficients and conditioning.
        timesteps: Total diffusion training steps K=1000.
            Config: diffusion.num_timesteps.
        beta_schedule: Noise schedule type ('cosine' or 'linear').
            Config: diffusion.beta_schedule.
        device: Compute device string.
        betas: Noise variance schedule, shape [T]. Registered buffer.
        alphas: 1 - betas, shape [T]. Registered buffer.
        alphas_cumprod: Cumulative product of alphas (alpha_bar_k), shape [T].
            Registered buffer.
        alphas_cumprod_prev: alphas_cumprod shifted right by 1 (prepend 1.0),
            shape [T]. Registered buffer. Used for DDIM t_prev=0 case.
        sqrt_alphas_cumprod: sqrt(alpha_bar_k), shape [T]. Registered buffer.
        sqrt_one_minus_alphas_cumprod: sqrt(1-alpha_bar_k), shape [T].
            Registered buffer.
        posterior_variance: DDPM posterior variance, shape [T]. Registered buffer.
        sqrt_recip_alphas_cumprod: 1/sqrt(alpha_bar_k), shape [T].
            Registered buffer. Used in predict_x0_from_eps.
        sqrt_recipm1_alphas_cumprod: sqrt(1/alpha_bar_k - 1), shape [T].
            Registered buffer. Used in predict_x0_from_eps.
    """

    def __init__(
        self,
        model: UNet,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        device: str = "cuda",
    ) -> None:
        """Initialize the Diffusion model.

        Computes and registers all diffusion schedule tensors as buffers.
        The schedule is computed once at construction and reused throughout
        training and inference.

        Args:
            model: U-Net denoising network. Must be constructed externally
                with the correct in_channels, out_channels, and cond_channels
                for the target experiment. Config: unet_1d.* or unet_3d.*.
            timesteps: Total number of diffusion training steps K.
                Config: diffusion.num_timesteps=1000.
            beta_schedule: Noise schedule type. 'cosine' uses the cosine
                schedule from Nichol & Dhariwal 2021 (recommended);
                'linear' uses linear interpolation from 1e-4 to 0.02.
                Config: diffusion.beta_schedule=cosine.
            device: Compute device string ('cuda', 'cpu', 'cuda:0', etc.).
                All schedule buffers are moved to this device.

        Raises:
            ValueError: If beta_schedule is not 'cosine' or 'linear'.
            ValueError: If timesteps <= 0.
        """
        super().__init__()

        if timesteps <= 0:
            raise ValueError(
                f"timesteps must be positive, got {timesteps}. "
                "Config: diffusion.num_timesteps=1000."
            )
        if beta_schedule not in ("cosine", "linear"):
            raise ValueError(
                f"beta_schedule must be 'cosine' or 'linear', got '{beta_schedule}'. "
                "Config: diffusion.beta_schedule=cosine."
            )

        self.model: UNet = model
        self.timesteps: int = timesteps
        self.beta_schedule: str = beta_schedule
        self.device: str = device

        # Compute all schedule tensors
        betas: torch.Tensor = self._make_beta_schedule()

        # Derived quantities
        alphas: torch.Tensor = 1.0 - betas
        alphas_cumprod: torch.Tensor = torch.cumprod(alphas, dim=0)

        # Shift alphas_cumprod right by 1, prepend 1.0 for t_prev=0 case
        # alphas_cumprod_prev[t] = alphas_cumprod[t-1], with [0] = 1.0
        alphas_cumprod_prev: torch.Tensor = F.pad(
            alphas_cumprod[:-1], (1, 0), value=1.0
        )  # shape [T]

        sqrt_alphas_cumprod: torch.Tensor = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod: torch.Tensor = torch.sqrt(1.0 - alphas_cumprod)

        # DDPM posterior variance: beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        posterior_variance: torch.Tensor = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod).clamp(min=1e-20)
        )

        # For predict_x0_from_eps:
        # x0_hat = (1/sqrt(alpha_bar_t)) * x_t - sqrt(1/alpha_bar_t - 1) * eps
        sqrt_recip_alphas_cumprod: torch.Tensor = torch.sqrt(
            1.0 / alphas_cumprod.clamp(min=1e-20)
        )
        sqrt_recipm1_alphas_cumprod: torch.Tensor = torch.sqrt(
            (1.0 / alphas_cumprod.clamp(min=1e-20)) - 1.0
        )

        # Register all as non-parameter buffers (moved with .to(device),
        # saved in state_dict, but not updated by optimizer)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod
        )

        logger.info(
            "Diffusion initialized: timesteps=%d, beta_schedule=%s, device=%s. "
            "beta range: [%.2e, %.2e], alpha_bar range: [%.4f, %.4f]",
            timesteps,
            beta_schedule,
            device,
            float(betas.min()),
            float(betas.max()),
            float(alphas_cumprod.min()),
            float(alphas_cumprod.max()),
        )

    # -----------------------------------------------------------------------
    # Beta schedule
    # -----------------------------------------------------------------------

    def _make_beta_schedule(self) -> torch.Tensor:
        """Compute the noise variance schedule beta_t for t in [0, T-1].

        Cosine schedule (Nichol & Dhariwal 2021, recommended):
            f(t) = cos((t/T + s) / (1 + s) * pi/2)^2
            alpha_bar_t = f(t) / f(0)
            beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
            Clipped to [0, 0.999] for numerical stability.

        Linear schedule (Ho et al. 2020, fallback):
            beta_t = linspace(1e-4, 0.02, T)

        Returns:
            Tensor of shape [timesteps] containing beta values. dtype=float32.
            Values are in [0, 0.999] for cosine or [1e-4, 0.02] for linear.
        """
        T: int = self.timesteps

        if self.beta_schedule == "cosine":
            # Cosine schedule: compute alpha_bar at T+1 points (0 to T inclusive)
            # t_steps: [0, 1, ..., T] normalized to [0, 1]
            t_steps: torch.Tensor = torch.linspace(0, T, T + 1, dtype=torch.float64)

            # f(t) = cos((t/T + s) / (1 + s) * pi/2)^2
            s: float = _COSINE_SCHEDULE_S
            f: torch.Tensor = torch.cos(
                (t_steps / T + s) / (1.0 + s) * (math.pi / 2.0)
            ) ** 2

            # Normalize: alpha_bar_t = f(t) / f(0)
            # f(0) = cos(s/(1+s) * pi/2)^2
            alphas_cumprod_full: torch.Tensor = f / f[0]

            # beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
            # alphas_cumprod_full has T+1 elements; betas has T elements
            betas: torch.Tensor = 1.0 - (
                alphas_cumprod_full[1:] / alphas_cumprod_full[:-1]
            )

            # Clip to prevent numerical issues
            betas = betas.clamp(min=_BETA_CLIP_MIN, max=_BETA_CLIP_MAX)

            return betas.float()

        else:  # linear
            # Linear schedule from Ho et al. 2020
            betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float32)
            return betas

    # -----------------------------------------------------------------------
    # Forward process
    # -----------------------------------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the forward diffusion process to add noise to clean data.

        Computes x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        as defined in paper Section 2.2.

        This is the reparameterized forward process that allows sampling
        x_t at any timestep t directly from x_0 without iterating through
        all intermediate steps.

        Args:
            x0: Clean wavelet coefficient tensor. Shape [B, C, T, X] for
                1D PDE experiments or [B, C, T, H, W] for 2D PDE experiments.
                This is the output of WaveletTransform.forward() on clean data.
                dtype=float32.
            t: Diffusion timestep indices, shape [B]. Integer values in
                [0, timesteps-1]. Sampled uniformly during training.
            noise: Optional pre-sampled Gaussian noise of same shape as x0.
                If None, samples noise = torch.randn_like(x0). Providing
                pre-sampled noise enables reproducible training.

        Returns:
            Noisy wavelet coefficient tensor x_t of same shape as x0.
            dtype=float32.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Gather schedule values for each batch element: [B]
        sqrt_alpha_bar: torch.Tensor = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar: torch.Tensor = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting: [B] → [B, 1, 1, ...] with x0.ndim-1 trailing dims
        n_trailing: int = x0.ndim - 1
        view_shape: Tuple[int, ...] = (-1,) + (1,) * n_trailing

        sqrt_alpha_bar = sqrt_alpha_bar.view(view_shape)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(view_shape)

        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    # -----------------------------------------------------------------------
    # Training loss
    # -----------------------------------------------------------------------

    def compute_loss(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor,
        cfg_dropout_prob: float = 0.1,
    ) -> torch.Tensor:
        """Compute the simplified ELBO training loss for DDPM.

        Implements the noise prediction loss from paper Section 2.2 (Eq. 2):
            L = E_{k,x0,eps}[||eps - eps_theta(sqrt(alpha_bar_k)*x0
                             + sqrt(1-alpha_bar_k)*eps, k)||^2]

        Classifier-free guidance (CFG) is implemented by randomly dropping
        the conditioning with probability cfg_dropout_prob, replacing it with
        zeros. This trains the model to handle both conditional and
        unconditional generation, enabling CFG at inference time.

        Args:
            x0: Clean wavelet coefficient tensor (denoising target).
                Shape [B, C, T, X] for 1D or [B, C, T, H, W] for 2D.
                This is WaveletTransform.forward() applied to clean PDE data.
                dtype=float32.
            cond: Conditioning wavelet coefficient tensor. Shape matches x0
                except for the channel dimension (cond_channels may differ
                from in_channels). For BRM: wavelet coefficients of (u0, f)
                or (u0, u_star). For SRM: wavelet coefficients of
                (W_low_dup, W_cond_high) concatenated.
                dtype=float32.
            cfg_dropout_prob: Probability of dropping the condition (replacing
                with zeros) for classifier-free guidance training.
                Config: diffusion.cfg_dropout_prob=0.1.

        Returns:
            Scalar MSE loss tensor. Backpropagatable for optimizer.step().

        Note:
            The timestep t is sampled uniformly from [0, timesteps-1] for
            each batch element independently, following standard DDPM training.
        """
        B: int = x0.shape[0]
        device: torch.device = x0.device

        # Sample random timesteps uniformly: [B] in [0, T-1]
        t: torch.Tensor = torch.randint(
            0, self.timesteps, (B,), device=device, dtype=torch.long
        )

        # Sample Gaussian noise
        noise: torch.Tensor = torch.randn_like(x0)

        # Apply forward diffusion: x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise
        x_noisy: torch.Tensor = self.q_sample(x0, t, noise)

        # CFG dropout: replace condition with zeros for cfg_dropout_prob fraction
        # of batch elements. This trains both conditional and unconditional branches.
        if cfg_dropout_prob > 0.0:
            # mask[i] = True means drop condition for sample i
            drop_mask: torch.Tensor = (
                torch.rand(B, device=device) < cfg_dropout_prob
            )  # [B], bool

            # Reshape mask for broadcasting over all condition dimensions
            n_trailing: int = cond.ndim - 1
            mask_view: Tuple[int, ...] = (-1,) + (1,) * n_trailing
            drop_mask_expanded: torch.Tensor = drop_mask.view(mask_view)

            # Zero out condition where mask is True
            cond_input: torch.Tensor = cond * (~drop_mask_expanded).float()
        else:
            cond_input = cond

        # Predict noise: eps_theta(x_t, t, cond)
        eps_pred: torch.Tensor = self.model(x_noisy, t, cond_input)

        # MSE loss between predicted and true noise
        return F.mse_loss(eps_pred, noise)

    # -----------------------------------------------------------------------
    # DDIM sampling
    # -----------------------------------------------------------------------

    def ddim_sample(
        self,
        shape: Tuple[int, ...],
        cond: torch.Tensor,
        ddim_steps: int = 50,
        eta: float = 1.0,
        cfg_weight: float = 1.0,
        guidance_fn: Optional[Callable[[torch.Tensor, int, int], torch.Tensor]] = None,
        guidance_lambda_schedule: Optional[Callable[[int, int], float]] = None,
    ) -> torch.Tensor:
        """Run the full DDIM reverse process to generate wavelet coefficients.

        Implements the DDIM sampling algorithm (Song et al. 2020) with
        classifier-free guidance. Starts from Gaussian noise and iteratively
        denoises to produce clean wavelet coefficients.

        For simulation: guidance_fn=None, pure DDIM with CFG.
        For control: guidance_fn provides the control objective gradient,
            applied after each DDIM step to steer generation toward lower I.

        DDIM uses a subsequence of the full [0, T] timesteps, enabling
        fast inference with ddim_steps << timesteps (e.g., 50 vs 1000).

        Args:
            shape: Output tensor shape (B, C, *spatial_dims). For 1D Burgers':
                (B, 4, 41, 60). For 2D fluid: (B, 8, 18, 34, 34).
                Must match the wavelet coefficient shape for the experiment.
            cond: Conditioning wavelet coefficient tensor. Shape [B, C_cond,
                *spatial_dims]. Passed to the U-Net at each denoising step.
                For CFG, the null condition (zeros) is also used internally.
            ddim_steps: Number of DDIM sampling steps. Config:
                inference.burgers.ddim_steps=50,
                inference.compressible_ns.ddim_steps=850,
                inference.fluid_2d.ddim_steps=100.
            eta: DDIM stochasticity parameter. eta=0: deterministic DDIM;
                eta=1: matches DDPM variance. Config:
                inference.burgers.ddim_eta=1.0.
            cfg_weight: Classifier-free guidance weight omega. cfg_weight=1.0
                means pure conditional (no unconditional mixing).
                cfg_weight=0.0 means pure unconditional.
                Config: diffusion.cfg_weight=1.0.
            guidance_fn: Optional control objective gradient function.
                Signature: (x0_hat: Tensor, step: int, total_steps: int) -> Tensor
                where the return is the gradient correction to add to eps.
                If None, no control guidance is applied (simulation mode).
                Used by WDNOPipeline._control_guidance_step.
            guidance_lambda_schedule: Optional callable returning the guidance
                weight lambda at each step.
                Signature: (step: int, total_steps: int) -> float.
                Only used when guidance_fn is not None.
                Config: inference.burgers.guidance_lambda=120000,
                inference.burgers.guidance_schedule=cosine.

        Returns:
            Generated wavelet coefficient tensor of shape ``shape``.
            dtype=float32. Apply WaveletTransform.inverse() to get PDE states.
        """
        device: torch.device = torch.device(self.device)
        B: int = shape[0]

        # Initialize from Gaussian noise: x_T ~ N(0, I)
        x: torch.Tensor = torch.randn(shape, device=device)

        # Build DDIM timestep subsequence
        # Map ddim_steps inference steps to the full [0, T-1] range
        # time_seq: [T-1, ..., 0] in ddim_steps steps (descending)
        time_seq: torch.Tensor = torch.linspace(
            0, self.timesteps - 1, ddim_steps + 1, dtype=torch.long
        )
        # Reverse to go from noisy (T-1) to clean (0)
        time_seq = time_seq.flip(0)  # [T-1, ..., 0], length ddim_steps+1

        # Create (t, t_prev) pairs: t is current step, t_prev is next (cleaner) step
        # time_seq[0] = T-1 (most noisy), time_seq[-1] = 0 (cleanest)
        # Pairs: (time_seq[0], time_seq[1]), (time_seq[1], time_seq[2]), ...
        time_pairs: List[Tuple[int, int]] = [
            (int(time_seq[i].item()), int(time_seq[i + 1].item()))
            for i in range(ddim_steps)
        ]

        total_steps: int = len(time_pairs)

        # Iterative denoising
        for step_idx, (t_val, t_prev_val) in enumerate(time_pairs):
            # Single DDIM step with CFG
            x = self._ddim_step(
                x=x,
                cond=cond,
                t=t_val,
                t_prev=t_prev_val,
                eta=eta,
                cfg_weight=cfg_weight,
            )

            # Apply control guidance if provided
            if guidance_fn is not None and guidance_lambda_schedule is not None:
                lambda_t: float = guidance_lambda_schedule(step_idx, total_steps)

                if lambda_t > 0.0:
                    # Compute denoised estimate for gradient computation
                    t_batch: torch.Tensor = torch.full(
                        (B,), t_val, device=device, dtype=torch.long
                    )
                    with torch.enable_grad():
                        x_grad: torch.Tensor = x.detach().requires_grad_(True)

                        # Get current eps prediction for x0_hat computation
                        cond_for_grad: torch.Tensor = cond.detach()
                        eps_for_grad: torch.Tensor = self.model(
                            x_grad, t_batch, cond_for_grad
                        )

                        # Estimate clean x0 from noisy x and predicted eps
                        x0_hat: torch.Tensor = self.predict_x0_from_eps(
                            x_grad, t_batch, eps_for_grad
                        )

                        # Compute gradient correction from guidance function
                        # guidance_fn returns the gradient correction tensor
                        grad_correction: torch.Tensor = guidance_fn(
                            x0_hat, step_idx, total_steps
                        )

                    # Apply guidance: subtract lambda * gradient from x
                    # (gradient descent on the objective I)
                    x = x.detach() - lambda_t * grad_correction.detach()

        return x

    def _ddim_step(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: int,
        t_prev: int,
        eta: float = 1.0,
        cfg_weight: float = 1.0,
    ) -> torch.Tensor:
        """Perform a single DDIM denoising step with classifier-free guidance.

        Implements the DDIM update rule (Song et al. 2020):
            x0_hat = (x_t - sqrt(1-alpha_bar_t)*eps) / sqrt(alpha_bar_t)
            sigma_t = eta * sqrt((1-alpha_bar_{t-1})/(1-alpha_bar_t)
                                 * (1 - alpha_bar_t/alpha_bar_{t-1}))
            dir_xt = sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * eps
            x_{t-1} = sqrt(alpha_bar_{t-1})*x0_hat + dir_xt + sigma_t*noise

        CFG combination (paper Section 2.2):
            eps = eps_uncond + cfg_weight * (eps_cond - eps_uncond)

        When cfg_weight=1.0 (config default), this simplifies to eps_cond.

        Args:
            x: Current noisy wavelet coefficients at timestep t.
                Shape [B, C, *spatial_dims]. dtype=float32.
            cond: Conditioning tensor. Shape [B, C_cond, *spatial_dims].
                dtype=float32.
            t: Current timestep index (integer, 0-indexed). Higher = noisier.
                In range [0, timesteps-1].
            t_prev: Previous (cleaner) timestep index. In range [0, t-1].
                When t_prev=0, this is the final denoising step.
            eta: DDIM stochasticity. eta=0: deterministic; eta=1: DDPM-like.
                Config: inference.burgers.ddim_eta=1.0.
            cfg_weight: CFG guidance weight omega. Config:
                diffusion.cfg_weight=1.0.

        Returns:
            Denoised tensor x_{t_prev} of same shape as x. dtype=float32.
        """
        B: int = x.shape[0]
        device: torch.device = x.device

        # Create batch of timestep indices
        t_batch: torch.Tensor = torch.full(
            (B,), t, device=device, dtype=torch.long
        )

        # --- Classifier-free guidance: combine conditional and unconditional ---
        with torch.no_grad():
            if cfg_weight == 1.0:
                # Pure conditional: skip unconditional forward pass for efficiency
                eps: torch.Tensor = self.model(x, t_batch, cond)
            elif cfg_weight == 0.0:
                # Pure unconditional
                null_cond: torch.Tensor = torch.zeros_like(cond)
                eps = self.model(x, t_batch, null_cond)
            else:
                # Mixed: run both and combine
                null_cond = torch.zeros_like(cond)
                eps_uncond: torch.Tensor = self.model(x, t_batch, null_cond)
                eps_cond: torch.Tensor = self.model(x, t_batch, cond)
                # CFG: eps_uncond + cfg_weight * (eps_cond - eps_uncond)
                eps = eps_uncond + cfg_weight * (eps_cond - eps_uncond)

        # --- Get schedule values for current and previous timesteps ---
        # These are scalars (same t for all batch elements during inference)
        alpha_bar_t: float = float(self.alphas_cumprod[t].item())
        alpha_bar_t_prev: float = float(self.alphas_cumprod_prev[t_prev].item())

        # --- Estimate clean x0 from noisy x_t and predicted eps ---
        # x0_hat = (x_t - sqrt(1-alpha_bar_t)*eps) / sqrt(alpha_bar_t)
        # Clamp to [-1, 1] for numerical stability (standard practice)
        sqrt_alpha_bar_t: float = math.sqrt(max(alpha_bar_t, 1e-20))
        sqrt_one_minus_alpha_bar_t: float = math.sqrt(max(1.0 - alpha_bar_t, 0.0))

        x0_hat: torch.Tensor = (
            x - sqrt_one_minus_alpha_bar_t * eps
        ) / sqrt_alpha_bar_t
        x0_hat =