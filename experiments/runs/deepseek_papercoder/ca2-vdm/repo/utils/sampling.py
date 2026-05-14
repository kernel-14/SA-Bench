## utils/sampling.py
"""
Noise scheduler implementing the diffusion process used in Ca2‑VDM.

This module provides a custom DDPMScheduler that:
- Pre‑computes a linear beta schedule (DDPM) with configurable endpoints.
- Adds noise during training according to the forward process.
- Supports timestep sub‑sampling for faster inference (improved DDPM with
  learned covariance).
- Implements the reverse step with a learnable variance (log‑variance
  output) following Nichol & Dhariwal (2021).

The schedule parameters are taken from ``config.yaml``:
    - ``num_timesteps``: 1000
    - ``beta_start``: 0.0001
    - ``beta_end``: 0.02
    - ``schedule``: "linear"

All internal computations are performed in float32 for numerical stability;
the outputs are cast to the dtype of the input tensors automatically.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn.functional as F


class DDPMScheduler:
    """
    Linear DDPM scheduler with optional learned covariance for improved DDPM.

    Parameters
    ----------
    num_train_timesteps : int
        Total number of diffusion timesteps (T), e.g. 1000.
    beta_start : float
        Smallest beta value (β₁), e.g. 1e-4.
    beta_end : float
        Largest beta value (β_T), e.g. 0.02.
    schedule : str
        Type of beta schedule – only "linear" is supported.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ) -> None:
        if schedule != "linear":
            raise NotImplementedError(
                f"Only 'linear' schedule is supported, got '{schedule}'."
            )
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")

        self.num_train_timesteps = num_train_timesteps

        # Build the linear schedule: β ∈ [β₁, β_T]
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Pre‑compute various scaled products required by add_noise and step
        self.register_schedule(betas, alphas, alphas_cumprod)

        # Placeholder for inference timesteps (populated by set_timesteps)
        self.timesteps: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Helper to register all derived tensors
    # ------------------------------------------------------------------
    def register_schedule(
        self,
        betas: torch.Tensor,
        alphas: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> None:
        """
        Store the fundamental arrays and compute frequently used projections.

        All tensors are kept in float32 and will be moved to the appropriate
        device when used.
        """
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod

        # For forward (noise addition)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # For computing pred_original_sample in reverse step (optional)
        self.sqrt_recip_alphas_cumprod = 1.0 / self.sqrt_alphas_cumprod
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod

        # Cumulative product of alphas for t-1 (shifted by one, with ᾱ₋₁=1)
        self.alphas_cumprod_prev = F.pad(
            alphas_cumprod[:-1], (1, 0), value=1.0
        )

    # ------------------------------------------------------------------
    # Forward operation – noise addition (training)
    # ------------------------------------------------------------------
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Corrupt clean samples with Gaussian noise according to the DDPM
        forward process:

            q(x_t | x_0) = 𝒩(√(ᾱ_t)·x_0, (1 - ᾱ_t)·I)

        Parameters
        ----------
        original_samples : Tensor
            Clean latent samples, shape ``(B, C, H, W)`` or ``(B, L, C, H, W)``.
            Only the last dimensions are treated as spatial; the timestep
            broadcasting works with any number of leading dims.
        noise : Tensor
            Standard Gaussian noise of the same shape as ``original_samples``.
        timesteps : LongTensor
            Integer indices into the schedule, shape ``(B,)``.  Each entry
            must satisfy ``0 <= t < num_train_timesteps``.

        Returns
        -------
        Tensor
            Noisy version of ``original_samples``, same shape and dtype
            (cast to match input).
        """
        if original_samples.shape != noise.shape:
            raise ValueError(
                f"Shape mismatch: original {original_samples.shape}, noise {noise.shape}"
            )
        if timesteps.dim() != 1:
            raise ValueError(f"timesteps must be 1‑D, got shape {timesteps.shape}")

        # Gather schedule values for each sample
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]  # (B,)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]  # (B,)

        # Reshape for broadcasting over spatial dimensions
        for _ in range(original_samples.dim() - 1):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        sqrt_alpha_prod = sqrt_alpha_prod.to(original_samples.dtype)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.to(original_samples.dtype)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    # ------------------------------------------------------------------
    # Inference timestep sub‑sampling
    # ------------------------------------------------------------------
    def set_timesteps(self, num_inference_steps: int) -> None:
        """
        Prepare the descending sequence of timesteps for accelerated inference.

        The timesteps are evenly spaced from ``num_train_timesteps - 1`` down
        to ``0``.  A minimum of one step is forced.

        Parameters
        ----------
        num_inference_steps : int
            Number of denoising steps used at inference (e.g., 100).
        """
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) exceeds "
                f"num_train_timesteps ({self.num_train_timesteps})"
            )

        max_t = self.num_train_timesteps - 1
        step_indices = torch.linspace(
            max_t, 0, num_inference_steps, dtype=torch.long
        )
        self.timesteps = step_indices

    # ------------------------------------------------------------------
    # Reverse step (improved DDPM with learned variance)
    # ------------------------------------------------------------------
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one reverse (denoising) step with a learned log‑variance.

        The model is assumed to output a tensor with twice the number of
        latent channels: the first half contains the noise prediction ε,
        the second half contains per‑component log‑variance ``log σ²``.

        The variance interpolation follows Nichol & Dhariwal (2021):

            σ² = exp(log_var) · β̃_t + (1 - exp(log_var)) · β_t

        where β̃_t is the posterior variance of the forward process.

        Parameters
        ----------
        model_output : Tensor
            Output of the denoising model, shape ``(B, 2·C, H, W)``.
        timestep : int
            Current diffusion timestep (an index in ``[0, T-1]``).  Must be
            one of the values returned by ``set_timesteps``.
        sample : Tensor
            Current noisy sample ``x_t``, shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Denoised sample ``x_{t-1}``, same shape and dtype as ``sample``.
        """
        if self.timesteps is None:
            raise RuntimeError("Timesteps not set. Call set_timesteps() first.")
        if timestep < 0 or timestep >= self.num_train_timesteps:
            raise ValueError(f"timestep {timestep} out of range [0, {self.num_train_timesteps-1}]")

        # ------------------------------------------------------------------
        # 1. Split model output into noise prediction and log‑variance
        # ------------------------------------------------------------------
        C = sample.shape[1]   # number of latent channels
        if model_output.shape[1] != 2 * C:
            raise ValueError(
                f"model_output must have {2*C} channels (got {model_output.shape[1]})"
            )
        noise_pred = model_output[:, :C, ...]
        log_var = model_output[:, C:, ...]

        # Clamp log‑variance to a safe range to avoid exploding exponentials
        log_var = log_var.clamp(-15.0, 15.0)

        # ------------------------------------------------------------------
        # 2. Compute predicted original sample x̂₀
        # ------------------------------------------------------------------
        alpha_prod_t = self.alphas_cumprod[timestep].to(sample.device)  # scalar
        sqrt_alpha_prod_t = alpha_prod_t.sqrt()
        sqrt_one_minus_alpha_prod_t = (1.0 - alpha_prod_t).sqrt()

        pred_original_sample = (
            sample - sqrt_one_minus_alpha_prod_t * noise_pred
        ) / sqrt_alpha_prod_t

        # ------------------------------------------------------------------
        # 3. Determine previous timestep (t-1 in the inference schedule)
        # ------------------------------------------------------------------
        # Locate the index of *timestep* inside self.timesteps
        match_mask = (self.timesteps == timestep).to(sample.device)
        if match_mask.sum() == 0:
            raise ValueError(
                f"Timestep {timestep} not found in the inference schedule. "
                "Did you call set_timesteps with the correct number of steps?"
            )
        idx = match_mask.nonzero(as_tuple=False)[0].item()

        if idx < len(self.timesteps) - 1:
            prev_t = self.timesteps[idx + 1].item()
            alpha_prod_t_prev = self.alphas_cumprod[prev_t].to(sample.device)
        else:
            # Final step (timestep == 0) – no further noise added
            prev_t = -1
            alpha_prod_t_prev = torch.tensor(1.0, device=sample.device)

        # ------------------------------------------------------------------
        # 4. Compute posterior mean
        # ------------------------------------------------------------------
        alpha_t = self.alphas[timestep].to(sample.device)
        beta_t = self.betas[timestep].to(sample.device)

        # Coefficients from the true posterior mean
        coef1 = alpha_prod_t_prev.sqrt() * beta_t / (1.0 - alpha_prod_t)
        coef2 = alpha_t.sqrt() * (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t)

        pred_mean = coef1 * pred_original_sample + coef2 * sample

        # ------------------------------------------------------------------
        # 5. Compute variance using the learned log‑variance
        # ------------------------------------------------------------------
        # β̃_t = posterior variance (lower bound)
        beta_tilde_t = (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) * beta_t

        # Interpolation between β̃_t and β_t
        variance = torch.exp(log_var) * beta_tilde_t + (1.0 - torch.exp(log_var)) * beta_t

        # For the very last step (t == 0) the variance must be zero
        if timestep == 0:
            variance = torch.zeros_like(variance)

        # ------------------------------------------------------------------
        # 6. Sample x_{t-1}
        # ------------------------------------------------------------------
        noise = torch.randn_like(sample)
        prev_sample = pred_mean + variance.sqrt() * noise

        return prev_sample


    # ------------------------------------------------------------------
    # Utility to obtain the full diffusion timesteps (for logging / eval)
    # ------------------------------------------------------------------
    def get_timesteps(self) -> Optional[torch.Tensor]:
        """
        Return the current inference timestep sequence, or ``None`` if
        ``set_timesteps`` has not been called.
        """
        return self.timesteps

