## training/losses.py
"""Diffusion noise schedule utilities and loss computation for Hi-MAR.

This module provides two public symbols consumed by both diffusion heads:

1. ``sinusoidal_embedding`` — a standalone function that converts integer
   timestep indices into sinusoidal embeddings. Defined here once to avoid
   duplication between ``MLPDiffusionHead`` and ``DiTDiffusionHead``.

2. ``DiffusionUtils`` — a plain Python class (not ``nn.Module``) that
   precomputes the linear DDPM beta schedule and exposes methods for forward
   diffusion (``q_sample``), loss computation (``compute_loss``), single
   reverse steps (``ddpm_sample_step``), and full sampling loops (``sample``).

Configuration alignment (config.yaml → diffusion section):
    diffusion.timesteps   = 100
    diffusion.beta_start  = 0.0001
    diffusion.beta_end    = 0.02
    diffusion.schedule    = linear
    diffusion.loss        = mse

Paper reference (Section 3.1):
    L(z_i, x_i) = E_{ε,t}[||ε - ε_θ(x_i^t | t, z_i)||²]
    where x_i^t is the noise-corrupted vector of x_i,
    ε ~ N(0, I), and t is a timestep of the noise schedule.
"""

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding (shared by both diffusion heads)
# ---------------------------------------------------------------------------


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Converts integer timestep indices into sinusoidal embeddings.

    Implements the standard sinusoidal position embedding formula adapted for
    scalar diffusion timesteps. Half the output dimensions encode ``sin``
    components and half encode ``cos`` components, with geometrically spaced
    frequencies.

    Formula:
        freq_i = exp(-log(10000) * i / (dim // 2))   for i in 0 … dim//2-1
        emb[b, 2i]   = sin(t[b] * freq_i)
        emb[b, 2i+1] = cos(t[b] * freq_i)

    This function is imported by both ``MLPDiffusionHead`` and
    ``DiTDiffusionHead`` to build their ``time_embed`` modules, ensuring a
    consistent timestep representation across both phases.

    Args:
        t: Integer timestep tensor of shape ``[B]``. Values are in
            ``{0, …, T-1}`` where ``T = diffusion.timesteps = 100``.
        dim: Embedding dimension. The MLP head uses 256 (per the Logic
            Analysis spec); the DiT head uses the same. Must be even.

    Returns:
        Float tensor of shape ``[B, dim]`` on the same device as ``t``.

    Raises:
        ValueError: If ``dim`` is odd (sinusoidal embedding requires even dim).
    """
    if dim % 2 != 0:
        raise ValueError(
            f"sinusoidal_embedding requires an even dimension, got dim={dim}."
        )

    device: torch.device = t.device
    half_dim: int = dim // 2

    # Geometric frequency sequence: shape [half_dim].
    # freq_i = exp(-log(10000) * i / half_dim)
    exponents: torch.Tensor = torch.arange(half_dim, dtype=torch.float32, device=device)
    freqs: torch.Tensor = torch.exp(
        -math.log(10000.0) * exponents / float(half_dim)
    )  # [half_dim]

    # Outer product: t [B] × freqs [half_dim] → args [B, half_dim].
    args: torch.Tensor = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half_dim]

    # Concatenate sin and cos components along the last dimension.
    embedding: torch.Tensor = torch.cat(
        [torch.sin(args), torch.cos(args)], dim=-1
    )  # [B, dim]

    return embedding


# ---------------------------------------------------------------------------
# DiffusionUtils
# ---------------------------------------------------------------------------


class DiffusionUtils:
    """Encapsulates the DDPM noise schedule and all diffusion computations.

    This is a plain Python class — not an ``nn.Module`` — because it holds no
    trainable parameters. All schedule tensors are precomputed constants stored
    as instance attributes with ``requires_grad=False``.

    The class is shared by both ``MLPDiffusionHead`` (Phase 1) and
    ``DiTDiffusionHead`` (Phase 2), ensuring a consistent noise schedule and
    loss formulation across the two phases.

    Configuration alignment (config.yaml):
        diffusion.timesteps  = 100
        diffusion.beta_start = 0.0001
        diffusion.beta_end   = 0.02
        diffusion.schedule   = linear

    Attributes:
        timesteps: Total number of diffusion timesteps ``T``.
        device: Compute device on which all schedule tensors reside.
        betas: Linear beta schedule, shape ``[T]``.
        alphas: ``1 - betas``, shape ``[T]``.
        alphas_cumprod: Cumulative product ``ᾱ_t``, shape ``[T]``.
        alphas_cumprod_prev: ``ᾱ_{t-1}`` with boundary ``ᾱ_0 = 1``,
            shape ``[T]``.
        sqrt_alphas_cumprod: ``√ᾱ_t``, shape ``[T]``.
        sqrt_one_minus_alphas_cumprod: ``√(1 - ᾱ_t)``, shape ``[T]``.
        posterior_variance: DDPM posterior variance ``β̃_t``, shape ``[T]``.
    """

    def __init__(
        self,
        timesteps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: Optional[torch.device] = None,
    ) -> None:
        """Precomputes the linear DDPM beta schedule and all derived quantities.

        Args:
            timesteps: Number of diffusion timesteps ``T``. Config specifies
                ``diffusion.timesteps = 100``.
            beta_start: Starting value of the linear beta schedule. Config
                specifies ``diffusion.beta_start = 0.0001``.
            beta_end: Ending value of the linear beta schedule. Config
                specifies ``diffusion.beta_end = 0.02``.
            device: Compute device for all schedule tensors. Defaults to CPU
                if not provided; callers should pass the training device.
        """
        self.timesteps: int = timesteps
        self.device: torch.device = device if device is not None else torch.device("cpu")

        # ------------------------------------------------------------------
        # 1. Linear beta schedule: β_t = linspace(beta_start, beta_end, T)
        # ------------------------------------------------------------------
        betas: torch.Tensor = torch.linspace(
            beta_start, beta_end, timesteps, dtype=torch.float64
        )
        # Store as float32 for compatibility with model activations.
        self.betas: torch.Tensor = betas.float().to(self.device)

        # ------------------------------------------------------------------
        # 2. Alphas: α_t = 1 - β_t
        # ------------------------------------------------------------------
        self.alphas: torch.Tensor = (1.0 - self.betas).to(self.device)

        # ------------------------------------------------------------------
        # 3. Cumulative product: ᾱ_t = ∏_{s=1}^{t} α_s
        # ------------------------------------------------------------------
        self.alphas_cumprod: torch.Tensor = torch.cumprod(
            self.alphas, dim=0
        ).to(self.device)

        # ------------------------------------------------------------------
        # 4. Shifted cumulative product for posterior: ᾱ_{t-1}
        #    Prepend 1.0 so that ᾱ_0 = 1 (clean data boundary).
        # ------------------------------------------------------------------
        self.alphas_cumprod_prev: torch.Tensor = torch.cat(
            [
                torch.ones(1, dtype=torch.float32, device=self.device),
                self.alphas_cumprod[:-1],
            ],
            dim=0,
        )  # shape [T]

        # ------------------------------------------------------------------
        # 5. Square-root quantities used in q_sample and ddpm_sample_step.
        # ------------------------------------------------------------------
        self.sqrt_alphas_cumprod: torch.Tensor = torch.sqrt(
            self.alphas_cumprod
        ).to(self.device)

        self.sqrt_one_minus_alphas_cumprod: torch.Tensor = torch.sqrt(
            1.0 - self.alphas_cumprod
        ).to(self.device)

        # ------------------------------------------------------------------
        # 6. DDPM posterior variance: β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        #    At t=0, (1 - ᾱ_0) = 0 and (1 - ᾱ_{-1}) = 0, so we clamp to
        #    avoid division by zero. The value at t=0 is never used in
        #    practice because we skip noise addition at the final step.
        # ------------------------------------------------------------------
        self.posterior_variance: torch.Tensor = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod).clamp(min=1e-20)
        ).to(self.device)

        # Ensure no gradients flow through schedule tensors.
        for attr in (
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
        ):
            getattr(self, attr).requires_grad_(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: corrupts clean tokens ``x0`` at timestep ``t``.

        Implements the closed-form forward process:
            x_t = √ᾱ_t · x0 + √(1 - ᾱ_t) · ε,   ε ~ N(0, I)

        Paper reference (Section 3.1):
            "x_i^t is the noise-corrupted vector of x_i"

        Args:
            x0: Clean latent tokens, shape ``[B, N, d]``. For Phase 1,
                ``N=64`` (low-res); for Phase 2, ``N=256`` (high-res).
                ``d = latent_channels = 16`` (KL-16 VAE, config vae.latent_channels).
            t: Integer timestep indices, shape ``[B]``. Each value is in
                ``{0, …, T-1}``. One timestep per sample in the batch.

        Returns:
            Tuple of:
                - ``x_noisy``: Noise-corrupted tokens, shape ``[B, N, d]``.
                - ``noise``: The Gaussian noise added, shape ``[B, N, d]``.
                  This is the training target ``ε`` for the MSE loss.
        """
        batch_size: int = x0.shape[0]
        device: torch.device = x0.device

        # Move schedule tensors to the same device as x0 (handles the case
        # where DiffusionUtils was constructed on CPU but x0 is on CUDA).
        sqrt_ac: torch.Tensor = self.sqrt_alphas_cumprod.to(device)[t]  # [B]
        sqrt_oneminus_ac: torch.Tensor = self.sqrt_one_minus_alphas_cumprod.to(device)[t]  # [B]

        # Reshape from [B] to [B, 1, 1] for broadcasting with [B, N, d].
        sqrt_ac = sqrt_ac.view(batch_size, 1, 1)
        sqrt_oneminus_ac = sqrt_oneminus_ac.view(batch_size, 1, 1)

        # Sample noise ε ~ N(0, I), same shape as x0.
        noise: torch.Tensor = torch.randn_like(x0)

        # Apply forward diffusion formula.
        x_noisy: torch.Tensor = sqrt_ac * x0 + sqrt_oneminus_ac * noise

        return x_noisy, noise

    def compute_loss(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes the MSE denoising loss ``||ε - ε_θ||²``.

        When a mask is provided, the loss is averaged only over masked token
        positions (``True`` in the mask), since only masked tokens are part of
        the generation objective. Unmasked tokens are already known and should
        not contribute to the gradient.

        Paper reference (Section 3.1):
            L(z_i, x_i) = E_{ε,t}[||ε - ε_θ(x_i^t | t, z_i)||²]

        Masking convention (Shared Knowledge):
            ``True = masked`` (token is hidden / to be predicted).

        Phase-specific behaviour:
            - Phase 1 (``MLPDiffusionHead``): mask is ``mask_lr [B, 64]``.
            - Phase 2 (``DiTDiffusionHead``): mask is ``mask_hr [B, 256]``.
              The DiT head processes all positions but loss is only on masked.

        Args:
            noise_pred: Predicted noise from the diffusion head,
                shape ``[B, N, d]``.
            noise_target: Ground-truth noise from ``q_sample``,
                shape ``[B, N, d]``.
            mask: Optional boolean tensor, shape ``[B, N]``. ``True`` at
                positions whose loss should be included. If ``None``, the
                loss is averaged over all ``B * N`` positions.

        Returns:
            Scalar loss tensor (mean MSE over selected token positions,
            averaged over the latent dimension ``d``).
        """
        # Element-wise squared error: [B, N, d].
        sq_err: torch.Tensor = (noise_pred - noise_target) ** 2

        # Average over the latent dimension d → per-token scalar loss [B, N].
        per_token_loss: torch.Tensor = sq_err.mean(dim=-1)

        if mask is not None:
            # Select only masked positions. mask is BoolTensor [B, N].
            # per_token_loss[mask] has shape [M] where M = sum(mask).
            masked_loss: torch.Tensor = per_token_loss[mask]
            if masked_loss.numel() == 0:
                # Edge case: no masked tokens in this batch (ratio=0).
                # Return zero loss to avoid NaN from empty mean.
                return torch.tensor(0.0, device=noise_pred.device, requires_grad=True)
            return masked_loss.mean()

        return per_token_loss.mean()

    def ddpm_sample_step(
        self,
        model_fn: Callable,
        x_t: torch.Tensor,
        t: int,
        cond: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Performs a single DDPM reverse diffusion step.

        Given noisy tokens ``x_t`` at timestep ``t``, predicts the noise via
        ``model_fn``, reconstructs the clean estimate ``x0_pred``, then
        computes the posterior mean and adds posterior noise to obtain
        ``x_{t-1}``.

        DDPM reverse step formula:
            x0_pred = (x_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t
            μ̃_t = coef1 · x0_pred + coef2 · x_t
            x_{t-1} = μ̃_t + √β̃_t · z   (z ~ N(0,I) if t > 0, else z = 0)

        where:
            coef1 = √ᾱ_{t-1} · β_t / (1 - ᾱ_t)
            coef2 = √α_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)

        Args:
            model_fn: Callable ``(x_noisy, t_tensor, cond) → noise_pred``.
                Either ``diff_head1.forward`` or ``diff_head2.forward``.
            x_t: Noisy tokens at timestep ``t``, shape ``[B, N, d]``.
            t: Current timestep as a Python ``int``, same for all batch
                elements during inference. Must be in ``{0, …, T-1}``.
            cond: Conditioning from the Transformer backbone. Shape depends
                on the head: ``[B, N, cond_dim]`` for DiT head or
                ``[B, N_masked, cond_dim]`` for MLP head.
            clip_denoised: If ``True``, clamps ``x0_pred`` to ``[-1, 1]``
                following standard DDPM practice. Defaults to ``True``.

        Returns:
            ``x_{t-1}``: Denoised tokens, shape ``[B, N, d]``, on the same
            device as ``x_t``.
        """
        batch_size: int = x_t.shape[0]
        device: torch.device = x_t.device

        # Create a batch of identical timestep tensors for model_fn.
        t_tensor: torch.Tensor = torch.full(
            (batch_size,), t, dtype=torch.long, device=device
        )

        # Predict noise ε_θ(x_t, t, cond).
        with torch.no_grad():
            noise_pred: torch.Tensor = model_fn(x_t, t_tensor, cond)

        # ------------------------------------------------------------------
        # Retrieve schedule values at timestep t and reshape for broadcasting.
        # All scalars → [B, 1, 1] for [B, N, d] tensors.
        # ------------------------------------------------------------------
        def _scalar_to_bcasted(arr: torch.Tensor, idx: int) -> torch.Tensor:
            """Extracts arr[idx] and reshapes to [B, 1, 1]."""
            val: torch.Tensor = arr.to(device)[idx]
            return val.view(1, 1, 1).expand(batch_size, 1, 1)

        sqrt_ac_t: torch.Tensor = _scalar_to_bcasted(self.sqrt_alphas_cumprod, t)
        sqrt_oneminus_ac_t: torch.Tensor = _scalar_to_bcasted(
            self.sqrt_one_minus_alphas_cumprod, t
        )
        alpha_t: torch.Tensor = _scalar_to_bcasted(self.alphas, t)
        alpha_cumprod_t: torch.Tensor = _scalar_to_bcasted(self.alphas_cumprod, t)
        alpha_cumprod_prev_t: torch.Tensor = _scalar_to_bcasted(
            self.alphas_cumprod_prev, t
        )
        beta_t: torch.Tensor = _scalar_to_bcasted(self.betas, t)
        post_var_t: torch.Tensor = _scalar_to_bcasted(self.posterior_variance, t)

        # ------------------------------------------------------------------
        # Reconstruct clean estimate x0_pred from noise prediction.
        # x0_pred = (x_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t
        # ------------------------------------------------------------------
        x0_pred: torch.Tensor = (
            x_t - sqrt_oneminus_ac_t * noise_pred
        ) / sqrt_ac_t.clamp(min=1e-8)

        if clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        # ------------------------------------------------------------------
        # Compute DDPM posterior mean coefficients.
        # coef1 = √ᾱ_{t-1} · β_t / (1 - ᾱ_t)
        # coef2 = √α_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        # ------------------------------------------------------------------
        denom: torch.Tensor = (1.0 - alpha_cumprod_t).clamp(min=1e-20)

        coef1: torch.Tensor = (
            torch.sqrt(alpha_cumprod_prev_t.clamp(min=0.0)) * beta_t / denom
        )
        coef2: torch.Tensor = (
            torch.sqrt(alpha_t.clamp(min=0.0))
            * (1.0 - alpha_cumprod_prev_t)
            / denom
        )

        # Posterior mean: μ̃_t = coef1 · x0_pred + coef2 · x_t
        mean: torch.Tensor = coef1 * x0_pred + coef2 * x_t

        # ------------------------------------------------------------------
        # Add posterior noise (only if t > 0; at t=0 we return the mean).
        # x_{t-1} = μ̃_t + √β̃_t · z
        # ------------------------------------------------------------------
        if t > 0:
            log_var: torch.Tensor = torch.log(post_var_t.clamp(min=1e-20))
            noise: torch.Tensor = torch.randn_like(x_t)
            x_prev: torch.Tensor = mean + torch.exp(0.5 * log_var) * noise
        else:
            x_prev = mean

        return x_prev

    def sample(
        self,
        model_fn: Callable,
        cond: torch.Tensor,
        shape: Tuple[int, ...],
        n_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Full reverse diffusion sampling loop.

        Starts from pure Gaussian noise and iteratively denoises over
        ``n_steps`` reverse steps to produce clean tokens. When
        ``n_steps < self.timesteps``, the full schedule is subsampled using
        evenly spaced timesteps (standard DDPM subsampling).

        This method is called by ``Generator._phase1_generate`` and
        ``Generator._phase2_generate`` for each AR step's newly unmasked
        token positions.

        Paper reference (Section 4.5):
            "we use 32 and 4 steps for the first and second phases with a
            cosine schedule."
            Note: these are AR steps; within each AR step, this method is
            called with the configured number of inner diffusion steps.

        Args:
            model_fn: Callable ``(x_noisy, t_tensor, cond) → noise_pred``.
                Either ``diff_head1.forward`` or ``diff_head2.forward``.
            cond: Conditioning tensor from the Transformer backbone. Shape
                depends on the head being used.
            shape: Output tensor shape, e.g. ``(B, N_masked, d)`` for Phase 1
                or ``(B, N_all, d)`` for Phase 2.
            n_steps: Number of reverse diffusion steps to perform. May be
                less than ``self.timesteps`` for faster inference.
            device: Target device for the output tensor.

        Returns:
            Denoised token tensor of shape ``shape`` on ``device``.
        """
        # Start from pure Gaussian noise.
        x: torch.Tensor = torch.randn(shape, device=device)

        # ------------------------------------------------------------------
        # Build the timestep sequence for subsampled reverse diffusion.
        # We create n_steps evenly spaced indices in [0, T-1] and iterate
        # in reverse (from high noise to low noise).
        # ------------------------------------------------------------------
        if n_steps >= self.timesteps:
            # Use the full schedule in reverse order.
            timestep_seq: list[int] = list(range(self.timesteps - 1, -1, -1))
        else:
            # Subsample: n_steps evenly spaced indices from 0 to T-1.
            # linspace gives n_steps values; we reverse for the denoising loop.
            indices: torch.Tensor = torch.linspace(
                0, self.timesteps - 1, n_steps, dtype=torch.long
            )
            timestep_seq = indices.long().tolist()
            # Sort in descending order (high noise → low noise).
            timestep_seq = sorted(timestep_seq, reverse=True)

        # ------------------------------------------------------------------
        # Reverse diffusion loop: x_T → x_{T-1} → … → x_0
        # ------------------------------------------------------------------
        for t in timestep_seq:
            x = self.ddpm_sample_step(
                model_fn=model_fn,
                x_t=x,
                t=int(t),
                cond=cond,
                clip_denoised=True,
            )

        return x
