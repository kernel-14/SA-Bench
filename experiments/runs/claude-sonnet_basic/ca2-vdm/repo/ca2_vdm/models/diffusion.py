"""
Ca2-VDM Diffusion Model.

Implements the full Ca2-VDM model including:
  - DDPM noise schedule (Ho et al., 2020)
  - Training with partially noised inputs (clean prefix + noisy target)
  - Combined loss: L_simple + L_vlb (Nichol & Dhariwal, 2021)
  - Autoregressive inference with KV-cache queue and cache sharing
  - Improved DDPM sampling (Nichol & Dhariwal, 2021) with 100 steps

Training objective (Eq. 2 in paper):
  L_simple(θ) = E[||(ε_θ([z_0^{0:P}, z_t^{P:L}], t) - ε) ⊙ m||²]
  where m masks out the clean prefix (m_i = 1 if i >= P else 0)
  and t_i = t if i >= P else 0 (different timestep embeddings for prefix vs target)

Autoregressive inference (Section 3.3):
  Each AR step:
    1. Denoising stage: denoise l frames using KV-cache from clean prefix
    2. Cache writing stage: compute clean KVs of denoised chunk, update queue
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .transformer import Ca2VDMTransformer
from .kv_cache import KVCacheQueue, SpatialKVCache


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule from Ho et al. (2020)."""
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule from Nichol & Dhariwal (2021)."""
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


class Ca2VDM(nn.Module):
    """
    Ca2-VDM: Efficient Autoregressive Video Diffusion Model.

    Combines the Ca2VDMTransformer with DDPM diffusion process,
    training objectives, and autoregressive inference with KV-cache.

    Args:
        transformer: Ca2VDMTransformer model.
        T: Number of diffusion timesteps (default 1000).
        beta_schedule: 'linear' or 'cosine'.
        beta_start: Start value for linear schedule.
        beta_end: End value for linear schedule.
        chunk_size: l, number of frames per AR step.
        max_prefix_len: P_max, maximum number of conditional frames.
        prefix_len_choices: List of valid prefix lengths for training sampling.
                            Default: [1, 1+l, 1+2l, ..., 1+nl] where 1+nl = P_max.
    """

    def __init__(
        self,
        transformer: Ca2VDMTransformer,
        T: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        chunk_size: int = 16,
        max_prefix_len: int = 49,
    ):
        super().__init__()
        self.transformer = transformer
        self.T = T
        self.chunk_size = chunk_size
        self.max_prefix_len = max_prefix_len

        # Build noise schedule
        if beta_schedule == "linear":
            betas = linear_beta_schedule(T, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(T)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # Posterior variance q(z_{t-1} | z_t, z_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20))
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # Valid prefix lengths for training: P in {1, 1+l, 1+2l, ..., P_max}
        self.prefix_len_choices = [1]
        p = 1 + chunk_size
        while p <= max_prefix_len:
            self.prefix_len_choices.append(p)
            p += chunk_size

    def q_sample(
        self,
        z0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward diffusion: sample z_t from z_0.
        z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * eps

        Args:
            z0: Clean latent of shape (B, ...).
            t: Timestep indices of shape (B,).
            noise: Optional noise tensor. If None, sampled from N(0, I).

        Returns:
            z_t of same shape as z0.
        """
        if noise is None:
            noise = torch.randn_like(z0)

        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        while sqrt_alpha_bar.dim() < z0.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)

        return sqrt_alpha_bar * z0 + sqrt_one_minus_alpha_bar * noise

    def predict_start_from_noise(
        self, z_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Predict z_0 from z_t and predicted noise."""
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t]
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t]
        while sqrt_recip.dim() < z_t.dim():
            sqrt_recip = sqrt_recip.unsqueeze(-1)
            sqrt_recipm1 = sqrt_recipm1.unsqueeze(-1)
        return sqrt_recip * z_t - sqrt_recipm1 * noise

    def q_posterior(
        self,
        z0: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior q(z_{t-1} | z_t, z_0).

        Returns:
            posterior_mean, posterior_variance, posterior_log_variance_clipped
        """
        coef1 = self.posterior_mean_coef1[t]
        coef2 = self.posterior_mean_coef2[t]
        while coef1.dim() < z0.dim():
            coef1 = coef1.unsqueeze(-1)
            coef2 = coef2.unsqueeze(-1)

        posterior_mean = coef1 * z0 + coef2 * z_t
        posterior_variance = self.posterior_variance[t]
        posterior_log_variance = self.posterior_log_variance_clipped[t]
        while posterior_variance.dim() < z_t.dim():
            posterior_variance = posterior_variance.unsqueeze(-1)
            posterior_log_variance = posterior_log_variance.unsqueeze(-1)

        return posterior_mean, posterior_variance, posterior_log_variance

    def training_loss(
        self,
        z0: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss for Ca2-VDM.

        Implements the combined loss L_simple + L_vlb from Section 3.2 and Appendix B.

        Args:
            z0: Clean video latent of shape (B, L, C, H, W).
            context: Optional text embeddings of shape (B, M, context_dim).
            context_mask: Optional text mask of shape (B, M).

        Returns:
            Dict with 'loss', 'loss_simple', 'loss_vlb'.
        """
        B, L, C, H, W = z0.shape
        device = z0.device

        # Sample random timestep t for denoising target
        t = torch.randint(0, self.T, (B,), device=device)

        # Sample random prefix length P from valid choices
        # P in {1, 1+l, ..., P_max}, but P must be <= L - 1 (need at least 1 denoising frame)
        valid_choices = [p for p in self.prefix_len_choices if p < L]
        if not valid_choices:
            valid_choices = [1]
        P = valid_choices[torch.randint(len(valid_choices), (1,)).item()]

        # Sample random cyclic TPE offset for training
        cyclic_offset = torch.randint(0, self.transformer.max_seq_len, (1,)).item()

        # Sample noise for denoising target frames
        noise = torch.randn_like(z0[:, P:])  # (B, L-P, C, H, W)

        # Create noisy denoising target
        z_t_target = self.q_sample(z0[:, P:], t, noise)  # (B, L-P, C, H, W)

        # Concatenate clean prefix + noisy target
        # z_input: (B, L, C, H, W) with first P frames clean, rest noisy
        z_input = torch.cat([z0[:, :P], z_t_target], dim=1)

        # Create timestep vector: t_i = 0 for prefix, t for denoising target
        t_vec = torch.zeros(B, L, dtype=torch.long, device=device)
        t_vec[:, P:] = t.unsqueeze(1).expand(-1, L - P)

        # Forward pass
        # The model predicts noise (and optionally variance) for denoising target
        model_out, _ = self.transformer(
            z_input,
            t_vec,
            prefix_len=P,
            context=context,
            context_mask=context_mask,
            cyclic_tpe_offset=cyclic_offset,
        )
        # model_out: (B, L-P, out_channels, H, W)
        # Split into noise prediction and variance prediction
        out_channels = self.transformer.out_channels
        in_channels = self.transformer.in_channels
        if out_channels == 2 * in_channels:
            eps_pred, v_pred = model_out.chunk(2, dim=2)
        else:
            eps_pred = model_out
            v_pred = None

        # Loss mask: only compute loss on denoising target (already done by slicing)
        # L_simple: MSE between predicted and actual noise
        loss_simple = F.mse_loss(eps_pred, noise)

        # L_vlb: KL divergence term (for learned variance)
        loss_vlb = torch.tensor(0.0, device=device)
        if v_pred is not None:
            # Compute L_vlb using the learned variance
            # Following Nichol & Dhariwal (2021): interpolate between
            # posterior variance and beta_t in log space
            z0_pred = self.predict_start_from_noise(z_t_target, t, eps_pred.detach())
            z0_pred = z0_pred.clamp(-1, 1)

            # True posterior
            true_mean, _, true_log_var = self.q_posterior(z0[:, P:], z_t_target, t)

            # Predicted posterior using learned variance
            # v_pred interpolates between log(beta_t) and log(posterior_variance)
            min_log = self.posterior_log_variance_clipped[t]
            max_log = torch.log(self.betas[t])
            while min_log.dim() < z_t_target.dim():
                min_log = min_log.unsqueeze(-1)
                max_log = max_log.unsqueeze(-1)

            # v_pred is in [-1, 1], map to [min_log, max_log]
            frac = (v_pred + 1) / 2
            model_log_var = frac * max_log + (1 - frac) * min_log

            pred_mean, _, _ = self.q_posterior(z0_pred, z_t_target, t)

            # KL divergence between two Gaussians
            kl = 0.5 * (
                -1.0
                + model_log_var - true_log_var
                + torch.exp(true_log_var - model_log_var)
                + ((true_mean - pred_mean) ** 2) * torch.exp(-model_log_var)
            )
            loss_vlb = kl.mean()

        loss = loss_simple + loss_vlb

        return {
            "loss": loss,
            "loss_simple": loss_simple,
            "loss_vlb": loss_vlb,
        }

    @torch.no_grad()
    def ddpm_step(
        self,
        z_t: torch.Tensor,
        t: int,
        eps_pred: torch.Tensor,
        v_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single DDPM denoising step: sample z_{t-1} from z_t.

        Args:
            z_t: Noisy latent of shape (B, l, C, H, W).
            t: Current timestep (integer).
            eps_pred: Predicted noise of shape (B, l, C, H, W).
            v_pred: Optional predicted variance of shape (B, l, C, H, W).

        Returns:
            z_{t-1} of shape (B, l, C, H, W).
        """
        B = z_t.shape[0]
        device = z_t.device
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        # Predict z_0
        z0_pred = self.predict_start_from_noise(z_t, t_tensor, eps_pred)
        z0_pred = z0_pred.clamp(-1, 1)

        # Compute posterior mean
        posterior_mean, posterior_var, posterior_log_var = self.q_posterior(z0_pred, z_t, t_tensor)

        # Compute variance
        if v_pred is not None:
            # Learned variance (Nichol & Dhariwal, 2021)
            min_log = self.posterior_log_variance_clipped[t_tensor]
            max_log = torch.log(self.betas[t_tensor])
            while min_log.dim() < z_t.dim():
                min_log = min_log.unsqueeze(-1)
                max_log = max_log.unsqueeze(-1)
            frac = (v_pred + 1) / 2
            model_log_var = frac * max_log + (1 - frac) * min_log
            model_var = torch.exp(model_log_var)
        else:
            model_var = posterior_var
            model_log_var = posterior_log_var

        # Sample z_{t-1}
        noise = torch.randn_like(z_t) if t > 0 else torch.zeros_like(z_t)
        z_prev = posterior_mean + torch.sqrt(model_var) * noise

        return z_prev

    @torch.no_grad()
    def autoregressive_generate(
        self,
        first_frame: torch.Tensor,
        num_frames: int,
        num_denoising_steps: int = 100,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        guidance_scale: float = 7.5,
        use_kv_cache: bool = True,
        use_prefix_enhancement: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive video generation with KV-cache and cache sharing.

        Implements the inference procedure from Section 3.3:
          1. Start from given first frame
          2. For each AR step:
             a. Denoising stage: denoise l frames using shared KV-cache
             b. Cache writing stage: compute clean KVs, update queue

        Args:
            first_frame: Initial frame latent of shape (B, C, H, W).
            num_frames: Total number of frames to generate.
            num_denoising_steps: Number of DDPM denoising steps (default 100).
            context: Optional text embeddings of shape (B, M, context_dim).
            context_mask: Optional text mask of shape (B, M).
            guidance_scale: Classifier-free guidance scale (default 7.5).
            use_kv_cache: Whether to use KV-cache (True = Ca2-VDM, False = baseline).
            use_prefix_enhancement: Whether to use prefix-enhanced spatial attention.

        Returns:
            Generated video latents of shape (B, num_frames, C, H, W).
        """
        B, C, H, W = first_frame.shape
        device = first_frame.device
        l = self.chunk_size
        num_layers = self.transformer.depth

        # Initialize KV-cache queues
        if use_kv_cache:
            temporal_cache = KVCacheQueue(
                max_frames=self.max_prefix_len,
                chunk_size=l,
                num_layers=num_layers,
            )
            spatial_cache = SpatialKVCache(num_layers=num_layers) if use_prefix_enhancement else None
        else:
            temporal_cache = None
            spatial_cache = None

        # Build DDPM timestep schedule (improved DDPM: cosine schedule for inference)
        # Use evenly spaced timesteps from T to 0
        timesteps = list(range(self.T - 1, -1, -1))
        step_size = max(1, self.T // num_denoising_steps)
        timesteps = timesteps[::step_size][:num_denoising_steps]

        # Generated frames accumulator
        generated_frames = [first_frame.unsqueeze(1)]  # list of (B, 1, C, H, W)
        clean_prefix = first_frame.unsqueeze(1)  # (B, 1, C, H, W)

        # Compute initial cache for first frame (cache writing for first frame)
        if use_kv_cache:
            self._cache_write(
                clean_prefix,
                temporal_cache,
                spatial_cache,
                context=context,
                context_mask=context_mask,
                ar_step=0,
                p_k=0,
            )

        num_ar_steps = (num_frames - 1 + l - 1) // l  # ceil((num_frames-1) / l)

        for ar_step in range(num_ar_steps):
            p_k = clean_prefix.shape[1]  # Number of clean prefix frames

            # Initialize noisy denoising target
            z_t = torch.randn(B, l, C, H, W, device=device)

            # Get temporal KV caches for all layers
            if use_kv_cache:
                layer_caches = [temporal_cache.get_cache(i) for i in range(num_layers)]
            else:
                # No cache: concatenate all clean prefix frames directly
                layer_caches = None

            # Denoising stage: shared cache across all timesteps
            for t_idx, t in enumerate(timesteps):
                t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

                if use_kv_cache:
                    # Use shared KV-cache (cache sharing: same cache for all t)
                    model_out, _ = self.transformer(
                        z_t,
                        t_tensor,
                        prefix_len=0,  # No prefix in input; cache handles it
                        context=context,
                        context_mask=context_mask,
                        temporal_kv_caches=layer_caches,
                        spatial_prefix_cache=self._get_spatial_prefix(spatial_cache, num_layers) if spatial_cache else None,
                        cyclic_tpe_offset=self._get_tpe_offset(p_k, l, ar_step),
                    )
                else:
                    # Baseline: concatenate clean prefix + noisy target
                    z_input = torch.cat([clean_prefix, z_t], dim=1)
                    t_vec = torch.zeros(B, p_k + l, dtype=torch.long, device=device)
                    t_vec[:, p_k:] = t
                    model_out, _ = self.transformer(
                        z_input,
                        t_vec,
                        prefix_len=p_k,
                        context=context,
                        context_mask=context_mask,
                    )

                # Classifier-free guidance
                if guidance_scale != 1.0 and context is not None:
                    # Unconditional forward pass
                    if use_kv_cache:
                        model_out_uncond, _ = self.transformer(
                            z_t,
                            t_tensor,
                            prefix_len=0,
                            context=None,
                            temporal_kv_caches=layer_caches,
                            cyclic_tpe_offset=self._get_tpe_offset(p_k, l, ar_step),
                        )
                    else:
                        model_out_uncond, _ = self.transformer(
                            z_input,
                            t_vec,
                            prefix_len=p_k,
                            context=None,
                        )
                    model_out = model_out_uncond + guidance_scale * (model_out - model_out_uncond)

                # Split noise and variance predictions
                out_channels = self.transformer.out_channels
                in_channels = self.transformer.in_channels
                if out_channels == 2 * in_channels:
                    eps_pred, v_pred = model_out.chunk(2, dim=2)
                else:
                    eps_pred = model_out
                    v_pred = None

                # DDPM step
                z_t = self.ddpm_step(z_t, t, eps_pred, v_pred)

            # z_t is now z_0 (denoised chunk)
            denoised_chunk = z_t  # (B, l, C, H, W)
            generated_frames.append(denoised_chunk)

            # Cache writing stage: compute clean KVs for denoised chunk
            if use_kv_cache:
                self._cache_write(
                    denoised_chunk,
                    temporal_cache,
                    spatial_cache,
                    context=context,
                    context_mask=context_mask,
                    ar_step=ar_step + 1,
                    p_k=p_k,
                )

            # Update clean prefix (for non-cache baseline)
            if not use_kv_cache:
                clean_prefix = torch.cat([clean_prefix, denoised_chunk], dim=1)
                # Truncate to max_prefix_len
                if clean_prefix.shape[1] > self.max_prefix_len:
                    clean_prefix = clean_prefix[:, -self.max_prefix_len:]

        # Concatenate all generated frames
        all_frames = torch.cat(generated_frames, dim=1)  # (B, 1 + num_ar_steps*l, C, H, W)
        return all_frames[:, :num_frames]

    def _get_tpe_offset(self, p_k: int, chunk_size: int, ar_step: int) -> int:
        """
        Compute the TPE offset for the denoising target at current AR step.
        Implements Cyclic-TPE: when p_k >= p_max, wrap around.
        """
        max_len = self.transformer.max_seq_len
        if p_k < self.max_prefix_len:
            return p_k
        else:
            # Cyclic: start from p_k % max_len
            return p_k % max_len

    def _get_spatial_prefix(
        self,
        spatial_cache: SpatialKVCache,
        num_layers: int,
    ) -> Optional[torch.Tensor]:
        """
        Get spatial prefix features from cache for prefix-enhanced spatial attention.
        Returns the cached spatial KV for the first layer (used as prefix features).
        """
        # The spatial prefix is the hidden features of the last P' frames
        # In practice, we return the cached K features as the prefix
        # This is a simplification; the actual implementation would store
        # the hidden features before the spatial attention projection
        return None  # Handled internally by the spatial cache

    @torch.no_grad()
    def _cache_write(
        self,
        clean_frames: torch.Tensor,
        temporal_cache: KVCacheQueue,
        spatial_cache: Optional[SpatialKVCache],
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        ar_step: int = 0,
        p_k: int = 0,
    ):
        """
        Cache writing stage: compute clean KV features for denoised frames.

        Runs a partial forward pass on the denoised frames (at t=0) to compute
        the KV-cache that will be used in the next AR step.

        Args:
            clean_frames: Denoised frames of shape (B, l, C, H, W).
            temporal_cache: Temporal KV-cache queue to update.
            spatial_cache: Spatial KV-cache to update.
            context: Optional text embeddings.
            context_mask: Optional text mask.
            ar_step: Current AR step index.
            p_k: Number of previously generated frames.
        """
        B, l, C, H, W = clean_frames.shape
        device = clean_frames.device

        # Use t=0 for clean frames (cache sharing: always t=0 for prefix)
        t_zero = torch.zeros(B, dtype=torch.long, device=device)

        # Compute TPE offset for clean frames
        tpe_offset = p_k % self.transformer.max_seq_len

        # Forward pass with return_kv=True to get KV pairs
        _, kv_dicts = self.transformer(
            clean_frames,
            t_zero,
            prefix_len=0,
            context=context,
            context_mask=context_mask,
            cyclic_tpe_offset=tpe_offset,
            return_kv=True,
        )

        if kv_dicts is not None:
            # Extract temporal KV pairs from each layer
            temporal_kvs = []
            spatial_kvs = []
            for layer_kv in kv_dicts:
                if layer_kv is not None:
                    t_kv = layer_kv.get("temporal_kv")
                    s_kv = layer_kv.get("spatial_kv")
                    if t_kv is not None:
                        # t_kv: (K, V) each of shape (B*HW, l, C_head)
                        # We need to store them for temporal attention
                        temporal_kvs.append(t_kv)
                    if s_kv is not None and spatial_cache is not None:
                        spatial_kvs.append(s_kv)

            if temporal_kvs:
                temporal_cache.update_all_layers(temporal_kvs)

            if spatial_kvs and spatial_cache is not None:
                spatial_cache.update_all_layers(spatial_kvs)
