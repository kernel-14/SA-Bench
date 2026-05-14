import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
from tqdm import tqdm

from model import Ca2VDM
from config import Config, get_t2v_config, get_video_prediction_config


class KVQueue:
    """KV-cache queue for temporal attention.
    
    Stores K/V pairs for up to P_max frames.
    New K/V are appended, oldest are dequeued when exceeding capacity.
    """

    def __init__(self, max_len: int):
        self.max_len = max_len
        self.queue_k: List[torch.Tensor] = []  # list of tensors, each is (B, chunk_len, HW, C)
        self.queue_v: List[torch.Tensor] = []
        self.total_len: int = 0

    def enqueue(self, k: torch.Tensor, v: torch.Tensor):
        """Add new K/V chunk to the queue."""
        self.queue_k.append(k)
        self.queue_v.append(v)
        self.total_len += k.shape[1]  # chunk length

        # Dequeue oldest if exceeding max_len
        while self.total_len > self.max_len and len(self.queue_k) > 0:
            removed_k = self.queue_k.pop(0)
            removed_v = self.queue_v.pop(0)
            self.total_len -= removed_k.shape[1]

    def get_full_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get concatenated K/V for all cached frames."""
        if len(self.queue_k) == 0:
            return None, None
        k_full = torch.cat(self.queue_k, dim=1)  # (B, total_len, HW, C)
        v_full = torch.cat(self.queue_v, dim=1)
        return k_full, v_full

    def clear(self):
        self.queue_k = []
        self.queue_v = []
        self.total_len = 0

    def __len__(self):
        return self.total_len


class Ca2VDMInference:
    """Autoregressive inference for Ca2-VDM with cache sharing.
    
    Implements the inference pipeline described in Sec. 3.3:
    - Each AR step has a denoising stage and a cache writing stage
    - Temporal KV-cache is stored in a queue (max length P_max)
    - Spatial KV-cache stores the most recent chunk's spatial K/V
    - Cache is shared across all denoising timesteps
    - Cyclic-TPEs for positional encoding beyond training length
    """

    def __init__(
        self,
        model: Ca2VDM,
        config: Config,
        device: torch.device = None,
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        self.l = config.inference.chunk_length
        self.P_max = config.inference.max_prefix_length
        self.num_inference_steps = config.inference.num_inference_steps
        self.guidance_scale = config.inference.guidance_scale

        # Initialize KV-cache queues (one per layer)
        self.temporal_kv_queues: List[KVQueue] = [
            KVQueue(self.P_max) for _ in range(self.model.num_layers)
        ]
        # Spatial KV-cache: only one chunk (most recent)
        self.spatial_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # Track cumulative generated frames
        self.P_k: int = 0  # total generated frames so far

        # Store generated latents
        self.generated_latents: List[torch.Tensor] = []

    def reset(self):
        """Reset caches for a new generation."""
        for q in self.temporal_kv_queues:
            q.clear()
        self.spatial_kv_cache = None
        self.P_k = 0
        self.generated_latents = []

    @torch.no_grad()
    def _ddpm_sampling_loop(
        self,
        noisy_latent: torch.Tensor,
        prefix_latent: torch.Tensor,
        tpe_offset: int,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Denoising loop for one AR step.
        
        Iteratively denoises noisy_latent conditioned on prefix_latent.
        Uses shared KV-cache across all timesteps.
        
        Args:
            noisy_latent: (B, l, C, H, W) pure noise at t=T
            prefix_latent: (B, P_k, C, H, W) clean prefix frames
            tpe_offset: TPE offset for cyclic assignment
            encoder_hidden_states: text embeddings for CFG
        
        Returns:
            denoised_latent: (B, l, C, H, W) at t=0
        """
        B, l, C, H, W = noisy_latent.shape
        P_k = prefix_latent.shape[1]
        device = self.device

        z_t = noisy_latent

        # Get temporal KV-cache
        temporal_kv_caches = []
        for q in self.temporal_kv_queues:
            k, v = q.get_full_cache()
            if k is not None:
                temporal_kv_caches.append((k, v))
            else:
                temporal_kv_caches.append(None)

        # DDPM timesteps
        timesteps = self._get_ddpm_timesteps()

        for i, t in enumerate(timesteps):
            # Build input: [clean_prefix, noisy_current]
            z_input = torch.cat([prefix_latent, z_t], dim=1)  # (B, P_k+l, C, H, W)
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Classifier-free guidance
            if encoder_hidden_states is not None and self.guidance_scale > 1.0:
                # Conditional
                output_cond = self.model(
                    z_input, t_batch, P_k, encoder_hidden_states, tpe_offset,
                    temporal_kv_caches=temporal_kv_caches,
                    spatial_kv_cache=self.spatial_kv_cache,
                    return_cache=False,
                )
                # Unconditional
                output_uncond = self.model(
                    z_input, t_batch, P_k, None, tpe_offset,
                    temporal_kv_caches=temporal_kv_caches,
                    spatial_kv_cache=self.spatial_kv_cache,
                    return_cache=False,
                )
                pred = output_uncond["pred"] + self.guidance_scale * (
                    output_cond["pred"] - output_uncond["pred"]
                )
            else:
                output = self.model(
                    z_input, t_batch, P_k, encoder_hidden_states, tpe_offset,
                    temporal_kv_caches=temporal_kv_caches,
                    spatial_kv_cache=self.spatial_kv_cache,
                    return_cache=False,
                )
                pred = output["pred"]  # (B, P_k+l, C, H, W)

            # Extract prediction for target frames only
            pred_target = pred[:, P_k:, :, :, :]  # (B, l, C, H, W)

            # DDPM step
            z_t = self._ddpm_step(z_t, pred_target, t, i < len(timesteps) - 1)

        return z_t

    def _ddpm_step(
        self,
        z_t: torch.Tensor,
        pred_noise: torch.Tensor,
        t: int,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """Single DDPM reverse step.
        
        Using improved DDPM (Nichol & Dhariwal 2021) with 100 steps.
        """
        device = z_t.device
        beta_start = 1e-4
        beta_end = 0.02
        T = 1000

        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        alpha_bar_t = alpha_bars[t]
        alpha_bar_t_prev = alpha_bars[t - 1] if t > 0 else torch.tensor(1.0, device=device)

        beta_t = betas[t]
        alpha_t = alphas[t]

        # Predict x_0
        pred_x0 = (z_t - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt().clamp(min=1e-8)

        # Compute mean
        pred_mean = (
            beta_t * alpha_bar_t_prev.sqrt() / (1 - alpha_bar_t) * pred_x0
            + (1 - alpha_bar_t_prev) * alpha_t.sqrt() / (1 - alpha_bar_t) * z_t
        )

        if add_noise:
            noise = torch.randn_like(z_t)
            # Use the posterior variance
            if t > 0:
                log_var = torch.log(beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)).clamp(max=20.0)
            else:
                log_var = torch.zeros_like(alpha_bar_t)
            z_prev = pred_mean + (0.5 * log_var).exp() * noise
        else:
            z_prev = pred_mean

        return z_prev

    def _get_ddpm_timesteps(self) -> List[int]:
        """Get timesteps for DDPM sampling.
        
        Uses improved DDPM with 100 steps, linearly spaced from T-1 down to 0.
        """
        T = 1000
        steps = self.num_inference_steps
        # Linearly spaced timesteps
        timesteps = torch.linspace(T - 1, 0, steps, dtype=torch.long)
        return timesteps.tolist()

    @torch.no_grad()
    def _cache_writing_stage(
        self,
        denoised_latent: torch.Tensor,
        prefix_latent: torch.Tensor,
        tpe_offset: int,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ):
        """Cache writing stage: compute clean K/V for the newly generated chunk.
        
        Runs a model forward with all frames at t=0 to compute and store
        temporal and spatial KV-caches for future AR steps.
        
        Args:
            denoised_latent: (B, l, C, H, W) newly generated clean frames
            prefix_latent: (B, P_k, C, H, W) all previously generated frames
            tpe_offset: TPE offset
        """
        B, l, C, H, W = denoised_latent.shape
        P_k = prefix_latent.shape[1]
        device = self.device

        # Build full clean input: all generated frames so far + new chunk
        z_full = torch.cat([prefix_latent, denoised_latent], dim=1)  # (B, P_k+l, C, H, W)
        L_total = P_k + l

        # All frames are at t=0 for cache writing
        t = torch.zeros(B, device=device, dtype=torch.long)

        # Get existing KV-caches
        temporal_kv_caches = []
        for q in self.temporal_kv_queues:
            k, v = q.get_full_cache()
            temporal_kv_caches.append((k, v) if k is not None else None)

        # Run model forward to compute new K/V
        output = self.model(
            z_full, t, P_k, encoder_hidden_states, tpe_offset,
            temporal_kv_caches=temporal_kv_caches,
            spatial_kv_cache=self.spatial_kv_cache,
            return_cache=True,
        )

        temporal_kv_new = output["temporal_kv_caches"]  # list of (K, V) per layer

        # Update temporal KV queues with new chunk's K/V
        for i, (k_new, v_new) in enumerate(temporal_kv_new):
            # Extract only the new chunk's K/V (last l frames)
            k_chunk = k_new[:, -l:, :, :]  # (B, l, HW, C)
            v_chunk = v_new[:, -l:, :, :]
            self.temporal_kv_queues[i].enqueue(k_chunk, v_chunk)

        # Update spatial KV-cache (only keep most recent chunk)
        # Spatial K/V are computed per frame with prefix enhancement
        # We cache the spatial K/V for the most recent P' frames
        # For simplicity, cache the full new chunk's spatial features
        self.spatial_kv_cache = self._compute_spatial_cache(denoised_latent, prefix_latent)

    @torch.no_grad()
    def _compute_spatial_cache(
        self,
        denoised_latent: torch.Tensor,
        prefix_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial KV-cache from the most recent chunk.
        
        Returns spatial K/V for the last P' frames for prefix-enhanced spatial attention.
        P' is model.prefix_len_enhance (default 3).
        """
        # Get the last P' frames of the generated sequence as spatial cache
        P_prime = self.model.prefix_len_enhance
        B, l, C, H, W = denoised_latent.shape

        # The spatial cache should be from frames P_k - P' to P_k (or last P' of prefix + new)
        # For simplicity, take the last P' frames from the total generated
        all_frames = torch.cat([prefix_latent, denoised_latent], dim=1)
        return all_frames[:, -P_prime:, :, :, :], all_frames[:, -P_prime:, :, :, :]

    @torch.no_grad()
    def generate(
        self,
        first_frame: torch.Tensor,
        num_ar_steps: int,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Autoregressive video generation.
        
        Args:
            first_frame: (B, 1, C, H, W) first frame latent
            num_ar_steps: number of autoregression steps
            encoder_hidden_states: text embeddings
            verbose: show progress bar
        
        Returns:
            video: (B, 1 + num_ar_steps * l, C, H, W) generated video
        """
        self.reset()

        B, _, C, H, W = first_frame.shape
        device = self.device
        first_frame = first_frame.to(device)

        # Initialize with first frame
        self.generated_latents.append(first_frame)
        self.P_k = 1

        # Compute initial KV-cache for the first frame
        # Cache writing for the single first frame
        t = torch.zeros(B, device=device, dtype=torch.long)
        z_input = first_frame  # (B, 1, C, H, W)
        output = self.model(
            z_input, t, 1, encoder_hidden_states, 0,
            temporal_kv_caches=[None] * self.model.num_layers,
            spatial_kv_cache=None,
            return_cache=True,
        )
        temporal_kv_new = output["temporal_kv_caches"]
        for i, (k, v) in enumerate(temporal_kv_new):
            self.temporal_kv_queues[i].enqueue(k, v)

        iterator = tqdm(range(num_ar_steps), desc="AR generation") if verbose else range(num_ar_steps)

        for step in iterator:
            # Sample noise for current chunk
            noisy_latent = torch.randn(B, self.l, C, H, W, device=device)

            # Build prefix (all previously generated frames)
            if len(self.generated_latents) > 1:
                prefix_latent = torch.cat(self.generated_latents, dim=1)  # (B, P_k, C, H, W)
            else:
                prefix_latent = self.generated_latents[0]

            # Compute TPE offset for cyclic assignment
            tpe_offset = self.P_k % self.model.max_train_len

            # Denoising stage
            denoised_latent = self._ddpm_sampling_loop(
                noisy_latent, prefix_latent, tpe_offset, encoder_hidden_states
            )

            # Cache writing stage
            self._cache_writing_stage(
                denoised_latent, prefix_latent, tpe_offset, encoder_hidden_states
            )

            # Update state
            self.generated_latents.append(denoised_latent)
            self.P_k += self.l

        # Concatenate all generated frames
        video = torch.cat(self.generated_latents, dim=1)  # (B, total_frames, C, H, W)
        return video


class BidirectionalInference:
    """Autoregressive inference for bidirectional baselines (OS-Fix and OS-Ext).
    
    OS-Fix: uses fixed-length conditional frames (P is fixed).
    OS-Ext: uses extendable conditional frames (P grows with AR step until P_max).
    No KV-cache — recomputes all conditional frames at each step.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        model_type: str = "os_ext",
        device: torch.device = None,
    ):
        self.model = model
        self.config = config
        self.model_type = model_type  # "os_fix" or "os_ext"
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        self.l = config.inference.chunk_length
        self.P_max = config.inference.max_prefix_length
        self.num_inference_steps = config.inference.num_inference_steps
        self.guidance_scale = config.inference.guidance_scale

        if model_type == "os_fix":
            # Fixed P: P = L_train / 2
            self.fixed_P = config.training.max_train_len // 2

    @torch.no_grad()
    def generate(
        self,
        first_frame: torch.Tensor,
        num_ar_steps: int,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        B, _, C, H, W = first_frame.shape
        device = self.device
        first_frame = first_frame.to(device)

        generated = [first_frame]
        P_k = 1

        iterator = tqdm(range(num_ar_steps), desc="AR (bidirectional)") if verbose else range(num_ar_steps)

        for step in iterator:
            # Determine prefix length
            if self.model_type == "os_fix":
                P = self.fixed_P
                if P > P_k:
                    # Pad with repeated first frame or zeros
                    pad_len = P - P_k
                    prefix = torch.cat(generated, dim=1)
                    if pad_len > 0:
                        pad = generated[-1][:, -1:].repeat(1, pad_len, 1, 1, 1)
                        prefix = torch.cat([pad, prefix], dim=1)
                    prefix = prefix[:, -P:]  # take last P frames
                else:
                    prefix = torch.cat(generated[-P:], dim=1)
            else:
                # OS-Ext: extendable condition
                P = min(P_k, self.P_max)
                prefix = torch.cat(generated[-P:], dim=1) if P > 0 else generated[0]

            # Sample noise
            noisy_latent = torch.randn(B, self.l, C, H, W, device=device)

            # Denoising loop
            z_t = noisy_latent
            timesteps = torch.linspace(999, 0, self.num_inference_steps, dtype=torch.long, device=device)

            for t in timesteps:
                z_input = torch.cat([prefix, z_t], dim=1)
                t_batch = torch.full((B,), t, device=device, dtype=torch.long)

                if encoder_hidden_states is not None and self.guidance_scale > 1.0:
                    output_cond = self.model(z_input, t_batch, P, encoder_hidden_states)
                    output_uncond = self.model(z_input, t_batch, P, None)
                    pred = output_uncond["pred"] + self.guidance_scale * (
                        output_cond["pred"] - output_uncond["pred"]
                    )
                else:
                    output = self.model(z_input, t_batch, P, encoder_hidden_states)
                    pred = output["pred"]

                pred_target = pred[:, P:, :, :, :]

                # DDPM step
                alpha_bar_t = self._get_alpha_bar(t, device)
                alpha_bar_t_prev = self._get_alpha_bar(t - 1, device) if t > 0 else torch.ones(1, device=device)

                pred_x0 = (z_t - (1 - alpha_bar_t).sqrt() * pred_target) / alpha_bar_t.sqrt().clamp(min=1e-8)
                beta_t = 1.0 - alpha_bar_t / alpha_bar_t_prev if t > 0 else torch.zeros(1, device=device)
                alpha_t = 1.0 - beta_t

                pred_mean = (
                    beta_t * alpha_bar_t_prev.sqrt() / (1 - alpha_bar_t) * pred_x0
                    + (1 - alpha_bar_t_prev) * alpha_t.sqrt() / (1 - alpha_bar_t) * z_t
                )

                if t > 0:
                    log_var = torch.log(beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)).clamp(max=20.0)
                    z_t = pred_mean + (0.5 * log_var).exp() * torch.randn_like(z_t)
                else:
                    z_t = pred_mean

            generated.append(z_t)
            P_k += self.l

        return torch.cat(generated, dim=1)

    @staticmethod
    def _get_alpha_bar(t: int, device: torch.device) -> torch.Tensor:
        T = 1000
        betas = torch.linspace(1e-4, 0.02, T, device=device)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        if t < 0:
            return torch.ones(1, device=device)
        if t >= T:
            t = T - 1
        return alpha_bars[t]


def decode_latents(vae_decoder, latents: torch.Tensor) -> torch.Tensor:
    """Decode VAE latents to pixel space.
    
    Args:
        vae_decoder: VAE decoder (e.g., from StableDiffusion)
        latents: (B, T, C, H, W) latent tensor
    
    Returns:
        video: (B, T, 3, H*8, W*8) pixel tensor normalized to [0, 1]
    """
    B, T, C, H, W = latents.shape
    latents = latents.view(B * T, C, H, W)
    latents = latents / 0.18215  # VAE scaling factor
    with torch.no_grad():
        video = vae_decoder(latents)
    video = video.view(B, T, *video.shape[1:])
    video = (video + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    video = video.clamp(0, 1)
    return video


def generate_video(
    model: Ca2VDM,
    config: Config,
    first_frame_latent: torch.Tensor,
    num_ar_steps: int,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    model_type: str = "ca2_vdm",
) -> torch.Tensor:
    """Convenience function for video generation.
    
    Args:
        model: Ca2VDM or bidirectional baseline
        config: configuration
        first_frame_latent: (B, 1, C, H, W)
        num_ar_steps: number of AR steps
        encoder_hidden_states: text embeddings
        model_type: "ca2_vdm", "os_fix", or "os_ext"
    
    Returns:
        video_latent: (B, total_frames, C, H, W)
    """
    if model_type == "ca2_vdm":
        inference = Ca2VDMInference(model, config)
    else:
        inference = BidirectionalInference(model, config, model_type)

    video = inference.generate(
        first_frame_latent,
        num_ar_steps,
        encoder_hidden_states,
        verbose=True,
    )
    return video
