## model/ca2_vdm.py
"""
Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing.

Top‑level model class that wraps:
- a frozen VAE for latent encoding/decoding,
- a frozen T5‑XXL text encoder,
- a DDPM noise scheduler,
- the spatial‑temporal Transformer with causal attention and KV‑cache.

The forward method supports both training (full clean+noisy sequence) and
autoregressive inference with optional caches.
The training_step implements the simplified + VLB loss with distinct
timestep embeddings for clean and noisy frames.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, AutoencoderKL
from transformers import AutoTokenizer, T5EncoderModel

from config import Config
from model.transformer import SpatialTemporalTransformer


# ---------------------------------------------------------------------------
# Helper: fold/unfold latent pixels into/from patch tokens
# ---------------------------------------------------------------------------

def unpatchify(x: torch.Tensor, patch_size: int, out_channels: int) -> torch.Tensor:
    """
    Convert token representations back to a spatial image.

    Args:
        x: Token tensor of shape ``(B, L, N, D_tok)`` where ``D_tok = out_channels * patch_size**2``.
        patch_size: Spatial extent of each patch (e.g. 2).
        out_channels: Number of output channels (e.g. 8 for noise + variance).

    Returns:
        Image tensor of shape ``(B, L, out_channels, H, W)``,
        where ``H = W = sqrt(N) * patch_size`` (e.g. 32 for N=256, patch_size=2).
    """
    B, L, N, D_tok = x.shape
    # N must be a perfect square representing H_p * W_p
    H_p = W_p = int(math.sqrt(N))
    if H_p * W_p != N:
        raise ValueError(f"Token number {N} is not a perfect square, cannot unpatchify.")
    p = patch_size
    # Reshape to (B, L, H_p, W_p, out_channels, p, p)
    x = x.view(B, L, H_p, W_p, out_channels, p, p)
    # Permute to interleave patches -> (B, L, out_channels, H_p, p, W_p, p)
    x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    # Merge spatial dimensions -> (B, L, out_channels, H_p * p, W_p * p)
    x = x.view(B, L, out_channels, H_p * p, W_p * p)
    return x


def patchify(latents: torch.Tensor, patch_size: int, embed_dim: int) -> torch.Tensor:
    """
    Convert latent frames into patch tokens using a 2D conv.

    Args:
        latents: Input tensor of shape ``(B, L, C, H, W)``.
        patch_size: Kernel size (and stride) of the conv.
        embed_dim: Output channel dimension of the convolution.

    Returns:
        Token tensor of shape ``(B, L, N, embed_dim)``,
        where ``N = (H/patch_size) * (W/patch_size)``.
    """
    B, L, C, H, W = latents.shape
    p = patch_size
    # Spatial dimensions must be divisible by patch_size
    if H % p != 0 or W % p != 0:
        raise ValueError(f"Latent spatial size {H}x{W} not divisible by patch_size {p}.")
    # Flatten batch and time dims
    x = latents.view(B * L, C, H, W)
    # Convolution: (B*L, embed_dim, H_p, W_p)
    # We will call this inside the main class; here we only provide a utility.
    # The actual patch embedding layer is part of Ca2VDM.
    raise NotImplementedError("Use Ca2VDM.patch_embed directly.")


# ---------------------------------------------------------------------------
# Ca2VDM
# ---------------------------------------------------------------------------

class Ca2VDM(nn.Module):
    """
    Complete Ca2-VDM model.

    Parameters
    ----------
    config : Config
        Global configuration object; must contain all necessary hyperparameters.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config = config
        self.latent_channels: int = 4  # SD latent channels
        self.patch_size: int = 2       # patch size for latent tokenisation
        self.num_patches: int = (config.data.latent_size // self.patch_size) ** 2  # e.g. 256
        self.hidden_dim: int = config.model.transformer.hidden_dim
        self.out_channels: int = 2 * self.latent_channels   # noise prediction + log variance

        # ------------------------------------------------------------------
        # 1. Frozen VAE
        # ------------------------------------------------------------------
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(config.model.vae_model)
        self.vae.requires_grad_(False).eval()
        # Move to device later (in main / Trainer). We keep here for convenience.
        # The VAE is run in float32; we may optionally upcast inside encode/decode.
        self.vae_dtype: torch.dtype = torch.float32

        # ------------------------------------------------------------------
        # 2. Frozen T5 text encoder
        # ------------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.text_encoder)
        self.text_encoder = T5EncoderModel.from_pretrained(config.model.text_encoder)
        self.text_encoder.requires_grad_(False).eval()
        self.text_hidden_size: int = self.text_encoder.config.d_model   # 4096 for T5-XXL
        # Project text hidden size to transformer hidden dimension
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_hidden_size, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # ------------------------------------------------------------------
        # 3. Patch embedding (latent -> tokens)
        # ------------------------------------------------------------------
        self.patch_embed = nn.Conv2d(
            in_channels=self.latent_channels,
            out_channels=self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # ------------------------------------------------------------------
        # 4. Spatial-Temporal Transformer
        # ------------------------------------------------------------------
        self.transformer = SpatialTemporalTransformer(config)

        # ------------------------------------------------------------------
        # 5. Output head (tokens -> latent prediction)
        # ------------------------------------------------------------------
        self.head = nn.Linear(self.hidden_dim, self.out_channels * self.patch_size**2)

        # ------------------------------------------------------------------
        # 6. Noise scheduler (DDPM)
        # ------------------------------------------------------------------
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.diffusion.num_timesteps,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            beta_schedule=config.diffusion.schedule,
            clip_sample=False,
        )
        # Cache schedule constants for VLB loss computation
        self.alphas_cumprod: torch.Tensor = self.noise_scheduler.alphas_cumprod.float()
        self.betas: torch.Tensor = self.noise_scheduler.betas.float()
        self.num_timesteps: int = config.diffusion.num_timesteps

        # Loss weighting for VLB (default 0.001 as in the paper)
        self.loss_vlb_weight: float = getattr(
            config.training.stage2, "loss_vlb_weight", 0.001
        ) if config.task == "t2v" else getattr(
            config.training.video_prediction, "loss_vlb_weight", 0.0
        )

        # Guidance scale (only for T2V)
        self.guidance_scale: float = config.inference.guidance_scale

        # Initialise weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialise weights following DiT convention: normal for linear,
        zero for biases, xavier for conv, normal for the last layer of
        text_proj with small std.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        # Last layer of text_proj gets smaller std for stability
        for name, param in self.text_proj.named_parameters():
            if "2.weight" in name:  # second linear layer weight
                nn.init.normal_(param, std=0.02 / math.sqrt(2.0 * self.hidden_dim))

    # ------------------------------------------------------------------
    # VAE helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_latents(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode pixel frames into latent codes.

        Args:
            frames: Tensor ``(B, L, 3, H_img, W_img)``, values in [-1, 1].

        Returns:
            Latent tensor ``(B, L, C_lat, H, W)``, scaled by the VAE factor.
            The tensor is float32 and on the same device as the VAE.
        """
        B, L, C_img, H_img, W_img = frames.shape
        # Flatten to (B*L, C, H, W)
        frames = frames.view(B * L, C_img, H_img, W_img).to(
            device=self.vae.device, dtype=self.vae_dtype
        )
        latent_dist = self.vae.encode(frames).latent_dist
        latents = latent_dist.mode()  # already scaled by VAE scaling factor (e.g. 0.18215)
        C_lat = latents.shape[1]
        H_lat, W_lat = latents.shape[2], latents.shape[3]
        latents = latents.view(B, L, C_lat, H_lat, W_lat)
        return latents

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes back to pixel frames.

        Args:
            latents: Tensor ``(B, L, C_lat, H_lat, W_lat)``, scaled by the VAE factor.

        Returns:
            Pixel frames ``(B, L, 3, H_img, W_img)`` in [0, 255] uint8.
        """
        B, L, C_lat, H_lat, W_lat = latents.shape
        latents = latents.view(B * L, C_lat, H_lat, W_lat).to(
            device=self.vae.device, dtype=self.vae_dtype
        )
        # VAE decode returns sample in [-1, 1]
        samples = self.vae.decode(latents).sample
        # Convert to [0, 255] uint8
        samples = (samples + 1.0) * 127.5
        samples = samples.clamp(0.0, 255.0).to(torch.uint8)
        C_img, H_img, W_img = samples.shape[1], samples.shape[2], samples.shape[3]
        samples = samples.view(B, L, C_img, H_img, W_img)
        return samples

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode a batch of text strings into embeddings.

        Args:
            text: List of length B with text prompts.

        Returns:
            Text embeddings ``(B, L_text_max, hidden_dim)``, padded/truncated
            to a fixed length (e.g. 120 tokens).
        """
        device = next(self.text_encoder.parameters()).device
        # Tokenise with padding and truncation
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=120,          # following PixArt‑α
        ).to(device)
        # Run T5 encoder (input_ids, attention_mask)
        outputs = self.text_encoder(**tokens, return_dict=True)
        last_hidden = outputs.last_hidden_state  # (B, seq_len, text_hidden_size)
        # Project to transformer hidden dimension
        text_emb = self.text_proj(last_hidden)    # (B, seq_len, hidden_dim)
        return text_emb

    # ------------------------------------------------------------------
    # Forward pass (training & inference)
    # ------------------------------------------------------------------
    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        temporal_cache: Optional[Dict[str, Any]] = None,
        spatial_cache: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Args:
            latents: Input latent frames ``(B, L, C_lat, H, W)``.
                     May contain clean prefix and noisy target in training,
                     or only noisy frames during inference denoising.
            timestep: Tensor of shape ``(B, L)`` with integer diffusion timesteps
                      (0 for clean frames, t for noisy).
            text_emb: Optional text embedding ``(B, L_text, D)``.
            temporal_cache: Optional dict with 'k' and 'v' per layer.
            spatial_cache: Optional dict with 'k' and 'v' per layer.

        Returns:
            Output tensor ``(B, L, 2*C_lat, H, W)`` where the first half
            contains noise prediction and the second half log‑variance
            (only the noise part is used in inference).
        """
        B, L, C, H, W = latents.shape
        device = latents.device

        # 1) Patchify latents -> tokens
        x = latents.reshape(B * L, C, H, W)
        x = self.patch_embed(x)                # (B*L, hidden_dim, H_p, W_p)
        H_p, W_p = x.shape[2], x.shape[3]
        N_tokens = H_p * W_p                   # e.g. 256
        x = x.flatten(2).transpose(1, 2)       # (B*L, N_tokens, hidden_dim)
        x = x.view(B, L, N_tokens, self.hidden_dim)

        # 2) Transformer
        x = self.transformer(
            x,
            timestep=timestep,
            text_emb=text_emb,
            temporal_cache=temporal_cache,
            spatial_cache=spatial_cache,
            write_cache=False,  # caching is handled by InferenceEngine
        )

        # 3) Output head
        x = self.head(x)                        # (B, L, N_tokens, out_channels * patch**2)
        # Unpatchify to image space
        out = unpatchify(
            x,
            patch_size=self.patch_size,
            out_channels=self.out_channels,     # 2*C_lat
        )  # (B, L, 2*C, H, W)

        return out

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute the diffusion loss for a single batch.

        Expected batch keys:
            ``"latents"``   : clean video latent tensor of shape ``(B, L, C, H, W)``.
            ``"text_emb"``  : text embedding ``(B, L_text, D)`` (may be None).
            ``"prefix_len"``: 1‑D tensor of shape ``(B,)`` with the number of
                              clean prefix frames for each sample.

        Returns:
            dict with ``"loss"``, and optionally ``"loss_simple"``, ``"loss_vlb"``
            for logging.
        """
        B, L, C, H, W = batch["latents"].shape
        device = batch["latents"].device

        # Ensure prefix_len is a 1‑D tensor
        prefix_len = batch["prefix_len"]
        if isinstance(prefix_len, int):
            prefix_len = torch.full((B,), prefix_len, device=device, dtype=torch.long)
        assert prefix_len.shape == (B,), "prefix_len must have shape (B,)"

        text_emb = batch.get("text_emb", None)

        # ---------------------------------------------------------------
        # 1. Sample diffusion timesteps
        # ---------------------------------------------------------------
        t = torch.randint(
            1, self.num_timesteps + 1, (B,), device=device, dtype=torch.long
        )

        # ---------------------------------------------------------------
        # 2. Build target mask and timestep vector
        # ---------------------------------------------------------------
        arange = torch.arange(L, device=device).unsqueeze(0)  # (1, L)
        target_mask = arange >= prefix_len.unsqueeze(1)       # (B, L)  – True for noisy frames
        timestep_vec = t.unsqueeze(1).expand(-1, L) * target_mask.long()   # (B, L)

        # ---------------------------------------------------------------
        # 3. Corrupt target frames with noise
        # ---------------------------------------------------------------
        latents_clean = batch["latents"]

        # Generate full noise; we will later zero‑out prefix frames for loss
        epsilon = torch.randn_like(latents_clean)

        # Compute noise schedule parameters per sample
        alpha_prod_t = self.alphas_cumprod[t]                    # (B,)
        sqrt_alpha_prod = alpha_prod_t.sqrt()                     # (B,)
        sqrt_one_minus_alpha_prod = (1 - alpha_prod_t).sqrt()     # (B,)

        # Expand for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod.view(B, 1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(B, 1, 1, 1, 1)

        # Noisy latents: used only for target frames
        noisy_latents = sqrt_alpha_prod * latents_clean + sqrt_one_minus_alpha_prod * epsilon

        # Combine clean prefix and noisy target
        # Create a broadcastable mask of shape (B, L, 1, 1, 1)
        target_mask_5d = target_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        z_input = torch.where(target_mask_5d, noisy_latents, latents_clean)

        # ---------------------------------------------------------------
        # 4. Forward pass
        # ---------------------------------------------------------------
        output = self.forward(
            z_input,             # (B, L, C, H, W)
            timestep_vec,        # (B, L)
            text_emb=text_emb,
            temporal_cache=None,
            spatial_cache=None,
        )  # (B, L, 2*C, H, W)

        # Split noise prediction and log variance
        noise_pred = output[:, :, :C, ...]          # (B, L, C, H, W)
        log_var = output[:, :, C:, ...]             # (B, L, C, H, W)

        # ---------------------------------------------------------------
        # 5. Compute simplified loss (masked, only on target frames)
        # ---------------------------------------------------------------
        # epsilon has noise for all frames; we zero out prefix part for correct loss
        epsilon_loss = epsilon.clone()
        epsilon_loss[~target_mask_5d.expand(-1, -1, C, H, W)] = 0.0

        loss_simple = F.mse_loss(noise_pred, epsilon_loss, reduction="none")
        # Mean over spatial dims, apply mask, then average over batch/target
        loss_simple = loss_simple.mean(dim=(-3, -2, -1))  # (B, L)
        loss_simple = (loss_simple * target_mask).sum() / target_mask.sum().clamp(min=1)

        # ---------------------------------------------------------------
        # 6. Compute VLB loss (if weight > 0)
        # ---------------------------------------------------------------
        if self.loss_vlb_weight > 0.0:
            # Extract target frames only (with same ordering as in target_mask)
            # Because batches can have varying prefix lengths, we compute VLB
            # per sample using a loop, which is acceptable because VLB is not
            # the primary loss.
            loss_vlb = torch.tensor(0.0, device=device)
            for b in range(B):
                Pb = prefix_len[b].item()
                if Pb >= L:
                    continue  # no target frames
                # Target latents
                z_0_target = latents_clean[b, Pb:]  # (l, C, H, W)
                z_t_target = z_input[b, Pb:]        # (l, C, H, W)
                noise_pred_target = noise_pred[b, Pb:]
                log_var_target = log_var[b, Pb:]
                epsilon_target = epsilon[b, Pb:]

                l_target = L - Pb
                # Timestep for this sample
                t_b = t[b].item()

                # Schedule coefficients
                alpha_prod = self.alphas_cumprod[t_b]
                alpha_prod_prev = self.alphas_cumprod[t_b - 1] if t_b > 1 else self.alphas_cumprod[0]
                beta = self.betas[t_b - 1]  # 0-indexed
                sqrt_alpha_prod_b = math.sqrt(alpha_prod)
                sqrt_alpha_prod_prev_b = math.sqrt(alpha_prod_prev)
                sqrt_one_minus_alpha_prod_b = math.sqrt(1 - alpha_prod)

                # True posterior mean & variance
                coef1 = (sqrt_alpha_prod_prev_b * beta) / (1 - alpha_prod)
                coef2 = (sqrt_alpha_prod_b * (1 - alpha_prod_prev)) / (1 - alpha_prod)
                mu_true = coef1 * z_0_target + coef2 * z_t_target
                var_true = (1 - alpha_prod_prev) / (1 - alpha_prod) * beta

                # Predicted mean (using noise prediction)
                mu_pred = (1 / sqrt_alpha_prod_b) * (
                    z_t_target - (beta / sqrt_one_minus_alpha_prod_b) * noise_pred_target
                )
                var_pred = torch.exp(log_var_target)

                # KL divergence: true || predicted
                kl = 0.5 * (
                    (mu_pred - mu_true) ** 2 / var_true
                    + var_pred / var_true
                    - 1.0
                    - (log_var_target + math.log(var_true + 1e-8))  # log(var_pred / var_true)
                )
                # mean over all spatial and channel dims
                kl_per_frame = kl.mean(dim=(-3, -2, -1))  # (l_target,)
                kl_per_frame = kl_per_frame.sum()          # sum over frames (they all have equal weight)
                loss_vlb = loss_vlb + kl_per_frame

            # Normalise over total target frames in batch
            total_target_frames = target_mask.sum()
            loss_vlb = loss_vlb / total_target_frames.clamp(min=1)
        else:
            loss_vlb = torch.tensor(0.0, device=device)

        # ---------------------------------------------------------------
        # 7. Total loss
        # ---------------------------------------------------------------
        loss = loss_simple + self.loss_vlb_weight * loss_vlb

        return {
            "loss": loss,
            "loss_simple": loss_simple.detach(),
            "loss_vlb": loss_vlb.detach() if isinstance(loss_vlb, torch.Tensor) else loss_vlb,
        }

