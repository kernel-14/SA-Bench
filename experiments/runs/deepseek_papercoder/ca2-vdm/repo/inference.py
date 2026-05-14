## inference.py
"""
Ca2‑VDM Autoregressive Inference Engine.

Implements :class:`InferenceEngine` which performs autoregressive video
generation with KV‑cache sharing, following the paper's algorithm.

The engine uses:
- A pre‑trained :class:`Ca2VDM` model,
- An improved DDPM scheduler (100 steps) with learned variance,
- A :class:`CacheManager` to store and retrieve temporal/spatial KV‑caches,
- Cyclic temporal positional embeddings consistent with training.

All hyperparameters are read from the global :class:`Config` object.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from config import Config
from model.ca2_vdm import Ca2VDM
from model.cache import CacheManager
from utils.sampling import DDPMScheduler

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Orchestrates autoregressive video generation with cache sharing.

    Parameters
    ----------
    model : Ca2VDM
        A fully trained Ca2‑VDM model (VAE, text encoder, transformer).
    config : Config
        Global configuration object.

    Attributes
    ----------
    model : Ca2VDM
    config : Config
    cache_manager : CacheManager
    scheduler : DDPMScheduler
    chunk_size : int
    max_prefix : int
    train_max_len : int
    p_prime : int
    denoising_steps : int
    guidance_scale : float
    null_text_emb : Optional[torch.Tensor]
        Pre‑computed embedding of the empty string for classifier‑free guidance.
    device : torch.device
    """

    def __init__(self, model: Ca2VDM, config: Config) -> None:
        if not model.training:
            model.eval()
        else:
            logger.warning("Model is in training mode; switching to eval mode for inference.")
            model.eval()

        self.model = model
        self.config = config

        # ------------------------------------------------------------------
        # Device & dtype
        # ------------------------------------------------------------------
        self.device = torch.device(config.system.device)
        self.model = self.model.to(self.device)

        # ------------------------------------------------------------------
        # Scheduler (improved DDPM with learned variance)
        # ------------------------------------------------------------------
        self.scheduler = DDPMScheduler(
            num_train_timesteps=config.diffusion.num_timesteps,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            schedule=config.diffusion.schedule,
        )
        self.denoising_steps: int = config.inference.denoising_steps
        self.scheduler.set_timesteps(self.denoising_steps)
        self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)

        # ------------------------------------------------------------------
        # Video generation parameters
        # ------------------------------------------------------------------
        self.chunk_size: int = config.video.chunk_size           # l
        self.max_prefix: int = config.video.max_prefix           # P_max
        self.train_max_len: int = config.video.train_max_len     # L_train
        self.p_prime: int = config.video.p_prime                 # p'
        self.guidance_scale: float = (
            config.inference.guidance_scale if config.task == "t2v" else 1.0
        )

        # ------------------------------------------------------------------
        # Null text embedding for classifier‑free guidance (T2V only)
        # ------------------------------------------------------------------
        self.null_text_emb: Optional[torch.Tensor] = None
        if config.task == "t2v":
            self.null_text_emb = self.model.encode_text([""]).detach()   # (1, seq_len, D)

        # ------------------------------------------------------------------
        # Cache manager
        # ------------------------------------------------------------------
        num_layers = len(self.model.transformer.blocks)
        self.cache_manager = CacheManager(
            max_temporal_length=self.max_prefix,
            p_prime=self.p_prime,
            chunk_size=self.chunk_size,
            num_temporal_layers=num_layers,
            num_spatial_layers=num_layers,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def autoregressive_generate(
        self,
        first_frame: torch.Tensor,
        text_prompt: Optional[str] = None,
        num_chunks: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a full video autoregressively starting from a given first frame.

        Parameters
        ----------
        first_frame : torch.Tensor
            Pixel frame of shape ``(3, H, W)`` in [‑1, 1], or a pre‑encoded
            latent of shape ``(C_lat, H_lat, W_lat)``.
        text_prompt : Optional[str]
            Text condition for T2V generation.  Ignored for video prediction.
        num_chunks : Optional[int]
            Number of autoregressive chunks to generate.  If ``None``, it is
            derived from ``config.inference.generate_frames // chunk_size``.

        Returns
        -------
        torch.Tensor
            Generated pixel video of shape ``(T, 3, H, W)``, dtype ``uint8``.
        """
        # ------------------------------------------------------------------
        # Determine output length
        # ------------------------------------------------------------------
        if num_chunks is None:
            total_frames = self.config.inference.generate_frames
            num_chunks = total_frames // self.chunk_size
            logger.info(
                "Using %d chunks to reach approximately %d frames "
                "(actual: %d after prepending first frame).",
                num_chunks,
                total_frames,
                1 + num_chunks * self.chunk_size,
            )
        assert num_chunks >= 1, "num_chunks must be at least 1"

        # ------------------------------------------------------------------
        # Encode first frame if it is in pixel space
        # ------------------------------------------------------------------
        first_frame = first_frame.to(self.device)
        if first_frame.dim() == 3 and first_frame.shape[0] not in (4,):
            # pixel: (3, H, W)
            pixel_frames = first_frame.unsqueeze(0).unsqueeze(0)   # (1, 1, 3, H, W)
            first_latents = self.model.encode_latents(pixel_frames).squeeze(0)  # (1, C, H_l, W_l)
        elif first_frame.shape[0] == self.model.latent_channels:
            # already latent: (C, H_l, W_l)
            first_latents = first_frame.unsqueeze(0)   # (1, C, H_l, W_l)
        else:
            raise ValueError(
                f"Unexpected first_frame shape {first_frame.shape}. "
                "Expected (3, H, W) pixel or (C_lat, H_lat, W_lat) latent."
            )

        # ------------------------------------------------------------------
        # Text embedding (same for the whole generation)
        # ------------------------------------------------------------------
        text_emb: Optional[torch.Tensor] = None
        if self.config.task == "t2v" and text_prompt is not None:
            text_emb = self.model.encode_text([text_prompt]).detach()

        # ------------------------------------------------------------------
        # Initial cache writing for the first frame
        # ------------------------------------------------------------------
        global_idx = 0
        cache_out = self._run_model_cache_writing(
            latents=first_latents.unsqueeze(0),   # (1, 1, C, H, W)
            global_idx=global_idx,
            text_emb=text_emb,
            temporal_cache_dict=None,
            spatial_cache_dict=None,
        )
        self.cache_manager.add_to_temporal_cache(
            cache_out["temporal"]["k"], cache_out["temporal"]["v"]
        )
        self.cache_manager.update_spatial_cache(
            cache_out["spatial"]["k"], cache_out["spatial"]["v"]
        )
        global_idx += 1   # next chunk starts at index 1

        # Decode the first frame to pixels (to be prepended later)
        first_pixel = self.model.decode_latents(first_latents.unsqueeze(0))[0]  # (3, H, W)

        # ------------------------------------------------------------------
        # Autoregressive loop
        # ------------------------------------------------------------------
        generated_pixel_chunks: List[torch.Tensor] = []
        for step in range(num_chunks):
            logger.info("Autoregressive step %d / %d", step + 1, num_chunks)

            # ---- Retrieve current caches ----------------------------------
            tk_list, tv_list = self.cache_manager.get_temporal_cache()
            sk_list, sv_list = self.cache_manager.get_spatial_cache()

            temp_dict_denoise = {
                "k": tk_list,
                "v": tv_list,
                "tpe_start_idx": global_idx,
            }
            spat_dict_denoise = {"k": sk_list, "v": sv_list}

            # ---- Denoising stage ------------------------------------------
            noise_latents = torch.randn(
                self.chunk_size,
                self.model.latent_channels,
                first_latents.shape[2],
                first_latents.shape[3],
                device=self.device,
            )
            clean_latents = self._denoising_step(
                noise_latents, global_idx, text_emb,
                temp_dict_denoise, spat_dict_denoise,
            )   # (chunk_size, C, H, W)

            # ---- Cache writing stage --------------------------------------
            cache_out = self._run_model_cache_writing(
                latents=clean_latents.unsqueeze(0),
                global_idx=global_idx,
                text_emb=text_emb,
                temporal_cache_dict=temp_dict_denoise,
                spatial_cache_dict=spat_dict_denoise,
            )
            self.cache_manager.add_to_temporal_cache(
                cache_out["temporal"]["k"], cache_out["temporal"]["v"]
            )
            self.cache_manager.update_spatial_cache(
                cache_out["spatial"]["k"], cache_out["spatial"]["v"]
            )

            # ---- Decode and accumulate ------------------------------------
            pixel_chunk = self.model.decode_latents(clean_latents.unsqueeze(0))  # (L, 3, H, W)
            generated_pixel_chunks.append(pixel_chunk)

            global_idx += self.chunk_size

        # ------------------------------------------------------------------
        # Assemble final video
        # ------------------------------------------------------------------
        final_video = torch.cat(
            [first_pixel.unsqueeze(0)] + generated_pixel_chunks, dim=0
        )   # (T, 3, H, W)
        return final_video

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _denoising_step(
        self,
        noise_latents: torch.Tensor,
        global_idx: int,
        text_emb: Optional[torch.Tensor],
        temp_dict: Dict[str, Any],
        spat_dict: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Run the full denoising loop for a single chunk using the given caches.

        Returns the clean latent ``z_0`` for the chunk.
        """
        B = 1   # we process one chunk at a time
        L = self.chunk_size
        device = noise_latents.device

        latents = noise_latents.unsqueeze(0)   # (1, L, C, H, W)

        for t in self.scheduler.timesteps:
            t_int = int(t.item())
            t_tensor = torch.full((B, L), t_int, dtype=torch.long, device=device)

            if self.config.task == "t2v" and self.guidance_scale != 1.0:
                pred_cond = self.model.forward(
                    latents, t_tensor, text_emb,
                    temporal_cache=temp_dict, spatial_cache=spat_dict,
                )
                pred_uncond = self.model.forward(
                    latents, t_tensor, self.null_text_emb,
                    temporal_cache=temp_dict, spatial_cache=spat_dict,
                )
                pred = pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)
            else:
                pred = self.model.forward(
                    latents, t_tensor, text_emb,
                    temporal_cache=temp_dict, spatial_cache=spat_dict,
                )

            # Flatten for scheduler
            pred_flat = pred.reshape(B * L, 2 * self.model.latent_channels, *pred.shape[-2:])
            latents_flat = latents.reshape(B * L, self.model.latent_channels, *latents.shape[-2:])
            latents_flat = self.scheduler.step(pred_flat, t_int, latents_flat).prev_sample
            latents = latents_flat.reshape(B, L, *latents.shape[2:])

        return latents.squeeze(0)   # (L, C, H, W)

    def _run_model_cache_writing(
        self,
        latents: torch.Tensor,
        global_idx: int,
        text_emb: Optional[torch.Tensor],
        temporal_cache_dict: Optional[Dict[str, Any]],
        spatial_cache_dict: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run the transformer in *cache writing* mode for a batch of clean latent frames.

        Returns the dictionary ``cache_out`` containing ``'temporal'`` and
        ``'spatial'`` caches for all attention layers.
        """
        B, L, C, H, W = latents.shape
        if B != 1:
            raise NotImplementedError("Cache writing only supports batch size 1.")

        device = latents.device
        num_layers = len(self.model.transformer.blocks)

        # ------------------------------------------------------------------
        # 1. Patchify
        # ------------------------------------------------------------------
        x = latents.view(B * L, C, H, W)
        x = self.model.patch_embed(x)                     # (B*L, D, Hp, Wp)
        Hp, Wp = x.shape[2], x.shape[3]
        N = Hp * Wp
        x = x.flatten(2).transpose(1, 2)                 # (B*L, N, D)
        x = x.view(B, L, N, x.shape[-1])

        # ------------------------------------------------------------------
        # 2. Timestep embedding (all frames are clean → t=0)
        # ------------------------------------------------------------------
        timestep = torch.zeros(B, L, dtype=torch.long, device=device)

        # ------------------------------------------------------------------
        # 3. Build temporal cache dict with the correct tpe_start_idx
        # ------------------------------------------------------------------
        if temporal_cache_dict is not None:
            temp_cache = {
                "k": temporal_cache_dict["k"],
                "v": temporal_cache_dict["v"],
                "tpe_start_idx": global_idx,
            }
        else:
            # Use lists of None of appropriate length
            temp_cache = {
                "k": [None] * num_layers,
                "v": [None] * num_layers,
                "tpe_start_idx": global_idx,
            }

        spat_cache = None
        if spatial_cache_dict is not None:
            spat_cache = {
                "k": spatial_cache_dict["k"],
                "v": spatial_cache_dict["v"],
            }
        else:
            spat_cache = {"k": [None] * num_layers, "v": [None] * num_layers}

        # ------------------------------------------------------------------
        # 4. Transformer forward with cache writing enabled
        # ------------------------------------------------------------------
        _, cache_out = self.model.transformer(
            x, timestep,
            text_emb=text_emb,
            temporal_cache=temp_cache,
            spatial_cache=spat_cache,
            write_cache=True,
        )
        return cache_out

