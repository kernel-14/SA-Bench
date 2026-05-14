```python
## inference.py
"""
Inference and video generation module.

Implements the `Sampler` class that uses the trained MM‑DiT and 3D VAE to perform
text‑to‑video and image‑to‑video generation.  Generation is autoregressive over latent
blocks (each block corresponds to 8 video frames due to VAE temporal compression).
Inside each block the pyramidal flow matching algorithm (spatial pyramid) is run with
classifier‑free guidance and temporal pyramid history conditioning.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from utils import (
    downsample,
    nearest_upsample,
    corrective_renoise,
    patchify,
    unpatchify,
)
from vae import ThreeDVAE
from model import MMDiT


# ---------------------------------------------------------------------------
# Text embedding helper for two encoders (T5 + CLIP) with projection layers.
# ---------------------------------------------------------------------------

class TextEmbedder:
    """
    Encodes a text prompt into a fixed‑dimensional context tensor that can be
    fed into the MM‑DiT model.  Uses frozen T5‑XXL and CLIP‑ViT‑L/14 and two
    trainable linear projections to map their hidden states to a common dimension.

    Args:
        cfg: The global configuration dict.
    """

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        from transformers import (
            T5EncoderModel,
            CLIPTextModel,
            T5Tokenizer,
            CLIPTokenizer,
        )

        self.cfg = cfg
        self.device = device
        t5_name = cfg["model"]["text_conditioning"]["t5_model_name"]
        clip_name = cfg["model"]["text_conditioning"]["clip_model_name"]

        # Tokenizers
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_name)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_name)

        # Encoder models – kept in eval mode, gradients off
        self.t5_encoder = T5EncoderModel.from_pretrained(t5_name).to(device).eval()
        self.clip_encoder = CLIPTextModel.from_pretrained(clip_name).to(device).eval()
        for p in self.t5_encoder.parameters():
            p.requires_grad = False
        for p in self.clip_encoder.parameters():
            p.requires_grad = False

        t5_dim = self.t5_encoder.config.d_model
        clip_dim = self.clip_encoder.config.hidden_size
        self.context_dim = cfg["model"]["text_conditioning"]["context_dim"]

        # Projection layers (trained together with the main model)
        self.t5_proj = nn.Linear(t5_dim, self.context_dim, bias=False).to(device)
        self.clip_proj = nn.Linear(clip_dim, self.context_dim, bias=False).to(device)

        # Cache for the unconditional (null) embedding
        self.null_ctx: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _encode(self, prompt: str) -> torch.Tensor:
        """Return combined (1, L, context_dim) tensor for a single prompt."""
        # T5 tokens
        t5_tokens = self.t5_tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        ).to(self.device)
        t5_emb = self.t5_encoder(**t5_tokens).last_hidden_state  # (1, L_t5, t5_dim)

        # CLIP tokens
        clip_tokens = self.clip_tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        ).to(self.device)
        clip_emb = self.clip_encoder(**clip_tokens).last_hidden_state  # (1, L_clip, clip_dim)

        # Project to common dimension and concatenate along the token length
        t5_proj = self.t5_proj(t5_emb)       # (1, L_t5, context_dim)
        clip_proj = self.clip_proj(clip_emb)   # (1, L_clip, context_dim)
        context = torch.cat([t5_proj, clip_proj], dim=1)  # (1, L_tot, context_dim)
        return context

    def encode(self, prompt: str, use_cache: bool = False) -> torch.Tensor:
        """Public method: returns the conditional context."""
        if use_cache and self.null_ctx is not None and prompt == "":
            return self.null_ctx
        ctx = self._encode(prompt)
        if prompt == "":
            self.null_ctx = ctx
        return ctx

    def get_unconditional_context(self) -> torch.Tensor:
        """Return the pre‑computed unconditional (empty string) context."""
        if self.null_ctx is None:
            self.null_ctx = self._encode("")
        return self.null_ctx


# ---------------------------------------------------------------------------
# Sampler – main inference class
# ---------------------------------------------------------------------------

class Sampler:
    """
    Generates videos using the pyramidal flow matching method.

    Args:
        cfg: The global configuration dict (OmegaConf or Dict).
        model: Trained MM‑DiT model (must be on target device, eval mode).
        vae: Trained 3D VAE (must be on target device, eval mode).
    """

    def __init__(self, cfg: Dict[str, Any], model: MMDiT, vae: ThreeDVAE) -> None:
        self.cfg = cfg
        self.model = model
        self.vae = vae
        self.device = next(model.parameters()).device
        model.eval()
        vae.eval()

        # -------- General parameters --------
        self.total_nfe: int = cfg["inference"]["total_nfe"]
        self.guidance_scale: float = cfg["inference"]["guidance_scale"]
        self.default_fps: int = cfg["inference"]["video"]["default_fps"]
        self.resolution: Tuple[int, int] = tuple(cfg["inference"]["video"]["default_resolution"])
        # spatial size after 8×8 VAE compression
        self.latent_resolution: Tuple[int, int] = (
            self.resolution[0] // 8,
            self.resolution[1] // 8,
        )
        self.latent_channels: int = cfg["vae"]["latent_channels"]
        self.patch_size: Tuple[int, int] = tuple(cfg["model"]["patch_size"])
        self.ph, self.pw = self.patch_size

        # -------- Pyramid schedule --------
        py_cfg = cfg["model"]["pyramid"]
        self.K: int = py_cfg["num_stages"]
        # s and e are stored finest‑first (k=0 is finest)
        self.s: List[float] = list(py_cfg["s"])
        self.e: List[float] = list(py_cfg["e"])
        assert len(self.s) == self.K == len(self.e)

        # -------- Temporal pyramid mapping --------
        tp_cfg = cfg["temporal_pyramid"]
        self.max_history_blocks: int = tp_cfg["max_history_frames"]
        # history_extra_factors[i] corresponds to offset = i+1
        self.history_extra_factors: List[int] = tp_cfg["history_factors"]
        assert len(self.history_extra_factors) == self.max_history_blocks

        # -------- Text embedder --------
        self.text_embedder = TextEmbedder(cfg, self.device)
        self.null_context = self.text_embedder.get_unconditional_context()

        # -------- Allocate NFE per stage --------
        self.stage_nfe: List[int] = self._allocate_nfe()

        # -------- Optional external text‑projection checkpoint --------
        text_proj_ckpt = cfg["inference"].get("text_projector_checkpoint", None)
        if text_proj_ckpt is not None:
            state = torch.load(text_proj_ckpt, map_location="cpu")
            self.text_embedder.t5_proj.load_state_dict(
                {k[len("t5_proj."):]: v for k, v in state.items() if k.startswith("t5_proj.")}
            )
            self.text_embedder.clip_proj.load_state_dict(
                {k[len("clip_proj."):]: v for k, v in state.items() if k.startswith("clip_proj.")}
            )

    # ------------------------------------------------------------------
    #  NFE distribution
    # ------------------------------------------------------------------
    def _allocate_nfe(self) -> List[int]:
        """Distribute total NFE among stages proportionally to their durations."""
        durations = [self.e[i] - self.s[i] for i in range(self.K)]
        total_dur = sum(durations)
        nfe_list = []
        remaining = self.total_nfe
        for i in range(self.K - 1):
            n = max(1, int(round(self.total_nfe * durations[i] / total_dur)))
            nfe_list.append(n)
            remaining -= n
        nfe_list.append(max(1, remaining))
        return nfe_list

    # ------------------------------------------------------------------
    #  Pyramidal flow solver
    # ------------------------------------------------------------------
    def _prepare_history_info(
        self, history: List[torch.Tensor], k: int
    ) -> List[Tuple[torch.Tensor, int, int, int]]:
        """
        For a given pyramid stage *k* (0 finest …