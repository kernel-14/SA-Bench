"""
inference.py

InferenceRunner for Hi‑MAR hierarchical masked autoregressive generation.

Implements the two‑phase autoregressive decoding with DDIM sampling,
classifier‑free guidance (CFG), and confidence‑ or random‑based token
unmasking.  The module is designed to operate on single samples (batch_size=1)
but can be extended to batch inference via ``run_on_dataloader``.

For each phase the Transformer is called once per step to produce conditional
tokens.  The diffusion heads are then invoked inside a custom DDIM loop that
applies CFG by combining conditional and unconditional noise predictions.

The unconditional context is obtained from a zero class embedding / empty text
embedding.  It is the caller’s responsibility to supply both ``context`` and
``context_uncond`` to ``generate``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

# Project imports – adjust paths according to your package structure
from config import InferencePhaseConfig
from masking import TokenMasker
from model import HiMARTransformer
from diffusion_heads import MLPDiffusionHead, DiffusionTransformerHead
from vae_tokenizer import VAETokenizer


class InferenceRunner:
    """Orchestrates autoregressive image generation for Hi‑MAR.

    Parameters
    ----------
    model : HiMARTransformer
        Scale‑aware Transformer backbone.
    head1 : MLPDiffusionHead
        Phase‑1 MLP‑based diffusion head.
    head2 : DiffusionTransformerHead
        Phase‑2 Diffusion‑Transformer head.
    vae : VAETokenizer
        Pre‑trained KL‑16 VAE for latent ↔ image conversion.
    config : InferencePhaseConfig
        Inference hyper‑parameters (steps, CFG scale, etc.).
    token_masker : TokenMasker
        Masking utility providing mask token, cosine schedule, and confidence
        estimation.
    """

    def __init__(
        self,
        model: HiMARTransformer,
        head1: MLPDiffusionHead,
        head2: DiffusionTransformerHead,
        vae: VAETokenizer,
        config: InferencePhaseConfig,
        token_masker: TokenMasker,
    ) -> None:
        super().__init__()

        self.model = model
        self.head1 = head1
        self.head2 = head2
        self.vae = vae
        self.token_masker = token_masker

        # Read configuration
        self.num_steps_p1: int = config.phase1_steps
        self.num_steps_p2: int = config.phase2_steps
        self.inner_steps: int = config.inner_diffusion_steps
        self.cfg_scale: float = config.cfg_scale
        self.confidence_metric: str = config.confidence_metric

        # Shared mask token (already in the latent space, dim = 16)
        self.mask_token: Tensor = model.get_mask_token()  # shape (1, 1, latent_dim)

        # Device (assume all parameters are on the same device)
        self.device = next(model.parameters()).device

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(
        self,
        context: Tensor,
        context_uncond: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate a single image from a conditional context.

        Parameters
        ----------
        context : Tensor
            Conditional context tokens of shape ``(1, L_ctx, hidden_size)``.
            For class‑conditional tasks this is the output of the class embedding
            (``(1, 1, hidden)``).  For text‑to‑image it is the projected text
            embedding (``(1, 77, hidden)``).
        context_uncond : Tensor, optional
            Unconditional counterpart of ``context``.  If ``None``, a tensor
            of zeros with the same shape is created internally.

        Returns
        -------
        Tensor
            RGB image of shape ``(3, 256, 256)`` with values in ``[0, 1]``.
        """
        if context_uncond is None:
            context_uncond = torch.zeros_like(context)

        # Phase 1 – low‑resolution autoregressive generation
        low_tokens, pivot_cond, pivot_uncond = self._phase1_generate(
            context, context_uncond
        )

        # Phase 2 – high‑resolution autoregressive generation guided by pivots
        high_tokens = self._phase2_generate(
            context, context_uncond, pivot_cond, pivot_uncond
        )

        # Decode final latents → image
        # high_tokens shape: (1, 256, 16)
        img = self.vae.decode(high_tokens)  # (1, 3, 256, 256) in [0,1]
        return img.squeeze(0)  # (3, 256, 256)

    @torch.no_grad()
    def run_on_dataloader(
        self,
        dataloader: DataLoader,
        use_cfg: bool = True,
    ) -> List[Tensor]:
        """Generate images for all samples in a dataloader.

        The dataloader is expected to yield batches containing either a
        ``class_id`` (LongTensor) or a ``text_emb`` (FloatTensor).  The
        unconditional context is obtained by zeroing the respective embedding.
        For simplicity, generation is performed one sample at a time.

        Parameters
        ----------
        dataloader : DataLoader
            Source of conditional information.
        use_cfg : bool
            If ``True``, classifier‑free guidance is applied; otherwise only the
            conditional branch is used (``cfg_scale`` is set to 1.0 internally).

        Returns
        -------
        List[Tensor]
            List of generated images, each of shape ``(3, 256, 256)`` in ``[0, 1]``.
        """
        generated: List[Tensor] = []

        # Temporarily override cfg_scale if disabled
        original_cfg = self.cfg_scale
        if not use_cfg:
            self.cfg_scale = 1.0

        for batch in dataloader:
            if "class_id" in batch:
                class_ids = batch["class_id"].to(self.device)  # (B,)
                cond_emb = self.model.class_embedding(class_ids)  # (B, hidden)
                uncond_emb = torch.zeros_like(cond_emb)
                for i in range(class_ids.size(0)):
                    c = cond_emb[i].unsqueeze(0)      # (1, hidden)
                    u = uncond_emb[i].unsqueeze(0)
                    # The class embedding must be transformed into a context token
                    # (the Transformer expects shape (1, L_ctx, hidden); here L_ctx=1)
                    c = c.unsqueeze(1)  # (1, 1, hidden)
                    u = u.unsqueeze(1)
                    img = self.generate(c, u)
                    generated.append(img)

            elif "text_emb" in batch:
                # batch["text_emb"] shape: (B, L, d_clip), e.g., (B, 77, 768)
                text_emb = batch["text_emb"].to(self.device)
                # Project to hidden size using the model's text projection
                cond_ctx = self.model.text_proj(text_emb)      # (B, L, hidden)
                uncond_ctx = torch.zeros_like(cond_ctx)
                for i in range(text_emb.size(0)):
                    c = cond_ctx[i].unsqueeze(0)   # (1, L, hidden)
                    u = uncond_ctx[i].unsqueeze(0)
                    img = self.generate(c, u)
                    generated.append(img)

            else:
                raise KeyError(
                    "Batch must contain either 'class_id' or 'text_emb'."
                )

        # Restore original cfg_scale
        self.cfg_scale = original_cfg
        return generated

    # ------------------------------------------------------------------ #
    # Phase 1 – low‑resolution autoregressive decoding
    # ------------------------------------------------------------------ #

    def _phase1_generate(
        self,
        ctx_cond: Tensor,
        ctx_uncond: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Execute Phase 1 and return pivots ``Z_s`` for both branches.

        Returns
        -------
        low_tokens : Tensor
            Decoded low‑res latents, shape ``(1, 64, 16)``.
        pivot_cond : Tensor
            Final conditional tokens ``Z_s`` for the conditional branch,
            shape ``(1, 64, hidden)``.
        pivot_uncond : Tensor
            Final conditional tokens ``Z_s`` for the unconditional branch,
            shape ``(1, 64, hidden)``.
        """
        B = ctx_cond.shape[0]  # must be 1
        N = 64  # number of low‑res tokens
        dim = self.mask_token.shape[-1]  # 16

        # Initialize with mask tokens
        low_tokens = self.mask_token.expand(B, N, dim).clone()
        mask = torch.ones(B, N, dtype=torch.bool, device=self.device)

        total_tokens = N
        already_unmasked = 0

        for step in range(self.num_steps_p1):
            # Cumulative fraction of tokens unmasked up to this step
            frac = self.token_masker.cosine_schedule_step(
                step, self.num_steps_p1
            )  # [0, 1]
            target = int(frac * total_tokens)
            num_to_unmask = max(0, target - already_unmasked)
            if num_to_unmask == 0 and step > 0:
                continue

            # Select indices of tokens to unmask
            if self.confidence_metric == "random":
                candidate = torch.where(mask[0])[0]
                perm = torch.randperm(len(candidate), device=self.device)
                selected = candidate[perm[: min(num_to_unmask, len(candidate))]]
            else:
                selected = self._confidence_phase1(
                    low_tokens, ctx_cond, ctx_uncond, mask, num_to_unmask
                )

            if len(selected) == 0:
                continue

            # Full Transformer forward (scale_id = 0) to get conditional tokens
            Z_cond = self.model(ctx_cond, low_tokens, scale_id=0)[:, :, :]    # (1,N,H)
            Z_uncond = self.model(ctx_uncond, low_tokens, scale_id=0)[:, :, :]

            # Extract conditionals for the selected positions
            z_c_sel = Z_cond[0, selected, :]      # (k, H)
            z_u_sel = Z_uncond[0, selected, :]

            # Sample clean latents for these positions using DDIM
            clean = self._ddim_sample_per_token_phase1(
                z_c_sel, z_u_sel
            )

            # Update sequence and mask
            low_tokens[0, selected] = clean
            mask[0, selected] = False
            already_unmasked += num_to_unmask

        # Final pivot extraction (both branches)
        pivot_cond = self.model(ctx_cond, low_tokens, scale_id=0)
        pivot_uncond = self.model(ctx_uncond, low_tokens, scale_id=0)

        return low_tokens, pivot_cond, pivot_uncond

    # ------------------------------------------------------------------ #
    # Phase 2 – high‑resolution autoregressive decoding
    # ------------------------------------------------------------------ #

    def _phase2_generate(
        self,
        ctx_cond: Tensor,
        ctx_uncond: Tensor,
        pivot_cond: Tensor,   # (1, 64, hidden)
        pivot_uncond: Tensor,
    ) -> Tensor:
        """Execute Phase 2 and return the full high‑res latents.

        The pivots ``Z_s`` are prepended to the context sequence so that the
        Transformer can attend to them.

        Returns
        -------
        high_tokens : Tensor
            Decoded high‑res latents, shape ``(1, 256, 16)``.
        """
        B = ctx_cond.shape[0]
        N = 256
        dim = self.mask_token.shape[-1]

        high_tokens = self.mask_token.expand(B, N, dim).clone()
        mask = torch.ones(B, N, dtype=torch.bool, device=self.device)

        # Concatenate pivots to the original context
        ctx_pivots_cond = torch.cat([ctx_cond, pivot_cond], dim=1)
        ctx_pivots_uncond = torch.cat([ctx_uncond, pivot_uncond], dim=1)

        total_tokens = N
        already_unmasked = 0

        for step in range(self.num_steps_p2):
            frac = self.token_masker.cosine_schedule_step(
                step, self.num_steps_p2
            )
            target = int(frac * total_tokens)
            num_to_unmask = max(0, target - already_unmasked)
            if num_to_unmask == 0 and step > 0:
                continue

            # Select positions to unmask
            if self.confidence_metric == "random":
                candidate = torch.where(mask[0])[0]
                perm = torch.randperm(len(candidate), device=self.device)
                selected = candidate[perm[: min(num_to_unmask, len(candidate))]]
            else:
                selected = self._confidence_phase2(
                    high_tokens,
                    mask,
                    ctx_pivots_cond,
                    ctx_pivots_uncond,
                    num_to_unmask,
                )

            if len(selected) == 0:
                continue

            # Transformer forward for Phase 2 (scale_id = 1)
            Z_cond = self.model(ctx_pivots_cond, high_tokens, scale_id=1)
            Z_uncond = self.model(ctx_pivots_uncond, high_tokens, scale_id=1)

            # Denoise only the masked positions using the Diffusion‑Transformer head
            high_tokens = self._ddim_sample_phase2_full(
                high_tokens, mask, Z_cond, Z_uncond
            )

            # Update mask
            mask[0, selected] = False
            already_unmasked += num_to_unmask

        return high_tokens

    # ------------------------------------------------------------------ #
    # DDIM sampling helpers
    # ------------------------------------------------------------------ #

    def _ddim_sample_per_token_phase1(
        self,
        z_cond: Tensor,
        z_uncond: Tensor,
    ) -> Tensor:
        """DDIM sampling for a batch of independent tokens using MLPDiffusionHead.

        Parameters
        ----------
        z_cond : Tensor
            Conditional tokens, shape ``(k, hidden_dim)``.
        z_uncond : Tensor
            Unconditional tokens, shape ``(k, hidden_dim)``.

        Returns
        -------
        Tensor
            Denoised latent tokens, shape ``(k, 16)``.
        """
        k, H = z_cond.shape
        dim = self.head1.latent_dim  # 16
        device = z_cond.device

        # Schedule
        alphas_cumprod = self.head1.alphas_cumprod  # (T,)
        T = len(alphas_cumprod) - 1
        times = torch.linspace(T, 0, self.inner_steps + 1, device=device).long()
        x = torch.randn(k, dim, device=device)

        for i in range(self.inner_steps):
            t = times[i]
            t_next = times[i + 1]
            t_norm = (t.float() / T).expand(k)  # (k,)

            # Expand to per‑token shape (k, 1, dim) for the head
            x_exp = x.unsqueeze(1)  # (k, 1, dim)
            noise_cond = self.head1(
                z_cond.unsqueeze(1), x_exp, t_norm
            )  # (k, 1, dim)
            noise_uncond = self.head1(
                z_uncond.unsqueeze(1), x_exp, t_norm
            )
            noise = noise_uncond + self.cfg_scale * (noise_cond - noise_uncond)
            noise = noise.squeeze(1)  # (k, dim)

            alpha_t = alphas_cumprod[t].float()
            x0 = (x - (1 - alpha_t).sqrt() * noise) / alpha_t.sqrt()

            if t_next >= 0:
                alpha_next = alphas_cumprod[t_next].float()
                x = alpha_next.sqrt() * x0 + (1 - alpha_next).sqrt() * noise
            else:
                x = x0

        return x

    def _ddim_sample_phase2_full(
        self,
        tokens: Tensor,
        mask: Tensor,
        z_cond_all: Tensor,
        z_uncond_all: Tensor,
    ) -> Tensor:
        """DDIM denoising for the full high‑res sequence using DiffusionTransformerHead.

        Only masked positions are denoised; unmasked positions are reset to
        their known clean values after each step.

        Parameters
        ----------
        tokens : Tensor
            Current token sequence ``(1, N, 16)``.  Masked positions contain the
            mask token embedding.
        mask : Tensor
            Boolean mask ``(1, N)`` where ``True`` indicates a masked position.
        z_cond_all : Tensor
            Conditional tokens for all positions, ``(1, N, hidden)``.
        z_uncond_all : Tensor
            Unconditional tokens, ``(1, N, hidden)``.

        Returns
        -------
        Tensor
            Updated token sequence, ``(1, N, 16)``.
        """
        B, N, dim = tokens.shape
        device = tokens.device

        # Replace mask‑token positions with random Gaussian noise (diffusion start)
        x = tokens.clone()
        x_init_noise = torch.randn(B, N, dim, device=device)
        x[mask] = x_init_noise[mask].to(x.dtype)

        # Clean latents for unmasked positions (taken from the original input)
        known_clean = tokens.clone()

        alphas_cumprod = self.head2.alphas_cumprod
        T = len(alphas_cumprod) - 1
        times = torch.linspace(T, 0, self.inner_steps + 1, device=device).long()

        for i in range(self.inner_steps):
            t = times[i]
            t_next = times[i + 1]
            t_norm = (t.float() / T).expand(B)

            noise_cond = self.head2(z_cond_all, x, t_norm)      # (1, N, dim)
            noise_uncond = self.head2(z_uncond_all, x, t_norm)
            noise = noise_uncond + self.cfg_scale * (noise_cond - noise_uncond)

            alpha_t = alphas_cumprod[t].float()
            x0 = (x - (1 - alpha_t).sqrt() * noise) / alpha_t.sqrt()

            if t_next >= 0:
                alpha_next = alphas_cumprod[t_next].float()
                x = alpha_next.sqrt() * x0 + (1 - alpha_next).sqrt() * noise
            else:
                x = x0

            # Restore unmasked positions to their original clean values
            x[~mask] = known_clean[~mask]

        return x

    # ------------------------------------------------------------------ #
    # Confidence estimation (heuristic placeholders)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _confidence_phase1(
        self,
        low_tokens: Tensor,
        ctx_cond: Tensor,
        ctx_uncond: Tensor,
        mask: Tensor,
        num_to_unmask: int,
    ) -> Tensor:
        """Confidence‑based selection for Phase 1.

        Replaces masked tokens with noise, runs one forward pass, and uses
        the negative noise‑prediction norm as confidence.
        """
        B, N, D = low_tokens.shape
        # Fixed high timestep
        t_val = 900  # out of 0..999
        t_norm = torch.tensor([t_val / 999.0], device=self.device)

        # Construct noisy input
        noise = torch.randn(B, N, D, device=self.device)
        x_t = low_tokens.clone()
        x_t[mask] = noise[mask]

        # Conditional tokens
        Z_cond = self.model(ctx_cond, x_t, scale_id=0)    # (B,N,H)
        Z_uncond = self.model(ctx_uncond, x_t, scale_id=0)

        # Predict noise for each position
        noise_cond = self.head1(Z_cond, x_t, t_norm.expand(B))[0]      # (N,D)
        noise_uncond = self.head1(Z_uncond, x_t, t_norm.expand(B))[0]
        noise_pred = noise_uncond + self.cfg_scale * (noise_cond - noise_uncond)

        # Confidence = -||noise||^2
        conf = -noise_pred.pow(2).sum(dim=-1)  # (N,)
        conf[~mask[0]] = -float("inf")

        _, indices = torch.topk(conf, num_to_unmask)
        return indices

    @torch.no_grad()
    def _confidence_phase2(
        self,
        high_tokens: Tensor,
        mask: Tensor,
        ctx_pivots_cond: Tensor,
        ctx_pivots_uncond: Tensor,
        num_to_unmask: int,
    ) -> Tensor:
        """Confidence‑based selection for Phase 2."""
        B, N, D = high_tokens.shape
        t_val = 900
        t_norm = torch.tensor([t_val / 999.0], device=self.device)

        noise = torch.randn(B, N, D, device=self.device)
        x_t = high_tokens.clone()
        x_t[mask] = noise[mask]

        Z_cond = self.model(ctx_pivots_cond, x_t, scale_id=1)
        Z_uncond = self.model(ctx_pivots_uncond, x_t, scale_id=1)

        noise_cond = self.head2(Z_cond, x_t, t_norm.expand(B))[0]
        noise_uncond = self.head2(Z_uncond, x_t, t_norm.expand(B))[0]
        noise_pred = noise_uncond + self.cfg_scale * (noise_cond - noise_uncond)

        conf = -noise_pred.pow(2).sum(dim=-1)
        conf[~mask[0]] = -float("inf")

        _, indices = torch.topk(conf, num_to_unmask)
        return indices

