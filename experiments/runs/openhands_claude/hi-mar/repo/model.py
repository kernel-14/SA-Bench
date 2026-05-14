"""
Hi-MAR: Hierarchical Masked Autoregressive Model

Architecture overview (Figure 2b):
  Phase 1 – low-resolution (small scale, N_s tokens):
    [class/text tokens] + [masked low-res tokens]
        → Hi-MAR Transformer (scale-aware blocks, scale=small)
        → conditional tokens Z^s
        → MLP Diffusion Head  → loss_1

  Phase 2 – high-resolution (large scale, N_l tokens):
    [class/text tokens] + [Z^s] + [masked high-res tokens]
        → Hi-MAR Transformer (scale-aware blocks, scale=large)
        → conditional tokens Z^l
        → Diffusion Transformer Head → loss_2
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import sinusoidal_embedding
from modules import (
    DiffusionTransformerHead,
    GaussianDiffusion,
    MLPDiffusionHead,
    ScaleAwareTransformerBlock,
)


# ---------------------------------------------------------------------------
# Positional embedding (2-D sinusoidal, flattened)
# ---------------------------------------------------------------------------

def build_2d_sincos_pos_embed(h: int, w: int, dim: int) -> torch.Tensor:
    """Returns (h*w, dim) sinusoidal 2-D positional embedding."""
    assert dim % 4 == 0
    y_pos = torch.arange(h, dtype=torch.float32)
    x_pos = torch.arange(w, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing="ij")

    half = dim // 2
    omega = torch.arange(half // 2, dtype=torch.float32) / (half // 2)
    omega = 1.0 / (10000 ** omega)

    y_emb = torch.outer(grid_y.flatten(), omega)
    x_emb = torch.outer(grid_x.flatten(), omega)

    pos_emb = torch.cat(
        [torch.sin(y_emb), torch.cos(y_emb), torch.sin(x_emb), torch.cos(x_emb)], dim=-1
    )
    return pos_emb  # (h*w, dim)


# ---------------------------------------------------------------------------
# Hi-MAR Transformer backbone
# ---------------------------------------------------------------------------

class HiMARTransformer(nn.Module):
    """
    Shared Transformer backbone for both phases.

    Scale awareness is injected via a learnable scale vector per resolution,
    which is encoded as a sinusoidal embedding and passed to every
    ScaleAwareTransformerBlock via AdaLN-Zero (paper §3.2).
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        token_dim: int,
        max_seq_len_small: int,
        max_seq_len_large: int,
        num_classes: int = 1000,
        text_embed_dim: int = 0,
        scale_emb_dim: int = 256,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_cfg: bool = True,
        cfg_dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.text_embed_dim = text_embed_dim

        # ---- Token projection ----
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # ---- Positional embeddings (fixed sinusoidal) ----
        h_s = w_s = int(math.isqrt(max_seq_len_small))
        h_l = w_l = int(math.isqrt(max_seq_len_large))
        pos_small = build_2d_sincos_pos_embed(h_s, w_s, hidden_dim)
        pos_large = build_2d_sincos_pos_embed(h_l, w_l, hidden_dim)
        self.register_buffer("pos_embed_small", pos_small.unsqueeze(0))  # (1, N_s, D)
        self.register_buffer("pos_embed_large", pos_large.unsqueeze(0))  # (1, N_l, D)

        # ---- Conditioning (class or text) ----
        if text_embed_dim > 0:
            # Text-to-image: project CLIP embeddings
            self.cond_proj = nn.Linear(text_embed_dim, hidden_dim)
        else:
            # Class-conditional: learnable class embeddings
            self.class_embed = nn.Embedding(num_classes + 1, hidden_dim)  # +1 for null class
            if use_cfg:
                self.cfg_dropout = cfg_dropout

        # ---- Scale-aware conditioning ----
        # Sinusoidal embedding of the resolution level (e.g., 8 or 16)
        self.scale_emb_dim = scale_emb_dim
        self.scale_mlp = nn.Sequential(
            nn.Linear(scale_emb_dim, scale_emb_dim),
            nn.SiLU(),
            nn.Linear(scale_emb_dim, scale_emb_dim),
        )

        # ---- Transformer blocks ----
        self.blocks = nn.ModuleList([
            ScaleAwareTransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                scale_dim=scale_emb_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(hidden_dim, eps=1e-6)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _get_scale_vec(self, resolution: int, batch_size: int) -> torch.Tensor:
        """Encode resolution as sinusoidal embedding → scale vector."""
        t = torch.tensor([resolution], dtype=torch.float32, device=self.mask_token.device)
        emb = sinusoidal_embedding(t, self.scale_emb_dim)  # (1, scale_emb_dim)
        emb = self.scale_mlp(emb)                          # (1, scale_emb_dim)
        return emb.expand(batch_size, -1)                  # (B, scale_emb_dim)

    def _get_cond_tokens(
        self,
        batch_size: int,
        class_labels: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
        cfg_force_uncond: bool = False,
    ) -> torch.Tensor:
        """Return conditioning tokens of shape (B, L_cond, hidden_dim)."""
        if self.text_embed_dim > 0:
            assert text_embeds is not None
            return self.cond_proj(text_embeds)  # (B, L_text, hidden_dim)
        else:
            assert class_labels is not None
            if self.training and self.use_cfg:
                # Randomly drop class labels for CFG training
                drop_mask = torch.rand(batch_size, device=class_labels.device) < self.cfg_dropout
                labels = class_labels.clone()
                labels[drop_mask] = self.num_classes  # null class
            elif cfg_force_uncond:
                labels = torch.full_like(class_labels, self.num_classes)
            else:
                labels = class_labels
            return self.class_embed(labels).unsqueeze(1)  # (B, 1, hidden_dim)

    def forward_phase(
        self,
        visual_tokens: torch.Tensor,
        mask: torch.Tensor,
        resolution: int,
        phase: int,
        class_labels: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
        pivot_tokens: Optional[torch.Tensor] = None,
        cfg_force_uncond: bool = False,
    ) -> torch.Tensor:
        """
        Run one phase of the Hi-MAR Transformer.

        Args:
            visual_tokens: (B, N, token_dim) continuous VAE tokens
            mask:          (B, N) bool, True = masked position
            resolution:    spatial resolution of the latent grid (e.g., 8 or 16)
            phase:         1 (low-res) or 2 (high-res)
            class_labels:  (B,) for class-conditional generation
            text_embeds:   (B, L, text_embed_dim) for text-to-image
            pivot_tokens:  (B, N_s, hidden_dim) conditional tokens from phase 1
                           (only used in phase 2)
            cfg_force_uncond: force unconditional generation (for CFG inference)
        Returns:
            cond_tokens: (B, N, hidden_dim) conditional tokens for diffusion head
        """
        B, N, _ = visual_tokens.shape

        # Project visual tokens; replace masked positions with mask token
        x = self.token_proj(visual_tokens)
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        x = torch.where(mask_expanded, self.mask_token.expand(B, N, -1), x)

        # Add positional embedding
        if phase == 1:
            x = x + self.pos_embed_small[:, :N, :]
        else:
            x = x + self.pos_embed_large[:, :N, :]

        # Conditioning tokens
        cond_tokens = self._get_cond_tokens(
            B, class_labels, text_embeds, cfg_force_uncond
        )  # (B, L_cond, D)

        # In phase 2, prepend pivot tokens (Z^s from phase 1)
        if phase == 2 and pivot_tokens is not None:
            # pivot_tokens: (B, N_s, hidden_dim)
            prefix = torch.cat([cond_tokens, pivot_tokens], dim=1)
        else:
            prefix = cond_tokens

        # Concatenate: [cond/pivot | visual tokens]
        seq = torch.cat([prefix, x], dim=1)  # (B, L_prefix + N, D)

        # Scale vector (same for all positions in this phase)
        scale_vec = self._get_scale_vec(resolution, B)  # (B, scale_emb_dim)

        # Transformer blocks
        for block in self.blocks:
            seq = block(seq, scale_vec)

        seq = self.norm_out(seq)

        # Return only the visual token positions
        L_prefix = prefix.shape[1]
        return seq[:, L_prefix:, :]  # (B, N, hidden_dim)


# ---------------------------------------------------------------------------
# Full Hi-MAR model
# ---------------------------------------------------------------------------

class HiMAR(nn.Module):
    """
    Hierarchical Masked Autoregressive Model (Hi-MAR).

    Two-phase generation:
      Phase 1: low-resolution tokens → MLP diffusion head
      Phase 2: high-resolution tokens + phase-1 pivots → Diffusion Transformer head
    """

    def __init__(
        self,
        # Transformer backbone
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        # Token dimensions
        token_dim: int,
        num_tokens_small: int,
        num_tokens_large: int,
        # Diffusion head phase 1 (MLP)
        diff_head1_layers: int,
        diff_head1_hidden: int,
        # Diffusion head phase 2 (Transformer)
        diff_head2_layers: int,
        diff_head2_hidden: int,
        diff_head2_heads: int = 8,
        # Conditioning
        num_classes: int = 1000,
        text_embed_dim: int = 0,
        # Diffusion
        num_diffusion_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        # Misc
        scale_emb_dim: int = 256,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_cfg: bool = True,
        cfg_dropout: float = 0.1,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.num_tokens_small = num_tokens_small
        self.num_tokens_large = num_tokens_large
        self.resolution_small = int(math.isqrt(num_tokens_small))
        self.resolution_large = int(math.isqrt(num_tokens_large))

        # Shared Transformer backbone
        self.transformer = HiMARTransformer(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            token_dim=token_dim,
            max_seq_len_small=num_tokens_small,
            max_seq_len_large=num_tokens_large,
            num_classes=num_classes,
            text_embed_dim=text_embed_dim,
            scale_emb_dim=scale_emb_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_cfg=use_cfg,
            cfg_dropout=cfg_dropout,
        )

        # Phase-1 MLP diffusion head
        self.diff_head1 = MLPDiffusionHead(
            token_dim=token_dim,
            cond_dim=hidden_dim,
            hidden_dim=diff_head1_hidden,
            num_layers=diff_head1_layers,
        )

        # Phase-2 Diffusion Transformer head
        self.diff_head2 = DiffusionTransformerHead(
            token_dim=token_dim,
            cond_dim=hidden_dim,
            hidden_dim=diff_head2_hidden,
            num_layers=diff_head2_layers,
            num_heads=diff_head2_heads,
        )

        # Shared noise schedule
        self.diffusion = GaussianDiffusion(
            num_timesteps=num_diffusion_timesteps,
            beta_schedule=beta_schedule,
        )

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens_small: torch.Tensor,
        tokens_large: torch.Tensor,
        mask_small: torch.Tensor,
        mask_large: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            tokens_small: (B, N_s, token_dim) low-res VAE tokens
            tokens_large: (B, N_l, token_dim) high-res VAE tokens
            mask_small:   (B, N_s) bool, True = masked
            mask_large:   (B, N_l) bool, True = masked
            class_labels: (B,) class indices (class-conditional)
            text_embeds:  (B, L, text_embed_dim) (text-to-image)
        Returns:
            dict with 'loss', 'loss_phase1', 'loss_phase2'
        """
        B = tokens_small.shape[0]
        device = tokens_small.device

        # ---- Phase 1: low-resolution ----
        cond_small = self.transformer.forward_phase(
            visual_tokens=tokens_small,
            mask=mask_small,
            resolution=self.resolution_small,
            phase=1,
            class_labels=class_labels,
            text_embeds=text_embeds,
        )  # (B, N_s, hidden_dim)

        # Diffusion loss on masked small tokens
        t1 = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)
        noise1 = torch.randn_like(tokens_small)
        x_noisy1 = self.diffusion.q_sample(tokens_small, t1, noise1)
        noise_pred1 = self.diff_head1(x_noisy1, t1, cond_small)
        loss1 = F.mse_loss(noise_pred1[mask_small], noise1[mask_small])

        # ---- Phase 2: high-resolution (conditioned on Z^s) ----
        # Use predicted conditional tokens (not ground-truth visual tokens)
        # to avoid training-inference discrepancy (paper §3.2)
        cond_large = self.transformer.forward_phase(
            visual_tokens=tokens_large,
            mask=mask_large,
            resolution=self.resolution_large,
            phase=2,
            class_labels=class_labels,
            text_embeds=text_embeds,
            pivot_tokens=cond_small.detach(),  # stop gradient through pivots
        )  # (B, N_l, hidden_dim)

        # Diffusion Transformer head loss on masked large tokens
        t2 = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)
        noise2 = torch.randn_like(tokens_large)
        x_noisy2 = self.diffusion.q_sample(tokens_large, t2, noise2)
        noise_pred2 = self.diff_head2(x_noisy2, t2, cond_large)
        loss2 = F.mse_loss(noise_pred2[mask_large], noise2[mask_large])

        total_loss = loss1 + loss2
        return {"loss": total_loss, "loss_phase1": loss1, "loss_phase2": loss2}

    # ------------------------------------------------------------------
    # Inference / generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        class_labels: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
        steps_phase1: int = 32,
        steps_phase2: int = 4,
        diff_steps: int = 100,
        cfg_scale: float = 1.5,
        cfg_scale_phase2: Optional[float] = None,
        temperature: float = 1.0,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Hierarchical masked autoregressive generation.

        Phase 1: iteratively unmask low-res tokens (cosine schedule, steps_phase1 steps)
        Phase 2: iteratively unmask high-res tokens (cosine schedule, steps_phase2 steps)

        Note (paper §4.3): for the "w/o CFG" setting, CFG is only turned off during
        phase 2 prediction (cfg_scale_phase2=1.0), since the first-stage output quality
        significantly affects Hi-MAR performance.

        Args:
            cfg_scale:        CFG scale for phase 1 (and phase 2 if cfg_scale_phase2 is None)
            cfg_scale_phase2: CFG scale for phase 2; if None, uses cfg_scale

        Returns:
            tokens_large: (B, N_l, token_dim) generated high-res tokens
        """
        if device is None:
            device = next(self.parameters()).device

        if cfg_scale_phase2 is None:
            cfg_scale_phase2 = cfg_scale

        N_s = self.num_tokens_small
        N_l = self.num_tokens_large

        # ---- Phase 1: generate low-res tokens ----
        tokens_small = torch.zeros(batch_size, N_s, self.token_dim, device=device)
        mask_small = torch.ones(batch_size, N_s, dtype=torch.bool, device=device)

        for step in range(steps_phase1):
            # Cosine masking schedule: ratio of tokens to keep unmasked
            ratio_unmasked = math.cos(math.pi / 2 * step / steps_phase1)
            num_to_unmask = max(1, int(N_s * (1.0 - ratio_unmasked)))

            # Forward pass
            cond_small = self._forward_with_cfg(
                tokens_small, mask_small,
                resolution=self.resolution_small, phase=1,
                class_labels=class_labels, text_embeds=text_embeds,
                cfg_scale=cfg_scale,
            )

            # Sample tokens for masked positions using diffusion head
            sampled = self._sample_tokens_mlp(
                cond_small, mask_small, diff_steps, temperature, device
            )

            # Update tokens at masked positions
            tokens_small = torch.where(
                mask_small.unsqueeze(-1).expand_as(tokens_small),
                sampled,
                tokens_small,
            )

            # Decide which tokens to keep unmasked (all if last step)
            if step == steps_phase1 - 1:
                mask_small = torch.zeros_like(mask_small)
            else:
                # Keep the num_to_unmask most-confident tokens unmasked
                # (confidence = negative reconstruction error, approximated by
                #  the norm of the predicted token)
                confidence = sampled.norm(dim=-1)  # (B, N_s)
                confidence[~mask_small] = float('inf')  # already unmasked → keep
                threshold = confidence.kthvalue(N_s - num_to_unmask + 1, dim=1).values
                mask_small = confidence < threshold.unsqueeze(1)

        # Final conditional tokens from phase 1
        cond_small_final = self._forward_with_cfg(
            tokens_small,
            torch.zeros(batch_size, N_s, dtype=torch.bool, device=device),
            resolution=self.resolution_small, phase=1,
            class_labels=class_labels, text_embeds=text_embeds,
            cfg_scale=1.0,  # no CFG needed for pivot extraction
        )

        # ---- Phase 2: generate high-res tokens ----
        tokens_large = torch.zeros(batch_size, N_l, self.token_dim, device=device)
        mask_large = torch.ones(batch_size, N_l, dtype=torch.bool, device=device)

        for step in range(steps_phase2):
            ratio_unmasked = math.cos(math.pi / 2 * step / steps_phase2)
            num_to_unmask = max(1, int(N_l * (1.0 - ratio_unmasked)))

            cond_large = self._forward_phase2_with_cfg(
                tokens_large, mask_large,
                pivot_tokens=cond_small_final,
                class_labels=class_labels, text_embeds=text_embeds,
                cfg_scale=cfg_scale_phase2,
            )

            sampled = self._sample_tokens_transformer(
                cond_large, mask_large, diff_steps, temperature, device
            )

            tokens_large = torch.where(
                mask_large.unsqueeze(-1).expand_as(tokens_large),
                sampled,
                tokens_large,
            )

            if step == steps_phase2 - 1:
                mask_large = torch.zeros_like(mask_large)
            else:
                confidence = sampled.norm(dim=-1)
                confidence[~mask_large] = float('inf')
                threshold = confidence.kthvalue(N_l - num_to_unmask + 1, dim=1).values
                mask_large = confidence < threshold.unsqueeze(1)

        return tokens_large

    def _forward_with_cfg(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        resolution: int,
        phase: int,
        class_labels: Optional[torch.Tensor],
        text_embeds: Optional[torch.Tensor],
        cfg_scale: float,
        pivot_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run transformer with classifier-free guidance."""
        if cfg_scale == 1.0:
            return self.transformer.forward_phase(
                tokens, mask, resolution, phase,
                class_labels=class_labels, text_embeds=text_embeds,
                pivot_tokens=pivot_tokens,
            )

        # Conditional
        cond = self.transformer.forward_phase(
            tokens, mask, resolution, phase,
            class_labels=class_labels, text_embeds=text_embeds,
            pivot_tokens=pivot_tokens,
        )
        # Unconditional
        uncond = self.transformer.forward_phase(
            tokens, mask, resolution, phase,
            class_labels=class_labels, text_embeds=text_embeds,
            pivot_tokens=pivot_tokens,
            cfg_force_uncond=True,
        )
        return uncond + cfg_scale * (cond - uncond)

    def _forward_phase2_with_cfg(
        self,
        tokens_large: torch.Tensor,
        mask_large: torch.Tensor,
        pivot_tokens: torch.Tensor,
        class_labels: Optional[torch.Tensor],
        text_embeds: Optional[torch.Tensor],
        cfg_scale: float,
    ) -> torch.Tensor:
        return self._forward_with_cfg(
            tokens_large, mask_large,
            resolution=self.resolution_large, phase=2,
            class_labels=class_labels, text_embeds=text_embeds,
            cfg_scale=cfg_scale,
            pivot_tokens=pivot_tokens,
        )

    def _sample_tokens_mlp(
        self,
        cond: torch.Tensor,
        mask: torch.Tensor,
        diff_steps: int,
        temperature: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample tokens using MLP diffusion head (phase 1)."""
        B, N, _ = cond.shape

        def model_fn(x_noisy, t, c):
            return self.diff_head1(x_noisy, t, c)

        sampled = self.diffusion.ddim_sample(
            model_fn=model_fn,
            shape=(B, N, self.token_dim),
            cond=cond,
            num_steps=diff_steps,
            device=device,
        )
        if temperature != 1.0:
            sampled = sampled * temperature
        return sampled

    def _sample_tokens_transformer(
        self,
        cond: torch.Tensor,
        mask: torch.Tensor,
        diff_steps: int,
        temperature: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample tokens using Diffusion Transformer head (phase 2)."""
        B, N, _ = cond.shape

        def model_fn(x_noisy, t, c):
            return self.diff_head2(x_noisy, t, c)

        sampled = self.diffusion.ddim_sample(
            model_fn=model_fn,
            shape=(B, N, self.token_dim),
            cond=cond,
            num_steps=diff_steps,
            device=device,
        )
        if temperature != 1.0:
            sampled = sampled * temperature
        return sampled


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "Hi-MAR-B": dict(
        num_layers=24,
        hidden_dim=768,
        num_heads=12,
        diff_head1_layers=6,
        diff_head1_hidden=1024,
        diff_head2_layers=6,
        diff_head2_hidden=512,
        diff_head2_heads=8,
    ),
    "Hi-MAR-L": dict(
        num_layers=32,
        hidden_dim=1024,
        num_heads=16,
        diff_head1_layers=8,
        diff_head1_hidden=1280,
        diff_head2_layers=8,
        diff_head2_hidden=512,
        diff_head2_heads=8,
    ),
    "Hi-MAR-H": dict(
        num_layers=40,
        hidden_dim=1280,
        num_heads=16,
        diff_head1_layers=12,
        diff_head1_hidden=1536,
        diff_head2_layers=12,
        diff_head2_hidden=768,
        diff_head2_heads=8,
    ),
    # Light-weight variant for MS-COCO text-to-image (comparable to U-ViT-S/2 Deep)
    "Hi-MAR-S": dict(
        num_layers=16,
        hidden_dim=512,
        num_heads=8,
        diff_head1_layers=4,
        diff_head1_hidden=768,
        diff_head2_layers=4,
        diff_head2_hidden=384,
        diff_head2_heads=6,
    ),
}


def build_himar(
    model_name: str = "Hi-MAR-B",
    token_dim: int = 16,
    num_tokens_small: int = 64,
    num_tokens_large: int = 256,
    num_classes: int = 1000,
    text_embed_dim: int = 0,
    **kwargs,
) -> HiMAR:
    cfg = MODEL_CONFIGS[model_name].copy()
    cfg.update(kwargs)
    return HiMAR(
        token_dim=token_dim,
        num_tokens_small=num_tokens_small,
        num_tokens_large=num_tokens_large,
        num_classes=num_classes,
        text_embed_dim=text_embed_dim,
        **cfg,
    )
