## Code: inference/generate.py

```python
## inference/generate.py
"""Autoregressive image generation for Hi-MAR at inference time.

This module implements the ``Generator`` class, which orchestrates the full
two-phase masked autoregressive generation pipeline. It bridges the trained
``HiMAR`` model with the ``VAETokenizer`` to produce final pixel-space images.

Two-phase generation overview:
    Phase 1 (32 AR steps, cosine schedule):
        - Generates 64 low-resolution tokens (128×128 → 8×8 latent)
        - Uses MLP diffusion head (per-token independent)
        - Produces conditional tokens Z^s as global structure pivots

    Phase 2 (4 AR steps, cosine schedule):
        - Generates 256 high-resolution tokens (256×256 → 16×16 latent)
        - Conditioned on Z^s from Phase 1
        - Uses Diffusion Transformer head (self-attention across all tokens)
        - Produces final high-resolution token predictions

Configuration alignment (config.yaml):
    inference.phase1_steps          = 32
    inference.phase2_steps          = 4
    inference.schedule              = cosine
    inference.cfg.scale             = 2.9
    inference.cfg.phase1_cfg_enabled = true
    inference.cfg.phase2_cfg_enabled = true
    inference.cfg.phase2_cfg_disabled_for_nocfg = true
    resolution.lr_seq_len           = 64
    resolution.hr_seq_len           = 256
    vae.latent_channels             = 16
    training_imagenet.n_classes     = 1000

Paper reference: Section 3, Section 4.2 (Inference), Section 4.5.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.himar import HiMAR
from models.vae_tokenizer import VAETokenizer
from training.masking_schedule import MaskingSchedule


class Generator:
    """Two-phase masked autoregressive image generator for Hi-MAR.

    Orchestrates the full inference pipeline:
    1. Build context tokens from class IDs (ImageNet) or text embeddings (COCO).
    2. Phase 1: Progressively unmask 64 low-resolution tokens over 32 AR steps.
    3. Phase 2: Progressively unmask 256 high-resolution tokens over 4 AR steps,
       conditioned on Phase 1 conditional tokens (pivots).
    4. Decode final high-resolution tokens to pixel images via the VAE.

    All public methods run under ``torch.no_grad()`` — generation never
    requires gradient computation.

    The ``Generator`` does not own the model or VAE; it receives them as
    constructor arguments and never modifies their parameters.

    Attributes:
        model: Trained HiMAR model in eval mode.
        vae: Frozen VAETokenizer for decoding latent tokens to images.
        masking: Stateless MaskingSchedule utility for cosine schedule logic.
        device: Compute device for all tensor operations.
        latent_dim: VAE latent channel dimension (16 for KL-16).
        lr_seq_len: Low-resolution token sequence length (64).
        hr_seq_len: High-resolution token sequence length (256).
        n_classes: Number of ImageNet classes (1000); index n_classes is null.
        hidden_size: Transformer hidden dimension for context construction.
    """

    def __init__(
        self,
        model: HiMAR,
        vae: VAETokenizer,
        device: torch.device,
    ) -> None:
        """Initialises the Generator.

        Sets the model to eval mode and extracts configuration constants
        from the model's config for use throughout generation.

        Args:
            model: Trained HiMAR model. Will be set to eval mode. The EMA
                transformer is used for generation when
                ``model.config.use_ema_for_inference=True`` (default).
            vae: Frozen VAETokenizer. Used only for decoding in
                ``decode_tokens()``. Never trained.
            device: Compute device for all tensor operations. Should match
                the device on which ``model`` and ``vae`` reside.
        """
        self.model: HiMAR = model
        self.vae: VAETokenizer = vae
        self.device: torch.device = device

        # Set model to eval mode — disables dropout and batch norm training
        # behaviour. Critical for deterministic generation.
        self.model.eval()

        # Stateless masking schedule utility for cosine unmasking logic.
        self.masking: MaskingSchedule = MaskingSchedule()

        # Extract configuration constants from the model config.
        # These are used throughout generation to avoid repeated attribute access.
        self.latent_dim: int = model.config.latent_dim          # 16 (KL-16 VAE)
        self.lr_seq_len: int = model.config.lr_seq_len          # 64 (8×8 latent)
        self.hr_seq_len: int = model.config.hr_seq_len          # 256 (16×16 latent)
        self.n_classes: int = model.config.n_classes            # 1000 (ImageNet-1K)
        self.hidden_size: int = model.config.hidden_size        # e.g., 768 (Hi-MAR-B)

    # ------------------------------------------------------------------
    # Public API: task-specific generation entry points
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_imagenet(
        self,
        class_ids: torch.Tensor,
        cfg_scale: float = 2.9,
        phase1_steps: int = 32,
        phase2_steps: int = 4,
        phase2_cfg_scale: Optional[float] = None,
        diff_sample_steps: int = 10,
    ) -> torch.Tensor:
        """Generates images conditioned on ImageNet class labels.

        Builds class embedding context from integer class IDs and delegates
        to ``generate_batch`` for the full two-phase generation pipeline.

        Paper reference (Section 4.2):
            "For class-conditional image generation, we validate Hi-MAR on
            ImageNet at 256×256 resolution."

        Config alignment:
            inference.cfg.scale = 2.9  → cfg_scale default
            inference.phase1_steps = 32
            inference.phase2_steps = 4
            training_imagenet.n_classes = 1000

        Args:
            class_ids: Integer class label tensor, shape ``[B]``. Values in
                ``{0, …, 999}`` for ImageNet-1K. Must be on ``self.device``.
            cfg_scale: Classifier-free guidance scale for Phase 1.
                Config: ``inference.cfg.scale = 2.9``. Default: 2.9.
            phase1_steps: Number of Phase 1 autoregressive steps.
                Config: ``inference.phase1_steps = 32``. Default: 32.
            phase2_steps: Number of Phase 2 autoregressive steps.
                Config: ``inference.phase2_steps = 4``. Default: 4.
            phase2_cfg_scale: CFG scale for Phase 2. If ``None``, uses
                ``cfg_scale`` (w/ CFG setting). Pass ``1.0`` for the w/o CFG
                setting (Phase 2 CFG disabled per paper).
                Config: ``inference.cfg.phase2_cfg_disabled_for_nocfg = true``.
            diff_sample_steps: Number of inner DDPM denoising steps per AR
                step. Fewer steps = faster but lower quality. Default: 10.

        Returns:
            Float tensor of shape ``[B, 3, 256, 256]`` in ``[-1, 1]`` range.
            Pixel-space images decoded from the predicted high-resolution
            latent tokens.
        """
        class_ids = class_ids.to(self.device)
        batch_size: int = class_ids.shape[0]

        # ------------------------------------------------------------------
        # Build conditioned context: class_embed(class_ids) → [B, 1, hidden_size]
        # The class embedding is a single-token context (one class per sample).
        # ------------------------------------------------------------------
        context: torch.Tensor = self.model.transformer.encode_class_context(
            class_ids
        )  # [B, 1, hidden_size]

        # ------------------------------------------------------------------
        # Build null context for CFG unconditional pass.
        # Null class index = n_classes (extra entry in class_embed, index 1000).
        # Config: training_imagenet.n_classes = 1000.
        # ------------------------------------------------------------------
        null_class_ids: torch.Tensor = torch.full(
            (batch_size,),
            fill_value=self.n_classes,
            dtype=torch.long,
            device=self.device,
        )
        null_context: torch.Tensor = self.model.transformer.encode_class_context(
            null_class_ids
        )  # [B, 1, hidden_size]

        # Resolve Phase 2 CFG scale.
        # Per paper: in w/o CFG setting, Phase 2 CFG is disabled (scale=1.0).
        # In w/ CFG setting, Phase 2 uses the same cfg_scale as Phase 1.
        p2_cfg: float = phase2_cfg_scale if phase2_cfg_scale is not None else cfg_scale

        return self.generate_batch(
            context=context,
            null_context=null_context,
            cfg_scale=cfg_scale,
            phase1_steps=phase1_steps,
            phase2_steps=phase2_steps,
            phase2_cfg_scale=p2_cfg,
            diff_sample_steps=diff_sample_steps,
        )

    @torch.no_grad()
    def generate_coco(
        self,
        text_embeddings: torch.Tensor,
        cfg_scale: float = 2.9,
        phase1_steps: int = 32,
        phase2_steps: int = 4,
        phase2_cfg_scale: Optional[float] = None,
        null_text_embedding: Optional[torch.Tensor] = None,
        diff_sample_steps: int = 10,
    ) -> torch.Tensor:
        """Generates images conditioned on CLIP text embeddings (MS-COCO).

        Projects CLIP text embeddings to the Transformer hidden space and
        delegates to ``generate_batch`` for the full two-phase generation.

        Paper reference (Section 4.2):
            "Following Stable Diffusion, we convert captions into a sequence
            of text embeddings with CLIP text encoder. Then the text embeddings
            act as context tokens and are fed into Hi-MAR."

        Config alignment:
            training_coco.text_encoder.model = openai/clip-vit-large-patch14
            training_coco.text_encoder.max_length = 77

        Args:
            text_embeddings: CLIP last_hidden_state embeddings, shape
                ``[B, 77, 768]``. Produced by the frozen CLIP ViT-L/14 model.
                Must be on ``self.device``.
            cfg_scale: Classifier-free guidance scale for Phase 1.
                Config: ``inference.cfg.scale = 2.9``. Default: 2.9.
            phase1_steps: Number of Phase 1 autoregressive steps. Default: 32.
            phase2_steps: Number of Phase 2 autoregressive steps. Default: 4.
            phase2_cfg_scale: CFG scale for Phase 2. If ``None``, uses
                ``cfg_scale``. Pass ``1.0`` for w/o CFG setting. Default: None.
            null_text_embedding: CLIP embedding of empty string ``''``, shape
                ``[1, 77, 768]`` or ``[B, 77, 768]``. Used for CFG null pass.
                If ``None``, zeros are used as the null embedding (suboptimal
                but functional). Callers should pass the pre-computed null
                embedding from ``COCODataset.null_text_embedding``.
            diff_sample_steps: Number of inner DDPM denoising steps per AR
                step. Default: 10.

        Returns:
            Float tensor of shape ``[B, 3, 256, 256]`` in ``[-1, 1]`` range.
        """
        text_embeddings = text_embeddings.to(self.device)
        batch_size: int = text_embeddings.shape[0]

        # ------------------------------------------------------------------
        # Build conditioned context: text_proj(text_embeddings) → [B, 77, H]
        # The text_proj linear layer lives in ScaleAwareTransformer.
        # ------------------------------------------------------------------
        context: torch.Tensor = self.model.transformer.encode_text_context(
            text_embeddings
        )  # [B, 77, hidden_size]

        # ------------------------------------------------------------------
        # Build null context for CFG unconditional pass.
        # Null text = CLIP embedding of empty string ''.
        # ------------------------------------------------------------------
        if null_text_embedding is not None:
            null_text: torch.Tensor = null_text_embedding.to(self.device)
            # Expand to batch size if needed: [1, 77, 768] → [B, 77, 768].
            if null_text.shape[0] == 1 and batch_size > 1:
                null_text = null_text.expand(batch_size, -1, -1)
            null_context: torch.Tensor = self.model.transformer.encode_text_context(
                null_text
            )  # [B, 77, hidden_size]
        else:
            # Fallback: zero null context. Suboptimal for CFG quality but
            # avoids requiring the caller to provide the null embedding.
            null_context = torch.zeros_like(context)

        # Resolve Phase 2 CFG scale.
        p2_cfg: float = phase2_cfg_scale if phase2_cfg_scale is not None else cfg_scale

        return self.generate_batch(
            context=context,
            null_context=null_context,
            cfg_scale=cfg_scale,
            phase1_steps=phase1_steps,
            phase2_steps=phase2_steps,
            phase2_cfg_scale=p2_cfg,
            diff_sample_steps=diff_sample_steps,
        )

    @torch.no_grad()
    def generate_batch(
        self,
        context: torch.Tensor,
        null_context: torch.Tensor,
        cfg_scale: float = 2.9,
        phase1_steps: int = 32,
        phase2_steps: int = 4,
        phase2_cfg_scale: float = 2.9,
        diff_sample_steps: int = 10,
    ) -> torch.Tensor:
        """Top-level two-phase generation orchestrator.

        Runs Phase 1 (low-resolution) and Phase 2 (high-resolution) generation
        sequentially, then decodes the final high-resolution tokens to images.

        This method is task-agnostic: it accepts pre-projected context tensors
        in the Transformer's hidden space, so it works for both ImageNet
        (class conditioning) and COCO (text conditioning).

        Paper reference (Section 3.2):
            "In the first phase, the masked low-resolution visual tokens along
            with the context tokens are fed into the Transformer, which outputs
            the conditional tokens Z^s. In the second phase, the Transformer
            takes the concatenation of context tokens, small scale conditional
            tokens and the masked dense visual tokens as input."

        Args:
            context: Pre-projected conditioned context tokens in hidden_size
                space, shape ``[B, C, hidden_size]``. C=1 for ImageNet class
                conditioning, C=77 for COCO text conditioning.
            null_context: Pre-projected null context for CFG unconditional
                pass, shape ``[B, C, hidden_size]``. Same shape as ``context``.
            cfg_scale: CFG scale for Phase 1. Default: 2.9.
            phase1_steps: Number of Phase 1 AR steps. Default: 32.
            phase2_steps: Number of Phase 2 AR steps. Default: 4.
            phase2_cfg_scale: CFG scale for Phase 2. Pass ``1.0`` to disable
                CFG for Phase 2 (w/o CFG setting per paper). Default: 2.9.
            diff_sample_steps: Number of inner DDPM denoising steps per AR
                step for both phases. Default: 10.

        Returns:
            Float tensor of shape ``[B, 3, 256, 256]`` in ``[-1, 1]`` range.
        """
        # ------------------------------------------------------------------
        # Phase 1: Generate 64 low-resolution conditional tokens.
        # Returns Z^s (transformer output), not raw visual tokens.
        # ------------------------------------------------------------------
        cond_tokens_lr: torch.Tensor = self._phase1_generate(
            context=context,
            null_context=null_context,
            cfg_scale=cfg_scale,
            n_steps=phase1_steps,
            diff_sample_steps=diff_sample_steps,
        )
        # cond_tokens_lr: [B, 64, hidden_size]

        # ------------------------------------------------------------------
        # Phase 2: Generate 256 high-resolution tokens conditioned on Z^s.
        # cond_tokens_lr is already detached (no_grad context ensures this).
        # ------------------------------------------------------------------
        tokens_hr: torch.Tensor = self._phase2_generate(
            cond_lr=cond_tokens_lr,
            context=context,
            null_context=null_context,
            cfg_scale=phase2_cfg_scale,
            n_steps=phase2_steps,
            diff_sample_steps=diff_sample_steps,
        )
        # tokens_hr: [B, 256, latent_dim]

        # ------------------------------------------------------------------
        # Decode high-resolution latent tokens to pixel images.
        # ------------------------------------------------------------------
        images: torch.Tensor = self.decode_tokens(tokens_hr)
        # images: [B, 3, 256, 256]

        return images

    # ------------------------------------------------------------------
    # Private: Phase 1 generation loop
    # ------------------------------------------------------------------

    def _phase1_generate(
        self,
        context: torch.Tensor,
        null_context: torch.Tensor,
        cfg_scale: float = 2.9,
        n_steps: int = 32,
        diff_sample_steps: int = 10,
    ) -> torch.Tensor:
        """Phase 1 masked autoregressive generation over 64 low-resolution tokens.

        Progressively unmasks 64 low-resolution tokens over ``n_steps``
        autoregressive steps using a cosine unmasking schedule. At each step:
        1. Run the transformer (with CFG) to get conditional tokens Z^s.
        2. Select which masked tokens to unmask (random selection).
        3. Sample predicted tokens for newly unmasked positions via the MLP
           diffusion head.
        4. Update the token buffer and mask.

        Returns the final conditional tokens Z^s (transformer output when all
        tokens are unmasked), which serve as global structure pivots for Phase 2.

        Paper reference (Section 3.2):
            "the first phase performs bidirectional autoregressive modeling over
            low-resolution visual tokens to capture the global structure."

        Paper reference (Section 4.5):
            "the FID decreases as the step number on the first phase increases
            and reaches an optimal value at 32 steps."

        Config alignment:
            inference.phase1_steps = 32
            resolution.lr_seq_len = 64
            vae.latent_channels = 16

        Args:
            context: Conditioned context tokens, shape ``[B, C, hidden_size]``.
            null_context: Null context for CFG, shape ``[B, C, hidden_size]``.
            cfg_scale: CFG scale. Default: 2.9.
            n_steps: Number of AR steps. Config: 32. Default: 32.
            diff_sample_steps: Inner DDPM denoising steps per AR step.
                Default: 10.

        Returns:
            Conditional tokens Z^s from the final transformer pass,
            shape ``[B, 64, hidden_size]``. These are the low-resolution
            pivots passed to Phase 2.
        """
        batch_size: int = context.shape[0]
        n_tokens: int = self.lr_seq_len  # 64

        # ------------------------------------------------------------------
        # Initialize token buffer and mask.
        # tokens_lr: accumulates predicted latent tokens as they are unmasked.
        # mask: True = masked (token not yet predicted).
        # ------------------------------------------------------------------
        tokens_lr: torch.Tensor = torch.zeros(
            batch_size, n_tokens, self.latent_dim,
            device=self.device, dtype=torch.float32,
        )  # [B, 64, 16]

        # All tokens start masked.
        mask: torch.Tensor = torch.ones(
            batch_size, n_tokens,
            device=self.device, dtype=torch.bool,
        )  # [B, 64], True = masked

        # Track the final conditional tokens from the last transformer pass.
        cond_tokens_lr: torch.Tensor = torch.zeros(
            batch_size, n_tokens, self.hidden_size,
            device=self.device, dtype=torch.float32,
        )  # [B, 64, hidden_size]

        # ------------------------------------------------------------------
        # Autoregressive generation loop.
        # ------------------------------------------------------------------
        for step in range(n_steps):
            # ------------------------------------------------------------------
            # Step 1: Compute cosine schedule — how many tokens to unmask now.
            # ratio_remaining = cos(π/2 * (step+1)/n_steps)
            # n_masked_target = ceil(ratio_remaining * n_tokens)
            # n_to_unmask[b] = max(0, current_n_masked[b] - n_masked_target)
            # ------------------------------------------------------------------
            progress: float = (step + 1) / float(n_steps)
            ratio_remaining: float = math.cos(math.pi / 2.0 * progress)
            n_masked_target: int = math.ceil(ratio_remaining * n_tokens)
            n_masked_target = max(0, min(n_masked_target, n_tokens))

            # Per-sample count of currently masked tokens.
            n_masked_current: torch.Tensor = mask.sum(dim=1)  # [B], int

            # Number of tokens to unmask at this step (per sample).
            # At the last step, force all remaining masked tokens to be unmasked.
            if step == n_steps - 1:
                n_to_unmask: torch.Tensor = n_masked_current  # unmask all remaining
            else:
                n_to_unmask = (n_masked_current - n_masked_target).clamp(min=0)
            # n_to_unmask: [B], int64

            # ------------------------------------------------------------------
            # Step 2: Run transformer with CFG to get conditional tokens Z^s.
            # _cfg_forward handles the conditioned + unconditioned passes and
            # applies the CFG formula: uncond + cfg_scale * (cond - uncond).
            # ------------------------------------------------------------------
            cond_tokens_lr = self._cfg_forward_phase1(
                tokens=tokens_lr,
                context=context,
                null_context=null_context,
                mask=mask,
                cfg_scale=cfg_scale,
            )
            # cond_tokens_lr: [B, 64, hidden_size]

            # ------------------------------------------------------------------
            # Step 3: Select which masked tokens to unmask at this step.
            # Random selection among currently masked positions.
            # ------------------------------------------------------------------
            newly_unmasked: torch.Tensor = self._select_tokens_to_unmask(
                mask=mask,
                n_to_unmask=n_to_unmask,
            )
            # newly_unmasked: BoolTensor [B, 64], True = unmask this token now

            # Skip diffusion sampling if no tokens to unmask at this step.
            if not newly_unmasked.any():
                continue

            # ------------------------------------------------------------------
            # Step 4: Sample predicted tokens for newly unmasked positions.
            # The MLP diffusion head processes tokens independently (per-token).
            # We run the full diffusion sampling loop for all positions and
            # then select only the newly unmasked ones.
            # ------------------------------------------------------------------
            predicted_lr: torch.Tensor = self._sample_phase1(
                cond_tokens=cond_tokens_lr,
                n_diff_steps=diff_sample_steps,
            )
            # predicted_lr: [B, 64, latent_dim]

            # ------------------------------------------------------------------
            # Step 5: Update token buffer — fill in newly unmasked positions.
            # torch.where: select predicted_lr where newly_unmasked=True,
            # else keep existing tokens_lr.
            # ------------------------------------------------------------------
            newly_unmasked_expanded: torch.Tensor = newly_unmasked.unsqueeze(-1)
            # [B, 64, 1] for broadcasting over latent_dim
            tokens_lr = torch.where(
                newly_unmasked_expanded,
                predicted_lr,
                tokens_lr,
            )  # [B, 64, latent_dim]

            # ------------------------------------------------------------------
            # Step 6: Update mask — mark newly unmasked positions as visible.
            # mask & ~newly_unmasked: keep masked except for newly unmasked.
            # ------------------------------------------------------------------
            mask = mask & ~newly_unmasked  # [B, 64]

        # ------------------------------------------------------------------
        # Final pass: run transformer with all tokens unmasked to get the
        # clean conditional tokens Z^s that Phase 2 will use as pivots.
        # This ensures Z^s reflects the fully predicted low-res structure.
        # ------------------------------------------------------------------
        final_mask: torch.Tensor = torch.zeros(
            batch_size, n_tokens,
            device=self.device, dtype=torch.bool,
        )  # All False — no masking

        cond_tokens_lr = self._cfg_forward_phase1(
            tokens=tokens_lr,
            context=context,
            null_context=null_context,
            mask=final_mask,
            cfg_scale=cfg_scale,
        )
        # cond_tokens_lr: [B, 64, hidden_size]

        return cond_tokens_lr

    # ------------------------------------------------------------------
    # Private: Phase 2 generation loop
    # ------------------------------------------------------------------

    def _phase2_generate(
        self,
        cond_lr: torch.Tensor,
        context: torch.Tensor,
        null_context: torch.Tensor,
        cfg_scale: float = 2.9,
        n_steps: int = 4,
        diff_sample_steps: int = 10,
    ) -> torch.Tensor:
        """Phase 2 masked autoregressive generation over 256 high-resolution tokens.

        Progressively unmasks 256 high-resolution tokens over ``n_steps``
        autoregressive steps, conditioned on Phase 1 conditional tokens
        ``cond_lr`` (the global structure pivots). Uses the Diffusion
        Transformer head which operates on ALL token positions simultaneously
        via self-attention.

        Paper reference (Section 3.2):
            "the Transformer takes the concatenation of context tokens, small
            scale conditional tokens and the masked dense visual tokens as input
            to generate dense conditional tokens, which are further fed into
            Diffusion Transformer head for token prediction."

        Paper reference (Section 4.5):
            "With the global structure provided by the first phase, the second
            phase can focus on the local fine-grained details and requires much
            fewer steps to generate satisfied results."
            "we use much fewer steps (e.g., 4 steps) in the second phase."

        Config alignment:
            inference.phase2_steps = 4
            resolution.hr_seq_len = 256
            vae.latent_channels = 16

        Args:
            cond_lr: Phase 1 conditional tokens Z^s (global structure pivots),
                shape ``[B, 64, hidden_size]``. These are the transformer
                outputs from Phase 1, not raw visual tokens.
            context: Conditioned context tokens, shape ``[B, C, hidden_size]``.
                Same context as Phase 1 (class or text conditioning).
            null_context: Null context for CFG, shape ``[B, C, hidden_size]``.
            cfg_scale: CFG scale for Phase 2. Pass ``1.0`` to disable CFG
                (w/o CFG setting per paper). Default: 2.9.
            n_steps: Number of AR steps. Config: 4. Default: 4.
            diff_sample_steps: Inner DDPM denoising steps per AR step.
                Default: 10.

        Returns:
            Predicted high-resolution latent tokens, shape
            ``[B, 256, latent_dim]``. Passed to ``decode_tokens()`` for
            pixel-space image reconstruction.
        """
        batch_size: int = cond_lr.shape[0]
        n_tokens: int = self.hr_seq_len  # 256

        # ------------------------------------------------------------------
        # Initialize high-resolution token buffer and mask.
        # ------------------------------------------------------------------
        tokens_hr: torch.Tensor = torch.zeros(
            batch_size, n_tokens, self.latent_dim,
            device=self.device, dtype=torch.float32,
        )  # [B, 256, 16]

        # All hr tokens start masked.
        mask_hr: torch.Tensor = torch.ones(
            batch_size, n_tokens,
            device=self.device, dtype=torch.bool,
        )  # [B, 256], True = masked

        # ------------------------------------------------------------------
        # Autoregressive generation loop.
        # ------------------------------------------------------------------
        for step in range(n_steps):
            # ------------------------------------------------------------------
            # Step 1: Compute cosine schedule for Phase 2.
            # ------------------------------------------------------------------
            progress: float = (step + 1) / float(n_steps)
            ratio_remaining: float = math.cos(math.pi / 2.0 * progress)
            n_masked_target: int = math.ceil(ratio_remaining * n_tokens)
            n_masked_target = max(0, min(n_masked_target, n_tokens))

            n_masked_current: torch.Tensor = mask_hr.sum(dim=1)  # [B]

            # At the last step, force all remaining masked tokens to be unmasked.
            if step == n_steps - 1:
                n_to_unmask: torch.Tensor = n_masked_current
            else:
                n_to_unmask = (n_masked_current - n_masked_target).clamp(min=0)
            # n_to_unmask: [B], int64

            # ------------------------------------------------------------------
            # Step 2: Run transformer with CFG for Phase 2.
            # The transformer receives [context | cond_lr | masked_hr_tokens].
            # _cfg_forward_phase2 handles the context concatenation internally.
            # ------------------------------------------------------------------
            cond_tokens_hr: torch.Tensor = self._cfg_forward_phase2(
                tokens_hr=tokens_hr,
                cond_lr=cond_lr,
                context=context,
                null_context=null_context,
                mask_hr=mask_hr,
                cfg_scale=cfg_scale,
            )
            # cond_tokens_hr: [B, 256, hidden_size]

            # ------------------------------------------------------------------
            # Step 3: Select which masked hr tokens to unmask at this step.
            # ------------------------------------------------------------------
            newly_unmasked: torch.Tensor = self._select_tokens_to_unmask(
                mask=mask_hr,
                n_to_unmask=n_to_unmask,
            )
            # newly_unmasked: BoolTensor [B, 256]

            # Skip diffusion sampling if no tokens to unmask.
            if not newly_unmasked.any():
                continue

            # ------------------------------------------------------------------
            # Step 4: Sample predicted hr tokens via DiT diffusion head.
            # The DiT head operates on ALL 256 positions simultaneously.
            # ------------------------------------------------------------------
            predicted_hr: torch.Tensor = self._sample_phase2(
                cond_tokens=cond_tokens_hr,
                n_diff_steps=diff_sample_steps,
            )
            # predicted_hr: [B, 256, latent_dim]

            # ------------------------------------------------------------------
            