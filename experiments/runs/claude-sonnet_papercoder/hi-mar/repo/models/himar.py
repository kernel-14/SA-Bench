```python
## models/himar.py
"""Top-level Hi-MAR model orchestrating the two-phase hierarchical masked autoregressive pipeline.

This module implements the complete Hi-MAR model described in the paper
"Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots".
It wires together the Scale-Aware Transformer backbone, MLP diffusion head
(Phase 1), and Diffusion Transformer head (Phase 2) into a unified nn.Module.

The critical design invariant throughout is that ``cond_tokens_lr`` from Phase 1
is always detached before Phase 2, ensuring training and inference behave
identically (avoiding the train/inference discrepancy described in Section 3.2).

Paper reference: Section 3, Figure 2(b).

Configuration alignment (config.yaml):
    active_model: himar_b
    models.himar_b.transformer.n_layers      = 24
    models.himar_b.transformer.hidden_size   = 768
    models.himar_b.transformer.n_heads       = 12
    models.himar_b.diff_head1.n_layers       = 6
    models.himar_b.diff_head1.hidden_size    = 1024
    models.himar_b.diff_head2.n_layers       = 6
    models.himar_b.diff_head2.hidden_size    = 512
    vae.latent_channels                      = 16
    diffusion.timesteps                      = 100
    resolution.lr_seq_len                    = 64
    resolution.hr_seq_len                    = 256
    inference.phase1_steps                   = 32
    inference.phase2_steps                   = 4
    inference.cfg.scale                      = 2.9
"""

import copy
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion_head_dit import DiTDiffusionHead
from models.diffusion_head_mlp import MLPDiffusionHead
from models.scale_aware_transformer import ScaleAwareTransformer, TransformerConfig
from training.losses import DiffusionUtils
from training.masking_schedule import MaskingSchedule


# ---------------------------------------------------------------------------
# HiMAR Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class HiMARConfig:
    """Flat configuration consumed by HiMAR.

    All fields have defaults matching the Hi-MAR-Base configuration from
    Table 1 of the paper. Values are sourced from config.yaml.

    Attributes:
        model_type: Model variant identifier. One of 'himar_b', 'himar_l',
            'himar_h', 'himar_s'. Config: active_model.
        n_layers: Number of ScaleAwareBlock layers in the backbone.
            Config: models.himar_b.transformer.n_layers = 24.
        hidden_size: Transformer hidden dimension D.
            Config: models.himar_b.transformer.hidden_size = 768.
        n_heads: Number of attention heads.
            Config: models.himar_b.transformer.n_heads = 12.
        mlp_ratio: FFN expansion ratio.
            Config: models.himar_b.transformer.mlp_ratio = 4.0.
        diff_head1_layers: Number of layers in Phase 1 MLP diffusion head.
            Config: models.himar_b.diff_head1.n_layers = 6.
        diff_head1_hidden: Hidden size of Phase 1 MLP diffusion head.
            Config: models.himar_b.diff_head1.hidden_size = 1024.
        diff_head2_layers: Number of layers in Phase 2 DiT diffusion head.
            Config: models.himar_b.diff_head2.n_layers = 6.
        diff_head2_hidden: Hidden size of Phase 2 DiT diffusion head.
            Config: models.himar_b.diff_head2.hidden_size = 512.
        n_classes: Number of ImageNet classes for class conditioning.
            Config: training_imagenet.n_classes = 1000.
            Index n_classes is the null class for CFG.
        latent_dim: VAE latent channel dimension.
            Config: vae.latent_channels = 16.
        lr_seq_len: Low-resolution token sequence length (Phase 1).
            Config: resolution.lr_seq_len = 64.
        hr_seq_len: High-resolution token sequence length (Phase 2).
            Config: resolution.hr_seq_len = 256.
        clip_dim: CLIP text embedding dimension for MS-COCO conditioning.
            Derived from openai/clip-vit-large-patch14 (768-dim).
        diff_timesteps: Number of diffusion timesteps.
            Config: diffusion.timesteps = 100.
        cfg_scale: Default classifier-free guidance scale.
            Config: inference.cfg.scale = 2.9.
        phase1_steps: Default number of Phase 1 AR steps at inference.
            Config: inference.phase1_steps = 32.
        phase2_steps: Default number of Phase 2 AR steps at inference.
            Config: inference.phase2_steps = 4.
        mask_ratio_min: Lower bound for Phase 1 uniform masking ratio.
            Config: training_imagenet.masking.phase1.ratio_min = 0.7.
        mask_ratio_max: Upper bound for Phase 1 uniform masking ratio.
            Config: training_imagenet.masking.phase1.ratio_max = 1.0.
        beta_alpha: Alpha parameter for Beta masking distribution (COCO).
            Config: training_coco.masking.phase1.beta_alpha = 4.0.
        beta_beta: Beta parameter for Beta masking distribution (COCO).
            Config: training_coco.masking.phase1.beta_beta = 1.0.
        phase1_masking_strategy: Masking strategy for Phase 1 training.
            'uniform' for ImageNet, 'beta' for COCO.
        use_ema_for_inference: Whether to use EMA transformer for generation.
    """

    model_type: str = "himar_b"
    n_layers: int = 24
    hidden_size: int = 768
    n_heads: int = 12
    mlp_ratio: float = 4.0
    diff_head1_layers: int = 6
    diff_head1_hidden: int = 1024
    diff_head2_layers: int = 6
    diff_head2_hidden: int = 512
    n_classes: int = 1000
    latent_dim: int = 16
    lr_seq_len: int = 64
    hr_seq_len: int = 256
    clip_dim: int = 768
    diff_timesteps: int = 100
    cfg_scale: float = 2.9
    phase1_steps: int = 32
    phase2_steps: int = 4
    mask_ratio_min: float = 0.7
    mask_ratio_max: float = 1.0
    beta_alpha: float = 4.0
    beta_beta: float = 1.0
    phase1_masking_strategy: str = "uniform"  # 'uniform' or 'beta'
    use_ema_for_inference: bool = True


# ---------------------------------------------------------------------------
# HiMAR
# ---------------------------------------------------------------------------


class HiMAR(nn.Module):
    """Hierarchical Masked Autoregressive Model.

    Orchestrates the full two-phase pipeline:
    - Phase 1: Masked autoregressive modeling over 64 low-resolution tokens
      using the Scale-Aware Transformer (scale_id=0) + MLP diffusion head.
      Produces conditional tokens Z^s that capture global structure.
    - Phase 2: Masked autoregressive modeling over 256 high-resolution tokens
      using the same Transformer (scale_id=1) conditioned on Z^s + DiT head.
      Produces the final high-resolution token predictions.

    The critical design invariant: Z^s (cond_tokens_lr) is always detached
    before Phase 2, ensuring training and inference behave identically.

    Paper reference: Section 3, Figure 2(b).

    Attributes:
        config: HiMARConfig instance with all hyperparameters.
        transformer: Shared Scale-Aware Transformer backbone.
        diff_head1: MLP-based diffusion head for Phase 1.
        diff_head2: Diffusion Transformer head for Phase 2.
        diff_utils: DDPM noise schedule utilities (not an nn.Module).
        ema_transformer: Exponential moving average copy of the transformer,
            used for inference. Updated by Trainer.update_ema().
    """

    def __init__(self, config: HiMARConfig) -> None:
        """Initialises Hi-MAR from a HiMARConfig.

        Instantiates all sub-components in dependency order and creates the
        EMA copy of the transformer backbone.

        Args:
            config: HiMARConfig instance. All fields have defaults matching
                Hi-MAR-Base. See HiMARConfig for field descriptions.
        """
        super().__init__()

        self.config: HiMARConfig = config

        # ------------------------------------------------------------------
        # 1. Scale-Aware Transformer backbone (shared across both phases).
        # The same weights are used for Phase 1 (scale_id=0) and Phase 2
        # (scale_id=1). The scale vector from AdaLN-Zero conditioning
        # distinguishes the two phases.
        # ------------------------------------------------------------------
        transformer_config: TransformerConfig = TransformerConfig(
            n_layers=config.n_layers,
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            mlp_ratio=config.mlp_ratio,
            n_classes=config.n_classes,
            latent_dim=config.latent_dim,
            lr_seq_len=config.lr_seq_len,
            hr_seq_len=config.hr_seq_len,
            clip_dim=config.clip_dim,
        )
        self.transformer: ScaleAwareTransformer = ScaleAwareTransformer(
            transformer_config
        )

        # ------------------------------------------------------------------
        # 2. Phase 1 MLP-based diffusion head.
        # Per-token independent denoising. Primarily optimises the transformer
        # to produce good conditional tokens Z^s (low-res pivots).
        # Config: models.himar_b.diff_head1.n_layers=6, hidden_size=1024.
        # cond_dim = backbone hidden_size (transformer output dimension).
        # ------------------------------------------------------------------
        self.diff_head1: MLPDiffusionHead = MLPDiffusionHead(
            n_layers=config.diff_head1_layers,
            hidden_size=config.diff_head1_hidden,
            input_dim=config.latent_dim,
            cond_dim=config.hidden_size,
            diff_timesteps=config.diff_timesteps,
        )

        # ------------------------------------------------------------------
        # 3. Phase 2 Diffusion Transformer head.
        # Self-attention across ALL 256 token positions for inter-token
        # dependency modeling. The key architectural innovation over MAR.
        # Config: models.himar_b.diff_head2.n_layers=6, hidden_size=512.
        # cond_dim = backbone hidden_size (transformer output dimension).
        # ------------------------------------------------------------------
        self.diff_head2: DiTDiffusionHead = DiTDiffusionHead(
            n_layers=config.diff_head2_layers,
            hidden_size=config.diff_head2_hidden,
            input_dim=config.latent_dim,
            cond_dim=config.hidden_size,
            diff_timesteps=config.diff_timesteps,
        )

        # ------------------------------------------------------------------
        # 4. Shared DDPM noise schedule utilities.
        # Not an nn.Module — holds no trainable parameters. Shared between
        # both diffusion heads to ensure consistent noise schedules.
        # Config: diffusion.timesteps=100, beta_start=0.0001, beta_end=0.02.
        # ------------------------------------------------------------------
        self.diff_utils: DiffusionUtils = DiffusionUtils(
            timesteps=config.diff_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
        )

        # ------------------------------------------------------------------
        # 5. EMA copy of the transformer backbone.
        # Used for inference (better generation quality than the online model).
        # Updated by Trainer.update_ema() during training.
        # All parameters are frozen (requires_grad=False) — EMA is not trained.
        # ------------------------------------------------------------------
        self.ema_transformer: ScaleAwareTransformer = copy.deepcopy(
            self.transformer
        )
        for param in self.ema_transformer.parameters():
            param.requires_grad_(False)

        # ------------------------------------------------------------------
        # 6. Null text embedding buffer for CFG in text-to-image generation.
        # Registered as a buffer so it moves with the model to the correct
        # device. Initialised to zeros; set by the caller before generation
        # via set_null_text_embed().
        # Shape: [1, 77, clip_dim] matching CLIP text encoder output.
        # ------------------------------------------------------------------
        self.register_buffer(
            "_null_text_embed",
            torch.zeros(1, 77, config.clip_dim),
            persistent=False,
        )

        # Flag indicating whether null text embed has been set by the caller.
        self._null_text_embed_set: bool = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_mask(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Zeros out masked token positions in latent space.

        Replaces masked positions with zeros in the raw VAE latent token
        space. The ScaleAwareTransformer.forward() then replaces these
        zeroed positions with the learnable mask_token embedding after its
        input projection (Linear(latent_dim, hidden_size)).

        This two-step approach keeps the masking logic clean: HiMAR zeros
        out masked latent tokens, and the transformer handles the actual
        mask token substitution in its own hidden space.

        Masking convention (Shared Knowledge):
            True = masked (token is hidden / to be predicted).
            False = unmasked (token is visible / already known).

        Args:
            tokens: Raw VAE latent tokens, shape ``[B, N, latent_dim]``.
                ``latent_dim = 16`` (KL-16 VAE).
            mask: Boolean tensor, shape ``[B, N]``. ``True`` at positions
                to be masked (zeroed out).

        Returns:
            Float tensor of shape ``[B, N, latent_dim]`` with masked
            positions set to zero. Non-destructive — does not modify
            the input ``tokens`` tensor.
        """
        # Clone to avoid modifying the input tensor in-place.
        masked_tokens: torch.Tensor = tokens.clone()

        # Zero out masked positions. mask is [B, N]; expand to [B, N, latent_dim]
        # for advanced indexing. Using boolean indexing directly:
        # masked_tokens[mask] selects all (b, n) pairs where mask[b, n] = True,
        # returning a 2D tensor of shape [M, latent_dim] where M = sum(mask).
        masked_tokens[mask] = 0.0

        return masked_tokens

    def _get_null_context(
        self,
        context: torch.Tensor,
        context_type: str = "class",
    ) -> torch.Tensor:
        """Constructs the null context for CFG unconditional forward pass.

        For class-conditional generation (ImageNet): the null context is the
        class embedding at index n_classes (the extra null class entry in
        self.transformer.class_embed).

        For text-conditional generation (COCO): the null context is the CLIP
        embedding of an empty string, stored in self._null_text_embed.

        Args:
            context: The conditioned context tensor, shape ``[B, C, hidden_size]``.
                Used to determine batch size and context length.
            context_type: Either 'class' (ImageNet) or 'text' (COCO).
                Determines which null embedding to use.

        Returns:
            Null context tensor of shape ``[B, C, hidden_size]``, matching
            the shape of the input ``context``.
        """
        batch_size: int = context.shape[0]
        device: torch.device = context.device

        if context_type == "class":
            # Null class index = n_classes (the extra entry in class_embed).
            # class_embed has n_classes+1 entries; index n_classes is null.
            null_class_ids: torch.Tensor = torch.full(
                (batch_size,),
                fill_value=self.config.n_classes,
                dtype=torch.long,
                device=device,
            )
            # encode_class_context returns [B, 1, hidden_size].
            null_ctx: torch.Tensor = self.transformer.encode_class_context(
                null_class_ids
            )
            return null_ctx

        elif context_type == "text":
            # Null text embedding: CLIP embedding of empty string.
            # _null_text_embed: [1, 77, clip_dim] → project to hidden_size.
            null_text: torch.Tensor = self._null_text_embed.to(device)
            # Expand to batch size: [1, 77, clip_dim] → [B, 77, clip_dim].
            null_text_expanded: torch.Tensor = null_text.expand(
                batch_size, -1, -1
            )
            # Project to hidden_size via text_proj.
            null_ctx = self.transformer.encode_text_context(null_text_expanded)
            return null_ctx

        else:
            raise ValueError(
                f"context_type must be 'class' or 'text', got '{context_type}'."
            )

    def _get_active_transformer(self) -> ScaleAwareTransformer:
        """Returns the transformer to use for the current operation.

        During training (self.training=True): returns self.transformer.
        During inference (self.training=False): returns self.ema_transformer
        if use_ema_for_inference=True, else self.transformer.

        Returns:
            The active ScaleAwareTransformer instance.
        """
        if not self.training and self.config.use_ema_for_inference:
            return self.ema_transformer
        return self.transformer

    # ------------------------------------------------------------------
    # Public API: Phase-specific forward passes
    # ------------------------------------------------------------------

    def forward_phase1(
        self,
        tokens_lr: torch.Tensor,
        context: torch.Tensor,
        mask_lr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Phase 1 forward pass: low-resolution masked autoregressive modeling.

        Processes 64 low-resolution tokens through the Scale-Aware Transformer
        (scale_id=0) and computes the MLP diffusion head loss on masked positions.
        Returns the full conditional token sequence Z^s for use as Phase 2 pivots.

        Paper reference (Section 3.2):
            "the first phase performs bidirectional autoregressive modeling over
            low-resolution visual tokens to capture the global structure."

        Args:
            tokens_lr: Ground-truth low-resolution latent tokens from the VAE,
                shape ``[B, 64, latent_dim]``. These are the Phase 1 training
                targets (what the diffusion head learns to reconstruct).
            context: Pre-projected context tokens in hidden_size space,
                shape ``[B, C, hidden_size]``. For ImageNet: class embeddings
                ``[B, 1, hidden_size]``. For COCO: projected text embeddings
                ``[B, 77, hidden_size]``.
            mask_lr: Boolean mask, shape ``[B, 64]``. ``True`` at positions
                that are masked (hidden). Sampled from uniform or Beta
                distribution during training.

        Returns:
            Tuple of:
                - ``cond_tokens_lr``: Conditional tokens Z^s from the
                  transformer, shape ``[B, 64, hidden_size]``. ALL positions
                  are returned (not just masked), since Phase 2 uses all 64
                  pivots. The caller (compute_loss) must detach this before
                  passing to forward_phase2.
                - ``loss1``: Scalar Phase 1 diffusion loss (MSE over masked
                  positions only). Differentiable w.r.t. transformer and
                  diff_head1 parameters.
        """
        # ------------------------------------------------------------------
        # Step 1: Apply mask — zero out masked positions in latent space.
        # The transformer's forward() replaces these zeros with mask_token
        # after its input projection.
        # ------------------------------------------------------------------
        masked_lr: torch.Tensor = self._apply_mask(tokens_lr, mask_lr)
        # masked_lr: [B, 64, latent_dim]

        # ------------------------------------------------------------------
        # Step 2: Run transformer (Phase 1, scale_id=0).
        # Input: masked low-res tokens + context (class/text).
        # Output: conditional tokens Z^s for all 64 positions.
        # ------------------------------------------------------------------
        cond_tokens_lr: torch.Tensor = self.transformer.forward(
            tokens=masked_lr,
            context=context,
            scale_id=0,
            mask=mask_lr,
        )
        # cond_tokens_lr: [B, 64, hidden_size]

        # ------------------------------------------------------------------
        # Step 3: Compute Phase 1 diffusion loss on masked positions only.
        # Extract masked positions for the per-token MLP head.
        # The MLP head processes tokens independently, so we can safely
        # flatten and select masked positions.
        # ------------------------------------------------------------------
        batch_size: int = tokens_lr.shape[0]
        n_lr: int = tokens_lr.shape[1]

        # Flatten batch and sequence dims for masked position selection.
        # mask_lr: [B, 64] → [B*64] for boolean indexing.
        mask_flat: torch.Tensor = mask_lr.view(batch_size * n_lr)  # [B*64]

        # Select masked conditional tokens: [B*64, hidden_size] → [M, hidden_size]
        cond_flat: torch.Tensor = cond_tokens_lr.view(
            batch_size * n_lr, self.config.hidden_size
        )[mask_flat]  # [M, hidden_size]

        # Select masked target tokens: [B*64, latent_dim] → [M, latent_dim]
        target_flat: torch.Tensor = tokens_lr.view(
            batch_size * n_lr, self.config.latent_dim
        )[mask_flat]  # [M, latent_dim]

        # Reshape to [1, M, *] for the diffusion head's expected [B, N, *] format.
        # The head processes all M masked tokens as a single "batch" of 1.
        cond_for_head: torch.Tensor = cond_flat.unsqueeze(0)    # [1, M, hidden_size]
        target_for_head: torch.Tensor = target_flat.unsqueeze(0)  # [1, M, latent_dim]

        # Compute Phase 1 loss. If no tokens are masked (edge case), return 0.
        if cond_flat.shape[0] == 0:
            loss1: torch.Tensor = torch.tensor(
                0.0, device=tokens_lr.device, requires_grad=True
            )
        else:
            loss1 = self.diff_head1.compute_loss(
                cond=cond_for_head,
                x_target=target_for_head,
                diff_utils=self.diff_utils,
            )

        return cond_tokens_lr, loss1

    def forward_phase2(
        self,
        tokens_hr: torch.Tensor,
        cond_tokens_lr: torch.Tensor,
        context: torch.Tensor,
        mask_hr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Phase 2 forward pass: high-resolution masked autoregressive modeling.

        Processes 256 high-resolution tokens through the Scale-Aware Transformer
        (scale_id=1) conditioned on Phase 1 conditional tokens (pivots) and
        computes the DiT diffusion head loss. The DiT head operates on ALL 256
        positions simultaneously (enabling inter-token self-attention), but the
        loss is computed only on masked positions.

        Paper reference (Section 3.2):
            "the Transformer takes the concatenation of context tokens, small
            scale conditional tokens and the masked dense visual tokens as input
            to generate dense conditional tokens."

        Critical: ``cond_tokens_lr`` MUST be detached by the caller before
        passing to this method. This prevents Phase 2 gradients from flowing
        back into Phase 1, matching the inference behavior where Phase 1 runs
        independently first.

        Args:
            tokens_hr: Ground-truth high-resolution latent tokens from the VAE,
                shape ``[B, 256, latent_dim]``. Phase 2 training targets.
            cond_tokens_lr: Conditional tokens Z^s from Phase 1 (detached),
                shape ``[B, 64, hidden_size]``. These are the low-resolution
                pivots that provide global structural guidance to Phase 2.
                Must be in hidden_size space (transformer output, not raw latents).
            context: Pre-projected context tokens in hidden_size space,
                shape ``[B, C, hidden_size]``. Same context as Phase 1.
            mask_hr: Boolean mask, shape ``[B, 256]``. ``True`` at positions
                that are masked (hidden). Sampled from cosine distribution.

        Returns:
            Tuple of:
                - ``cond_tokens_hr``: Conditional tokens Z^l from the
                  transformer, shape ``[B, 256, hidden_size]``. ALL positions
                  are returned (not just masked), since the DiT head needs
                  all positions for self-attention.
                - ``loss2``: Scalar Phase 2 diffusion loss (MSE over masked
                  positions only, but forward pass over all positions).
                  Differentiable w.r.t. transformer and diff_head2 parameters.
        """
        # ------------------------------------------------------------------
        # Step 1: Apply mask — zero out masked positions in latent space.
        # ------------------------------------------------------------------
        masked_hr: torch.Tensor = self._apply_mask(tokens_hr, mask_hr)
        # masked_hr: [B, 256, latent_dim]

        # ------------------------------------------------------------------
        # Step 2: Build Phase 2 context.
        # Paper: "the Transformer takes the concatenation of context tokens,
        # small scale conditional tokens and the masked dense visual tokens."
        # → context_phase2 = [original_context | cond_tokens_lr]
        # Both are in hidden_size space, so direct concatenation is valid.
        # cond_tokens_lr: [B, 64, hidden_size] (Phase 1 pivots)
        # context: [B, C, hidden_size] (class/text conditioning)
        # context_phase2: [B, C+64, hidden_size]
        # ------------------------------------------------------------------
        context_phase2: torch.Tensor = torch.cat(
            [context, cond_tokens_lr], dim=1
        )
        # context_phase2: [B, C+64, hidden_size]

        # ------------------------------------------------------------------
        # Step 3: Run transformer (Phase 2, scale_id=1).
        # Input: masked high-res tokens + extended context (class/text + pivots).
        # Output: conditional tokens Z^l for all 256 positions.
        # ------------------------------------------------------------------
        cond_tokens_hr: torch.Tensor = self.transformer.forward(
            tokens=masked_hr,
            context=context_phase2,
            scale_id=1,
            mask=mask_hr,
        )
        # cond_tokens_hr: [B, 256, hidden_size]

        # ------------------------------------------------------------------
        # Step 4: Compute Phase 2 diffusion loss.
        # The DiT head forward pass runs over ALL 256 positions (enabling
        # global self-attention), but the MSE loss is computed only on
        # masked positions (mask_hr=True).
        # ------------------------------------------------------------------
        loss2: torch.Tensor = self.diff_head2.compute_loss(
            cond=cond_tokens_hr,
            x_target=tokens_hr,
            mask=mask_hr,
            diff_utils=self.diff_utils,
        )

        return cond_tokens_hr, loss2

    def compute_loss(
        self,
        tokens_lr: torch.Tensor,
        tokens_hr: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Full training forward pass computing the combined two-phase loss.

        Called by Trainer.train_step() for each training batch. Samples masks
        for both phases, runs Phase 1 and Phase 2 forward passes, and returns
        the sum of both losses.

        The critical detach boundary is enforced here: cond_tokens_lr is
        detached after Phase 1 before being passed to Phase 2. This ensures
        Phase 2 gradients do not flow back into Phase 1, matching inference
        behavior where Phase 1 runs independently first.

        Paper reference (Section 3.2):
            "To mitigate such training-inference discrepancy, we take the
            conditional tokens output from the Hi-MAR Transformer of
            low-resolution visual tokens for the second phase instead."

        Masking strategies (config.yaml):
            Phase 1 (ImageNet): uniform ratio in [0.7, 1.0]
            Phase 1 (COCO): Beta(alpha=4, beta=1)
            Phase 2 (both): cosine schedule following MaskGIT

        Args:
            tokens_lr: Low-resolution latent tokens from VAETokenizer,
                shape ``[B, 64, latent_dim]``. Phase 1 training targets.
            tokens_hr: High-resolution latent tokens from VAETokenizer,
                shape ``[B, 256, latent_dim]``. Phase 2 training targets.
            context: Pre-projected context tokens in hidden_size space,
                shape ``[B, C, hidden_size]``. For ImageNet: class embeddings
                ``[B, 1, hidden_size]``. For COCO: projected text embeddings
                ``[B, 77, hidden_size]``.

        Returns:
            Scalar total loss ``loss1 + loss2``. Differentiable w.r.t. all
            trainable parameters (transformer, diff_head1, diff_head2).
        """
        batch_size: int = tokens_lr.shape[0]
        device: torch.device = tokens_lr.device

        # ------------------------------------------------------------------
        # Step 1: Sample Phase 1 masks.
        # Strategy depends on the training task (ImageNet vs. COCO).
        # Config: training_imagenet.masking.phase1.strategy = 'uniform'
        #         training_coco.masking.phase1.strategy = 'beta'
        # ------------------------------------------------------------------
        if self.config.phase1_masking_strategy == "uniform":
            # ImageNet: uniform ratio in [mask_ratio_min, mask_ratio_max].
            # Config: ratio_min=0.7, ratio_max=1.0.
            ratio_lr: torch.Tensor = MaskingSchedule.sample_uniform_ratio(
                batch_size=batch_size,
                r_min=self.config.mask_ratio_min,
                r_max=self.config.mask_ratio_max,
            ).to(device)
        elif self.config.phase1_masking_strategy == "beta":
            # COCO: Beta(alpha=4, beta=1) following AutoNAT-L.
            # Config: beta_alpha=4.0, beta_beta=1.0.
            ratio_lr = MaskingSchedule.sample_beta_ratio(
                batch_size=batch_size,
                alpha=self.config.beta_alpha