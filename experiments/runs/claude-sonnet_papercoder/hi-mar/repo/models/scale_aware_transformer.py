## Code: models/scale_aware_transformer.py

```python
## models/scale_aware_transformer.py
"""Scale-aware Transformer backbone for Hi-MAR.

This module implements the shared Transformer backbone used in both Phase 1
(low-resolution, 64 tokens) and Phase 2 (high-resolution, 256 tokens) of
Hi-MAR. The two phases are distinguished by a ``scale_id`` (0 or 1) that
drives AdaLN-Zero conditioning, allowing a single set of weights to behave
differently at each resolution level.

Two public classes are provided:

1. ``ScaleAwareBlock`` — a single Transformer block with AdaLN-Zero scale
   conditioning, implementing the equations from Section 3.2 of the paper:
       ṽ = a·v + b
       [α₁, β₁, γ₁, α₂, β₂, γ₂] = split(ṽ)
       z_a = z^i + γ₁ · Attention(α₁ · LN(z^i) + β₁)
       z^{i+1} = z_a + γ₂ · FFN(α₂ · LN(z_a) + β₂)

2. ``ScaleAwareTransformer`` — the full backbone stacking ``n_layers``
   ``ScaleAwareBlock`` instances, handling input projection, masking,
   positional embeddings, context prepending, and output stripping.

Configuration alignment (config.yaml):
    models.himar_b.transformer.n_layers      = 24
    models.himar_b.transformer.hidden_size   = 768
    models.himar_b.transformer.n_heads       = 12
    models.himar_b.transformer.mlp_ratio     = 4.0
    training_imagenet.n_classes              = 1000
    vae.latent_channels                      = 16
    resolution.lr_seq_len                    = 64
    resolution.hr_seq_len                    = 256

Paper reference: Section 3.2, Figure 2(b) and 2(c).
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config dataclass consumed by ScaleAwareTransformer
# ---------------------------------------------------------------------------


@dataclass
class TransformerConfig:
    """Flat configuration consumed by ScaleAwareTransformer.

    This dataclass mirrors the fields extracted from config.yaml by the
    top-level ``Config.from_yaml()`` method. Default values match the
    Hi-MAR-Base configuration from Table 1 of the paper.

    Attributes:
        n_layers: Number of ScaleAwareBlock layers in the backbone.
            Paper Table 1: 24 (Base), 32 (Large), 40 (Huge), 16 (Small).
        hidden_size: Transformer hidden dimension D.
            Paper Table 1: 768 (Base), 1024 (Large), 1280 (Huge), 512 (Small).
        n_heads: Number of attention heads.
            Config: 12 (Base), 16 (Large), 20 (Huge), 8 (Small).
        mlp_ratio: FFN expansion ratio. Config: 4.0 for all variants.
        n_classes: Number of ImageNet classes for class conditioning.
            Config training_imagenet.n_classes = 1000.
            The class embedding has n_classes+1 entries; index n_classes is
            the null class used for CFG unconditional passes.
        latent_dim: VAE latent channel dimension.
            Config vae.latent_channels = 16 (KL-16).
        lr_seq_len: Low-resolution token sequence length (Phase 1).
            Config resolution.lr_seq_len = 64 (8×8 latent from 128×128 image).
        hr_seq_len: High-resolution token sequence length (Phase 2).
            Config resolution.hr_seq_len = 256 (16×16 latent from 256×256 image).
        clip_dim: CLIP text embedding dimension for MS-COCO conditioning.
            Derived from config text_encoder.model = openai/clip-vit-large-patch14
            which produces 768-dimensional embeddings.
    """

    n_layers: int = 24
    hidden_size: int = 768
    n_heads: int = 12
    mlp_ratio: float = 4.0
    n_classes: int = 1000
    latent_dim: int = 16
    lr_seq_len: int = 64
    hr_seq_len: int = 256
    clip_dim: int = 768  # openai/clip-vit-large-patch14 output dimension


# ---------------------------------------------------------------------------
# ScaleAwareBlock
# ---------------------------------------------------------------------------


class ScaleAwareBlock(nn.Module):
    """Single Transformer block with AdaLN-Zero scale conditioning.

    Implements the scale-aware block described in Section 3.2 of the paper
    and illustrated in Figure 2(c). The block replaces standard LayerNorm
    with AdaLN-Zero: a scale vector ``v`` (derived from the resolution scale
    ID) is linearly projected to produce six modulation parameters
    ``[α₁, β₁, γ₁, α₂, β₂, γ₂]`` that control the LayerNorm scale/shift
    and the residual gate for both the attention and FFN sub-layers.

    The AdaLN-Zero trick (from DiT, Peebles & Xie 2023) zero-initialises the
    ``adaLN_modulation`` linear layer so that at the start of training all
    residual gates ``γ₁, γ₂ = 0``, making each block an identity function.
    This provides stable gradient flow at initialisation.

    Paper equations (Section 3.2):
        ṽ = a·v + b                                  (adaLN_modulation)
        [α₁, β₁, γ₁, α₂, β₂, γ₂] = split(ṽ)
        z_a = z^i + γ₁ · Attention(α₁ · LN(z^i) + β₁)
        z^{i+1} = z_a + γ₂ · FFN(α₂ · LN(z_a) + β₂)

    Attributes:
        norm1: LayerNorm without learnable affine parameters (AdaLN handles it).
        norm2: Same as norm1 for the FFN sub-layer.
        attn: Multi-head self-attention with batch_first=True.
        ffn: Two-layer FFN with GELU activation.
        adaLN_modulation: Linear layer producing 6×hidden_size modulation
            parameters from the scale vector. Zero-initialised.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
    ) -> None:
        """Initialises the ScaleAwareBlock.

        Args:
            hidden_size: Transformer hidden dimension D. Must be divisible by
                ``n_heads``.
            n_heads: Number of attention heads. Must divide ``hidden_size``
                evenly.
            mlp_ratio: FFN hidden dimension multiplier. The FFN intermediate
                dimension is ``int(hidden_size * mlp_ratio)``. Config: 4.0.
        """
        super().__init__()

        self.hidden_size: int = hidden_size
        self.n_heads: int = n_heads
        self.mlp_ratio: float = mlp_ratio

        # ------------------------------------------------------------------
        # LayerNorm without learnable affine parameters.
        # AdaLN provides scale and shift externally via the modulation params.
        # eps=1e-6 matches DiT convention for numerical stability.
        # ------------------------------------------------------------------
        self.norm1: nn.LayerNorm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.norm2: nn.LayerNorm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )

        # ------------------------------------------------------------------
        # Multi-head self-attention.
        # batch_first=True: input/output shape is [B, N, D] throughout,
        # avoiding transposes. Bidirectional (no causal mask) per BERT-style
        # masked autoregressive modelling.
        # ------------------------------------------------------------------
        self.attn: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.0,
        )

        # ------------------------------------------------------------------
        # Two-layer FFN with GELU activation.
        # Intermediate dimension: int(hidden_size * mlp_ratio).
        # ------------------------------------------------------------------
        ffn_hidden: int = int(hidden_size * mlp_ratio)
        self.ffn: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_size),
        )

        # ------------------------------------------------------------------
        # AdaLN-Zero modulation layer.
        # Maps scale vector [B, D] → [B, 6*D] producing:
        #   [α₁, β₁, γ₁, α₂, β₂, γ₂], each of shape [B, D].
        # Zero-initialised so all residual gates γ start at 0 (identity init).
        # ------------------------------------------------------------------
        self.adaLN_modulation: nn.Linear = nn.Linear(
            hidden_size, 6 * hidden_size, bias=True
        )
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(
        self,
        x: torch.Tensor,
        scale_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the scale-aware Transformer block.

        Args:
            x: Input token sequence, shape ``[B, N, D]``. ``N`` is the
                combined sequence length (context + image tokens).
            scale_vec: Scale conditioning vector, shape ``[B, D]``. Produced
                by ``ScaleAwareTransformer.get_scale_vector()``. The same
                vector is used for all token positions and all blocks in a
                forward pass.

        Returns:
            Output token sequence, shape ``[B, N, D]``, on the same device
            as ``x``.
        """
        # ------------------------------------------------------------------
        # Step 1: Compute 6 modulation parameters from the scale vector.
        # adaLN_modulation: [B, D] → [B, 6*D]
        # chunk(6, dim=-1): 6 tensors of shape [B, D]
        # unsqueeze(1): [B, D] → [B, 1, D] for broadcasting over N tokens.
        # ------------------------------------------------------------------
        modulation: torch.Tensor = self.adaLN_modulation(scale_vec)  # [B, 6*D]
        chunks = modulation.chunk(6, dim=-1)  # 6 × [B, D]
        alpha1: torch.Tensor = chunks[0].unsqueeze(1)  # [B, 1, D]
        beta1: torch.Tensor = chunks[1].unsqueeze(1)   # [B, 1, D]
        gamma1: torch.Tensor = chunks[2].unsqueeze(1)  # [B, 1, D]
        alpha2: torch.Tensor = chunks[3].unsqueeze(1)  # [B, 1, D]
        beta2: torch.Tensor = chunks[4].unsqueeze(1)   # [B, 1, D]
        gamma2: torch.Tensor = chunks[5].unsqueeze(1)  # [B, 1, D]

        # ------------------------------------------------------------------
        # Step 2: Attention sub-layer with AdaLN-Zero.
        # Paper: z_a = z^i + γ₁ · Attention(α₁ · LN(z^i) + β₁)
        # ------------------------------------------------------------------
        normed1: torch.Tensor = self.norm1(x)                    # [B, N, D]
        modulated1: torch.Tensor = alpha1 * normed1 + beta1      # [B, N, D]
        # Self-attention: Q = K = V = modulated1 (bidirectional, no mask).
        attn_out: torch.Tensor
        attn_out, _ = self.attn(modulated1, modulated1, modulated1)  # [B, N, D]
        z_a: torch.Tensor = x + gamma1 * attn_out                # [B, N, D]

        # ------------------------------------------------------------------
        # Step 3: FFN sub-layer with AdaLN-Zero.
        # Paper: z^{i+1} = z_a + γ₂ · FFN(α₂ · LN(z_a) + β₂)
        # ------------------------------------------------------------------
        normed2: torch.Tensor = self.norm2(z_a)                  # [B, N, D]
        modulated2: torch.Tensor = alpha2 * normed2 + beta2      # [B, N, D]
        ffn_out: torch.Tensor = self.ffn(modulated2)             # [B, N, D]
        z_out: torch.Tensor = z_a + gamma2 * ffn_out             # [B, N, D]

        return z_out


# ---------------------------------------------------------------------------
# ScaleAwareTransformer
# ---------------------------------------------------------------------------


class ScaleAwareTransformer(nn.Module):
    """Full scale-aware Transformer backbone shared across both Hi-MAR phases.

    This module is the central component of Hi-MAR. It is called twice per
    training step:
    - Phase 1 (``scale_id=0``): processes 64 low-resolution tokens with class
      or text context, producing conditional tokens ``Z^s`` used as pivots.
    - Phase 2 (``scale_id=1``): processes 256 high-resolution tokens with
      class/text context AND the Phase 1 conditional tokens as additional
      context, producing conditional tokens ``Z^l`` for the DiT diffusion head.

    The same weights are used in both phases; the ``scale_id`` drives the
    AdaLN-Zero conditioning that makes the blocks behave differently at each
    resolution level.

    Architecture overview (Figure 2b, 2c):
        tokens [B,N,16] → input_proj → [B,N,H]
                        → replace masked positions with mask_token
                        → add positional embedding (lr or hr)
        context [B,C,H] → prepend → [B,C+N,H]
                        → N × ScaleAwareBlock(scale_vec)
                        → LayerNorm
                        → strip context prefix → [B,N,H]

    Attributes:
        blocks: ModuleList of ``n_layers`` ScaleAwareBlock instances.
        scale_embed: Embedding mapping scale_id ∈ {0,1} to hidden_size.
        scale_mlp: MLP expanding scale embedding to rich modulation signal.
        pos_embed_lr: Learnable positional embedding for 64 low-res positions.
        pos_embed_hr: Learnable positional embedding for 256 high-res positions.
        input_proj: Linear projection from latent_dim (16) to hidden_size.
        class_embed: Embedding for ImageNet class conditioning (n_classes+1
            entries; index n_classes is the null class for CFG).
        text_proj: Linear projection from CLIP dim (768) to hidden_size.
        mask_token: Learnable mask token embedding, shape [1, 1, hidden_size].
        norm: Final LayerNorm applied after all blocks.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialises the ScaleAwareTransformer from a TransformerConfig.

        Args:
            config: Flat configuration dataclass. All fields have defaults
                matching Hi-MAR-Base. See ``TransformerConfig`` for field
                descriptions and config.yaml alignment.
        """
        super().__init__()

        self.config: TransformerConfig = config
        hidden_size: int = config.hidden_size
        n_heads: int = config.n_heads
        mlp_ratio: float = config.mlp_ratio
        n_layers: int = config.n_layers
        latent_dim: int = config.latent_dim
        lr_seq_len: int = config.lr_seq_len
        hr_seq_len: int = config.hr_seq_len
        n_classes: int = config.n_classes
        clip_dim: int = config.clip_dim

        # ------------------------------------------------------------------
        # 1. Transformer blocks: n_layers ScaleAwareBlock instances.
        # ------------------------------------------------------------------
        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                ScaleAwareBlock(hidden_size, n_heads, mlp_ratio)
                for _ in range(n_layers)
            ]
        )

        # ------------------------------------------------------------------
        # 2. Scale conditioning.
        # scale_embed: maps scale_id ∈ {0, 1} → [B, hidden_size]
        # scale_mlp: expands embedding for richer modulation capacity.
        #   Architecture: Linear → SiLU → Linear (following DiT convention).
        # ------------------------------------------------------------------
        self.scale_embed: nn.Embedding = nn.Embedding(2, hidden_size)
        self.scale_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # ------------------------------------------------------------------
        # 3. Learnable positional embeddings.
        # Separate embeddings for low-res (64 positions) and high-res (256).
        # Initialised with small normal noise so positions are distinguishable
        # from the start of training (std=0.02 follows ViT convention).
        # ------------------------------------------------------------------
        self.pos_embed_lr: nn.Parameter = nn.Parameter(
            torch.zeros(1, lr_seq_len, hidden_size)
        )
        self.pos_embed_hr: nn.Parameter = nn.Parameter(
            torch.zeros(1, hr_seq_len, hidden_size)
        )
        nn.init.normal_(self.pos_embed_lr, std=0.02)
        nn.init.normal_(self.pos_embed_hr, std=0.02)

        # ------------------------------------------------------------------
        # 4. Input projection: latent_dim (16) → hidden_size.
        # Applied to raw VAE latent tokens before masking and pos embedding.
        # NOT applied to context tokens (class/text/pivot), which are already
        # in hidden_size space when passed to forward().
        # ------------------------------------------------------------------
        self.input_proj: nn.Linear = nn.Linear(latent_dim, hidden_size)

        # ------------------------------------------------------------------
        # 5. Class conditioning for ImageNet (class-conditional generation).
        # n_classes + 1 entries: indices 0…999 are real classes, index 1000
        # is the null class used for CFG unconditional passes.
        # ------------------------------------------------------------------
        self.class_embed: nn.Embedding = nn.Embedding(n_classes + 1, hidden_size)

        # ------------------------------------------------------------------
        # 6. Text conditioning for MS-COCO (text-to-image generation).
        # Projects CLIP ViT-L/14 embeddings (768-dim) to hidden_size.
        # ------------------------------------------------------------------
        self.text_proj: nn.Linear = nn.Linear(clip_dim, hidden_size)

        # ------------------------------------------------------------------
        # 7. Learnable mask token.
        # A single [1, 1, hidden_size] parameter broadcast to all masked
        # positions. Initialised to zeros; learned during training.
        # ------------------------------------------------------------------
        self.mask_token: nn.Parameter = nn.Parameter(
            torch.zeros(1, 1, hidden_size)
        )

        # ------------------------------------------------------------------
        # 8. Final LayerNorm applied after all blocks.
        # ------------------------------------------------------------------
        self.norm: nn.LayerNorm = nn.LayerNorm(hidden_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_scale_vector(
        self,
        scale_id: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Computes the scale conditioning vector for a given resolution phase.

        The scale vector is a global conditioning signal (same for all token
        positions) that drives the AdaLN-Zero modulation in every
        ``ScaleAwareBlock``. It encodes which resolution phase is being
        processed (0 = low-res Phase 1, 1 = high-res Phase 2).

        Paper reference (Section 3.2):
            "we first represent the scale information by sinusoidal embedding.
            The sinusoidal embedding is fed into MLP layers to generate scale
            vector v."
            Note: the paper mentions sinusoidal embedding but the implementation
            uses a learnable nn.Embedding for simplicity and trainability,
            which is equivalent in practice.

        Args:
            scale_id: Integer scale identifier. 0 = low-resolution (Phase 1,
                64 tokens), 1 = high-resolution (Phase 2, 256 tokens).
            batch_size: Number of samples in the current batch. Used to
                construct the index tensor for the embedding lookup.
            device: Target device for the output tensor. Must match the device
                of the model parameters.

        Returns:
            Float tensor of shape ``[B, hidden_size]`` where ``B = batch_size``.
            This tensor is passed to every ``ScaleAwareBlock.forward()`` call
            in the current forward pass.
        """
        # Create a batch of identical scale_id indices: [B].
        scale_id_tensor: torch.Tensor = torch.full(
            (batch_size,),
            fill_value=scale_id,
            dtype=torch.long,
            device=device,
        )

        # Embedding lookup: [B] → [B, hidden_size].
        scale_emb: torch.Tensor = self.scale_embed(scale_id_tensor)

        # MLP expansion for richer modulation capacity: [B, H] → [B, H].
        scale_vec: torch.Tensor = self.scale_mlp(scale_emb)

        return scale_vec

    def encode_class_context(
        self,
        class_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Encodes integer class IDs into context token sequences.

        Convenience method for ImageNet class-conditional generation. The
        class embedding is looked up and unsqueezed to produce a single-token
        context sequence ``[B, 1, hidden_size]``.

        For CFG, pass ``class_ids = torch.full((B,), n_classes)`` to get the
        null class embedding (index ``n_classes = 1000``).

        Args:
            class_ids: Integer tensor of shape ``[B]`` with values in
                ``{0, …, n_classes}`` (inclusive). Index ``n_classes`` is the
                null class for CFG.

        Returns:
            Float tensor of shape ``[B, 1, hidden_size]``.
        """
        # class_embed: [B] → [B, hidden_size] → [B, 1, hidden_size]
        return self.class_embed(class_ids).unsqueeze(1)

    def encode_text_context(
        self,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Projects CLIP text embeddings into the Transformer hidden space.

        Convenience method for MS-COCO text-to-image generation. The CLIP
        ViT-L/14 embeddings (768-dim, sequence of 77 tokens) are projected
        to ``hidden_size`` via ``text_proj``.

        Args:
            text_embeddings: Float tensor of shape ``[B, 77, 768]`` from the
                frozen CLIP text encoder (``openai/clip-vit-large-patch14``).

        Returns:
            Float tensor of shape ``[B, 77, hidden_size]``.
        """
        return self.text_proj(text_embeddings)

    def forward(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        scale_id: int,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the scale-aware Transformer backbone.

        Processes a sequence of image tokens (with masked positions replaced
        by the learnable mask token) conditioned on context tokens and the
        resolution scale. Returns conditional tokens for the diffusion head.

        Input contract:
            - ``tokens``: raw VAE latent tokens, shape ``[B, N, latent_dim]``.
              N = 64 for Phase 1 (scale_id=0), N = 256 for Phase 2 (scale_id=1).
            - ``context``: pre-projected context tokens in hidden_size space,
              shape ``[B, C, hidden_size]``. The caller is responsible for
              projecting class or text embeddings to hidden_size before calling
              forward (use ``encode_class_context`` or ``encode_text_context``).
              For Phase 2, context should include the Phase 1 conditional tokens
              (pivots) concatenated with class/text tokens:
              ``context = torch.cat([class_ctx, cond_tokens_lr], dim=1)``.
            - ``mask``: boolean tensor ``[B, N]``, ``True = masked``.
            - ``scale_id``: 0 for Phase 1, 1 for Phase 2.

        Processing pipeline:
            (a) Project tokens: [B, N, latent_dim] → [B, N, hidden_size]
            (b) Replace masked positions with mask_token
            (c) Add positional embedding (pos_embed_lr or pos_embed_hr)
            (d) Prepend context tokens: [B, C+N, hidden_size]
            (e) Compute scale vector: [B, hidden_size]
            (f) Pass through all ScaleAwareBlocks
            (g) Apply final LayerNorm
            (h) Strip context prefix, return [B, N, hidden_size]

        Args:
            tokens: Raw VAE latent tokens, shape ``[B, N, latent_dim]``.
                ``latent_dim = 16`` (KL-16 VAE, config vae.latent_channels).
            context: Pre-projected context tokens in hidden_size space,
                shape ``[B, C, hidden_size]``. ``C`` is the context sequence
                length (1 for class conditioning, 77 for text conditioning,
                or 1+64=65 / 77+64=141 for Phase 2 with pivots).
            scale_id: Integer scale identifier. 0 = Phase 1 (low-res),
                1 = Phase 2 (high-res).
            mask: Boolean tensor ``[B, N]``. ``True`` at positions that are
                masked (hidden). Masked positions are replaced by mask_token.

        Returns:
            Conditional tokens, shape ``[B, N, hidden_size]``. These are the
            ``Z^s`` (Phase 1) or ``Z^l`` (Phase 2) conditional tokens fed into
            the diffusion heads. Context tokens are stripped from the output.

        Raises:
            ValueError: If ``scale_id`` is not 0 or 1.
            ValueError: If the token sequence length does not match the
                expected length for the given ``scale_id``.
        """
        if scale_id not in (0, 1):
            raise ValueError(
                f"scale_id must be 0 (low-res) or 1 (high-res), got {scale_id}."
            )

        batch_size: int = tokens.shape[0]
        n_tokens: int = tokens.shape[1]
        device: torch.device = tokens.device

        # Validate sequence length against scale_id.
        expected_len: int = (
            self.config.lr_seq_len if scale_id == 0 else self.config.hr_seq_len
        )
        if n_tokens != expected_len:
            raise ValueError(
                f"Token sequence length {n_tokens} does not match expected "
                f"length {expected_len} for scale_id={scale_id}. "
                f"Phase 1 expects {self.config.lr_seq_len} tokens (low-res), "
                f"Phase 2 expects {self.config.hr_seq_len} tokens (high-res)."
            )

        # ------------------------------------------------------------------
        # Step (a): Project tokens from latent_dim to hidden_size.
        # input_proj: [B, N, latent_dim] → [B, N, hidden_size]
        # This projection is applied ONLY to the raw image tokens, not to
        # context tokens (which are already in hidden_size space).
        # ------------------------------------------------------------------
        x: torch.Tensor = self.input_proj(tokens)  # [B, N, hidden_size]

        # ------------------------------------------------------------------
        # Step (b): Replace masked positions with the learnable mask token.
        # mask: [B, N] (True = masked)
        # mask_token: [1, 1, hidden_size] → broadcast to [B, N, hidden_size]
        # ------------------------------------------------------------------
        mask_expanded: torch.Tensor = mask.unsqueeze(-1).expand_as(x)  # [B, N, H]
        mask_token_expanded: torch.Tensor = self.mask_token.expand_as(x)  # [B, N, H]
        x = torch.where(mask_expanded, mask_token_expanded, x)  # [B, N, H]

        # ------------------------------------------------------------------
        # Step (c): Add positional embeddings.
        # pos_embed_lr: [1, 64, H] for Phase 1 (scale_id=0)
        # pos_embed_hr: [1, 256, H] for Phase 2 (scale_id=1)
        # Broadcasting over batch dimension is automatic.
        # Positional embedding is added BEFORE context prepending so that
        # only image tokens receive positional information.
        # ------------------------------------------------------------------
        if scale_id == 0:
            x = x + self.pos_embed_lr  # [B, 64, H]
        else:
            x = x + self.pos_embed_hr  # [B, 256, H]

        # ------------------------------------------------------------------
        # Step (d): Prepend context tokens.
        # context: [B, C, hidden_size] (pre-projected by caller)
        # x: [B, N, hidden_size]
        # After cat: [B, C+N, hidden_size]
        # ------------------------------------------------------------------
        context_len: int = context.shape[1]
        x = torch.cat([context, x], dim=1)  # [B, C+N, H]

        # ------------------------------------------------------------------
        # Step (e): Compute scale conditioning vector.
        # scale_vec: [B, hidden_size] — same for all token positions.
        # ------------------------------------------------------------------
        scale_vec: torch.Tensor = self.get_scale_vector(
            scale_id=scale_id,
            batch_size=batch_size,
            device=device,
        )  # [B, H]

        # ------------------------------------------------------------------
        # Step (f): Pass through all ScaleAwareBlocks.
        # Each block receives the full sequence [B, C+N, H] and the scale
        # vector [B, H]. The scale vector is constant across all blocks.
        # ------------------------------------------------------------------
        for block in self.blocks:
            x = block(x, scale_vec)  # [B, C+N, H]

        # ------------------------------------------------------------------
        # Step (g): Apply final LayerNorm.
        # ------------------------------------------------------------------
        x = self.norm(x)  # [B, C+N, H]

        # ------------------------------------------------------------------
        # Step (h): Strip context prefix, return only image token positions.
        # x[:, :context_