"""
model.py

Hi‑MAR Transformer – the core scale‑aware bidirectional Transformer for
Hierarchical Masked Autoregressive image generation.

This module provides two classes:
  - ``ScaleAwareTransformerBlock`` : a Transformer block modulated by a global
    scale vector via adaLN‑Zero.
  - ``HiMARTransformer`` : the full stack of blocks, including input projection,
    positional embeddings, class/text embedding, and the mask token definition.

The same transformer is used for both Phase 1 (low‑resolution) and Phase 2
(high‑resolution) by simply passing a different ``scale_id`` and appropriately
prepared ``context`` and ``image_tokens``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# Project imports
from config import ModelConfig
from utils import ScaleVectorMLP, chunk_adaLN_parameters, get_sinusoidal_embedding


# ---------------------------------------------------------------------------
#  Scale‑Aware Transformer Block
# ---------------------------------------------------------------------------

class ScaleAwareTransformerBlock(nn.Module):
    """
    A single Transformer block whose layer‑norm and residual paths are modulated
    by a global scale vector ``v``, exactly as described in Section 3.2 of the
    Hi‑MAR paper.

    The block implements adaLN‑Zero: the modulation parameters are generated from
    ``v`` by a linear projection that is initialised to zero, so that at the start
    of training the block behaves as an identity.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        v_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            hidden_size:  Dimensionality of the token representations.
            num_heads:    Number of attention heads.
            v_dim:        Dimensionality of the global scale vector ``v``.
            mlp_ratio:    Expansion ratio for the feed‑forward hidden layer.
            dropout:      Dropout probability (applied after attention and FFN).
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.v_dim = v_dim

        # ------------------------------------------------------------------
        # adaLN‑Zero modulation projection
        # Generates (α₁,β₁,γ₁,α₂,β₂,γ₂) from v.
        # ------------------------------------------------------------------
        self.adaLN_modulation = nn.Linear(v_dim, 6 * hidden_size)
        # Zero‑initialise so that the block starts as identity
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

        # Layer norms without learnable affine parameters (they come from adaLN)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        # Self‑attention (batch_first for (B, T, C) input)
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed‑forward network
        ffn_hidden = int(hidden_size * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Apply the scale‑aware Transformer block to a token sequence.

        Args:
            x:  Input tokens of shape ``(B, T, hidden_size)``.
            v:  Global scale vector of shape ``(v_dim,)`` (shared across
                the batch and all positions).

        Returns:
            Output tokens of shape ``(B, T, hidden_size)``.
        """
        B = x.shape[0]

        # 1. Compute modulation parameters from v
        #    v is (v_dim,) → expand to (B, 6*hidden_size)
        mod = self.adaLN_modulation(v)          # (v_dim,) -> (6*C,)
        mod = mod.unsqueeze(0).expand(B, -1)     # (B, 6*C)
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = chunk_adaLN_parameters(
            mod, self.hidden_size
        )
        # Each now (B, C), will broadcast over the token dimension.

        # 2. Self‑attention sub‑block
        h = self.norm1(x)                       # (B, T, C)
        h = h * alpha1.unsqueeze(1) + beta1.unsqueeze(1)   # adaLN
        h, _ = self.attn(h, h, h)               # (B, T, C)
        x = x + gamma1.unsqueeze(1) * h         # residual with gate

        # 3. Feed‑forward sub‑block
        h = self.norm2(x)
        h = h * alpha2.unsqueeze(1) + beta2.unsqueeze(1)
        h = self.ffn(h)
        x = x + gamma2.unsqueeze(1) * h

        return x


# ---------------------------------------------------------------------------
#  Hi‑MAR Transformer
# ---------------------------------------------------------------------------

class HiMARTransformer(nn.Module):
    """
    The full Hi‑MAR Transformer backbone.

    It processes a sequence composed of *context tokens* (class embedding or
    projected text embeddings, optionally preceded by pivot tokens Zₛ from
    Phase 1) followed by *image tokens* (continuous VAE latents, possibly
    masked).  The Transformer uses full bidirectional self‑attention and
    injects scale‑aware modulation via a global scale vector.

    The output is the sequence of *conditional tokens* for the image positions,
    which are subsequently fed to the appropriate diffusion head.
    """

    def __init__(
        self,
        config: ModelConfig,
        latent_dim: int,
        num_classes: int = 0,
        text_encoder_dim: int = 0,
        v_dim: Optional[int] = None,
        scale_sin_embed_dim: int = 256,
        scale_hidden_dim: int = 512,
    ) -> None:
        """
        Args:
            config:             Model configuration (variant, layers, hidden, etc.).
            latent_dim:         Dimension of the continuous VAE latent (usually 16).
            num_classes:        Number of classes for class‑conditional generation
                                (set to 0 for text‑to‑image).
            text_encoder_dim:   Dimension of the text encoder (e.g., 768 for CLIP‑L/14)
                                (set to 0 for class‑conditional).
            v_dim:              Dimension of the global scale vector ``v``.
                                Defaults to ``hidden_size``.
            scale_sin_embed_dim:Sinusoidal embedding dimension for the scale id.
            scale_hidden_dim:   Hidden dimension of the small MLP that produces ``v``.
        """
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.mlp_ratio = config.mlp_ratio
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.text_encoder_dim = text_encoder_dim

        # Determine number of attention heads (default: hidden // 64, at least 1)
        num_heads = max(1, self.hidden_size // 64)

        # Dimension of the global scale vector
        if v_dim is None:
            v_dim = self.hidden_size
        self.v_dim = v_dim

        # ------------------------------------------------------------------
        # Input projection: maps raw VAE latents (dim 16) → hidden_size
        # ------------------------------------------------------------------
        self.input_proj = nn.Linear(latent_dim, self.hidden_size)

        # ------------------------------------------------------------------
        # Context encodings
        # ------------------------------------------------------------------
        if num_classes > 0:
            self.class_embedding = nn.Embedding(num_classes, self.hidden_size)
        else:
            self.class_embedding = None

        if text_encoder_dim > 0:
            self.text_proj = nn.Linear(text_encoder_dim, self.hidden_size)
        else:
            self.text_proj = None

        # ------------------------------------------------------------------
        # Learnable 1‑D positional embeddings for image tokens
        # Low‑res: up to 64 tokens; High‑res: up to 256 tokens
        # ------------------------------------------------------------------
        self.pos_embed_low = nn.Parameter(torch.randn(1, 64, self.hidden_size) * 0.02)
        self.pos_embed_high = nn.Parameter(torch.randn(1, 256, self.hidden_size) * 0.02)

        # ------------------------------------------------------------------
        # Shared learnable mask token (stored in the latent space, dim=16)
        # ------------------------------------------------------------------
        self.mask_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)

        # ------------------------------------------------------------------
        # Scale‑embedding MLP: scale_id (0/1) → global scale vector v
        # ------------------------------------------------------------------
        self.scale_embedding_mlp = ScaleVectorMLP(
            sin_embed_dim=scale_sin_embed_dim,
            hidden_dim=scale_hidden_dim,
            out_dim=v_dim,
        )

        # ------------------------------------------------------------------
        # Stack of scale‑aware Transformer blocks
        # ------------------------------------------------------------------
        self.layers = nn.ModuleList([
            ScaleAwareTransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=num_heads,
                v_dim=v_dim,
                mlp_ratio=self.mlp_ratio,
                dropout=0.0,   # No dropout in the backbone (kept for simplicity)
            )
            for _ in range(self.num_layers)
        ])

        # Initialise weights (excluding zero‑init adaLN layers inside blocks)
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """
        Initialise linear layers and embeddings with standard practices,
        following the paper and DiT conventions.
        """
        # Standard initialisation for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                if module.elementwise_affine and module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                    nn.init.constant_(module.bias, 0)

        # Override: the input projection uses a smaller spread due to low latent dim
        nn.init.normal_(self.input_proj.weight, std=0.02 / math.sqrt(self.num_layers))
        if self.input_proj.bias is not None:
            nn.init.constant_(self.input_proj.bias, 0)

        # Text projection similar treatment
        if self.text_proj is not None:
            nn.init.normal_(self.text_proj.weight, std=0.02)
            if self.text_proj.bias is not None:
                nn.init.constant_(self.text_proj.bias, 0)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        context: Optional[Tensor],
        image_tokens: Tensor,
        scale_id: int,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Process a batch of (masked) image tokens together with optional context.

        Args:
            context:
                - For **class‑conditional**: a LongTensor of class indices,
                  shape ``(B,)``.
                - For **text‑to‑image**: a FloatTensor of CLIP text embeddings,
                  shape ``(B, L_text, d_text)``.
                - For **Phase 2**, the caller prepends the low‑res conditional
                  tokens Zₛ to this argument, yielding shape
                  ``(B, L_text + N_low, hidden_size)`` or
                  ``(B, 1 + N_low, hidden_size)``.
                - May be ``None`` if no context is needed (e.g., unconditional CFG
                  forward pass).
            image_tokens:
                Tensor of continuous latent tokens, shape ``(B, N, latent_dim)``.
                Already contains the shareable ``mask_token`` at masked positions
                when training or during inference.
            scale_id:
                Integer tensor of shape ``(B,)`` or a single Python ``int``
                specifying the resolution: ``0`` for low‑res, ``1`` for high‑res.
            mask:
                Optional boolean tensor of shape ``(B, N)`` indicating which
                positions are currently masked.  Not used inside the Transformer;
                forwarded for convenience.

        Returns:
            Conditional tokens for the image positions, shape ``(B, N, hidden_size)``.
            These are the features that will be fed to the diffusion heads.
        """
        B, N, D = image_tokens.shape
        device = image_tokens.device
        assert D == self.latent_dim, (
            f"Expected latent dim {self.latent_dim}, got {D}"
        )

        # 1. Normalise scale_id to a 1‑D tensor
        if isinstance(scale_id, int):
            scale_id = torch.full((B,), scale_id, dtype=torch.long, device=device)
        elif scale_id.ndim == 0:
            scale_id = scale_id.unsqueeze(0).expand(B)
        else:
            scale_id = scale_id.to(device=device, dtype=torch.long)

        # 2. Compute global scale vector v from scale_id
        #    scale_embedding_mlp expects LongTensor of shape (B,)
        v = self.scale_embedding_mlp(scale_id)          # (B, v_dim)
        # For our blocks we need a single v shared across batch.
        # Following the paper, the same scale vector is used for the whole batch,
        # so we just take the first row (all are identical for same scale_id).
        # Actually, batch may mix scales? No, single scale per forward.
        v = v[0]                                        # (v_dim,)

        # 3. Project image tokens
        x = self.input_proj(image_tokens)               # (B, N, hidden_size)

        # 4. Add positional embeddings (only to image positions)
        if scale_id[0].item() == 0:
            pos_embed = self.pos_embed_low[:, :N, :]
        else:
            pos_embed = self.pos_embed_high[:, :N, :]
        x = x + pos_embed                                # (B, N, hidden_size)

        # 5. Construct context tokens
        context_tokens_list = []

        if context is not None:
            if self.class_embedding is not None and context.dtype == torch.long:
                # Class‑conditional: context shape (B,) of class ids
                c = self.class_embedding(context)        # (B, hidden_size)
                c = c.unsqueeze(1)                       # (B, 1, hidden_size)
                context_tokens_list.append(c)

            elif self.text_proj is not None and context.dtype in (torch.float32, torch.float16, torch.bfloat16):
                # Text‑to‑image: context shape (B, L_text, d_text)
                c = self.text_proj(context)             # (B, L_text, hidden_size)
                context_tokens_list.append(c)

            else:
                # Context is already pre‑projected (e.g., pivots Z_s from Phase 1).
                # In this case, context is expected to be (B, L_ctx, hidden_size).
                if context.ndim == 3 and context.shape[-1] == self.hidden_size:
                    context_tokens_list.append(context)
                else:
                    raise ValueError(
                        "`context` must be either LongTensor (class ids), "
                        "FloatTensor (text embeddings), or pre‑projected "
                        f"(B, L, hidden_size). Got shape {context.shape}, "
                        f"dtype {context.dtype}."
                    )

        # 6. Concatenate context + image tokens
        if context_tokens_list:
            context_tokens = torch.cat(context_tokens_list, dim=1)   # (B, L_ctx, H)
            tokens = torch.cat([context_tokens, x], dim=1)           # (B, L_ctx+N, H)
        else:
            tokens = x

        # 7. Pass through all scale‑aware blocks
        for block in self.layers:
            tokens = block(tokens, v)

        # 8. Extract the image‑position outputs (conditional tokens)
        L_ctx = tokens.shape[1] - N
        if L_ctx < 0:
            # Should not happen; safety fallback
            conditional_tokens = tokens
        else:
            conditional_tokens = tokens[:, L_ctx:, :]   # (B, N, hidden_size)

        return conditional_tokens


    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def get_mask_token(self) -> Tensor:
        """
        Return the learnable mask token in the latent space.

        Shape: ``(1, 1, latent_dim)``.
        """
        return self.mask_token

    def get_num_tokens(self, scale_id: int) -> int:
        """
        Return the number of image tokens for a given scale.

        Args:
            scale_id: ``0`` → 64 (low‑res), ``1`` → 256 (high‑res).

        Returns:
            Number of spatial tokens.
        """
        if scale_id == 0:
            return 64
        elif scale_id == 1:
            return 256
        else:
            raise ValueError(f"scale_id must be 0 or 1, got {scale_id}")

    def get_hidden_size(self) -> int:
        """Return the Transformer hidden size."""
        return self.hidden_size

