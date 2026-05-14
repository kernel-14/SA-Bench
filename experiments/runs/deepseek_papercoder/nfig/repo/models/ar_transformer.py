"""
models/ar_transformer.py

Autoregressive transformer for the Next-Frequency Image Generation (NFIG) framework.

A decoder‑only, block‑causal transformer that predicts frequency‑band tokens
in a coarse‑to‑fine order.  Principal components:

    - AdaLNLayer : Adaptive layer normalisation that modulates features
                   using a class‑conditioned scale and shift vector.
    - VARTransformer : Multi‑scale autoregressive generator with
                       block‑wise causal attention, classifier‑free guidance,
                       and top‑k sampling.

All hyperparameters are read from a configuration dictionary compatible with
the project's config.yaml.  The implementation closely follows the VAR
architecture adapted for frequency‑ordered token generation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Adaptive Layer Normalisation (AdaLN) with class conditioning
# ----------------------------------------------------------------------

class AdaLNLayer(nn.Module):
    """
    Adaptive layer normalisation conditioned on a class label.

    The layer first embeds a class index into a `dim`-dimensional vector.
    This vector is linearly projected to produce scale and shift parameters
    that are applied after a standard LayerNorm (without learned affine parameters).

    Args:
        dim: Number of features (the normalisation dimension).
        num_classes: Total number of class labels.  An additional slot is
            reserved for the "unconditional" index (index = num_classes).
    """

    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        # We reserve one extra index for the unconditional case
        self.num_classes = num_classes
        self.total_classes = num_classes + 1

        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.embed = nn.Embedding(self.total_classes, dim)
        self.ada_scale = nn.Linear(dim, dim)
        self.ada_shift = nn.Linear(dim, dim)

        # Initialise the projection weights to zero to keep the initial behaviour
        # as identity.
        nn.init.zeros_(self.ada_scale.weight)
        nn.init.zeros_(self.ada_scale.bias)
        nn.init.zeros_(self.ada_shift.weight)
        nn.init.zeros_(self.ada_shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive normalisation.

        Args:
            x: Input tensor of shape (B, N, dim).
            cond: Class label tensor of shape (B,).  Values must be in
                [0, self.total_classes - 1].

        Returns:
            Output tensor of the same shape as `x`.
        """
        B, N, D = x.shape
        # Embed the class indices
        cond_emb = self.embed(cond)                            # (B, D)

        scale = self.ada_scale(cond_emb).unsqueeze(1)          # (B, 1, D)
        shift = self.ada_shift(cond_emb).unsqueeze(1)          # (B, 1, D)

        x_norm = self.norm(x)
        # Modulation: factor = 1 + scale to ensure initial identity
        return x_norm * (1.0 + scale) + shift


# ----------------------------------------------------------------------
# Main generator transformer
# ----------------------------------------------------------------------

class VARTransformer(nn.Module):
    """
    Next‑Frequency Prediction Transformer.

    Args:
        config: Dictionary with keys:
            - generator: dim, depth, num_heads, mlp_ratio, dropout,
                         vocab_size
            - tokenizer: scale_sizes (list of ints)
            - data: num_classes
            - training_generator: cfg_drop_prob (unused here; used by trainer)
            - inference: cfg_scale, top_k (unused at init, used in generate)
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        # ---- Unpack configuration ----
        gen = config["generator"]
        self.dim: int = gen["dim"]                           # 1024
        self.depth: int = gen["depth"]                       # 16
        self.num_heads: int = gen["num_heads"]               # 16
        self.mlp_ratio: float = gen["mlp_ratio"]             # 4.0
        self.dropout_rate: float = gen["dropout"]            # 0.1
        self.vocab_size: int = gen["vocab_size"]             # 4096

        # Frequency scales
        scale_cfg = config["tokenizer"]
        self.scale_sizes: List[int] = scale_cfg["scale_sizes"]  # [1,2,...,16]
        self.num_scales: int = len(self.scale_sizes)

        # Class conditioning
        data_cfg = config["data"]
        self.num_classes: int = data_cfg["num_classes"]      # 1000
        # The unconditional index is the last slot (num_classes)
        self.uncond_idx: int = self.num_classes

        # ---- Special mask token (used only during generation) ----
        self.mask_token_id: int = self.vocab_size          # index = 4096

        # ---- Token embedding (vocabulary + mask) ----
        self.token_emb = nn.Embedding(
            self.vocab_size + 1, self.dim, padding_idx=None
        )

        # ---- Scale embedding ----
        self.scale_emb = nn.Embedding(self.num_scales, self.dim)

        # ---- Spatial position embedding (master 16×16 grid) ----
        self.master_h: int = 16
        self.master_w: int = 16
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.master_h, self.master_w, self.dim) * 0.02
        )

        # ---- Build transformer blocks (each contains AdaLN + attention + MLP) ----
        self.blocks = nn.ModuleList([
            self._make_block(block_idx)
            for block_idx in range(self.depth)
        ])

        # ---- Final layer norm and output projection ----
        self.ln_f = nn.LayerNorm(self.dim)
        self.output_proj = nn.Linear(self.dim, self.vocab_size, bias=False)

        # ---- Block‑causal attention mask (pre‑computed once) ----
        self.register_buffer("block_mask", self._build_block_mask(), persistent=False)

        # Initialise weights
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------
    # Transformer block factory
    # ------------------------------------------------------------------
    def _make_block(self, idx: int) -> nn.Module:
        """
        Create a single decoder block with two sub‑layers:
            - Self‑attention with AdaLN
            - MLP with AdaLN
        All AdaLN layers use the same number of classes.
        """
        return TransformerBlock(
            dim=self.dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout_rate,
            num_classes=self.num_classes,   # AdaLN will internally add 1 for unconditional
        )

    # ------------------------------------------------------------------
    # Position embedding helper
    # ------------------------------------------------------------------
    def _get_scale_position_embed(self, scale_idx: int) -> torch.Tensor:
        """
        Return a 2D spatial position embedding for the given scale,
        interpolated from the master 16×16 grid.

        Args:
            scale_idx: index into self.scale_sizes.

        Returns:
            Tensor of shape (1, h_i * w_i, dim).
        """
        h_i = self.scale_sizes[scale_idx]
        w_i = h_i
        # (1, 16, 16, dim) -> (1, dim, 16, 16)
        pos = self.pos_embed.permute(0, 3, 1, 2)
        # interpolate to target size (h_i, w_i)
        pos = F.interpolate(
            pos, size=(h_i, w_i), mode="bilinear", align_corners=False
        )
        # -> (1, h_i, w_i, dim)
        pos = pos.permute(0, 2, 3, 1)
        # flatten spatial dims: (1, h_i * w_i, dim)
        pos = pos.reshape(1, h_i * w_i, self.dim)
        return pos

    # ------------------------------------------------------------------
    # Block‑causal mask construction
    # ------------------------------------------------------------------
    def _build_block_mask(self) -> torch.Tensor:
        """
        Construct a block‑causal attention mask of shape (S, S).

        Tokens are ordered by increasing scale (low‑to‑high frequency).
        Tokens at scale i can attend to all tokens from scales 0 .. i
        (bidirectional within the same scale) and must be masked from
        tokens at scales > i.

        Returns:
            Float mask with 0.0 for allowed positions and -inf for masked.
        """
        S = sum(s * s for s in self.scale_sizes)        # total tokens
        mask = torch.zeros(S, S)

        start_i = 0
        for i, s_i in enumerate(self.scale_sizes):
            n_i = s_i * s_i
            end_i = start_i + n_i

            # Allow attention to scales <= i
            start_j_allowed = 0
            end_j_allowed = 0
            for j, s_j in enumerate(self.scale_sizes):
                n_j = s_j * s_j
                end_j_allowed += n_j
                if j == i:
                    break

            # Mask tokens of scales > i: set those positions to -inf
            if end_j_allowed < S:
                mask[start_i:end_i, end_j_allowed:] = float("-inf")

            start_i = end_i

        return mask

    def get_block_mask(self) -> torch.Tensor:
        """
        Public accessor for the block‑causal mask.

        Returns:
            Float mask tensor of shape (S, S).
        """
        return self.block_mask

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        token_ids: List[torch.Tensor],
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute logits for the next‑token prediction task.

        Args:
            token_ids: List of tensors, one per scale, each of shape
                (B, h_i * w_i) containing token indices in [0, vocab_size).
                All scales must be present (even if some are mask tokens
                during generation; they are embedded normally).
            class_labels: Tensor of shape (B,) containing class indices
                in [0, num_classes-1] or the unconditional index
                (num_classes).

        Returns:
            Logits tensor of shape (B, S, vocab_size), where S is the
            total number of tokens over all scales.
        """
        B = class_labels.size(0)
        device = class_labels.device

        # ---- 1. Build the unified token sequence ----
        all_embeddings: List[torch.Tensor] = []
        for i, t_ids in enumerate(token_ids):
            # Ensure the token ids are on the correct device
            t_ids = t_ids.to(device=device, dtype=torch.long)
            # Token embedding
            tok_emb = self.token_emb(t_ids)                     # (B, n_i, dim)
            # Scale embedding (broadcasted across tokens of this scale)
            scale_id = torch.full(
                (B, 1), i, device=device, dtype=torch.long
            )
            scale_emb = self.scale_emb(scale_id)                 # (B, 1, dim)
            tok_emb = tok_emb + scale_emb
            # Spatial position embedding (interpolated)
            pos = self._get_scale_position_embed(i).to(device)   # (1, n_i, dim)
            tok_emb = tok_emb + pos

            all_embeddings.append(tok_emb)

        # Concatenate along the token axis: shape (B, S, dim)
        x = torch.cat(all_embeddings, dim=1)

        # ---- 2. Apply transformer blocks ----
        # Use the pre‑computed block mask; broadcast to all batches and heads.
        mask = self.block_mask.to(device)                       # (S, S)
        # The attention function expects a float mask; -inf values are fine.
        # We need to ensure the head dimension: (B, 1, S, S) or (1, 1, S, S).
        # Here we just pass the 2D mask, and inside each block it will be
        # broadcast automatically.

        for block in self.blocks:
            x = block(x, class_labels, attn_mask=mask)

        # ---- 3. Final layer norm + output projection ----
        x = self.ln_f(x)
        logits = self.output_proj(x)                            # (B, S, vocab_size)
        return logits

    # ------------------------------------------------------------------
    # Generation with classifier‑free guidance and top‑k sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        class_label: int,
        top_k: int = 990,
        cfg_scale: float = 4.5,
    ) -> List[torch.Tensor]:
        """
        Autoregressively generate token sequences from low to high frequency.

        Args:
            class_label: Integer class index (0 .. num_classes-1).
            top_k: Number of highest probability tokens to keep per position.
            cfg_scale: Classifier‑free guidance scale.  Must be >= 0.

        Returns:
            A list of tensors, one per scale, each shaped (1, h_i*w_i) with
            sampled token indices.
        """
        device = next(self.parameters()).device
        # Total sequence length
        S = sum(s * s for s in self.scale_sizes)

        # Start with all tokens set to the mask token
        token_ids_flat = torch.full(
            (1, S), self.mask_token_id, dtype=torch.long, device=device
        )

        # Prepare the list of tensors for the forward call
        # Initially, we split the flat sequence into per‑scale slices
        scale_slices = []
        offset = 0
        for s_i in self.scale_sizes:
            n_i = s_i * s_i
            scale_slices.append(token_ids_flat[:, offset:offset + n_i])
            offset += n_i

        # Unconditional class index
        uncond_idx = torch.tensor([self.uncond_idx], device=device)
        cond_idx = torch.tensor([class_label], device=device)

        # Iterate over scales
        for i in range(self.num_scales):
            # ----- Conditional forward -----
            logits_cond = self.forward(scale_slices, cond_idx)         # (1, S, vocab_size)

            # ----- Unconditional forward -----
            logits_uncond = self.forward(scale_slices, uncond_idx)     # (1, S, vocab_size)

            # ----- CFG combination -----
            logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)

            # Extract logits for the current scale i
            n_i = self.scale_sizes[i] * self.scale_sizes[i]
            # Determine the start column for this scale
            start_col = 0
            for j in range(i):
                start_col += self.scale_sizes[j] * self.scale_sizes[j]

            logits_scale = logits[:, start_col:start_col + n_i, :]   # (1, n_i, vocab_size)

            # ---- Top‑k filtering ----
            if top_k > 0:
                k_vals, _ = torch.topk(logits_scale, top_k, dim=-1)
                # Minimum value among top‑k for each position
                min_val = k_vals[:, :, -1:]                     # (1, n_i, 1)
                # Mask out values below the threshold
                logits_scale = torch.where(
                    logits_scale >= min_val, logits_scale, torch.tensor(float("-inf"), device=device)
                )

            # Sample from the filtered distribution
            probs = F.softmax(logits_scale, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, self.vocab_size), num_samples=1
            ).view(1, n_i)                                               # (1, n_i)

            # Update the flat token_ids with the new samples
            token_ids_flat[0, start_col:start_col + n_i] = sampled.squeeze()
            # Update the corresponding slice in scale_slices for the next iterations
            scale_slices[i] = token_ids_flat[:, start_col:start_col + n_i].clone()

        # Return generated token lists (one per scale) for decoding
        gen_tokens: List[torch.Tensor] = []
        offset = 0
        for s_i in self.scale_sizes:
            n_i = s_i * s_i
            gen_tokens.append(token_ids_flat[:, offset:offset + n_i])
            offset += n_i
        return gen_tokens


# ----------------------------------------------------------------------
# Transformer Block (used internally by VARTransformer)
# ----------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    A single transformer block with two sub‑layers:

        - Multi‑head self‑attention preceded by AdaLN
        - MLP preceded by AdaLN

    Args:
        dim: Hidden dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Expansion factor for the MLP hidden layer.
        dropout: Dropout probability.
        num_classes: Total number of class labels (excluding unconditional).
            The AdaLN layer will add an extra slot for the unconditional index.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # ---- AdaLN for attention ----
        self.adaln1 = AdaLNLayer(dim, num_classes)
        # ---- Self‑attention ----
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        # ---- AdaLN for MLP ----
        self.adaln2 = AdaLNLayer(dim, num_classes)
        # ---- MLP ----
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        class_labels: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply the transformer block.

        Args:
            x: Input tensor of shape (B, N, dim).
            class_labels: Class label indices, shape (B,).
            attn_mask: Attention mask broadcastable to (B, N, N).  Default None.

        Returns:
            Output tensor of shape (B, N, dim).
        """
        # ---- Self‑attention sub‑layer ----
        # AdaLN layer expects class labels, not an embedded vector
        x_norm = self.adaln1(x, class_labels)
        # MultiheadAttention expects attn_mask as a 2D float tensor with -inf for masked positions.
        # The mask must be of shape (N, N) or (B, N, N). We have (N, N). We'll pass it as is.
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_out)

        # ---- MLP sub‑layer ----
        x_norm = self.adaln2(x, class_labels)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out

        return x


# ----------------------------------------------------------------------
# Quick sanity test (not executed when imported)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple test with dummy config and random tensors
    mock_config = {
        "generator": {
            "dim": 1024,
            "depth": 2,                # small for test
            "num_heads": 16,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "vocab_size": 4096,
        },
        "tokenizer": {
            "scale_sizes": [1, 2, 4, 8],   # reduced for test
        },
        "data": {
            "num_classes": 1000,
        },
        "training_generator": {
            "cfg_drop_prob": 0.1,
        },
        "inference": {
            "cfg_scale": 4.5,
            "top_k": 990,
        },
    }
    model = VARTransformer(mock_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Simulate token inputs: batch of 2 images
    B = 2
    token_ids_list = []
    for s in mock_config["tokenizer"]["scale_sizes"]:
        n = s * s
        # Random token indices (no mask token)
        t = torch.randint(0, 4096, (B, n))
        token_ids_list.append(t)

    labels = torch.randint(0, 1000, (B,))
    logits = model(token_ids_list, labels)
    print("Logits shape:", logits.shape)

    # Test generation
    gen_tokens = model.generate(class_label=42, top_k=50, cfg_scale=3.0)
    print("Generated token shapes:")
    for i, t in enumerate(gen_tokens):
        print(f"  Scale {i}: {t.shape}")

