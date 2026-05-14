"""
NFIG Autoregressive Transformer for Next-Frequency Image Generation.
Implements block-wise causal attention and adaptive layer normalization (AdaLN)
as described in Section 3.2 of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization for class-conditional generation.
    AdaLN modulates the normalized features using scale and shift
    derived from the class embedding and the current frequency band.
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale_proj = nn.Linear(cond_dim, dim)
        self.shift_proj = nn.Linear(cond_dim, dim)

        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        scale = self.scale_proj(cond)
        shift = self.shift_proj(cond)
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with optional causal mask."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with AdaLN and feed-forward network."""

    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0
    ):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(x, mask)
        x = x + self.mlp(x)
        return x


class AdaLNTransformerBlock(nn.Module):
    """Transformer block with AdaLN for class conditioning."""

    def __init__(
        self, dim: int, num_heads: int, cond_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0
    ):
        super().__init__()
        self.adaln_attn = AdaLayerNorm(dim, cond_dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.adaln_mlp = AdaLayerNorm(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.adaln_attn(x, cond), mask)
        x = x + self.mlp(self.adaln_mlp(x, cond))
        return x


def create_blockwise_causal_mask(
    block_sizes: List[int], device: torch.device
) -> torch.Tensor:
    """
    Create block-wise causal attention mask.
    Tokens in block i can attend to tokens in blocks 0..i,
    but not to tokens in blocks > i.
    Within a block, tokens can attend to each other.

    Args:
        block_sizes: List of token counts per frequency band.
    Returns:
        Boolean mask of shape (1, 1, total_tokens, total_tokens), True = allowed.
    """
    total_tokens = sum(block_sizes)
    mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool, device=device)

    start_i = 0
    for i, size_i in enumerate(block_sizes):
        end_i = start_i + size_i
        start_j = 0
        for j, size_j in enumerate(block_sizes):
            end_j = start_j + size_j
            if j <= i:
                mask[start_i:end_i, start_j:end_j] = True
            start_j = end_j
        start_i = end_i

    return mask.unsqueeze(0).unsqueeze(0)


class FrequencyBandEmbedding(nn.Module):
    """Embeds frequency band index for positional information."""

    def __init__(self, num_bands: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_bands, dim)

    def forward(self, band_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(band_indices)


class NFIGTransformer(nn.Module):
    """
    Next-Frequency Image Generation Transformer.
    Autoregressively predicts tokens for each frequency band,
    conditioned on all previous frequency bands.

    Architecture:
    - Token embeddings from codebook indices
    - Block-wise causal attention
    - AdaLN for class conditioning
    - Predicts next-frequency tokens hierarchically
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 16,
        num_classes: int = 1000,
        scale_factors: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        feature_map_size: int = 16,
        dropout: float = 0.1,
        use_adaln: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.scale_factors = scale_factors
        self.num_bands = len(scale_factors)
        self.feature_map_size = feature_map_size
        self.use_adaln = use_adaln

        # Compute token counts per band
        # Scale factor s directly gives the resolution: h_i = s, w_i = s
        self.block_sizes = [s * s for s in scale_factors]
        self.total_tokens = sum(self.block_sizes)

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)

        # Positional embeddings: learnable per-position, plus frequency band embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, total, hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.band_embed = FrequencyBandEmbedding(self.num_bands, hidden_dim)

        # Class condition embedding
        self.class_embed = nn.Embedding(num_classes, hidden_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Start-of-sequence token (for the initial generation step)
        self.sos_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.sos_token, std=0.02)

        # Learnable frequency band query tokens
        self.band_queries = nn.Parameter(
            torch.zeros(self.num_bands, hidden_dim)
        )
        nn.init.normal_(self.band_queries, std=0.02)

        # Transformer blocks
        if use_adaln:
            self.blocks = nn.ModuleList([
                AdaLNTransformerBlock(hidden_dim, num_heads, hidden_dim, dropout=dropout)
                for _ in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ])

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def _get_band_token_positions(self, device: torch.device) -> torch.Tensor:
        """Get the start/end positions of tokens for each frequency band."""
        positions = []
        pos = 0
        for size in self.block_sizes:
            positions.append((pos, pos + size))
            pos += size
        return torch.tensor(positions, device=device)

    def forward(
        self,
        tokens: List[torch.Tensor],
        class_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            tokens: List of token tensors, one per frequency band.
                    Each is shape (B, h_i, w_i) with values in [0, vocab_size).
            class_ids: (B,) class labels.
        Returns:
            logits: (B, total_tokens, vocab_size) prediction logits.
        """
        B = class_ids.shape[0]
        device = class_ids.device

        # Flatten all tokens into one sequence
        token_list = []
        for t in tokens:
            token_list.append(t.reshape(B, -1))
        all_tokens = torch.cat(token_list, dim=1)  # (B, total_tokens)

        # Embed tokens
        x = self.token_embed(all_tokens)  # (B, total_tokens, hidden_dim)

        # Add positional embeddings
        x = x + self.pos_embed[:, : x.shape[1], :]

        # Add frequency band embeddings
        band_ids = []
        for i, size in enumerate(self.block_sizes):
            band_ids.append(torch.full((size,), i, dtype=torch.long, device=device))
        band_ids = torch.cat(band_ids, dim=0)  # (total_tokens,)
        band_emb = self.band_embed(band_ids)  # (total_tokens, hidden_dim)
        x = x + band_emb.unsqueeze(0)

        # Class conditioning
        class_emb = self.class_embed(class_ids)  # (B, hidden_dim)
        cond = self.cond_proj(class_emb)  # (B, hidden_dim)

        # Block-wise causal mask
        if mask is None:
            mask = create_blockwise_causal_mask(self.block_sizes, device)

        # Transformer blocks
        for block in self.blocks:
            if self.use_adaln:
                x = block(x, cond, mask)
            else:
                x = block(x, mask)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, total_tokens, vocab_size)

        return logits

    def generate(
        self,
        class_ids: torch.Tensor,
        cfg_scale: float = 4.5,
        top_k: int = 990,
        temperature: float = 1.0,
        use_cfg: bool = True,
    ) -> List[torch.Tensor]:
        """
        Autoregressive generation: predict tokens for each frequency band sequentially.
        Implements Classifier-Free Guidance (CFG) as described in the paper.

        Args:
            class_ids: (B,) class labels.
            cfg_scale: CFG scale factor (default 4.5 from paper).
            top_k: Top-k sampling parameter (default 990 from paper).
            temperature: Sampling temperature.
        Returns:
            List of token tensors per frequency band.
        """
        B = class_ids.shape[0]
        device = class_ids.device

        generated_tokens = []
        accumulated_embeddings = []

        # Pre-compute mask once
        mask = create_blockwise_causal_mask(self.block_sizes, device)

        # Null class embedding for CFG
        null_class = torch.full_like(class_ids, self.num_classes)  # use out-of-range class
        null_class_embeds = None  # Will be computed if needed

        for band_idx in range(self.num_bands):
            h_i = self.scale_factors[band_idx]
            w_i = self.scale_factors[band_idx]
            num_tokens = h_i * w_i

            if band_idx == 0:
                # First band: start with SOS token and band 0 query
                x_in = self.sos_token.expand(B, 1, -1)
                # Add band query
                band_query = self.band_queries[band_idx].unsqueeze(0).unsqueeze(1).expand(B, 1, -1)
                x_in = x_in + band_query + self.pos_embed[:, :1, :]
            else:
                # Build context from all previously generated tokens
                flat_tokens = []
                for t in generated_tokens:
                    flat_tokens.append(t.reshape(B, -1))
                context_tokens = torch.cat(flat_tokens, dim=1)  # (B, total_previous_tokens)

                # Start with accumulated embeddings from previous steps
                x_context = self.token_embed(context_tokens)
                x_context = x_context + self.pos_embed[:, : x_context.shape[1], :]

                # Add band embeddings for context
                band_ids_context = []
                for i in range(band_idx):
                    size_i = self.block_sizes[i]
                    band_ids_context.append(torch.full((size_i,), i, dtype=torch.long, device=device))
                band_ids_context = torch.cat(band_ids_context, dim=0)
                band_emb_context = self.band_embed(band_ids_context).unsqueeze(0)
                x_context = x_context + band_emb_context

                # New band query for the current band
                band_query = self.band_queries[band_idx].unsqueeze(0).repeat(1, num_tokens, 1)
                band_query = band_query + self.pos_embed[
                    :, sum(self.block_sizes[:band_idx]) : sum(self.block_sizes[:band_idx]) + num_tokens, :
                ]

                x_in = torch.cat([x_context, band_query], dim=1)

            # Class conditioning
            class_emb = self.class_embed(class_ids)
            cond = self.cond_proj(class_emb)

            # Create sub-mask for current sequence length
            curr_len = x_in.shape[1]
            sub_mask = mask[:, :, :curr_len, :curr_len]

            # Forward pass
            x = x_in
            for block in self.blocks:
                if self.use_adaln:
                    x = block(x, cond, sub_mask)
                else:
                    x = block(x, sub_mask)

            x = self.ln_f(x)

            # Get logits for the new tokens only
            new_logits = self.head(x[:, -num_tokens:, :])  # (B, num_tokens, vocab_size)

            # Apply CFG if enabled
            if use_cfg and cfg_scale > 1.0:
                if null_class_embeds is None:
                    null_class_embeds = self.class_embed(null_class)
                    null_cond = self.cond_proj(null_class_embeds)

                # Forward with null class
                x_null = x_in
                for block in self.blocks:
                    if self.use_adaln:
                        x_null = block(x_null, null_cond, sub_mask)
                    else:
                        x_null = block(x_null, sub_mask)
                x_null = self.ln_f(x_null)
                null_logits = self.head(x_null[:, -num_tokens:, :])

                new_logits = null_logits + cfg_scale * (new_logits - null_logits)

            # Apply temperature
            new_logits = new_logits / temperature

            # Top-k sampling
            if top_k > 0 and top_k < self.vocab_size:
                top_values, _ = torch.topk(new_logits, top_k, dim=-1)
                min_top = top_values[:, :, -1:]
                new_logits = torch.where(
                    new_logits < min_top,
                    torch.full_like(new_logits, float("-inf")),
                    new_logits,
                )

            # Sample
            probs = F.softmax(new_logits, dim=-1)
            sampled_tokens = torch.multinomial(
                probs.reshape(-1, self.vocab_size), num_samples=1
            ).reshape(B, num_tokens)
            sampled_tokens_2d = sampled_tokens.reshape(B, h_i, w_i)

            generated_tokens.append(sampled_tokens_2d)

        return generated_tokens

    def compute_loss(
        self,
        logits: torch.Tensor,
        target_tokens: List[torch.Tensor],
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss between predicted logits and target tokens.
        Following the paper: L(T, \tilde{T}) = -Σ t_i log(\tilde{t}_i)
        """
        # Flatten target tokens
        targets = torch.cat([t.reshape(t.shape[0], -1) for t in target_tokens], dim=1)
        targets = targets.to(dtype=torch.long)

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=ignore_index,
        )
        return loss


class NFIGTransformerSmall(NFIGTransformer):
    """Smaller variant for faster experiments."""

    def __init__(
        self,
        vocab_size: int = 4096,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        num_classes: int = 1000,
        scale_factors: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        feature_map_size: int = 16,
        dropout: float = 0.1,
        use_adaln: bool = True,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            scale_factors=scale_factors,
            feature_map_size=feature_map_size,
            dropout=dropout,
            use_adaln=use_adaln,
        )
