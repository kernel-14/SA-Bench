"""
NFIG Transformer: Next-Frequency Image Generation autoregressive model.

Based on VAR (Visual AutoRegressive) transformer backbone with frequency-aware
block-wise causal attention. Generates image tokens from low to high frequency bands.

Reference: "NFIG: Multi-Scale Autoregressive Image Generation via Frequency Ordering"

Token counts per band (scale_factors = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]):
  [1, 4, 9, 16, 25, 36, 64, 100, 169, 256] = 680 total tokens
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


def build_causal_mask(token_counts: List[int], device: torch.device) -> torch.Tensor:
    """
    Build block-wise causal attention mask for next-frequency prediction.

    Tokens within the same frequency band can attend to each other (full attention),
    and all tokens can attend to tokens from earlier (lower-frequency) bands.
    Tokens cannot attend to tokens from later (higher-frequency) bands.

    Args:
        token_counts: list of token counts per frequency band [n1, n2, ..., nk]
        device: target device
    Returns:
        mask: (total_tokens, total_tokens) boolean mask where True = masked (no attention)
    """
    total = sum(token_counts)
    # Start with all masked (no attention)
    mask = torch.ones(total, total, dtype=torch.bool, device=device)

    start = 0
    for count in token_counts:
        end = start + count
        # Allow attention from current band to all previous bands + current band
        mask[start:end, :end] = False
        start = end

    return mask  # True = masked


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional causal mask."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, n_heads, T, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_heads, T, T)

        if attn_mask is not None:
            # attn_mask: (T, T) bool, True = masked
            attn = attn.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm and AdaLN for class conditioning."""

    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        # AdaLN: class-conditional scale and shift (6 parameters: shift/scale/gate for attn and mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) token sequence
            c: (B, C) class conditioning embedding
            attn_mask: optional causal mask
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        # Attention with AdaLN
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(h, attn_mask)
        # MLP with AdaLN
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class NFIGTransformer(nn.Module):
    """
    Next-Frequency Image Generation Transformer.

    Decoder-only transformer with block-wise causal attention that generates
    image tokens from low to high frequency bands autoregressively.

    Architecture follows VAR with:
    - Shared token embedding across all frequency bands
    - Learned positional embeddings per frequency band
    - AdaLN class conditioning
    - Block-wise causal attention mask (tokens in band i can attend to all
      tokens in bands 0..i, but not bands i+1..n)

    Model sizes:
    - NFIG-310M: depth=16, embed_dim=1024, n_heads=16
    - NFIG-600M: depth=20, embed_dim=1152, n_heads=16
    """

    def __init__(
        self,
        codebook_size: int = 4096,
        token_counts: Optional[List[int]] = None,
        n_classes: int = 1000,
        embed_dim: int = 1024,
        depth: int = 16,
        n_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        class_dropout_prob: float = 0.1,
    ):
        """
        Args:
            codebook_size: vocabulary size (K)
            token_counts: number of tokens per frequency band
                          Default: [1, 4, 9, 16, 25, 36, 64, 100, 169, 256] (680 total)
            n_classes: number of ImageNet classes
            embed_dim: transformer hidden dimension
            depth: number of transformer layers
            n_heads: number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: dropout probability
            class_dropout_prob: probability of dropping class label (for CFG training)
        """
        super().__init__()

        if token_counts is None:
            # Default: scale_factors=[1,2,3,4,5,6,8,10,13,16] -> s^2 tokens per band
            token_counts = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]

        self.token_counts = token_counts
        self.n_bands = len(token_counts)
        self.total_tokens = sum(token_counts)
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.class_dropout_prob = class_dropout_prob
        self.n_classes = n_classes

        # Token embedding (shared across all bands)
        self.tok_emb = nn.Embedding(codebook_size, embed_dim)

        # Learned positional embeddings per band
        self.pos_embs = nn.ParameterList([
            nn.Parameter(torch.zeros(1, n, embed_dim))
            for n in token_counts
        ])

        # Class embedding (+1 for unconditional token used in CFG)
        self.cls_emb = nn.Embedding(n_classes + 1, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Output head: predict next token logits
        self.norm_out = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, codebook_size, bias=False)

        # Pre-compute and cache the full causal mask
        self._causal_mask = None

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for p in self.pos_embs:
            nn.init.trunc_normal_(p, std=0.02)

    def get_causal_mask(self, device: torch.device) -> torch.Tensor:
        """Get (or build and cache) the full block-wise causal mask."""
        if self._causal_mask is None or self._causal_mask.device != device:
            self._causal_mask = build_causal_mask(self.token_counts, device)
        return self._causal_mask

    def _get_class_embedding(self, class_labels: torch.Tensor,
                              training: bool = True) -> torch.Tensor:
        """Get class embedding with optional dropout for CFG training."""
        if training and self.class_dropout_prob > 0:
            mask = torch.rand(class_labels.shape, device=class_labels.device)
            class_labels = torch.where(
                mask < self.class_dropout_prob,
                torch.full_like(class_labels, self.n_classes),  # unconditional token
                class_labels
            )
        return self.cls_emb(class_labels)  # (B, embed_dim)

    def forward(self, token_sequences: List[torch.Tensor],
                class_labels: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            token_sequences: list of (B, n_i) token index tensors per band
            class_labels: (B,) class indices
        Returns:
            logits: (B, total_tokens, codebook_size)
        """
        B = class_labels.shape[0]
        device = class_labels.device

        # Embed all tokens with positional embeddings
        token_embeds = []
        for tokens, pos_emb in zip(token_sequences, self.pos_embs):
            emb = self.tok_emb(tokens) + pos_emb  # (B, n_i, C)
            token_embeds.append(emb)

        x = torch.cat(token_embeds, dim=1)  # (B, total_tokens, C)

        # Class conditioning
        c = self._get_class_embedding(class_labels, training=self.training)  # (B, C)

        # Block-wise causal mask
        attn_mask = self.get_causal_mask(device)

        # Transformer forward
        for block in self.blocks:
            x = block(x, c, attn_mask)

        x = self.norm_out(x)
        logits = self.head(x)  # (B, total_tokens, codebook_size)

        return logits

    @torch.no_grad()
    def generate_fast(
        self,
        class_labels: torch.Tensor,
        cfg_scale: float = 4.5,
        top_k: int = 990,
        temperature: float = 1.0,
    ) -> List[torch.Tensor]:
        """
        Fast autoregressive generation with classifier-free guidance.

        Generates all tokens in each band simultaneously (band-parallel generation),
        following the VAR approach. This is the primary inference method.

        Args:
            class_labels: (B,) class indices
            cfg_scale: classifier-free guidance scale (paper uses 4.5)
            top_k: top-k sampling parameter (paper uses 990)
            temperature: sampling temperature
        Returns:
            generated_tokens: list of (B, n_i) token tensors per band
        """
        B = class_labels.shape[0]
        device = class_labels.device
        uncond_labels = torch.full_like(class_labels, self.n_classes)

        generated_tokens = []
        past_embeds = None  # (B, tokens_so_far, C)

        c_cond = self.cls_emb(class_labels)
        c_uncond = self.cls_emb(uncond_labels)

        for band_idx in range(self.n_bands):
            n_tokens = self.token_counts[band_idx]
            pos_emb = self.pos_embs[band_idx]  # (1, n_tokens, C)

            if past_embeds is None:
                # First band: use positional embeddings as input
                x = pos_emb.expand(B, -1, -1)  # (B, n_tokens, C)
            else:
                # Append positional embeddings for current band
                x = torch.cat([past_embeds, pos_emb.expand(B, -1, -1)], dim=1)

            # Build causal mask for current sequence length
            counts_so_far = [self.token_counts[i] for i in range(band_idx)] + [n_tokens]
            attn_mask = build_causal_mask(counts_so_far, device)

            # Conditional forward
            x_cond = x.clone()
            for block in self.blocks:
                x_cond = block(x_cond, c_cond, attn_mask)
            x_cond = self.norm_out(x_cond)
            logits_cond = self.head(x_cond[:, -n_tokens:, :])  # (B, n_tokens, V)

            # Unconditional forward
            x_uncond = x.clone()
            for block in self.blocks:
                x_uncond = block(x_uncond, c_uncond, attn_mask)
            x_uncond = self.norm_out(x_uncond)
            logits_uncond = self.head(x_uncond[:, -n_tokens:, :])

            # Classifier-free guidance
            logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
            logits = logits / temperature

            # Top-k sampling
            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                topk_logits, _ = torch.topk(logits, top_k_val, dim=-1)
                threshold = topk_logits[:, :, -1:].expand_as(logits)
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            # Sample tokens
            B_n, T_n, V = probs.shape
            tokens = torch.multinomial(probs.reshape(B_n * T_n, V), 1).reshape(B_n, T_n)

            generated_tokens.append(tokens)

            # Update past_embeds with actual token embeddings
            tok_embs = self.tok_emb(tokens) + pos_emb.expand(B, -1, -1)
            if past_embeds is None:
                past_embeds = tok_embs
            else:
                past_embeds = torch.cat([past_embeds, tok_embs], dim=1)

        return generated_tokens


def nfig_310m(codebook_size: int = 4096, n_classes: int = 1000,
              token_counts: Optional[List[int]] = None) -> NFIGTransformer:
    """
    NFIG-310M model.
    depth=16, embed_dim=1024, n_heads=16
    Achieves FID=2.81, IS=332.42 on ImageNet-256 after 350 epochs.
    """
    return NFIGTransformer(
        codebook_size=codebook_size,
        token_counts=token_counts,
        n_classes=n_classes,
        embed_dim=1024,
        depth=16,
        n_heads=16,
        mlp_ratio=4.0,
        dropout=0.0,
        class_dropout_prob=0.1,
    )


def nfig_600m(codebook_size: int = 4096, n_classes: int = 1000,
              token_counts: Optional[List[int]] = None) -> NFIGTransformer:
    """
    NFIG-600M model.
    depth=20, embed_dim=1280, n_heads=16 (~600M parameters)
    """
    return NFIGTransformer(
        codebook_size=codebook_size,
        token_counts=token_counts,
        n_classes=n_classes,
        embed_dim=1280,
        depth=20,
        n_heads=16,
        mlp_ratio=4.0,
        dropout=0.0,
        class_dropout_prob=0.1,
    )
