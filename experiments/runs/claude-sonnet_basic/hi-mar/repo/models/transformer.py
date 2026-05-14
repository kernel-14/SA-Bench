"""
Scale-aware Transformer blocks for Hi-MAR.
Implements the masked autoregressive Transformer backbone with scale conditioning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial


def modulate(x, shift, scale):
    """Apply AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    Generate 2D sinusoidal positional embeddings.
    grid_size: int of the grid height and width
    return: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class ScaleEmbedder(nn.Module):
    """
    Embeds scale/resolution information into a vector.
    Uses sinusoidal embedding followed by MLP layers.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def scale_embedding(scale, dim, max_period=10000):
        """Create sinusoidal scale embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=scale.device)
        args = scale[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, scale):
        """
        Args:
            scale: scale indices [B] (e.g., 0 for low-res, 1 for high-res)
        """
        scale_freq = self.scale_embedding(scale, self.frequency_embedding_size)
        scale_emb = self.mlp(scale_freq)
        return scale_emb


class ScaleAwareTransformerBlock(nn.Module):
    """
    Scale-aware Transformer block with AdaLN-Zero conditioning.
    Incorporates scale information via learnable scale vectors.

    From the paper (Eq. 2):
        v_tilde = a * v + b
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = split(v_tilde)
        z_a = z^i + gamma1 * Attention(alpha1 * LN(z^i) + beta1)
        z^{i+1} = z_a + gamma2 * FFN(alpha2 * LN(z_a) + beta2)
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # AdaLN-Zero: produces alpha1, beta1, gamma1, alpha2, beta2, gamma2
        # Initialized to zero so initial behavior is identity
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, attn_mask=None):
        """
        Args:
            x: input tokens [B, N, hidden_size]
            c: conditioning vector [B, hidden_size] (scale + class/text condition)
            attn_mask: optional attention mask
        Returns:
            output tokens [B, N, hidden_size]
        """
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.adaLN_modulation(c).chunk(6, dim=-1)

        # Self-attention with AdaLN
        x_norm = alpha1.unsqueeze(1) * self.norm1(x) + beta1.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + gamma1.unsqueeze(1) * attn_out

        # FFN with AdaLN
        x_norm = alpha2.unsqueeze(1) * self.norm2(x) + beta2.unsqueeze(1)
        x = x + gamma2.unsqueeze(1) * self.ff(x_norm)

        return x


class ClassEmbedder(nn.Module):
    """Class label embedder for class-conditional generation."""

    def __init__(self, num_classes, hidden_size, dropout_prob=0.1):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """Drops labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train=True, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class MaskedAutoregressiveTransformer(nn.Module):
    """
    Masked Autoregressive Transformer backbone for Hi-MAR.
    Processes tokens with bidirectional attention and scale-aware conditioning.
    """

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_channels=16,
        hidden_size=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=1000,
        dropout=0.0,
        class_dropout_prob=0.1,
        mask_ratio_min=0.7,
        mask_ratio_max=1.0,
        grad_checkpointing=False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.grad_checkpointing = grad_checkpointing

        # Number of tokens
        self.seq_len = (img_size // patch_size) ** 2

        # Token embedding
        self.token_embed = nn.Linear(in_channels, hidden_size)

        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, hidden_size), requires_grad=False
        )

        # Class embedding
        self.class_embed = ClassEmbedder(num_classes, hidden_size, class_dropout_prob)

        # Scale embedding (for scale-aware conditioning)
        self.scale_embed = ScaleEmbedder(hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ScaleAwareTransformerBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Output norm
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            int(self.seq_len ** 0.5),
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize mask token
        nn.init.normal_(self.mask_token, std=0.02)

        # Initialize linear layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask, class_labels, scale_id, train=True):
        """
        Args:
            x: input tokens [B, N, in_channels]
            mask: binary mask [B, N], 1 = masked, 0 = unmasked
            class_labels: class labels [B]
            scale_id: scale identifier [B] (0 for low-res, 1 for high-res)
            train: whether in training mode (for class dropout)
        Returns:
            conditional tokens [B, N, hidden_size]
        """
        B, N, C = x.shape

        # Embed tokens
        x = self.token_embed(x)  # [B, N, hidden_size]

        # Add positional embedding
        x = x + self.pos_embed[:, :N, :]

        # Replace masked tokens with mask token
        mask_tokens = self.mask_token.expand(B, N, -1)
        mask_expanded = mask.unsqueeze(-1).float()
        x = x * (1 - mask_expanded) + mask_tokens * mask_expanded

        # Get class conditioning
        class_emb = self.class_embed(class_labels, train=train)  # [B, hidden_size]

        # Get scale conditioning
        scale_emb = self.scale_embed(scale_id)  # [B, hidden_size]

        # Combined conditioning: class + scale
        c = class_emb + scale_emb  # [B, hidden_size]

        # Process through Transformer blocks
        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(block, x, c)
            else:
                x = block(x, c)

        x = self.norm(x)
        return x
