"""Vision Transformer (ViT) backbone for PEFT study.

Based on Dosovitskiy et al. "An Image is Worth 16x16 Words" (ICLR 2021).
Supports ImageNet-21K and CLIP pretrained variants.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # B, E, H/P, W/P
        x = x.flatten(2).transpose(1, 2)  # B, N, E
        return x


class MlpBlock(nn.Module):
    """MLP block: GELU(x*W1+b1)*W2+b2."""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim, num_heads=12, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, heads, N, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerNorm(nn.Module):
    """Layer Normalization with optional bias."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class DropPath(nn.Module):
    """Stochastic depth (drop path)."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Block(nn.Module):
    """Transformer block with MSA + MLP + two LNs + two residual connections."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = LayerNorm(dim)
        self.mlp = MlpBlock(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # Intermediate feature hooks for PEFT methods:
        # h1 = Z_{m-1} (input)
        # h2 = norm1(h1)         -> LN before MSA
        # h3 = qkv(h2)           -> Q/K/V projections
        # h4 = attention output  (within MSA)
        # h5 = dropout(h4) + h1  -> after MSA residual
        # h6 = h5                (same as h5 for notation)
        # h7 = norm2(h6)         -> LN before MLP
        # h8 = fc1(h7)           -> MLP first FC
        # h9 = MLP output + h5   -> after MLP residual
        # h10 = output

    def forward(self, x):
        # h1 = x
        h1 = x
        # h2
        h2 = self.norm1(h1)
        # h3 (handled inside attn)
        # h4
        h4 = self.attn(h2)
        # h5
        h5 = self.drop_path1(h4) + h1
        # h7
        h7 = self.norm2(h5)
        # h8, h9
        h9 = self.mlp(h7)
        h9 = self.drop_path2(h9) + h5
        # h10 = output
        return h9


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) model."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        use_cls_token=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])

        self.norm = LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1) if self.cls_token is not None else None
        if cls_tokens is not None:
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if cls_tokens is not None:
            x = x[:, 0]  # class token
        else:
            x = x.mean(dim=1)  # global average pooling
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def vit_base_patch16_224(**kwargs):
    """ViT-B/16: 86M parameters."""
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        **kwargs,
    )
    return model
