"""
U-Net architecture for Flow Matching and Diffusion models.
Based on the architecture from Rombach et al. 2022 (Stable Diffusion),
adapted for latent space Flow Matching.
"""
import math
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool = True, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool = True, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = GroupNorm32(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, -1)
        qkv = self.qkv(h)
        q, k, v = qkv.reshape(B * self.num_heads, C // self.num_heads * 3, -1).split(C // self.num_heads, dim=1)
        scale = 1.0 / math.sqrt(C // self.num_heads)
        attn = torch.einsum("bci,bcj->bij", q, k) * scale
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bij,bcj->bci", attn, v)
        h = h.reshape(B, C, -1)
        h = self.proj_out(h)
        return (x + h.reshape(B, C, H, W)).to(x.dtype)


class CrossAttention(nn.Module):
    """
    Cross-attention for text conditioning.
    """
    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, H * W).permute(0, 2, 1)
        
        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q = q.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, -1, C)
        
        return self.to_out(out).permute(0, 2, 1).reshape(B, C, H, W)


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.in_layers = nn.Sequential(
            GroupNorm32(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        
        if use_scale_shift_norm:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_channels, 2 * self.out_channels),
            )
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_channels, self.out_channels),
            )

        self.out_layers = nn.Sequential(
            GroupNorm32(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        if self.use_scale_shift_norm:
            scale, shift = emb_out.chunk(2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out
            
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class TransformerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, context_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn1 = AttentionBlock(channels, num_heads)
        self.attn2 = CrossAttention(channels, context_dim, num_heads, dropout)
        self.ff = nn.Sequential(
            GroupNorm32(32, channels),
            nn.Conv2d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, 1),
        )
        self.norm1 = GroupNorm32(32, channels)
        self.norm2 = GroupNorm32(32, channels)
        self.norm3 = GroupNorm32(32, channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        x = x + self.ff(self.norm3(x))
        return x


class UNetModel(nn.Module):
    """
    U-Net model for Flow Matching with text conditioning.
    Architecture based on Stable Diffusion's U-Net.
    """
    def __init__(
        self,
        in_channels: int = 4,
        model_channels: int = 320,
        out_channels: int = 4,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [4, 2, 1],
        dropout: float = 0.0,
        channel_mult: List[int] = [1, 2, 4, 4],
        num_heads: int = 8,
        transformer_depth: int = 1,
        context_dim: int = 768,
        use_linear_in_transformer: bool = True,
        image_size: int = 64,
        conv_resample: bool = True,
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.context_dim = context_dim
        self.transformer_depth = transformer_depth
        
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])
        
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch, time_embed_dim, model_channels * mult,
                        dropout=dropout, use_scale_shift_norm=use_scale_shift_norm
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    for _ in range(transformer_depth):
                        layers.append(TransformerBlock(ch, num_heads, context_dim, dropout))
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(Downsample(ch, conv_resample))
                input_block_chans.append(ch)
                ds *= 2
        
        self.middle_block = nn.Sequential(
            ResBlock(ch, time_embed_dim, dropout=dropout, use_scale_shift_norm=use_scale_shift_norm),
            *[TransformerBlock(ch, num_heads, context_dim, dropout) for _ in range(transformer_depth)],
            ResBlock(ch, time_embed_dim, dropout=dropout, use_scale_shift_norm=use_scale_shift_norm),
        )
        
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich, time_embed_dim, model_channels * mult,
                        dropout=dropout, use_scale_shift_norm=use_scale_shift_norm
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    for _ in range(transformer_depth):
                        layers.append(TransformerBlock(ch, num_heads, context_dim, dropout))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
        
        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, out_channels, 3, padding=1)),
        )

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)
        
        h = x
        for module in self.input_blocks:
            if isinstance(module, (ResBlock, nn.Sequential)):
                h = module[0](h, emb) if isinstance(module, nn.Sequential) else module(h, emb)
                for sub_module in (module[1:] if isinstance(module, nn.Sequential) else []):
                    if isinstance(sub_module, TransformerBlock):
                        h = sub_module(h, context)
                    elif isinstance(sub_module, ResBlock):
                        h = sub_module(h, emb)
            else:
                h = module(h)
            hs.append(h)
        
        h = self.middle_block[0](h, emb)
        for layer in self.middle_block[1:]:
            if isinstance(layer, TransformerBlock):
                h = layer(h, context)
            else:
                h = layer(h, emb)
        
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module[0](h, emb)
            for sub_module in module[1:]:
                if isinstance(sub_module, TransformerBlock):
                    h = sub_module(h, context)
                elif isinstance(sub_module, ResBlock):
                    h = sub_module(h, emb)
                else:
                    h = sub_module(h)
        
        return self.out(h)
