"""
1D U-Net architecture for WDNO.

This implements the denoising U-Net used in 1D PDE experiments
(Burgers' equation, advection, compressible Navier-Stokes).

Architecture follows DDPM (Ho et al., 2020) with:
- Initial dimension: 128
- 4 downsampling/upsampling layers
- Convolution kernel size: 3
- Dimension multiplier: [1, 2, 4, 8]
- ResNet block groups: 8
- Attention hidden dimension: 32
- Attention heads: 4
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time step embeddings for diffusion models."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResnetBlock(nn.Module):
    """ResNet block with group normalization and time embedding."""
    
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    
    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)
        
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Block(nn.Module):
    """Basic block with group norm and SiLU activation."""
    
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
    
    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        
        x = self.act(x)
        return x


class LinearAttention(nn.Module):
    """Linear attention for efficient computation."""
    
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim)
        )
    
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        
        q = q * self.scale
        v = v / (h * w)
        
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        
        q = q * self.scale
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)
    
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Downsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        self.conv = nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, default(dim_out, dim), 4, 2, 1)
    
    def forward(self, x):
        return self.conv(x)


class UNet1D(nn.Module):
    """
    2D U-Net for 1D PDE data (treating spatiotemporal data as 2D images).
    
    Used for 1D Burgers' equation, advection, and compressible NS experiments.
    Input data is treated as 2D (time x space) after wavelet transform.
    
    Architecture:
    - Initial dim: 128
    - 4 downsampling/upsampling layers
    - Kernel size: 3
    - Dim multiplier: [1, 2, 4, 8]
    - ResNet block groups: 8
    - Attention hidden dim: 32
    - Attention heads: 4
    """
    
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        cond_channels=0,
        resnet_block_groups=8,
        learned_variance=False,
        attn_dim_head=32,
        attn_heads=4,
    ):
        super().__init__()
        
        self.channels = channels
        self.cond_channels = cond_channels
        input_channels = channels + cond_channels
        
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)
        
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        block_klass = lambda dim_in, dim_out: ResnetBlock(
            dim_in, dim_out, time_emb_dim=dim * 4, groups=resnet_block_groups
        )
        
        # Time embeddings
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Downsampling
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                PreNorm(dim_in, LinearAttention(dim_in, heads=attn_heads, dim_head=attn_dim_head)),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
            ]))
        
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = PreNorm(mid_dim, Attention(mid_dim, heads=attn_heads, dim_head=attn_dim_head))
        self.mid_block2 = block_klass(mid_dim, mid_dim)
        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out),
                block_klass(dim_out + dim_in, dim_out),
                PreNorm(dim_out, LinearAttention(dim_out, heads=attn_heads, dim_head=attn_dim_head)),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
            ]))
        
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        
        self.final_res_block = block_klass(dim * 2, dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
    
    def forward(self, x, time, cond=None):
        """
        Forward pass.
        
        Args:
            x: Noisy wavelet coefficients (B, C, H, W)
            time: Diffusion timestep (B,)
            cond: Conditioning information (B, cond_C, H, W)
        
        Returns:
            Predicted noise (B, C, H, W)
        """
        if exists(cond):
            x = torch.cat([x, cond], dim=1)
        
        x = self.init_conv(x)
        r = x.clone()
        
        t = self.time_mlp(time)
        
        h = []
        
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            
            x = downsample(x)
        
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            
            x = upsample(x)
        
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)
