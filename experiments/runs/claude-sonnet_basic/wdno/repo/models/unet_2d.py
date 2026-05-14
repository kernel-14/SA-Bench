"""
3D U-Net architecture for WDNO 2D experiments.

This implements the 3D denoising U-Net used in 2D PDE experiments
(incompressible fluid, ERA5).

Architecture inspired by Video Diffusion Models (Ho et al., 2022) with:
- 3D convolutions (spatial-temporal)
- Number of attention heads: 4
- Kernel size of conv3d: (3, 3, 3)
- Padding of conv3d: (1, 1, 1)
- Stride of conv3d: (1, 1, 1)
- Kernel size of downsampling: (1, 4, 4)
- Padding of downsampling: (0, 1, 1)
- Stride of downsampling: (1, 2, 2)
- DDIM sampling iterations: 100
- eta of DDIM Sampling: 1
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


class Block3D(nn.Module):
    """Basic 3D block with group norm and SiLU activation."""
    
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (3, 3, 3), padding=(1, 1, 1))
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


class ResnetBlock3D(nn.Module):
    """3D ResNet block with group normalization and time embedding."""
    
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        
        self.block1 = Block3D(dim, dim_out, groups=groups)
        self.block2 = Block3D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    
    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            scale_shift = time_emb.chunk(2, dim=1)
        
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention3D(nn.Module):
    """Linear attention for 3D data."""
    
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv3d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim)
        )
    
    def forward(self, x):
        b, c, t, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda tensor: rearrange(tensor, "b (heads c) t h w -> b heads c (t h w)", heads=self.heads),
            qkv
        )
        
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        
        q = q * self.scale
        v = v / (t * h * w)
        
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (t x y) -> b (h c) t x y", h=self.heads, t=t, x=h, y=w)
        return self.to_out(out)


class Attention3D(nn.Module):
    """Multi-head self-attention for 3D data."""
    
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)
    
    def forward(self, x):
        b, c, t, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda tensor: rearrange(tensor, "b (heads c) t h w -> b heads c (t h w)", heads=self.heads),
            qkv
        )
        
        q = q * self.scale
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (t x y) d -> b (h d) t x y", t=t, x=h, y=w)
        return self.to_out(out)


class PreNorm3D(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)
    
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Downsample3D(nn.Module):
    """3D downsampling: only spatial dimensions are downsampled."""
    
    def __init__(self, dim, dim_out=None):
        super().__init__()
        # Kernel (1, 4, 4), padding (0, 1, 1), stride (1, 2, 2)
        # Only downsamples spatial dimensions, not temporal
        self.conv = nn.Conv3d(
            dim, default(dim_out, dim),
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1)
        )
    
    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D upsampling: only spatial dimensions are upsampled."""
    
    def __init__(self, dim, dim_out=None):
        super().__init__()
        # Kernel (1, 4, 4), padding (0, 1, 1), stride (1, 2, 2)
        self.conv = nn.ConvTranspose3d(
            dim, default(dim_out, dim),
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for 2D PDE data (treating spatiotemporal data as 3D volumes).
    
    Used for 2D incompressible fluid and ERA5 experiments.
    Input data is treated as 3D (time x height x width) after wavelet transform.
    
    Architecture inspired by Video Diffusion Models (Ho et al., 2022).
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
        attn_dim_head=32,
        attn_heads=4,
    ):
        super().__init__()
        
        self.channels = channels
        self.cond_channels = cond_channels
        input_channels = channels + cond_channels
        
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv3d(input_channels, init_dim, (3, 3, 3), padding=(1, 1, 1))
        
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        def make_block(dim_in, dim_out):
            return ResnetBlock3D(dim_in, dim_out, time_emb_dim=dim * 4, groups=resnet_block_groups)
        
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
                make_block(dim_in, dim_in),
                make_block(dim_in, dim_in),
                PreNorm3D(dim_in, LinearAttention3D(dim_in, heads=attn_heads, dim_head=attn_dim_head)),
                Downsample3D(dim_in, dim_out) if not is_last else nn.Conv3d(dim_in, dim_out, (3, 3, 3), padding=(1, 1, 1)),
            ]))
        
        mid_dim = dims[-1]
        self.mid_block1 = make_block(mid_dim, mid_dim)
        self.mid_attn = PreNorm3D(mid_dim, Attention3D(mid_dim, heads=attn_heads, dim_head=attn_dim_head))
        self.mid_block2 = make_block(mid_dim, mid_dim)
        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            
            self.ups.append(nn.ModuleList([
                make_block(dim_out + dim_in, dim_out),
                make_block(dim_out + dim_in, dim_out),
                PreNorm3D(dim_out, LinearAttention3D(dim_out, heads=attn_heads, dim_head=attn_dim_head)),
                Upsample3D(dim_out, dim_in) if not is_last else nn.Conv3d(dim_out, dim_in, (3, 3, 3), padding=(1, 1, 1)),
            ]))
        
        self.out_dim = default(out_dim, channels)
        
        self.final_res_block = make_block(dim * 2, dim)
        self.final_conv = nn.Conv3d(dim, self.out_dim, 1)
    
    def forward(self, x, time, cond=None):
        """
        Forward pass.
        
        Args:
            x: Noisy wavelet coefficients (B, C, T, H, W)
            time: Diffusion timestep (B,)
            cond: Conditioning information (B, cond_C, T, H, W)
        
        Returns:
            Predicted noise (B, C, T, H, W)
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
