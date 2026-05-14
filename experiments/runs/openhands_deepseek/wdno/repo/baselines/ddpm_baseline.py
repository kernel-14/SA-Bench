"""DDPM baseline implementation for comparison.

This implements a standard DDPM in the original space-time domain (not wavelet).
Used as comparison in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResnetBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock2D(nn.Module):
    def __init__(self, channels, heads=4, head_dim=32):
        super().__init__()
        self.heads = heads
        self.hidden_dim = heads * head_dim
        self.scale = head_dim ** -0.5
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.to_qkv = nn.Conv2d(channels, self.hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(self.hidden_dim, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.to_qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(B, self.heads, -1, H * W) * self.scale
        k = k.reshape(B, self.heads, -1, H * W)
        v = v.reshape(B, self.heads, -1, H * W)
        attn = torch.einsum('bhid,bhjd->bhij', q, k).softmax(-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.reshape(B, -1, H, W)
        return self.to_out(out) + x


class DDPMUNet2D(nn.Module):
    """Simple 2D U-Net for DDPM baseline in original domain."""
    
    def __init__(self, in_channels, cond_channels=0, init_dim=128, dim_mult=(1, 2, 4, 8)):
        super().__init__()
        total_in = in_channels + cond_channels
        time_dim = init_dim * 4
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(init_dim),
            nn.Linear(init_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        self.init_conv = nn.Conv2d(total_in, init_dim, 3, padding=1)
        
        # Downsample
        self.downs = nn.ModuleList()
        dims = [init_dim]
        cur = init_dim
        for i, mult in enumerate(dim_mult):
            out = init_dim * mult
            self.downs.append(nn.ModuleList([
                ResnetBlock2D(cur, out, time_dim),
                ResnetBlock2D(out, out, time_dim),
                AttentionBlock2D(out) if i >= 2 else nn.Identity(),
                nn.Conv2d(out, out, 3, stride=2, padding=1),
            ]))
            dims.append(out)
            cur = out
        
        # Middle
        mid = init_dim * dim_mult[-1]
        self.mid1 = ResnetBlock2D(mid, mid, time_dim)
        self.mid_attn = AttentionBlock2D(mid)
        self.mid2 = ResnetBlock2D(mid, mid, time_dim)
        
        # Upsample
        self.ups = nn.ModuleList()
        for i in reversed(range(len(dim_mult))):
            out = init_dim * dim_mult[i]
            skip_dim = dims[i]
            self.ups.append(nn.ModuleList([
                ResnetBlock2D(cur + skip_dim, out, time_dim),
                ResnetBlock2D(out, out, time_dim),
                AttentionBlock2D(out) if i >= 2 else nn.Identity(),
                nn.ConvTranspose2d(out, out, 4, stride=2, padding=1),
            ]))
            cur = out
        
        self.final = nn.Sequential(
            nn.GroupNorm(8, init_dim),
            nn.SiLU(),
            nn.Conv2d(init_dim, in_channels, 3, padding=1),
        )
    
    def forward(self, x, time, cond=None):
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        t_emb = self.time_mlp(time)
        h = self.init_conv(x)
        skips = [h]
        
        for res1, res2, attn, downsample in self.downs:
            h = res1(h, t_emb)
            h = res2(h, t_emb)
            if isinstance(attn, AttentionBlock2D):
                h = attn(h)
            skips.append(h)
            h = downsample(h)
        
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)
        
        for res1, res2, attn, upsample in self.ups:
            skip = skips.pop()
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode='nearest')
            h = torch.cat([h, skip], dim=1)
            h = res1(h, t_emb)
            h = res2(h, t_emb)
            if isinstance(attn, AttentionBlock2D):
                h = attn(h)
            h = upsample(h)
        
        if h.shape[2:] != skips[0].shape[2:]:
            h = F.interpolate(h, size=skips[0].shape[2:], mode='nearest')
        return self.final(h)


class DDPMBaseline:
    """DDPM baseline operating in original space-time domain."""
    
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.timesteps = timesteps
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    def to(self, device):
        for attr in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                     'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self
    
    def training_loss(self, denoise_fn, x_start, t, cond=None):
        noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]
        while sqrt_alpha.dim() < x_start.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        x_t = sqrt_alpha * x_start + sqrt_one_minus * noise
        pred = denoise_fn(x_t, t, cond=cond)
        return (noise - pred).pow(2).mean()
