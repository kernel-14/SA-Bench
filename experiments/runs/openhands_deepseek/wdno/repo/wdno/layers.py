import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for diffusion timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResnetBlock(nn.Module):
    """Residual block with time embedding for 1D/2D U-Net."""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        h = h + self.time_emb_proj(time_emb)[:, :, None, None]
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class ResnetBlock3D(nn.Module):
    """Residual block with time embedding for 3D U-Net."""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        h = h + self.time_emb_proj(time_emb)[:, :, None, None, None]
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Multi-head attention block for 2D feature maps."""
    
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
        q, k, v = rearrange(qkv, 'b (qkv heads d) h w -> qkv b heads (h w) d', 
                           qkv=3, heads=self.heads)
        
        q = q * self.scale
        attn = torch.einsum('bhid,bhjd->bhij', q, k)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b heads (h w) d -> b (heads d) h w', h=H, w=W)
        return self.to_out(out) + x


class AttentionBlock3D(nn.Module):
    """Multi-head attention block for 3D feature maps. 
    Uses axial attention: applies spatial attention per frame.
    """
    
    def __init__(self, channels, heads=4, head_dim=32):
        super().__init__()
        self.heads = heads
        self.hidden_dim = heads * head_dim
        self.scale = head_dim ** -0.5
        
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.to_qkv = nn.Conv3d(channels, self.hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(self.hidden_dim, channels, 1)
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        h = self.norm(x)
        h = rearrange(h, 'b c t h w -> (b t) c h w')
        qkv = self.to_qkv(h)
        qkv = rearrange(qkv, '(b t) (qkv heads d) h w -> qkv (b t) heads (h w) d',
                       qkv=3, heads=self.heads, t=T)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = torch.einsum('bhid,bhjd->bhij', q, k)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, '(b t) heads (h w) d -> (b t) (heads d) h w', 
                       t=T, h=H, w=W)
        out = rearrange(out, '(b t) c h w -> b c t h w', b=B)
        return self.to_out(out) + x


class Downsample(nn.Module):
    """Downsampling layer for 2D."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer for 2D."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class Downsample3D(nn.Module):
    """Downsampling layer for 3D (spatial-only downsampling: H, W).
    From Table 20: kernel (1,4,4), padding (1,2,2), stride different.
    Actually paper says stride (0,1,1) which is unusual - we interpret as
    stride (1,2,2) for spatial downsampling.
    """
    
    def __init__(self, channels, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2)):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
    
    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    """Upsampling layer for 3D (spatial-only upsampling: H, W).
    From Table 20: kernel (1,4,4), padding (0,1,1), stride (1,2,2) - same pattern.
    """
    
    def __init__(self, channels, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2)):
        super().__init__()
        self.conv = nn.ConvTranspose3d(channels, channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding)
    
    def forward(self, x):
        return self.conv(x)


class TemporalDownsample3D(nn.Module):
    """Temporal-only downsampling for 3D data."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), 
                              stride=(2, 1, 1), padding=(1, 0, 0))
    
    def forward(self, x):
        return self.conv(x)


class TemporalUpsample3D(nn.Module):
    """Temporal-only upsampling for 3D data."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose3d(channels, channels, kernel_size=(4, 1, 1),
                                       stride=(2, 1, 1), padding=(1, 0, 0))
    
    def forward(self, x):
        return self.conv(x)
