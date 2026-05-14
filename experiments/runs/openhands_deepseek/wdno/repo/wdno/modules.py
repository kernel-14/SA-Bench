import torch
import torch.nn as nn
from .layers import (
    SinusoidalPositionEmbedding, ResnetBlock, ResnetBlock3D,
    AttentionBlock, AttentionBlock3D,
    Downsample, Upsample, Downsample3D, Upsample3D,
    TemporalDownsample3D, TemporalUpsample3D
)


class UNet2D(nn.Module):
    """2D U-Net for 1D PDE problems (time x space).
    Used as epsilon_theta and epsilon_phi in WDNO.
    
    Architecture: input -> initial conv -> downsample blocks -> middle block -> upsample blocks -> output conv
    
    From paper Table 18:
    - init_dim: 128
    - down/up layers: 4
    - kernel_size: 3
    - dim_mult: [1, 2, 4, 8]
    - resnet_groups: 8
    - attn_hidden_dim: 32
    - attn_heads: 4
    """
    
    def __init__(self, in_channels, out_channels=None, cond_channels=0,
                 init_dim=128, dim_mult=(1, 2, 4, 8), down_up_layers=4,
                 kernel_size=3, resnet_groups=8, attn_heads=4, attn_hidden_dim=32,
                 dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = out_channels or in_channels
        total_in = in_channels + cond_channels
        
        # Time embedding
        time_dim = init_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(init_dim),
            nn.Linear(init_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(total_in, init_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        
        # Downsample path
        self.downs = nn.ModuleList([])
        dims = [init_dim]
        current_dim = init_dim
        for i in range(down_up_layers):
            out_dim = init_dim * dim_mult[i]
            self.downs.append(nn.ModuleList([
                ResnetBlock(current_dim, out_dim, time_dim, resnet_groups, dropout),
                ResnetBlock(out_dim, out_dim, time_dim, resnet_groups, dropout),
                AttentionBlock(out_dim, attn_heads, attn_hidden_dim) if i >= 2 else nn.Identity(),
                Downsample(out_dim),
            ]))
            dims.append(out_dim)
            current_dim = out_dim
        
        # Middle block
        mid_dim = init_dim * dim_mult[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_dim, resnet_groups, dropout)
        self.mid_attn = AttentionBlock(mid_dim, attn_heads, attn_hidden_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_dim, resnet_groups, dropout)
        
        # Upsample path
        self.ups = nn.ModuleList([])
        for i in reversed(range(down_up_layers)):
            out_dim = init_dim * dim_mult[i]
            skip_dim = dims[i]
            self.ups.append(nn.ModuleList([
                ResnetBlock(current_dim + skip_dim, out_dim, time_dim, resnet_groups, dropout),
                ResnetBlock(out_dim, out_dim, time_dim, resnet_groups, dropout),
                AttentionBlock(out_dim, attn_heads, attn_hidden_dim) if i >= 2 else nn.Identity(),
                Upsample(out_dim),
            ]))
            current_dim = out_dim
        
        # Output convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(resnet_groups, init_dim),
            nn.SiLU(),
            nn.Conv2d(init_dim, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
        )
    
    def forward(self, x, time, cond=None):
        """
        Args:
            x: (B, C_in, H, W) noisy input
            time: (B,) timestep
            cond: (B, C_cond, H, W) optional conditioning
        Returns:
            (B, C_out, H, W) predicted noise
        """
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        
        time_emb = self.time_mlp(time)
        
        h = self.init_conv(x)
        skip_connections = [h]
        
        for resnet1, resnet2, attn, downsample in self.downs:
            h = resnet1(h, time_emb)
            h = resnet2(h, time_emb)
            if isinstance(attn, AttentionBlock):
                h = attn(h)
            skip_connections.append(h)
            h = downsample(h)
        
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)
        
        for resnet1, resnet2, attn, upsample in self.ups:
            skip = skip_connections.pop()
            if h.shape[2:] != skip.shape[2:]:
                h = nn.functional.interpolate(h, size=skip.shape[2:], mode='nearest')
            h = torch.cat([h, skip], dim=1)
            h = resnet1(h, time_emb)
            h = resnet2(h, time_emb)
            if isinstance(attn, AttentionBlock):
                h = attn(h)
            h = upsample(h)
        
        # Match final skip connection size
        if h.shape[2:] != skip_connections[0].shape[2:]:
            h = nn.functional.interpolate(h, size=skip_connections[0].shape[2:], mode='nearest')
        
        return self.final_conv(h)


class UNet3D(nn.Module):
    """3D U-Net for 2D PDE problems (time x space_x x space_y).
    Used for 2D incompressible fluid and ERA5.
    
    Architecture adapted from Table 20 and paper description.
    Uses 3D convolutions throughout.
    init_dim: 100
    kernel_size: (3,3,3)
    attn_heads: 4
    """
    
    def __init__(self, in_channels, out_channels=None, cond_channels=0,
                 init_dim=100, dim_mult=(1, 2, 2, 4), down_up_layers=3,
                 attn_heads=4, attn_hidden_dim=32, dropout=0.0,
                 spatial_downsample_only=True):
        super().__init__()
        self.in_channels = in_channels
        out_channels = out_channels or in_channels
        total_in = in_channels + cond_channels
        self.spatial_downsample_only = spatial_downsample_only
        
        # Time embedding
        time_dim = init_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(init_dim),
            nn.Linear(init_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Initial convolution
        self.init_conv = nn.Conv3d(total_in, init_dim, kernel_size=3, padding=1)
        
        # Downsample path
        self.downs = nn.ModuleList([])
        dims = [init_dim]
        current_dim = init_dim
        for i in range(down_up_layers):
            out_dim = init_dim * dim_mult[i]
            self.downs.append(nn.ModuleList([
                ResnetBlock3D(current_dim, out_dim, time_dim, groups=8, dropout=dropout),
                ResnetBlock3D(out_dim, out_dim, time_dim, groups=8, dropout=dropout),
                AttentionBlock3D(out_dim, attn_heads, attn_hidden_dim),
                Downsample3D(out_dim) if spatial_downsample_only else nn.Identity(),
                TemporalDownsample3D(out_dim) if not spatial_downsample_only else nn.Identity(),
            ]))
            dims.append(out_dim)
            current_dim = out_dim
        
        # Middle block
        mid_dim = init_dim * dim_mult[-1]
        self.mid_block1 = ResnetBlock3D(mid_dim, mid_dim, time_dim, groups=8, dropout=dropout)
        self.mid_attn = AttentionBlock3D(mid_dim, attn_heads, attn_hidden_dim)
        self.mid_block2 = ResnetBlock3D(mid_dim, mid_dim, time_dim, groups=8, dropout=dropout)
        
        # Upsample path
        self.ups = nn.ModuleList([])
        for i in reversed(range(down_up_layers)):
            out_dim = init_dim * dim_mult[i]
            skip_dim = dims[i]
            self.ups.append(nn.ModuleList([
                ResnetBlock3D(current_dim + skip_dim, out_dim, time_dim, groups=8, dropout=dropout),
                ResnetBlock3D(out_dim, out_dim, time_dim, groups=8, dropout=dropout),
                AttentionBlock3D(out_dim, attn_heads, attn_hidden_dim),
                Upsample3D(out_dim) if spatial_downsample_only else nn.Identity(),
                TemporalUpsample3D(out_dim) if not spatial_downsample_only else nn.Identity(),
            ]))
            current_dim = out_dim
        
        # Output convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, init_dim),
            nn.SiLU(),
            nn.Conv3d(init_dim, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x, time, cond=None):
        """
        Args:
            x: (B, C_in, T, H, W) noisy input
            time: (B,) timestep
            cond: (B, C_cond, T, H, W) optional conditioning
        Returns:
            (B, C_out, T, H, W) predicted noise
        """
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        
        time_emb = self.time_mlp(time)
        
        h = self.init_conv(x)
        skip_connections = [h]
        
        for resnet1, resnet2, attn, ds_spatial, ds_temporal in self.downs:
            h = resnet1(h, time_emb)
            h = resnet2(h, time_emb)
            h = attn(h)
            skip_connections.append(h)
            h = ds_spatial(h)
            h = ds_temporal(h)
        
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)
        
        for resnet1, resnet2, attn, us_spatial, us_temporal in self.ups:
            h = us_temporal(h)
            h = us_spatial(h)
            skip = skip_connections.pop()
            if h.shape[2:] != skip.shape[2:]:
                h = nn.functional.interpolate(h, size=skip.shape[2:], mode='trilinear', align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = resnet1(h, time_emb)
            h = resnet2(h, time_emb)
            h = attn(h)
        
        if h.shape[2:] != skip_connections[0].shape[2:]:
            h = nn.functional.interpolate(h, size=skip_connections[0].shape[2:], mode='trilinear', align_corners=False)
        
        return self.final_conv(h)
