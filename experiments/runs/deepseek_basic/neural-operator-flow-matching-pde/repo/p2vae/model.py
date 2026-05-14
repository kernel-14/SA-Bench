"""P2VAE: Pretrained Physics Variational Autoencoder.

Based on SD-VAE architecture (Rombach et al., 2022).
Compresses c3p128 PDE snapshots to c16p16 latent grids (12x compression).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

from .config import P2VAEConfig


class GroupNorm(nn.GroupNorm):
    """GroupNorm with optional fp32 computation."""
    def forward(self, x):
        if x.dtype == torch.float16:
            return super().forward(x.float()).to(x.dtype)
        return super().forward(x)


class ResnetBlock(nn.Module):
    """Residual block with two convolutions and optional time embedding."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 temb_channels: Optional[int] = None, 
                 dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1)
        
        if temb_channels is not None:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.temb_proj = None
            
        self.norm2 = GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1)
        
        self.nin_shortcut = None
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=1, stride=1, padding=0)
    
    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        if self.temb_proj is not None and temb is not None:
            h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        
        return x + h


class Downsample(nn.Module):
    """Downsampling layer with 2x spatial reduction."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, 
                              stride=2, padding=0)
    
    def forward(self, x):
        # padding: left=1, right=0, top=1, bottom=0
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer with 2x spatial increase."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3,
                              stride=1, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


class AttnBlock(nn.Module):
    """Self-attention block for 2D feature maps."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.norm = GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5
    
    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        
        B, C, H, W = q.shape
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # B, HW, C
        k = k.reshape(B, C, H * W)  # B, C, HW
        v = v.reshape(B, C, H * W).permute(0, 2, 1)  # B, HW, C
        
        attn = torch.bmm(q, k) * self.scale  # B, HW, HW
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(attn, v)  # B, HW, C
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        out = self.proj_out(out)
        
        return x + out


class Encoder(nn.Module):
    """Encoder of P2VAE.
    
    Encodes c3p128 -> c16p16 latent (mu, logvar).
    """
    
    def __init__(self, config: P2VAEConfig):
        super().__init__()
        self.config = config
        ch = config.base_dim
        ch_mult = config.ch_mult
        num_res_blocks = config.num_res_blocks
        z_channels = config.z_channels
        in_channels = config.in_channels
        
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, 
                                 stride=1, padding=1)
        
        # Resolution chain: 128 -> 64 -> 32 -> 16 -> 16
        # ch_mult = (1, 2, 4, 4) gives channel counts: ch, 2ch, 4ch, 4ch
        self.down = nn.ModuleList()
        block_in = ch
        for i_level, mult in enumerate(ch_mult):
            block_out = ch * mult
            for _ in range(num_res_blocks):
                self.down.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            if i_level < len(ch_mult) - 1:
                self.down.append(Downsample(block_in))
        
        # Middle block
        self.mid = nn.ModuleList([
            ResnetBlock(block_in, block_in),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in),
        ])
        
        # Output
        self.norm_out = GroupNorm(32, block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3,
                                   stride=1, padding=1)
    
    def forward(self, x):
        h = self.conv_in(x)
        for layer in self.down:
            h = layer(h)
        for layer in self.mid:
            h = layer(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        # Split into mu and logvar
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder of P2VAE.
    
    Decodes c16p16 latent -> c3p128.
    """
    
    def __init__(self, config: P2VAEConfig):
        super().__init__()
        self.config = config
        ch = config.base_dim
        ch_mult = config.ch_mult
        num_res_blocks = config.num_res_blocks
        z_channels = config.z_channels
        out_channels = config.out_channels
        
        # Reverse multipliers for decoder
        ch_mult_rev = tuple(reversed(ch_mult))
        block_in = ch * ch_mult_rev[0]
        
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3,
                                 stride=1, padding=1)
        
        # Middle block
        self.mid = nn.ModuleList([
            ResnetBlock(block_in, block_in),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in),
        ])
        
        # Upsampling blocks
        self.up = nn.ModuleList()
        for i_level, mult in enumerate(ch_mult_rev):
            block_out = ch * mult
            for _ in range(num_res_blocks + 1):
                self.up.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            if i_level < len(ch_mult_rev) - 1:
                self.up.append(Upsample(block_in))
        
        # Output
        self.norm_out = GroupNorm(32, block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3,
                                   stride=1, padding=1)
    
    def forward(self, z):
        h = self.conv_in(z)
        for layer in self.mid:
            h = layer(h)
        for layer in self.up:
            h = layer(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class P2VAE(nn.Module):
    """Pretrained Physics Variational Autoencoder.
    
    Compresses PDE snapshots with shape (B, 3, 128, 128) to latent codes 
    with shape (B, 16, 16, 16), achieving 12x compression.
    
    Follows SD-VAE architecture with downsampling/upsampling blocks,
    residual blocks, and self-attention at the bottleneck.
    """
    
    def __init__(self, config: P2VAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        # KL weight as in the paper: β = 1e-3
        self.kl_weight = config.kl_weight
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"P2VAE parameter count: {n_params:,} ({n_params/1e6:.1f}M)")
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor (B, 3, 128, 128)
            
        Returns:
            mu, logvar: Distribution parameters (B, 16, 16, 16)
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to reconstructed input.
        
        Args:
            z: Latent tensor (B, 16, 16, 16)
            
        Returns:
            Reconstructed tensor (B, 3, 128, 128)
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor, 
                sample_posterior: bool = True) -> dict:
        """Forward pass with reparameterization.
        
        Args:
            x: Input tensor (B, 3, 128, 128)
            sample_posterior: If True, sample z ~ q(z|x); 
                              if False, use mu directly
            
        Returns:
            Dictionary with 'reconstruction', 'mu', 'logvar', 'z', 'kl_loss'
        """
        mu, logvar = self.encode(x)
        
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu
        
        reconstruction = self.decode(z)
        
        # KL divergence loss (Eq. from paper: KL(q_ω(y|x) || p(y)))
        # Prior p(y) = N(0, I)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 
                                    dim=[1, 2, 3])
        kl_loss = kl_loss.mean()
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'kl_loss': kl_loss,
        }
    
    def compute_loss(self, x: torch.Tensor) -> dict:
        """Compute the VAE loss.
        
        L_VAE = 1/2 E[||x - x_hat||^2] + β KL(q_ω(y|x) || p(y))
        
        Args:
            x: Input tensor (B, 3, 128, 128)
            
        Returns:
            Dictionary with 'loss', 'recon_loss', 'kl_loss'
        """
        output = self.forward(x)
        
        # Reconstruction loss: 1/2 ||x - x_hat||^2
        recon_loss = 0.5 * F.mse_loss(output['reconstruction'], x, 
                                       reduction='none')
        recon_loss = recon_loss.mean()
        
        # Total loss
        loss = recon_loss + self.kl_weight * output['kl_loss']
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': output['kl_loss'],
            'reconstruction': output['reconstruction'],
            'z': output['z'],
        }


def build_p2vae(config: P2VAEConfig) -> P2VAE:
    """Build a P2VAE model from configuration."""
    return P2VAE(config)
