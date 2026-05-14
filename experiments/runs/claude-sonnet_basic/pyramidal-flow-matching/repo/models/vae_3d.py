"""
3D Variational Autoencoder for video compression.

Based on MAGVIT-v2 architecture (Yu et al., 2024) with:
- 3D causal convolutions (each frame depends only on preceding frames)
- Asymmetric encoder-decoder
- KL regularization
- 8x8x8 compression ratio (spatial x spatial x temporal)

Trained on WebVid-10M and 6.9M SAM images from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalConv3d(nn.Module):
    """
    3D causal convolution that ensures each frame only depends on preceding frames.
    
    Implements temporal causality by padding only the past in the temporal dimension.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        temporal_stride: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.temporal_stride = temporal_stride
        
        # Temporal padding: pad only the past (causal)
        self.temporal_pad = kernel_size - 1
        
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=(temporal_stride, stride, stride),
            padding=(0, padding, padding),  # No temporal padding (handled manually)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)
        
        Returns:
            (B, C_out, T', H', W')
        """
        # Pad temporal dimension causally (only past)
        if self.temporal_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, self.temporal_pad, 0))
        return self.conv(x)


class ResBlock3D(nn.Module):
    """3D residual block with group normalization."""
    
    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = CausalConv3d(channels, channels)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.conv2 = CausalConv3d(channels, channels)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class AttentionBlock3D(nn.Module):
    """3D self-attention block for the bottleneck."""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        x = x.view(B, C, T * H * W)
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, T * H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, self.head_dim, T * H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, self.head_dim, T * H * W).transpose(2, 3)
        
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).contiguous().view(B, C, T * H * W)
        out = self.proj(out)
        out = out.view(B, C, T, H, W)
        
        return out + residual


class Encoder3D(nn.Module):
    """
    3D encoder for video compression.
    
    Achieves 8x8x8 compression ratio through:
    - 3 spatial downsampling stages (2x each = 8x total)
    - 3 temporal downsampling stages (2x each = 8x total)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        latent_channels: int = 16,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        channels = [base_channels * m for m in channel_multipliers]
        
        # Initial convolution
        self.conv_in = CausalConv3d(in_channels, channels[0])
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            block = nn.ModuleList()
            # Residual blocks
            for _ in range(num_res_blocks):
                block.append(ResBlock3D(channels[i]))
            # Downsampling (spatial and temporal)
            block.append(CausalConv3d(
                channels[i], channels[i + 1],
                kernel_size=3, stride=2, padding=1,
                temporal_stride=2,
            ))
            self.down_blocks.append(block)
        
        # Bottleneck
        self.mid_block1 = ResBlock3D(channels[-1])
        self.mid_attn = AttentionBlock3D(channels[-1])
        self.mid_block2 = ResBlock3D(channels[-1])
        
        # Output
        self.norm_out = nn.GroupNorm(32, channels[-1])
        self.act_out = nn.SiLU()
        self.conv_out = CausalConv3d(channels[-1], 2 * latent_channels)  # mean + logvar
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, T, H, W) video tensor
        
        Returns:
            Tuple of (mean, logvar) each of shape (B, latent_channels, T//8, H//8, W//8)
        """
        x = self.conv_in(x)
        
        for block in self.down_blocks:
            for layer in block:
                x = layer(x)
        
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
        
        x = self.act_out(self.norm_out(x))
        x = self.conv_out(x)
        
        mean, logvar = x.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30, 20)
        
        return mean, logvar


class Decoder3D(nn.Module):
    """
    3D decoder for video reconstruction.
    
    Asymmetric design (larger than encoder) for better reconstruction quality.
    """
    
    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        latent_channels: int = 16,
        num_res_blocks: int = 3,  # More blocks than encoder (asymmetric)
    ):
        super().__init__()
        
        channels = [base_channels * m for m in reversed(channel_multipliers)]
        
        # Initial convolution from latent
        self.conv_in = CausalConv3d(latent_channels, channels[0])
        
        # Bottleneck
        self.mid_block1 = ResBlock3D(channels[0])
        self.mid_attn = AttentionBlock3D(channels[0])
        self.mid_block2 = ResBlock3D(channels[0])
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            block = nn.ModuleList()
            # Residual blocks
            for _ in range(num_res_blocks):
                block.append(ResBlock3D(channels[i]))
            # Upsampling (spatial and temporal)
            block.append(nn.Sequential(
                nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),
                CausalConv3d(channels[i], channels[i + 1]),
            ))
            self.up_blocks.append(block)
        
        # Output
        self.norm_out = nn.GroupNorm(32, channels[-1])
        self.act_out = nn.SiLU()
        self.conv_out = CausalConv3d(channels[-1], out_channels)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_channels, T//8, H//8, W//8) latent tensor
        
        Returns:
            (B, C, T, H, W) reconstructed video
        """
        x = self.conv_in(z)
        
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
        
        for block in self.up_blocks:
            for layer in block:
                x = layer(x)
        
        x = self.act_out(self.norm_out(x))
        x = self.conv_out(x)
        
        return x


class VideoVAE(nn.Module):
    """
    3D Variational Autoencoder for video compression.
    
    Achieves 8x8x8 compression ratio (spatial x spatial x temporal).
    Architecture similar to MAGVIT-v2 with 3D causal convolutions.
    
    Trained on WebVid-10M and 6.9M SAM images from scratch.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        encoder_num_res_blocks: int = 2,
        decoder_num_res_blocks: int = 3,
        kl_weight: float = 1e-6,
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.kl_weight = kl_weight
        
        self.encoder = Encoder3D(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            latent_channels=latent_channels,
            num_res_blocks=encoder_num_res_blocks,
        )
        
        self.decoder = Decoder3D(
            out_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            latent_channels=latent_channels,
            num_res_blocks=decoder_num_res_blocks,
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode video to latent space.
        
        Args:
            x: (B, C, T, H, W) video tensor, normalized to [-1, 1]
        
        Returns:
            Tuple of (mean, logvar) latent distributions
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to video.
        
        Args:
            z: (B, latent_channels, T//8, H//8, W//8) latent tensor
        
        Returns:
            (B, C, T, H, W) reconstructed video
        """
        return self.decoder(z)
    
    def reparameterize(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from the latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode, sample, decode.
        
        Args:
            x: (B, C, T, H, W) input video
            sample_posterior: Whether to sample from posterior or use mean
        
        Returns:
            Tuple of (reconstruction, mean, logvar)
        """
        mean, logvar = self.encode(x)
        
        if sample_posterior:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean
        
        reconstruction = self.decode(z)
        
        return reconstruction, mean, logvar
    
    def compute_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute VAE loss = reconstruction loss + KL divergence.
        
        Args:
            x: Original video
            reconstruction: Reconstructed video
            mean, logvar: Latent distribution parameters
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Reconstruction loss (L1 + perceptual)
        recon_loss = F.l1_loss(reconstruction, x)
        
        # KL divergence: -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return total_loss, {
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item(),
        }
    
    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video to latent space (inference mode, returns mean).
        
        Args:
            video: (B, C, T, H, W) video tensor
        
        Returns:
            Latent tensor (B, latent_channels, T//8, H//8, W//8)
        """
        mean, _ = self.encode(video)
        return mean
    
    @torch.no_grad()
    def decode_video(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to video (inference mode).
        
        Args:
            z: Latent tensor (B, latent_channels, T//8, H//8, W//8)
        
        Returns:
            Reconstructed video (B, C, T, H, W)
        """
        return self.decode(z)
