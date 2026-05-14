## models/p2vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Any

from config import Config

# --- Internal Building Blocks for SD-VAE-like architecture (simplified) ---
# In a real project, these would typically be in models/components.py
# and imported. They are defined here to make models/p2vae.py self-contained
# for this specific task, reflecting the typical structure of SD-VAEs.

class _ConvBlock(nn.Module):
    """
    A basic convolution block: Conv2d -> GroupNorm -> SiLU (activation).
    Used as a helper within ResnetBlock and for initial/final convolutions.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, use_norm: bool = True, use_act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_norm)
        self.norm = nn.GroupNorm(32, out_channels) if use_norm else nn.Identity()
        self.act = nn.SiLU() if use_act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class _ResnetBlock(nn.Module):
    """
    A Residual Block with two ConvBlocks and a shortcut connection.
    Features GroupNorm and SiLU activations, common in SD-VAEs.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = _ConvBlock(in_channels, out_channels, use_norm=False, use_act=False) # Norm and act are external

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = _ConvBlock(out_channels, out_channels, use_norm=False, use_act=False)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)

class _Downsample(nn.Module):
    """
    Spatially downsamples feature maps by a factor of 2 using a stride-2 convolution.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class _Upsample(nn.Module):
    """
    Spatially upsamples feature maps by a factor of 2 using nearest-neighbor interpolation
    followed by a convolution.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class _SpatialAttentionBlock(nn.Module):
    """
    Self-attention block applied to feature maps (B, C, H, W).
    Flattens spatial dimensions, applies attention, then reshapes back.
    Based on typical implementations in SD-VAEs.
    """
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        if self.head_dim == 0:
            raise ValueError(f"Channels ({channels}) must be divisible by num_heads ({num_heads})")

        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        self.query = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.key = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        h_ = self.proj_in(h_)

        B, C, H, W = h_.shape
        # Reshape to (B, C, H*W) and then (B, H*W, C) for attention operation
        # Split into heads for Q, K, V
        q = self.query(h_).view(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2) # (B, H, H*W, D_head)
        k = self.key(h_).view(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)   # (B, H, H*W, D_head)
        v = self.value(h_).view(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)  # (B, H, H*W, D_head)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale # (B, H, H*W, H*W)
        attn = F.softmax(attn, dim=-1)

        h_ = torch.matmul(attn, v) # (B, H, H*W, D_head)

        # Concatenate heads and reshape back to (B, C, H, W)
        h_ = h_.transpose(-1, -2).reshape(B, C, H, W)

        h_ = self.proj_out(h_)

        return x + h_

# --- End internal building blocks ---


class P2VAEModel(nn.Module):
    """
    Pretrained Physics Variational Autoencoder (P2VAE) model.
    Compresses high-resolution physical fields into a compact latent representation
    and reconstructs them, based on the SD-VAE architecture.
    """

    def __init__(self, config: Config):
        """
        Initializes the P2VAE model, constructing its encoder and decoder components.

        Args:
            config (Config): An instance of the Config class, providing all necessary model hyperparameters.
        """
        super().__init__()

        # Retrieve essential parameters from the config object
        self.input_channels: int = config.get('dataset.target_channels', 3)
        self.input_resolution: Tuple[int, int] = tuple(config.get('dataset.target_resolution', [128, 128]))
        self.base_channels: int = config.get('p2vae_model.base_channels', 64)
        self.latent_channels: int = config.get('p2vae_model.latent_channels', 16)
        self.latent_resolution: Tuple[int, int] = tuple(config.get('p2vae_model.latent_resolution', [16, 16]))
        self.num_heads: int = config.get('fmt_model.num_heads', 8) # Reusing this for spatial attention as it's a general param
        
        # Determine the data type for model parameters and computations
        dtype_str: str = config.get('global.dtype', 'float16')
        self.dtype: torch.dtype = getattr(torch, dtype_str)
        
        if self.input_resolution[0] % self.latent_resolution[0] != 0:
            raise ValueError(f"Input resolution {self.input_resolution} must be divisible by latent resolution {self.latent_resolution}.")
        self.num_down_sampling_steps = int(torch.log2(torch.tensor(float(self.input_resolution[0]) / self.latent_resolution[0])).item())

        # Channel multipliers for different resolutions (e.g., [1, 2, 4, 4] for 3 downsampling steps)
        channel_mults = [1, 2, 4, 4] 
        
        # Ensure channel_mults is long enough for the number of downsampling steps + 1 (for the initial base_channels stage)
        while len(channel_mults) < self.num_down_sampling_steps + 1:
            channel_mults.append(channel_mults[-1]) # Extend by repeating the last multiplier

        # --- Encoder ---
        self.encoder = nn.ModuleList()
        # Initial convolution (input_channels -> base_channels)
        self.encoder.append(_ConvBlock(self.input_channels, self.base_channels, kernel_size=3, padding=1))
        
        in_ch = self.base_channels
        for i in range(self.num_down_sampling_steps):
            out_ch = self.base_channels * channel_mults[i] # Current output channels
            
            # Two ResnetBlocks
            self.encoder.append(nn.ModuleList([
                _ResnetBlock(in_ch, out_ch),
                _ResnetBlock(out_ch, out_ch)
            ]))
            
            # Optional: Add attention block at certain intermediate resolutions
            if self.input_resolution[0] // (2**(i+1)) <= 32: # e.g. at 32x32, 16x16, 8x8 resolutions
                 self.encoder.append(_SpatialAttentionBlock(out_ch, self.num_heads))

            # Downsample layer
            self.encoder.append(_Downsample(out_ch))
            in_ch = out_ch # Update in_ch for next stage
        
        # Middle Block for Encoder (at latent resolution)
        # The number of channels at this stage is `in_ch` from the last downsampling step.
        self.encoder.append(nn.ModuleList([
            _ResnetBlock(in_ch, in_ch), 
            _SpatialAttentionBlock(in_ch, self.num_heads), # Attention at lowest resolution
            _ResnetBlock(in_ch, in_ch)
        ]))
        
        # Final encoder layers: GroupNorm, SiLU, Conv to 2*latent_channels (for mu and log_var)
        self.encoder.append(nn.Sequential(
            nn.GroupNorm(32, in_ch), # Norm before final activation
            nn.SiLU(), # Activation
            nn.Conv2d(in_ch, 2 * self.latent_channels, kernel_size=3, padding=1) # Output mu and log_var
        ))

        # --- Decoder ---
        self.decoder = nn.ModuleList()
        # Initial convolution from latent channels to the highest channel depth of the encoder's latent bottleneck
        # The `in_ch` from the encoder's middle block is the starting point for the decoder's middle block.
        self.decoder.append(_ConvBlock(self.latent_channels, in_ch, kernel_size=3, padding=1))
        
        # Middle Block for Decoder (at latent resolution), mirrors encoder's middle block
        self.decoder.append(nn.ModuleList([
            _ResnetBlock(in_ch, in_ch),
            _SpatialAttentionBlock(in_ch, self.num_heads),
            _ResnetBlock(in_ch, in_ch)
        ]))

        # Reverse order of channel_mults for decoder's upsampling stages
        # We need `num_down_sampling_steps` stages, corresponding to `channel_mults` from `num_down_sampling_steps-1` down to `0`.
        channel_mults_rev = channel_mults[self.num_down_sampling_steps-1::-1]
        
        for i in range(self.num_down_sampling_steps):
            out_ch = self.base_channels * channel_mults_rev[i] # Current output channels after resnet blocks
            
            # Two ResnetBlocks
            self.decoder.append(nn.ModuleList([
                _ResnetBlock(in_ch, out_ch),
                _ResnetBlock(out_ch, out_ch)
            ]))
            
            # Optional: Add attention block at certain intermediate resolutions (mirroring encoder)
            if self.input_resolution[0] // (2**(self.num_down_sampling_steps - i)) <= 32: # e.g. at 16x16, 32x32 resolutions
                self.decoder.append(_SpatialAttentionBlock(out_ch, self.num_heads))
            
            # Upsample layer
            self.decoder.append(_Upsample(out_ch))
            in_ch = out_ch # Update in_ch for next stage
        
        # Final decoder blocks: Two ResnetBlocks to bring channels back to base_channels
        self.decoder.append(nn.ModuleList([
            _ResnetBlock(in_ch, self.base_channels),
            _ResnetBlock(self.base_channels, self.base_channels)
        ]))

        # Final output layers: GroupNorm, SiLU, Conv to input_channels
        self.decoder.append(nn.Sequential(
            nn.GroupNorm(32, self.base_channels), # Norm before final activation
            nn.SiLU(), # Activation
            nn.Conv2d(self.base_channels, self.input_channels, kernel_size=3, padding=1) # Output reconstructed image
        ))

        # Move model to specified data type
        self.to(self.dtype)

    def _sample_latent(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Implements the reparameterization trick to sample a latent vector z
        from the learned mean and log-variance.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a full forward pass through the P2VAE, encoding an input x,
        sampling a latent z, and decoding it back to x_reco.

        Args:
            x (torch.Tensor): Input tensor representing a physical field,
                              with shape (batch_size, input_channels, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - x_reco (torch.Tensor): Reconstructed field.
                - mu (torch.Tensor): Mean of the latent distribution.
                - log_var (torch.Tensor): Log-variance of the latent distribution.
                - z (torch.Tensor): Sampled latent vector.
        """
        x = x.to(self.dtype)
        
        # Encoder pass
        # The encoder is designed as a ModuleList where each item could be:
        # 1. An individual _ConvBlock (the first layer)
        # 2. A nn.ModuleList containing ResnetBlocks and potentially an AttentionBlock
        # 3. An _Downsample layer
        # 4. A nn.Sequential for the final GroupNorm+SiLU+Conv
        
        # Apply initial convolution
        current_x = self.encoder[0](x)
        
        # Apply downsampling stages and middle block
        # Start from index 1 to skip the initial conv
        # The last element is nn.Sequential for final mu/log_var projection
        for module in self.encoder[1:-1]: 
            if isinstance(module, nn.ModuleList): # Handles blocks inside each stage
                for sub_module in module:
                    current_x = sub_module(current_x)
            else: # Handles downsample layers and attention blocks directly
                current_x = module(current_x)
        
        # Apply final encoder projection for mu and log_var
        encoded_output = self.encoder[-1](current_x)
        
        # Split final encoder output into mu and log_var
        mu, log_var = torch.chunk(encoded_output, 2, dim=1) 

        # Sample latent z
        z = self._sample_latent(mu, log_var)

        # Decoder pass
        # Start decoder with sampled latent, apply initial conv
        current_x = self.decoder[0](z)

        # Apply middle block and upsampling stages
        # The last element is nn.Sequential for final output projection
        for module in self.decoder[1:-1]:
            if isinstance(module, nn.ModuleList): # Handles blocks inside each stage
                for sub_module in module:
                    current_x = sub_module(current_x)
            else: # Handles upsample layers and attention blocks directly
                current_x = module(current_x)
        
        # Apply final decoder projection for reconstruction
        x_reco = self.decoder[-1](current_x)

        return x_reco, mu, log_var, z

    @torch.no_grad()
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Provides a public interface to obtain only the sampled latent representation z
        for a given input x. This is typically used when the P2VAE weights are frozen
        during FMT training.

        Args:
            x (torch.Tensor): Input tensor representing a physical field.

        Returns:
            torch.Tensor: Sampled latent vector z.
        """
        self.eval() # Set model to evaluation mode
        x = x.to(self.dtype)

        # Encoder pass (similar to forward, but only up to mu/log_var)
        current_x = self.encoder[0](x)
        
        for module in self.encoder[1:-1]:
            if isinstance(module, nn.ModuleList):
                for sub_module in module:
                    current_x = sub_module(current_x)
            else:
                current_x = module(current_x)
        
        encoded_output = self.encoder[-1](current_x)
        mu, log_var = torch.chunk(encoded_output, 2, dim=1)

        # Sample latent z
        z = self._sample_latent(mu, log_var)
        
        return z

