
import torch
import torch.nn as nn
from einops.layers.torch import EinMix as Rearrange
from typing import Optional, List

from adjoint_matching.config import Config

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: Optional[int] = None):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
        self.act = nn.SiLU()

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, out_channels)
        else:
            self.time_proj = None

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None):
        h = self.act(self.norm1(self.conv1(x)))
        if self.time_proj is not None and time_emb is not None:
            h += self.time_proj(time_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip_connection(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(B, self.num_heads * 3, C // self.num_heads, H, W)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        k = k.view(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        v = v.view(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)

        attn = (q @ k.transpose(-1, -2)) * (C // self.num_heads)**-0.5
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(-1, -2).reshape(B, C, H, W)
        return self.proj(out) + x

class TimestepEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
        self.freq_bands = 16

    def forward(self, t):
        t_freq = torch.exp(torch.arange(0, self.freq_bands, device=t.device) * -(torch.log(torch.tensor(10000.0)) / (self.freq_bands - 1)))
        t_emb = t[:, None] * t_freq[None, :]
        t_emb = torch.cat([t_emb.sin(), t_emb.cos()], dim=-1)
        return self.mlp(t_emb)

class UNet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        unet_config = config.UNET_CONFIG
        self.in_channels = unet_config["in_channels"]
        self.out_channels = unet_config["out_channels"]
        
        # Time embedding
        self.time_embedder = TimestepEmbedder(unet_config["block_out_channels"][0] * 4) # Arbitrary choice for dim
        time_emb_dim = unet_config["block_out_channels"][0] * 4

        # Initial convolution
        self.conv_in = nn.Conv2d(self.in_channels, unet_config["block_out_channels"][0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.mid_block = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        channels = unet_config["block_out_channels"]
        num_blocks = len(channels)

        # Down blocks
        for i in range(num_blocks):
            in_c = channels[i-1] if i > 0 else channels[0]
            out_c = channels[i]
            self.down_blocks.append(nn.ModuleList([
                ResBlock(in_c, out_c, time_emb_dim),
                ResBlock(out_c, out_c, time_emb_dim),
                AttentionBlock(out_c) if out_c == unet_config["cross_attention_dim"] else nn.Identity(), # Simple conditional attention
                nn.Conv2d(out_c, out_c, 3, stride=2, padding=1) if i < num_blocks - 1 else nn.Identity()
            ]))
        
        # Mid block
        self.mid_block.append(ResBlock(channels[-1], channels[-1], time_emb_dim))
        self.mid_block.append(AttentionBlock(channels[-1]))
        self.mid_block.append(ResBlock(channels[-1], channels[-1], time_emb_dim))

        # Up blocks
        for i in reversed(range(num_blocks)):
            in_c = channels[i] * 2 if i < num_blocks - 1 else channels[i] # For skip connection
            out_c = channels[i]
            self.up_blocks.append(nn.ModuleList([
                ResBlock(in_c, out_c, time_emb_dim),
                ResBlock(out_c, out_c, time_emb_dim),
                AttentionBlock(out_c) if out_c == unet_config["cross_attention_dim"] else nn.Identity(),
                nn.ConvTranspose2d(out_c, out_c, 4, stride=2, padding=1) if i > 0 else nn.Identity()
            ]))

        # Final convolution
        self.conv_out = nn.Conv2d(channels[0], self.out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_conditioning: Optional[torch.Tensor] = None):
        # x: latent input (N, C, H, W)
        # t: timestep (N,)
        # text_conditioning: (N, L, D) where L is sequence length, D is embedding dim
        
        time_emb = self.time_embedder(t)
        
        # Initial convolution
        x = self.conv_in(x)

        skips = []
        # Downsampling path
        for block_list in self.down_blocks:
            res1, res2, attn, downsample = block_list
            x = res1(x, time_emb)
            x = res2(x, time_emb)
            x = attn(x) # Add text conditioning here if needed
            skips.append(x)
            x = downsample(x)

        # Mid path
        for block in self.mid_block:
            if isinstance(block, ResBlock):
                x = block(x, time_emb)
            else: # AttentionBlock
                x = block(x) # Add text conditioning here if needed

        # Upsampling path
        for i, block_list in enumerate(self.up_blocks):
            res1, res2, attn, upsample = block_list
            x = torch.cat([x, skips.pop()], dim=1) # Skip connection
            x = res1(x, time_emb)
            x = res2(x, time_emb)
            x = attn(x) # Add text conditioning here if needed
            x = upsample(x)

        # Final convolution
        return self.conv_out(x)

class RewardModel(nn.Module):
    """
    Placeholder for a Reward Model (e.g., ImageReward).
    In a real scenario, this would be a pre-trained model like ImageReward.
    For reproduction purposes, we'll create a dummy model that returns a scalar.
    """
    def __init__(self):
        super().__init__()
        self.dummy_linear = nn.Linear(512, 1) # Assuming input features of 512, output scalar reward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x would typically be a processed image / latent representation
        # For a dummy, we'll just flatten and pass through linear
        return self.dummy_linear(x.mean(dim=[-1, -2])) # Simple reduction for dummy

class FlowMatchingModel(nn.Module):
    """
    Combines the U-Net as the velocity field predictor.
    The forward pass will return the predicted velocity field v(x,t).
    """
    def __init__(self, config: Config):
        super().__init__()
        self.unet = UNet(config)

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_conditioning: Optional[torch.Tensor] = None):
        # The U-Net directly predicts the velocity field v(x,t)
        return self.unet(x, t, text_conditioning)

