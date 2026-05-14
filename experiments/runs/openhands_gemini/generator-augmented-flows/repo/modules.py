
import torch
from torch import nn
from layers import SinusoidalPositionalEmbedding, ResidualBlock, Upsample, Downsample, AttentionBlock
from einops import rearrange

class UNet(nn.Module):
    """
    A U-Net like architecture, often referred to as SongUNet or NCSN++ in diffusion models,
    which is a common choice for score-based and consistency models.
    """
    def __init__(
        self,
        image_size,
        in_channels,
        out_channels,
        model_channels,
        num_blocks, # Can be a single int or a list for different resolutions
        channel_multiplicative_factor,
        attn_resolutions,
        dropout=0.0,
        num_res_blocks=2, # Number of residual blocks per resolution level
        time_emb_dim=256
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.dropout = dropout

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        dims = [model_channels]
        curr_dim = model_channels
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(channel_multiplicative_factor)

        # Downsampling path
        for i in range(num_resolutions):
            mult = channel_multiplicative_factor[i]
            if isinstance(num_blocks, list):
                num_blocks_at_res = num_blocks[i]
            else:
                num_blocks_at_res = num_blocks

            for _ in range(num_blocks_at_res):
                self.downs.append(nn.ModuleList([
                    ResidualBlock(curr_dim, curr_dim * mult, time_emb_dim),
                    AttentionBlock(curr_dim * mult) if image_size // (2**i) in attn_resolutions else nn.Identity()
                ]))
                curr_dim = curr_dim * mult
                dims.append(curr_dim)
            
            if i < num_resolutions - 1:
                self.downs.append(nn.ModuleList([
                    Downsample(curr_dim)
                ]))
                dims.append(curr_dim)

        # Bottleneck
        self.mid_block1 = ResidualBlock(curr_dim, curr_dim, time_emb_dim)
        self.mid_attn = AttentionBlock(curr_dim)
        self.mid_block2 = ResidualBlock(curr_dim, curr_dim, time_emb_dim)

        # Upsampling path
        for i in reversed(range(num_resolutions)):
            mult = channel_multiplicative_factor[i]
            if isinstance(num_blocks, list):
                num_blocks_at_res = num_blocks[i]
            else:
                num_blocks_at_res = num_blocks

            for _ in range(num_blocks_at_res):
                self.ups.append(nn.ModuleList([
                    ResidualBlock(curr_dim + dims.pop(), curr_dim // mult, time_emb_dim),
                    AttentionBlock(curr_dim // mult) if image_size // (2**i) in attn_resolutions else nn.Identity()
                ]))
                curr_dim = curr_dim // mult
            
            if i > 0:
                self.ups.append(nn.ModuleList([
                    Upsample(curr_dim)
                ]))
                dims.pop() # Remove the last dim as it was for the downsample before upsample

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, time):
        # x: (batch_size, in_channels, H, W)
        # time: (batch_size,)

        t = self.time_mlp(time)
        h = self.init_conv(x)
        hs = [h]

        # Downsampling
        for module in self.downs:
            if isinstance(module, nn.ModuleList):
                # ResidualBlock and AttentionBlock
                h = module[0](h, t)
                h = module[1](h)
            else:
                # Downsample
                h = module(h)
            hs.append(h)

        # Bottleneck
        h = self.mid_block1(h, t)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t)

        # Upsampling
        for module in self.ups:
            if isinstance(module, nn.ModuleList):
                # ResidualBlock and AttentionBlock
                h = torch.cat([h, hs.pop()], dim=1) # Skip connection
                h = module[0](h, t)
                h = module[1](h)
            else:
                # Upsample
                h = module(h)

        return self.final_conv(h)


