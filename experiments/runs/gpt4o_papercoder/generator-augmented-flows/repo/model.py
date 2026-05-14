## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings for encoding noise levels (sigma_t).
    """
    def __init__(self, embed_dim: int):
        """
        Initializes the positional embedding module.

        Args:
            embed_dim (int): Embedding dimension for sinusoidal positional encoding.
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, sigma: torch.Tensor):
        """
        Computes sinusoidal embeddings for input sigma values.

        Args:
            sigma (torch.Tensor): Noise levels (shape: [batch_size, 1]).

        Returns:
            torch.Tensor: Sinusoidal embeddings (shape: [batch_size, embed_dim]).
        """
        half_dim = self.embed_dim // 2
        emb = torch.exp(-torch.arange(half_dim, dtype=torch.float32).to(sigma.device) * (torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = sigma.unsqueeze(1) * emb
        return torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)


class ResidualBlock(nn.Module):
    """
    Basic residual block with two convolutional layers.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        """
        Initializes the ResidualBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout (float): Dropout rate.
        """
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class UNet(nn.Module):
    """
    UNet-like architecture based on NCSN++ with configurable blocks and dropout.
    """
    def __init__(self, channels: int, num_blocks: int, dropout: float):
        """
        Initializes the UNet.

        Args:
            channels (int): Initial number of channels.
            num_blocks (int): Number of residual blocks per stage.
            dropout (float): Dropout rate for residual blocks.
        """
        super(UNet, self).__init__()
        ch_mult = [1, 2, 2]  # Channel multiplier for downsampling and upsampling stages
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.bottleneck = ResidualBlock(channels * ch_mult[-1], channels * ch_mult[-1], dropout)

        in_ch = channels
        # Downsampling path
        for mult in ch_mult:
            out_ch = channels * mult
            self.down.append(ResidualBlock(in_ch, out_ch, dropout))
            self.down.append(nn.AvgPool2d(kernel_size=2))
            in_ch = out_ch

        # Upsampling path
        for mult in reversed(ch_mult):
            out_ch = channels * mult
            self.up.append(ResidualBlock(in_ch, out_ch, dropout))
            self.up.append(nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2))
            in_ch = out_ch

        self.final_layer = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, sigma_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of UNet.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].
            sigma_embedding (torch.Tensor): Positional embedding of sigma.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, channels, height, width].
        """
        skips = []
        for layer in self.down:
            x = layer(x)
            skips.append(x)

        x = self.bottleneck(x)

        for layer in self.up:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x)

        return self.final_layer(x)


class ConsistencyModel(nn.Module):
    """
    Consistency model based on the NCSN++ UNet-like architecture.
    """
    def __init__(self, params: Dict):
        """
        Initializes the ConsistencyModel.

        Args:
            params (Dict): Dictionary containing configuration parameters.
        """
        super(ConsistencyModel, self).__init__()
        # Model configuration
        self.channels = params["model"].get("channels", 128)
        self.dropout = params["model"].get("dropout", 0.0)
        self.num_blocks = params["model"].get("blocks_per_resolution", 3)
        self.embed_dim = 2 * self.channels  # Embedding dimension for sigma

        # Pre-computed variance for skip coefficients
        self.sigma_d = 1.0  # Assume variance is 1.0; may need adjustment for specific datasets

        # Positional embedding for sigma
        self.embedding = PositionalEmbedding(embed_dim=self.embed_dim)

        # UNet backbone
        self.unet = UNet(channels=self.channels, num_blocks=self.num_blocks, dropout=self.dropout)

    def forward(self, x: torch.Tensor, sigma_t: float) -> torch.Tensor:
        """
        Forward pass for the consistency model.

        Args:
            x (torch.Tensor): Noisy input tensor (x_t) of shape [batch_size, channels, height, width].
            sigma_t (float): Noise schedule value (sigma_t).

        Returns:
            torch.Tensor: Endpoint prediction tensor of shape [batch_size, channels, height, width].
        """
        # Convert sigma_t to a tensor if it's not
        if not isinstance(sigma_t, torch.Tensor):
            sigma_t = torch.tensor([sigma_t], dtype=torch.float32, device=x.device).view(-1, 1)

        # Compute positional embedding
        sigma_embedding = self.embedding(sigma_t)

        # Pass through UNet
        unet_output = self.unet(x, sigma_embedding)

        # Compute skip and output coefficients
        c_skip = (self.sigma_d ** 2) / (self.sigma_d ** 2 + (sigma_t - 0.002) ** 2)
        c_out = (self.sigma_d * (sigma_t - 0.002)) / torch.sqrt(self.sigma_d ** 2 + (sigma_t - 0.002) ** 2)

        # Combine skip and output terms
        return c_skip * x + c_out * unet_output
