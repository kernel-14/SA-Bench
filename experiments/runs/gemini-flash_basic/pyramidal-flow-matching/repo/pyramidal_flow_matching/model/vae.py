import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, 
                 in_channels: int = 3,
                 out_channels: int = 3,
                 down_block_types: tuple = ("DownEncoderBlock3D", "DownEncoderBlock3D", "DownEncoderBlock3D", "DownEncoderBlock3D"),
                 up_block_types: tuple = ("UpDecoderBlock3D", "UpDecoderBlock3D", "UpDecoderBlock3D", "UpDecoderBlock3D"),
                 block_out_channels: tuple = (128, 256, 512, 512),
                 latent_channels: int = 4,
                 sample_size: int = 512,
                 downsampling_ratio: tuple = (8, 8, 8)): # Spatiotemporal downsampling ratio
        super().__init__()
        self.downsampling_ratio = downsampling_ratio
        self.latent_channels = latent_channels
        
        # Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            latent_channels=latent_channels,
            sample_size=sample_size
        )

        # Decoder
        self.decoder = Decoder(
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            latent_channels=latent_channels,
            sample_size=sample_size
        )

    def encode(self, x: torch.Tensor) -> "DiagonalGaussianDistribution":
        # Encode input video to latent space
        # The output should be a distribution from which we can sample
        h = self.encoder(x)
        # A 1x1x1 convolution to map the encoder output to mean and logvar of the latent distribution
        moments = nn.Conv3d(h.shape[1], 2 * self.latent_channels, kernel_size=1)(h)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Decode latent representation back to video pixels
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior = self.encode(x)
        z = posterior.sample()
        recon = self.decode(z)
        # KL divergence and negative log-likelihood are typically computed for VAE training
        return recon, posterior.kl().mean(), posterior.nll(recon).mean()

class Encoder(nn.Module):
    def __init__(self, in_channels, down_block_types, block_out_channels, latent_channels, sample_size):
        super().__init__()
        # Based on typical VAE Encoder structures, similar to Stable Diffusion's VAE
        self.conv_in = nn.Conv3d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)
        
        self.down_blocks = nn.ModuleList()
        current_channels = block_out_channels[0]
        for i in range(len(down_block_types)):
            next_channels = block_out_channels[i]
            is_last_block = i == len(down_block_types) - 1
            self.down_blocks.append(
                DownEncoderBlock3D(current_channels, next_channels, has_downsample=not is_last_block)
            )
            current_channels = next_channels
        
        self.mid_block = UNetMidBlock3D(block_out_channels[-1])

        self.conv_norm_out = nn.GroupNorm(32, block_out_channels[-1])
        self.conv_act = nn.SiLU()
        # The final conv_out maps to the latent features before splitting into mean/logvar
        self.conv_out = nn.Conv3d(block_out_channels[-1], block_out_channels[-1], kernel_size=1) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)

        for down_block in self.down_blocks:
            x = down_block(x)
        
        x = self.mid_block(x)

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x

class Decoder(nn.Module):
    def __init__(self, out_channels, up_block_types, block_out_channels, latent_channels, sample_size):
        super().__init__()
        # Based on typical VAE Decoder structures
        self.conv_in = nn.Conv3d(latent_channels, block_out_channels[-1], kernel_size=1)

        self.mid_block = UNetMidBlock3D(block_out_channels[-1])
        
        self.up_blocks = nn.ModuleList()
        reversed_block_out_channels = list(reversed(block_out_channels))
        current_channels = reversed_block_out_channels[0]
        for i in range(len(up_block_types)):
            next_channels = reversed_block_out_channels[min(i + 1, len(reversed_block_out_channels) - 1)]
            is_first_block = i == 0 # The first up block does not upsample
            self.up_blocks.append(
                UpDecoderBlock3D(current_channels, next_channels, has_upsample=not is_first_block)
            )
            current_channels = next_channels
        
        self.conv_norm_out = nn.GroupNorm(32, block_out_channels[0])
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)

        x = self.mid_block(x)
        
        for up_block in self.up_blocks:
            x = up_block(x)

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x

# Dummy Blocks and Distribution for VAE (simplified for conceptual replication)
class DownEncoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, has_downsample=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU()
        )
        
        if has_downsample:
            self.downsample = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        x = self.block(x)
        return self.downsample(x)

class UpDecoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, has_upsample=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU()
        )

        if has_upsample:
            # Adjust ConvTranspose3d to correctly map input channels to output channels for upsampling
            self.upsample = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.upsample = nn.Identity()

    def forward(self, x):
        if hasattr(self, 'upsample') and not isinstance(self.upsample, nn.Identity):
            x = self.upsample(x)
        x = self.block(x)
        return x

class UNetMidBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.block(x)


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0) # Clamp logvar for numerical stability
        self.std = torch.exp(0.5 * self.logvar)
        self.variance = torch.exp(self.logvar)

    def sample(self) -> torch.Tensor:
        # Sample from the distribution (reparameterization trick)
        return self.mean + self.std * torch.randn(self.mean.shape, device=self.mean.device)

    def kl(self) -> torch.Tensor:
        # KL divergence with a standard Gaussian
        # KL = 0.5 * sum(exp(logvar) + mean^2 - 1 - logvar)
        return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.variance - 1.0 - self.logvar, dim=[1, 2, 3, 4])

    def nll(self, sample: torch.Tensor) -> torch.Tensor:
        # Negative log-likelihood of the sample under the distribution
        # NLL = 0.5 * sum(log(2*pi*variance) + (sample - mean)^2 / variance)
        log_prob = -0.5 * (torch.log(2 * torch.pi * self.variance) + torch.pow(sample - self.mean, 2) / self.variance)
        return -torch.sum(log_prob, dim=[1, 2, 3, 4])
