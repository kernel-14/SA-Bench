## vae.py
"""
3D Causal Variational Autoencoder (VAE) for video compression.

Architecture inspired by MAGVIT-v2 but with KL-regularized continuous latent space.
The encoder uses causal 3D convolutions to ensure no information leakage from future frames.
The decoder reconstructs the full video from the latent code.

Usage:
    vae = ThreeDVAE(cfg)
    recon, mu, logvar = vae(video_tensor)
    loss_dict = vae.compute_loss(recon, video_tensor, mu, logvar, perceptual_loss_fn)
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv3d(nn.Module):
    """
    3D convolution with temporal causality: the operation at time t depends only on
    inputs at times <= t. Achieved by asymmetric padding (all padding on the left/time-before
    side) and zero padding on the right.
    
    Supports standard stride (1 or 2) while maintaining the causal property.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int, int],
        stride: int | Tuple[int, int, int] = 1,
        bias: bool = False,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kt = kh = kw = kernel_size
        else:
            kt, kh, kw = kernel_size

        if isinstance(stride, int):
            st = sh = sw = stride
        else:
            st, sh, sw = stride

        # Padding order for F.pad: (left, right, top, bottom, front, back)
        # front/back correspond to depth (time) dimension
        self.pad = (kw // 2, kw // 2, kh // 2, kh // 2, kt - 1, 0)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.pad, mode="constant", value=0)
        return self.conv(x)


class ResBlock3D(nn.Module):
    """
    Residual block with two 3D causal convolutions, GroupNorm, and SiLU activation.
    """

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        gn_groups = min(num_groups, channels)
        self.norm1 = nn.GroupNorm(gn_groups, channels)
        self.act1 = nn.SiLU()
        self.conv1 = CausalConv3d(channels, channels, kernel_size=3, stride=1)

        self.norm2 = nn.GroupNorm(gn_groups, channels)
        self.act2 = nn.SiLU()
        self.conv2 = CausalConv3d(channels, channels, kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x + residual


class ThreeDVAE(nn.Module):
    """
    3D Causal VAE with 8x8x8 spatial-temporal compression.

    Expects input of shape (B, C, T, H, W) and produces a latent code of shape
    (B, latent_channels, T//8, H//8, W//8). Supports KL regularization.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        self.input_channels = cfg["input_channels"]
        self.latent_channels = cfg["latent_channels"]
        self.encoder_channels = cfg["encoder_channels"]
        num_res_blocks = cfg["num_res_blocks"]
        self.kl_weight = cfg.get("kl_weight", 1e-6)
        self.perceptual_weight = cfg.get("perceptual_loss_weight", 1.0)

        # ---------- Encoder ----------
        # Initial convolution (no spatial or temporal downsampling)
        self.enc_conv_in = CausalConv3d(self.input_channels, self.encoder_channels[0], kernel_size=3, stride=1)

        # For each resolution level: stack of ResBlocks, then (optionally) a downsampling layer.
        self.enc_level_blocks = nn.ModuleList()
        self.enc_downs = nn.ModuleList()
        for i, ch in enumerate(self.encoder_channels):
            blocks = nn.ModuleList([ResBlock3D(ch) for _ in range(num_res_blocks)])
            self.enc_level_blocks.append(blocks)
            if i < len(self.encoder_channels) - 1:
                next_ch = self.encoder_channels[i + 1]
                self.enc_downs.append(CausalConv3d(ch, next_ch, kernel_size=3, stride=2))
            else:
                self.enc_downs.append(None)

        # Final projection to mean and logvar
        self.enc_norm = nn.GroupNorm(min(32, self.encoder_channels[-1]), self.encoder_channels[-1])
        self.enc_act = nn.SiLU()
        self.conv_mu = CausalConv3d(self.encoder_channels[-1], self.latent_channels, kernel_size=3, stride=1)
        self.conv_logvar = CausalConv3d(self.encoder_channels[-1], self.latent_channels, kernel_size=3, stride=1)

        # ---------- Decoder ----------
        decoder_layers = []
        # Start from the compressed latent
        decoder_layers.append(CausalConv3d(self.latent_channels, self.encoder_channels[-1], kernel_size=3, stride=1))

        # Process each level in reverse order
        for idx in range(len(self.encoder_channels) - 1, -1, -1):
            current_ch = self.encoder_channels[idx]
            # Upsampling to the previous (finer) level
            if idx > 0:
                prev_ch = self.encoder_channels[idx - 1]
                decoder_layers.append(
                    nn.ConvTranspose3d(
                        current_ch,
                        prev_ch,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    )
                )
                active_ch = prev_ch
            else:
                active_ch = current_ch

            # Residual blocks at this resolution
            for _ in range(num_res_blocks):
                decoder_layers.append(ResBlock3D(active_ch))

        # Final output projection to RGB/video space
        decoder_layers.append(nn.GroupNorm(min(32, self.encoder_channels[0]), self.encoder_channels[0]))
        decoder_layers.append(nn.SiLU())
        decoder_layers.append(CausalConv3d(self.encoder_channels[0], self.input_channels, kernel_size=3, stride=1))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a video into latent distribution parameters.

        Args:
            video: Tensor of shape (B, C, T, H, W).

        Returns:
            mu, logvar: each of shape (B, latent_channels, T//8, H//8, W//8)
        """
        x = self.enc_conv_in(video)
        for i, blocks in enumerate(self.enc_level_blocks):
            for block in blocks:
                x = block(x)
            if self.enc_downs[i] is not None:
                x = self.enc_downs[i](x)
        x = self.enc_norm(x)
        x = self.enc_act(x)
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * eps.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent code to a reconstructed video.

        Args:
            latent: Tensor of shape (B, latent_channels, T_l, H_l, W_l).

        Returns:
            Reconstructed video of shape (B, C, T, H, W).
        """
        return self.decoder(latent)

    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode -> reparameterize -> decode.

        Returns:
            recon: reconstructed video
            mu: posterior mean
            logvar: posterior log variance
        """
        mu, logvar = self.encode(video)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def compute_loss(
        self,
        recon: torch.Tensor,
        video: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        perceptual_loss_fn: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the total VAE loss.

        Args:
            recon: Reconstructed video, same shape as video.
            video: Original video.
            mu, logvar: Latent distribution parameters.
            perceptual_loss_fn: Optional LPIPS module. If None, perceptual loss is skipped.

        Returns:
            Dictionary with keys 'loss', 'recon_loss', 'kl_loss', 'perceptual_loss'.
        """
        mse = F.mse_loss(recon, video)

        # KL divergence averaged over batch and all dimensions except channels?
        # The paper uses sum over latent dims, mean over batch. Standard: 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar) averaged.
        kl = 0.5 * torch.mean(
            torch.sum(
                torch.exp(logvar) + mu.pow(2) - 1.0 - logvar,
                dim=[1, 2, 3, 4],
            )
        )

        total_loss = mse + self.kl_weight * kl

        if perceptual_loss_fn is not None and self.perceptual_weight > 0:
            # LPIPS expects 4D tensors (B, C, H, W). Process frames independently.
            B, C, T, H, W = recon.shape
            recon_frames = recon.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            video_frames = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            p_loss = perceptual_loss_fn(recon_frames, video_frames).mean()
            total_loss = total_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor(0.0, device=recon.device)

        return {
            "loss": total_loss,
            "recon_loss": mse,
            "kl_loss": kl,
            "perceptual_loss": p_loss,
        }

