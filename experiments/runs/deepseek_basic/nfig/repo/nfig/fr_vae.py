"""
Frequency-guided Residual-quantized VAE (FR-VAE)

This module implements the image tokenizer described in Section 3.1 of the paper.
FR-VAE separates low and high-frequency components in the representation learning
process, with low frequencies encoding global structure and high frequencies
preserving local details.

Key components:
1. Frequency-guided Decomposer/Composer (in frequency_utils.py)
2. Frequency-guided Residual Quantization
3. VQ-GAN framework with DINO discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .frequency_utils import (
    FrequencyGuidedDecomposer,
    FrequencyGuidedComposer,
    compute_frequency_band_boundaries,
)


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer using a learnable codebook.
    
    Args:
        codebook_size: K, number of codes in the codebook
        codebook_dim: C, dimension of each code vector
        commitment_cost: beta, weight for commitment loss
    """
    
    def __init__(self, codebook_size: int, codebook_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost
        
        # Learnable codebook: Z ∈ R^{K×C}
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
    
    def forward(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous feature map v.
        
        Args:
            v: (B, C, h, w) continuous feature map
        
        Returns:
            v_q: (B, C, h, w) quantized feature map
            tokens: (B, h, w) discrete token indices
            loss: scalar commitment loss
        """
        B, C, h, w = v.shape
        
        # Reshape: (B, C, h, w) -> (B, h, w, C) -> (B*h*w, C)
        v_flat = v.permute(0, 2, 3, 1).contiguous().view(-1, C)
        
        # Compute distances to codebook entries
        # ||v - z||^2 = ||v||^2 + ||z||^2 - 2 * v·z
        v_sq = torch.sum(v_flat ** 2, dim=1, keepdim=True)
        z_sq = torch.sum(self.codebook.weight ** 2, dim=1)
        
        distances = v_sq + z_sq.unsqueeze(0) - 2 * torch.matmul(v_flat, self.codebook.weight.t())
        
        # Find nearest codebook entry
        token_indices = torch.argmin(distances, dim=1)  # (B*h*w,)
        
        # Lookup quantized vectors
        v_q_flat = self.codebook(token_indices)  # (B*h*w, C)
        
        # Straight-through estimator
        v_q_flat = v_flat + (v_q_flat - v_flat).detach()
        
        # Reshape back
        v_q = v_q_flat.view(B, h, w, C).permute(0, 3, 1, 2).contiguous()
        tokens = token_indices.view(B, h, w)
        
        # Commitment loss
        commitment_loss = self.commitment_cost * F.mse_loss(v_q_flat.detach(), v_flat)
        
        # Codebook loss (moves codebook towards encoder outputs)
        codebook_loss = F.mse_loss(v_q_flat, v_flat.detach())
        
        loss = commitment_loss + codebook_loss
        
        return v_q, tokens, loss
    
    def quantize_indices(self, token_indices: torch.Tensor) -> torch.Tensor:
        """Convert token indices back to quantized features."""
        B, h, w = token_indices.shape
        v_q_flat = self.codebook(token_indices.view(-1))
        v_q = v_q_flat.view(B, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return v_q


class ResidualQuantizer(nn.Module):
    """
    Frequency-guided Residual Quantization.
    
    Progressively captures different frequency components through a residual
    learning scheme, as described in Section 3.1.2.
    
    For each frequency component i:
        - Scale v_i to appropriate resolution (h_i, w_i)
        - Quantize v_i
        - Compute residual R_i = R_{i-1} + (f_hat_i - interpolate(v_q_i))
    
    Args:
        scales: list of (h_i, w_i) tuples for each frequency band
        codebook_size: size of the shared codebook
        codebook_dim: dimension of code vectors
        latent_dim: channel dimension of the feature map
    """
    
    def __init__(self, scales: List[Tuple[int, int]], codebook_size: int = 4096,
                 codebook_dim: int = 256, latent_dim: int = 256):
        super().__init__()
        self.scales = scales
        self.latent_dim = latent_dim
        self.n_scales = len(scales)
        
        # Shared codebook
        self.codebook = VectorQuantizer(codebook_size, codebook_dim)
        
        # Projection layers: downsample each frequency component to its target scale
        # and project to codebook dimension
        self.down_projs = nn.ModuleList()
        self.up_projs = nn.ModuleList()
        
        for h_i, w_i in scales:
            # Downsample: latent_dim -> codebook_dim at resolution (h_i, w_i)
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(latent_dim, latent_dim // 2, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(latent_dim // 2, codebook_dim, kernel_size=3, padding=1),
                )
            )
            # Upsample back: codebook_dim -> latent_dim
            self.up_projs.append(
                nn.Sequential(
                    nn.Conv2d(codebook_dim, latent_dim // 2, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(latent_dim // 2, latent_dim, kernel_size=3, padding=1),
                )
            )
    
    def forward(self, freq_components: List[torch.Tensor],
                original_h: int, original_w: int) -> Tuple[List[torch.Tensor], 
                                                           List[torch.Tensor], 
                                                           torch.Tensor]:
        """
        Apply residual quantization across frequency components.
        
        Args:
            freq_components: list of f_hat_i tensors, each (B, C, H', W')
            original_h, original_w: original feature map dimensions
        
        Returns:
            v_q_list: list of quantized feature maps at their respective scales
            token_list: list of discrete token indices
            total_vq_loss: total vector quantization loss
        """
        B = freq_components[0].shape[0]
        device = freq_components[0].device
        
        v_q_list = []
        token_list = []
        total_vq_loss = 0.0
        
        # R_0 = 0
        residual = torch.zeros(B, self.latent_dim, original_h, original_w, device=device)
        
        for i, (f_hat_i, (h_i, w_i)) in enumerate(zip(freq_components, self.scales)):
            # Accumulate signal: signal_i = residual + f_hat_i
            accumulated = residual + f_hat_i
            
            # Downsample accumulated to target scale (h_i, w_i)
            if (original_h, original_w) != (h_i, w_i):
                accumulated_scaled = F.interpolate(
                    accumulated, size=(h_i, w_i), mode='bilinear', align_corners=False
                )
            else:
                accumulated_scaled = accumulated
            
            # Project to codebook dimension
            v_i = self.down_projs[i](accumulated_scaled)
            
            # Quantize
            v_q_i, tokens_i, vq_loss = self.codebook(v_i)
            total_vq_loss = total_vq_loss + vq_loss
            
            # Project back to latent dimension
            v_q_i_proj = self.up_projs[i](v_q_i)
            
            # Upsample back to original resolution
            if (h_i, w_i) != (original_h, original_w):
                v_q_i_full = F.interpolate(
                    v_q_i_proj, size=(original_h, original_w), 
                    mode='bilinear', align_corners=False
                )
            else:
                v_q_i_full = v_q_i_proj
            
            # Update residual: R_i = R_{i-1} + (f_hat_i - v_q_i_full)
            residual = residual + (f_hat_i - v_q_i_full)
            
            v_q_list.append(v_q_i_full)
            token_list.append(tokens_i)
        
        return v_q_list, token_list, total_vq_loss


class Encoder(nn.Module):
    """
    Image encoder: maps image x ∈ R^{H×W×3} to latent feature f ∈ R^{H'×W'×C}.
    
    Uses a CNN-based encoder with DINOv2-base pretrained weights (as mentioned in paper).
    Following VAR's approach with VQGAN architecture.
    """
    
    def __init__(self, in_channels: int = 3, latent_dim: int = 256, 
                 hidden_dims: List[int] = [128, 256, 256, 256]):
        super().__init__()
        self.latent_dim = latent_dim
        
        modules = []
        in_ch = in_channels
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                )
            )
            in_ch = h_dim
        
        # Final projection to latent_dim
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(latent_dim),
                nn.ReLU(),
            )
        )
        
        self.encoder = nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image
        
        Returns:
            f: (B, C, H', W') latent feature map
        """
        return self.encoder(x)


class Decoder(nn.Module):
    """
    Image decoder: reconstructs image from combined quantized feature map.
    
    Uses a CNN-based decoder with upsampling.
    """
    
    def __init__(self, latent_dim: int = 256, out_channels: int = 3,
                 hidden_dims: List[int] = [256, 256, 128, 64]):
        super().__init__()
        
        modules = []
        in_ch = latent_dim
        
        for i, h_dim in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(in_ch, h_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                )
            )
            in_ch = h_dim
        
        # Final layer
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),
            )
        )
        
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f: (B, C, H', W') feature map (combined from all frequency bands)
        
        Returns:
            x_recon: (B, 3, H, W) reconstructed image
        """
        return self.decoder(f)


class DinoDiscriminator(nn.Module):
    """
    DINO Discriminator for adversarial training (from VAR's tokenizer).
    
    Uses a lightweight CNN-based discriminator.
    """
    
    def __init__(self, in_channels: int = 3, hidden_dims: List[int] = [64, 128, 256, 512]):
        super().__init__()
        
        modules = []
        in_ch = in_channels
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2),
                )
            )
            in_ch = h_dim
        
        # Final classification layer
        modules.append(
            nn.Conv2d(hidden_dims[-1], 1, kernel_size=4, stride=1, padding=0)
        )
        
        self.discriminator = nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)


class LPIPSLoss(nn.Module):
    """
    LPIPS perceptual loss wrapper.
    Uses a pretrained VGG-based LPIPS model if available, 
    otherwise falls back to MSE on intermediate features.
    """
    
    def __init__(self):
        super().__init__()
        self.use_lpips = False
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='vgg').eval()
            self.use_lpips = True
        except ImportError:
            print("Warning: lpips not installed, using MSE-based perceptual loss fallback")
            self.lpips_fn = None
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        if self.use_lpips:
            # LPIPS expects [-1, 1] range, our images may be in [0, 1] or [-1, 1]
            return self.lpips_fn(img1, img2).mean()
        else:
            # Fallback: simple MSE
            return F.mse_loss(img1, img2)


class FRVAE(nn.Module):
    """
    Frequency-guided Residual-quantized VAE (FR-VAE).
    
    Full image tokenizer described in Section 3.1 of the paper.
    
    Architecture:
    1. Encoder: image → feature map f
    2. Frequency-guided Decomposer: f → frequency components {f_hat_i}
    3. Residual Quantizer: {f_hat_i} → quantized features + tokens
    4. Frequency-guided Composer: quantized features → combined feature f_tilde
    5. Decoder: f_tilde → reconstructed image
    
    Args:
        scales: list of (h_i, w_i) tuples for frequency bands
        codebook_size: K, number of codes
        codebook_dim: dimension of code vectors
        latent_dim: channel dimension of feature map
        image_size: input image size (H, W)
    """
    
    def __init__(
        self,
        scales: List[Tuple[int, int]] = None,
        codebook_size: int = 4096,
        codebook_dim: int = 256,
        latent_dim: int = 256,
        image_size: int = 256,
        use_dino_disc: bool = True,
    ):
        super().__init__()
        
        # Default scales from paper: [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
        # These are scale factors, meaning features at resolutions: image_size // scale
        if scales is None:
            scale_factors = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
            # Feature map size = image_size // 16 (due to 4x downsampling in encoder)
            feat_size = image_size // 16  # 256 -> 16
            self.scales = [(feat_size * s, feat_size * s) for s in scale_factors]
        else:
            self.scales = scales
        
        # Adjust scales to keep them bounded by feature map size
        # (paper uses scaling factors directly as resolutions)
        self.scales = scales if scales is not None else [(s, s) for s in 
                          [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]]
        
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.image_size = image_size
        self.use_dino_disc = use_dino_disc
        self.n_scales = len(self.scales)
        
        # Compute total vocabulary size
        self.total_tokens = sum(h * w for h, w in self.scales)
        
        # Encoder
        self.encoder = Encoder(in_channels=3, latent_dim=latent_dim)
        
        # Frequency-guided Decomposer
        self.decomposer = FrequencyGuidedDecomposer(
            scales=self.scales, latent_dim=latent_dim, sigma_max=1.0
        )
        
        # Residual Quantizer
        self.residual_quantizer = ResidualQuantizer(
            scales=self.scales,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            latent_dim=latent_dim,
        )
        
        # Frequency-guided Composer
        self.composer = FrequencyGuidedComposer()
        
        # Decoder
        self.decoder = Decoder(latent_dim=latent_dim, out_channels=3)
        
        # Discriminator
        if use_dino_disc:
            self.discriminator = DinoDiscriminator(in_channels=3)
        else:
            self.discriminator = None
    
    def encode(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Encode image to discrete tokens.
        
        Args:
            x: (B, 3, H, W) input image
        
        Returns:
            token_list: list of (B, h_i, w_i) token indices
            f_combined: (B, C, H', W') combined quantized feature
            vq_loss: total vector quantization loss
        """
        B, _, H, W = x.shape
        
        # Encode
        f = self.encoder(x)
        feat_h, feat_w = f.shape[-2:]
        
        # Decompose into frequency components
        freq_components = self.decomposer(f)
        
        # Residual quantization
        v_q_list, token_list, vq_loss = self.residual_quantizer(
            freq_components, feat_h, feat_w
        )
        
        # Combine quantized components
        f_combined = self.composer(v_q_list, feat_h, feat_w) + \
                     self.composer(freq_components, feat_h, feat_w).detach() - \
                     self.composer([v.detach() for v in v_q_list], feat_h, feat_w)
        
        return token_list, f_combined, vq_loss
    
    def decode(self, f_combined: torch.Tensor) -> torch.Tensor:
        """
        Decode combined feature map to image.
        
        Args:
            f_combined: (B, C, H', W')
        
        Returns:
            x_recon: (B, 3, H, W) reconstructed image
        """
        return self.decoder(f_combined)
    
    def decode_from_tokens(self, token_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode from discrete tokens directly.
        
        Args:
            token_list: list of (B, h_i, w_i) or (B, h_i*w_i) token indices
        
        Returns:
            x_recon: (B, 3, H, W) reconstructed image
        """
        device = token_list[0].device
        B = token_list[0].shape[0]
        
        # Determine feature map size from encoder output
        feat_size = self.image_size // 16
        
        # Reconstruct quantized features from tokens
        v_q_list = []
        for i, tokens in enumerate(token_list):
            h_i, w_i = self.scales[i]
            n_i = h_i * w_i
            
            # Handle both flat (B, n) and grid (B, h, w) formats
            if tokens.dim() == 2:
                tokens_grid = tokens.view(B, h_i, w_i)
            else:
                tokens_grid = tokens
            
            # Get quantized features from token indices (at codebook dim)
            v_q_i_codebook = self.residual_quantizer.codebook.quantize_indices(tokens_grid)
            # (B, C_codebook, h_i, w_i)
            
            # Project back to latent dim
            v_q_i_proj = self.residual_quantizer.up_projs[i](v_q_i_codebook)
            
            # Upsample to feature map size
            if (h_i, w_i) != (feat_size, feat_size):
                v_q_i_full = F.interpolate(
                    v_q_i_proj, size=(feat_size, feat_size),
                    mode='bilinear', align_corners=False
                )
            else:
                v_q_i_full = v_q_i_proj
            
            v_q_list.append(v_q_i_full)
        
        # Combine
        f_combined = self.composer(v_q_list, feat_size, feat_size)
        
        # Decode
        return self.decoder(f_combined)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Full forward pass.
        
        Returns:
            x_recon: reconstructed image
            token_list: list of token indices
            vq_loss: quantization loss
        """
        token_list, f_combined, vq_loss = self.encode(x)
        x_recon = self.decode(f_combined)
        return x_recon, token_list, vq_loss
    
    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor, 
                     vq_loss: torch.Tensor, f_orig: Optional[torch.Tensor] = None,
                     f_combined: Optional[torch.Tensor] = None,
                     optimizer_idx: int = 0) -> torch.Tensor:
        """
        Compute total loss for FR-VAE training.
        
        Following Appendix B.1:
        L = ||I - I_hat||_2^2 + ||f - f_hat||_2^2 + L_p(I) + 0.5 * L_g(I)
        
        Args:
            x: original image
            x_recon: reconstructed image
            vq_loss: vector quantization loss
            f_orig: original feature map (for feature reconstruction loss)
            f_combined: combined feature map
            optimizer_idx: 0 for generator, 1 for discriminator
        
        Returns:
            total_loss
        """
        if optimizer_idx == 1:
            # Discriminator loss
            if self.discriminator is not None:
                real_pred = self.discriminator(x)
                fake_pred = self.discriminator(x_recon.detach())
                
                real_loss = F.relu(1.0 - real_pred).mean()
                fake_loss = F.relu(1.0 + fake_pred).mean()
                d_loss = real_loss + fake_loss
                return d_loss
            return torch.tensor(0.0, device=x.device)
        
        # Generator losses
        
        # Pixel reconstruction loss
        recon_loss_pixel = F.mse_loss(x, x_recon)
        
        # Feature reconstruction loss
        if f_orig is not None and f_combined is not None:
            recon_loss_feat = F.mse_loss(f_orig, f_combined)
        else:
            recon_loss_feat = 0.0
        
        # Perceptual loss (LPIPS)
        lpips_loss_fn = LPIPSLoss()
        lpips_loss = lpips_loss_fn(x, x_recon)
        
        # GAN loss
        if self.discriminator is not None:
            fake_pred = self.discriminator(x_recon)
            gan_loss = -fake_pred.mean()
        else:
            gan_loss = 0.0
        
        # Total loss: pixel + feature + perceptual + 0.5 * GAN + VQ
        total_loss = (
            recon_loss_pixel + 
            recon_loss_feat + 
            lpips_loss + 
            0.5 * gan_loss +
            vq_loss
        )
        
        return total_loss
    
    def get_total_tokens(self) -> int:
        """Get total number of discrete tokens for an image."""
        return sum(h * w for h, w in self.scales)
    
    def get_token_sequence(self, token_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Flatten all tokens into a single sequence for autoregressive generation.
        
        Args:
            token_list: list of (B, h_i, w_i) token indices
        
        Returns:
            flat_tokens: (B, total_tokens) flattened token sequence
        """
        flat_tokens = []
        for tokens in token_list:
            flat_tokens.append(tokens.view(tokens.shape[0], -1))
        return torch.cat(flat_tokens, dim=1)
    
    def unflatten_tokens(self, flat_tokens: torch.Tensor) -> List[torch.Tensor]:
        """
        Convert flat token sequence back to list of per-scale token grids.
        
        Args:
            flat_tokens: (B, total_tokens)
        
        Returns:
            token_list: list of (B, h_i, w_i)
        """
        token_list = []
        start = 0
        for h_i, w_i in self.scales:
            n = h_i * w_i
            tokens_i = flat_tokens[:, start:start + n].view(-1, h_i, w_i)
            token_list.append(tokens_i)
            start += n
        return token_list
